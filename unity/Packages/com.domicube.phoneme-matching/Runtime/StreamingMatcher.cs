using System;
using System.Collections.Generic;

namespace DomiCube.PhonemeMatching
{
    /// <summary>A confirmed answer.</summary>
    public sealed class StreamingHit
    {
        public MatchResult Result;
        public int Frames;  // frames consumed before confirming
        public int Streak;  // consecutive frames the winner held
        public int RequiredFrames = 2;
        public PronunciationFeedback Feedback; // snapshot of the confirming frame
    }

    /// <summary>
    /// Continuous-listening wrapper around <see cref="Matcher"/>,
    /// mirroring python/runtime/matching/streaming.py.
    ///
    /// Scoring runs many times per session - a 0.4s hop over 10 seconds
    /// with four candidates is ~100 chances to be wrong, so a 1%
    /// per-frame false-accept rate compounds to roughly 63% per session.
    /// Requiring the SAME answer to win several consecutive frames
    /// collapses that: a chance alignment drifts to another candidate on
    /// the next frame, while a real utterance stays in the rolling window
    /// for its whole duration and keeps winning.
    ///
    /// Free of audio and model dependencies: it consumes phoneme lists,
    /// so the caller owns the microphone, the rolling window and the
    /// recogniser.
    /// </summary>
    public sealed class StreamingMatcher
    {
        private string _streakId;
        private int _streak;
        private int _frames;
        private readonly ModelErrorProfile _feedbackErrors;

        public Matcher Matcher { get; }
        public IReadOnlyList<Answer> Candidates { get; }

        /// <summary>
        /// Frames the same answer must win before committing. Must be
        /// reachable given the matcher's context limit - see
        /// <see cref="Matcher.ContextSlice"/>.
        ///
        /// 2 by default. Swept over rolling sessions built from all 36
        /// golden clips: 1 and 2 both detect 18/36, but 2 false-accepts
        /// 16/612 against 19/612; 3 drops to 15/36 detections to save
        /// another 6. Losing a child who did say the word is the worse
        /// failure, so 2. Measured on adult TTS - child speech scores
        /// less steadily and may want a longer streak.
        /// </summary>
        public int Consecutive { get; }

        /// <summary>Frames the current leader has held; 0 if none.</summary>
        public int Streak => _streak;

        public StreamingMatcher(
            Matcher matcher, IReadOnlyList<Answer> candidates,
            int consecutive = 2, ModelErrorProfile feedbackErrors = null)
        {
            if (matcher == null)
            {
                throw new ArgumentNullException(nameof(matcher));
            }

            if (consecutive < 1)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(consecutive), consecutive,
                    "consecutive must be >= 1");
            }

            Matcher = matcher;
            Candidates = candidates ?? new List<Answer>();
            Consecutive = consecutive;
            _feedbackErrors = feedbackErrors;
            foreach (var candidate in Candidates) AnswerPolicy.Validate(candidate);
            Reset();
        }

        /// <summary>Word override; ambiguous or absent IDs use the session default.</summary>
        public int RequiredFrames(string targetId = null)
        {
            if (targetId == null) return Consecutive;
            Answer winner = null;
            foreach (var candidate in Candidates)
            {
                if (candidate.Id != targetId) continue;
                if (winner != null) return Consecutive;
                winner = candidate;
            }
            if (winner == null) return Consecutive;
            AnswerPolicy.Validate(winner);
            return winner.ConfirmationFrames ?? Consecutive;
        }

        private static double ContextOf(double? value)
        {
            return value.HasValue ? value.Value : 0.0;
        }

        /// <summary>
        /// A lone frame that reproduced the target exactly.
        ///
        /// Everything is re-read from the winning candidate on the frame
        /// that would confirm, so nothing approved when this object was
        /// built can go stale if the caller edits a candidate or the
        /// matcher afterwards. Everything is checked against the snapshot
        /// calibration wrote - the matrix it named, its streaming profile,
        /// the streak length, the word and its threshold - never against
        /// whatever matrix this matcher happens to hold. A caller who tuned
        /// skip cost or coverage by hand, or who is running a different
        /// matrix, is outside what was measured, so the escape switches
        /// itself off.
        /// </summary>
        private bool IsExactSingleFrame(MatchResult result)
        {
            if (!result.Passed || result.Score != 1.0 || Consecutive != 2
                || result.TargetId == null)
            {
                return false;
            }

            // An id shared by two candidates cannot say which one won, so
            // neither may borrow the other's approval.
            Answer winner = null;
            foreach (var candidate in Candidates)
            {
                if (candidate != null && string.Equals(
                        candidate.Id, result.TargetId, StringComparison.Ordinal))
                {
                    if (winner != null)
                    {
                        return false;
                    }

                    winner = candidate;
                }
            }

            if (winner == null || winner.ConfirmationFrames.HasValue || !winner.AllowExactSingleFrame)
            {
                return false;
            }

            // The snapshot calibration wrote, not whatever matrix this
            // matcher happens to hold: an approval belongs to one word,
            // scored one way, under one matrix.
            var snapshot = winner.ExactSingleFrameCalibration;
            if (snapshot == null || snapshot.Consecutive != 2
                || !string.Equals(snapshot.Text, winner.Text, StringComparison.Ordinal)
                || snapshot.Threshold != winner.Threshold
                || snapshot.Phonemes == null || snapshot.Phonemes.Count == 0
                || !SamePhonemes(snapshot.Phonemes, winner.Phonemes))
            {
                return false;
            }

            if (Matcher.Mode != MatchMode.Substring || Matcher.Matrix == null)
            {
                return false;
            }

            // Both sides have to name a matrix, and the same one. An
            // anonymous file read through LoadMatrix(json) with no id to
            // fall back on is "unknown", which identifies nothing: it must
            // not pass the comparison, or every unnamed matrix would
            // inherit this word's approval. The shipped child matrix JSON
            // carries "matrix_id", so the one-argument call names itself.
            if (!ConfusionMatrix.NamesAMatrix(Matcher.Matrix.MatrixId)
                || !ConfusionMatrix.NamesAMatrix(snapshot.MatrixId)
                || !string.Equals(Matcher.Matrix.MatrixId, snapshot.MatrixId,
                                  StringComparison.Ordinal))
            {
                return false;
            }

            // All three values, checked here rather than only in the JSON
            // parser: the snapshot fields are public and mutable, and a
            // context multiplier that has gone missing must not read as the
            // 0 an unset matcher would report. Python refuses the same
            // snapshot.
            if (!Measured(snapshot.SkipCost) || !Measured(snapshot.Coverage)
                || !Measured(snapshot.ContextMult))
            {
                return false;
            }

            return Matcher.SkipCost == snapshot.SkipCost.Value
                   && Matcher.Coverage == snapshot.Coverage.Value
                   && ContextOf(Matcher.ContextMult)
                      == ContextOf(snapshot.ContextMult);
        }

        /// <summary>A profile value that is present and a real number.</summary>
        private static bool Measured(double? value)
        {
            return value.HasValue && !double.IsNaN(value.Value)
                   && !double.IsInfinity(value.Value);
        }

        private static bool SamePhonemes(
            IReadOnlyList<string> a, IReadOnlyList<string> b)
        {
            if (a == null || b == null || a.Count != b.Count)
            {
                return false;
            }

            for (int i = 0; i < a.Count; i++)
            {
                if (!string.Equals(a[i], b[i], StringComparison.Ordinal))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>Clear the streak. Call between questions.</summary>
        public void Reset()
        {
            _streakId = null;
            _streak = 0;
            _frames = 0;
        }

        /// <summary>
        /// Score one frame of recognised phonemes; returns a hit once the
        /// streak is long enough.
        ///
        /// The phonemes should come from re-recognising the most recent
        /// audio window, NOT from recognising isolated chunks and
        /// concatenating - wav2vec2 is a context model and mangles short
        /// fragments (measured: 0.5s chunks lost a quarter of the
        /// phonemes and garbled whole words).
        ///
        /// After a hit the streak stays intact, so further pushes keep
        /// returning hits; call <see cref="Reset"/> when moving on.
        /// </summary>
        public StreamingHit Push(IReadOnlyList<string> phonemes,
            IReadOnlyDictionary<string, AcousticEvidence> acousticEvidence = null)
        {
            _frames++;
            var result = Matcher.BestMatch(phonemes, Candidates, acousticEvidence);

            if (result.Passed && result.TargetId != null)
            {
                if (string.Equals(result.TargetId, _streakId,
                        StringComparison.Ordinal))
                {
                    _streak++;
                }
                else
                {
                    _streakId = result.TargetId;
                    _streak = 1;
                }
            }
            else
            {
                _streakId = null;
                _streak = 0;
            }

            if (_streak >= RequiredFrames(result.TargetId) || IsExactSingleFrame(result))
            {
                return new StreamingHit
                {
                    Result = result,
                    Frames = _frames,
                    // The real streak, not a fabricated second frame: a
                    // display that said "2 of 2" for one frame would lie.
                    Streak = _streak,
                    RequiredFrames = RequiredFrames(result.TargetId),
                    Feedback = PronunciationFeedback.FromMatch(result, _frames, _streak, Matcher.Matrix.MatrixId, _feedbackErrors)
                };
            }

            return null;
        }
    }
}
