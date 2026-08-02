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

        public Matcher Matcher { get; }
        public IReadOnlyList<Answer> Candidates { get; }

        /// <summary>
        /// Frames the same answer must win before committing. Must be
        /// reachable given the matcher's context limit - see
        /// <see cref="Matcher.ContextSlice"/>.
        /// </summary>
        public int Consecutive { get; }

        /// <summary>Frames the current leader has held; 0 if none.</summary>
        public int Streak => _streak;

        public StreamingMatcher(
            Matcher matcher, IReadOnlyList<Answer> candidates,
            int consecutive = 3)
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
            Reset();
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
        public StreamingHit Push(IReadOnlyList<string> phonemes)
        {
            _frames++;
            var result = Matcher.BestMatch(phonemes, Candidates);

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

            if (_streak >= Consecutive)
            {
                return new StreamingHit
                {
                    Result = result,
                    Frames = _frames,
                    Streak = _streak
                };
            }

            return null;
        }
    }
}
