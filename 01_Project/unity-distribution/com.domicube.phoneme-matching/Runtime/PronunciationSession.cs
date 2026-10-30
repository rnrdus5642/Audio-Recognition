using System;
using System.Collections.Generic;

namespace DomiCube.PhonemeMatching
{
    /// <summary>What one scored window produced.</summary>
    public sealed class FrameScore
    {
        /// <summary>Raw recogniser text; empty when nothing was heard.</summary>
        public string Text = string.Empty;

        public List<string> Phonemes = new List<string>();

        /// <summary>Selected candidate for this window, with alignment:
        /// highest-scoring pass, or highest-scoring failure if none pass.</summary>
        public MatchResult Best;

        /// <summary>Consecutive windows the leader has held.</summary>
        public int Streak;
        /// <summary>Required windows for the selected word, including its override.</summary>
        public int RequiredFrames;

        /// <summary>The answer is confirmed and the session has ended.</summary>
        public bool Confirmed;

        /// <summary>Confirmation snapshot, or null before confirmation.</summary>
        public PronunciationFeedback Feedback;

        /// <summary>
        /// Which 16 kHz sample this window ended on, or -1 when the
        /// window was handed over directly and the session has no idea
        /// where it came from.
        /// </summary>
        public long EndSampleIndex = -1;

        /// <summary>Unmodified input window levels, before model preprocessing.</summary>
        public double AudioRms;
        public double AudioPeak;

        /// <summary>Why this frame did or did not confirm; never a pronunciation diagnosis.</summary>
        public OrthographicWordResult WordDecision;

        public string DecisionReason => WordDecision != null ? WordDecision.DecisionReason : Confirmed ? "confirmed"
            : Phonemes == null || Phonemes.Count == 0 ? "no_recognized_phonemes"
            : Best == null || Best.TargetId == null ? "no_candidate"
            : Best.Score < Best.Threshold ? "below_threshold"
            : Best.WholeRejected ? "whole_utterance_rejected"
            : Best.CompetitorRejected ? "other_action_scored_higher"
            : Best.Passed ? "waiting_for_confirmation" : "rejected";
    }

    /// <summary>
    /// Audio in, confirmation out.
    ///
    /// Build one and keep it - the object lives as long as the lesson
    /// does, and each <see cref="Begin(string)"/> to <see cref="End"/>
    /// is one question.
    ///
    /// Owns no microphone and no coroutine, so the application decides
    /// when judging happens and where the audio comes from - its own
    /// capture system, a recorded clip, a network stream. Nothing here
    /// runs until <see cref="Push"/> is called.
    ///
    ///     var session = new PronunciationSession(matrix, catalog, recognizer);
    ///     session.Begin("사과");
    ///     // every hop, hand over the most recent window:
    ///     var frame = session.Push(window);
    ///     if (frame.Confirmed) { ... }   // session is over
    ///
    /// When the audio arrives in chunks instead, give the constructor an
    /// <see cref="AudioWindowBuffer"/> built with a hop and drain whole
    /// frames - then how often the capture callback fires cannot change
    /// when the answer confirms:
    ///
    ///     var buffer = new AudioWindowBuffer(2.5f, 0.5f);
    ///     var session = new PronunciationSession(m, c, r, 2, buffer);
    ///     buffer.Append(chunk, rate);
    ///     while (session.TryPush(out var frame)) { ... }
    ///
    /// Feed the most recent window each time, NOT successive chunks:
    /// wav2vec2 is a context model and short isolated fragments come back
    /// garbled (measured on a 4.4 s utterance, 0.5 s chunks lost a
    /// quarter of the phonemes).
    ///
    /// <see cref="Unity.PronunciationListener"/> is this class plus a
    /// microphone, for callers that would rather not own one.
    /// </summary>
    public sealed class PronunciationSession
    {
        private readonly ConfusionMatrix _matrix;
        private readonly TargetCatalog _catalog;
        private readonly IPhonemeRecognizer _recognizer;
        private readonly int _consecutive;
        private readonly AudioWindowBuffer _buffer;
        private readonly ModelErrorProfile _feedbackErrors;
        private readonly OrthographicWordPolicy _wordPolicy;
        private double _audioSeconds;
        public bool UsesOrthographicWordPolicy => _wordPolicy != null;
        public FrameScore LastFrame { get; private set; }

        private Matcher _matcher;
        private StreamingMatcher _streaming;
        private List<Answer> _candidates;

        /// <param name="buffer">
        /// Optional. When given, <see cref="Begin"/> clears it and
        /// <see cref="Push()"/> reads from it, so the previous answer
        /// cannot linger in the window and be scored against the next
        /// word. Leave null to hand windows over yourself.
        /// </param>
        public PronunciationSession(
            ConfusionMatrix matrix,
            TargetCatalog catalog,
            IPhonemeRecognizer recognizer,
            int consecutive = 2,
            AudioWindowBuffer buffer = null, ModelErrorProfile feedbackErrors = null,
            bool useOrthographicWordPolicy = false, double wordThreshold = .85)
        {
            _buffer = buffer;
            _feedbackErrors = feedbackErrors;
            _matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
            _catalog = catalog ?? throw new ArgumentNullException(nameof(catalog));
            _recognizer = recognizer
                ?? throw new ArgumentNullException(nameof(recognizer));
            if (consecutive < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(consecutive));
            }

            _consecutive = consecutive;
            if (useOrthographicWordPolicy) _wordPolicy = new OrthographicWordPolicy(matrix, wordThreshold, consecutive);
        }

        /// <summary>True between <see cref="Begin"/> and the confirmation
        /// (or <see cref="End"/>).</summary>
        public bool IsActive { get; private set; }

        public static PronunciationSession ForOrthographicWord(ConfusionMatrix matrix,
            IPhonemeRecognizer recognizer, double threshold = .85, int confirmations = 2,
            AudioWindowBuffer buffer = null) => new PronunciationSession(matrix,
                new TargetCatalog(), recognizer, confirmations, buffer, null, true, threshold);

        /// <summary>
        /// Windows the same answer must win before confirming. Exposed so
        /// a UI can show progress towards it - a display that hardcodes
        /// the number goes stale the moment the default changes.
        /// </summary>
        public int Consecutive => _consecutive;

        /// <summary>The word being asked for, or null.</summary>
        public string TargetText { get; private set; }

        /// <summary>Last confirmation; cleared when a new question starts.</summary>
        public PronunciationFeedback LastConfirmation { get; private set; }

        /// <summary>The answers that count as correct this question.</summary>
        public IReadOnlyList<Answer> Candidates => _candidates;

        /// <summary>
        /// Start judging <paramref name="targetText"/>. Only that word is
        /// scored: it passes when its own score clears its own threshold
        /// for <see cref="Consecutive"/> windows running.
        /// </summary>
        public void Begin(string targetText)
        {
            Begin(new[] { targetText });
        }

        /// <summary>
        /// Start judging with several acceptable answers - synonyms, or a
        /// question that takes any of a set. Each frame chooses the highest
        /// score among answers that pass their own thresholds. The SAME answer
        /// must be selected for <see cref="Consecutive"/> windows running;
        /// <see cref="FrameScore.Best"/> says which. If none pass, Best is the
        /// highest-scoring failure and the streak resets.
        /// </summary>
        public void Begin(IReadOnlyList<string> targetWords)
        {
            LastConfirmation = null;
            LastFrame = null;
            _audioSeconds = 0;
            if (_wordPolicy != null)
            {
                IsActive = false;
                _wordPolicy.Begin(targetWords);
                _candidates = new List<Answer>();
                foreach (var word in _wordPolicy.Targets)
                    _candidates.Add(new Answer { Id = word, Text = word,
                        Threshold = _wordPolicy.Threshold,
                        Phonemes = Korean.JamoIpa.ToPhonemes(Korean.Rules.Apply(word)) });
                TargetText = _candidates.Count == 1 ? _candidates[0].Text : null;
                _buffer?.Reset();
                IsActive = true;
                return;
            }
            if (targetWords == null || targetWords.Count == 0)
            {
                throw new ArgumentException(
                    "at least one word is required", nameof(targetWords));
            }

            var answers = new List<Answer>(targetWords.Count);
            foreach (var word in targetWords)
            {
                var answer = FindAnswer(word);
                if (answer == null)
                {
                    throw new ArgumentException(
                        $"'{word}' is not in targets.json. Build it with "
                        + "python -m python.build.build_targets",
                        nameof(targetWords));
                }

                answers.Add(answer);
            }

            _candidates = answers;
            TargetText = targetWords.Count == 1 ? targetWords[0] : null;
            StartSession();
        }

        private void StartSession()
        {
            _matcher = Matcher.ForStreaming(_matrix);
            _streaming = new StreamingMatcher(
                _matcher, _candidates, _consecutive, _feedbackErrors);

            // Whatever is still in the window belongs to the last
            // question - usually the answer just confirmed, which would
            // otherwise be scored again against the new word.
            _buffer?.Reset();
            IsActive = true;
        }

        /// <summary>
        /// Stop without a confirmation.
        ///
        /// Empties the buffer this session was given, if it has one: an
        /// unread frame and a half-finished hop both belong to the
        /// question that just ended, and either one carried into the next
        /// question would put audio nobody asked for at the front of it.
        ///
        /// <see cref="LastConfirmation"/> survives. It is the record of
        /// what happened, and callers read it after ending.
        /// </summary>
        public void End()
        {
            IsActive = false;
            _streaming = null;
            _wordPolicy?.End();
            _buffer?.Reset();
        }

        /// <summary>
        /// Abandon the session without touching the buffer, so a framing
        /// fault stays readable on the buffer that raised it.
        /// </summary>
        private void Invalidate()
        {
            IsActive = false;
            _streaming = null;
            _wordPolicy?.End();
        }

        /// <summary>
        /// Score one window of 16 kHz mono audio.
        ///
        /// The window must be the length the acoustic model expects - a
        /// statically exported graph accepts exactly one size.
        /// </summary>
        public FrameScore Push(float[] audioWindow16kMono)
            => Push(audioWindow16kMono, _audioSeconds + OrthographicWordPolicy.HopSeconds);

        public FrameScore Push(float[] audioWindow16kMono, double elapsedAudioSeconds)
        {
            if (!IsActive)
            {
                throw new InvalidOperationException(
                    "call Begin() before Push()");
            }

            if (audioWindow16kMono == null)
            {
                throw new ArgumentNullException(nameof(audioWindow16kMono));
            }

            double voiceFraction = 0;
            if (_wordPolicy != null)
            {
                if (audioWindow16kMono.Length != OrthographicWordPolicy.WindowSamples)
                    throw new ArgumentException("Web v4 requires 40000 samples of 16kHz mono PCM.");
                voiceFraction = OrthographicWordPolicy.VoiceFraction(audioWindow16kMono);
            }
            _recognizer.RecognizeWithText(
                audioWindow16kMono, out string text, out var phonemes);

            if (_wordPolicy != null)
            {
                var decision = _wordPolicy.Push(text, elapsedAudioSeconds, voiceFraction);
                var scored = new FrameScore { Text = text ?? string.Empty,
                    Phonemes = phonemes ?? new List<string>(), Best = _wordPolicy.ToMatchResult(decision),
                    Streak = decision.Streak, RequiredFrames = _wordPolicy.Confirmations,
                    Confirmed = decision.Confirmed, WordDecision = decision };
                double wordEnergy = 0;
                foreach (float sample in audioWindow16kMono)
                {
                    wordEnergy += (double)sample * sample;
                    scored.AudioPeak = Math.Max(scored.AudioPeak, Math.Abs((double)sample));
                }
                scored.AudioRms = Math.Sqrt(wordEnergy / audioWindow16kMono.Length);
                _audioSeconds = elapsedAudioSeconds;
                if (scored.Confirmed)
                {
                    scored.Feedback = PronunciationFeedback.FromMatch(scored.Best, decision.Frames, decision.Streak);
                    LastConfirmation = scored.Feedback;
                    End();
                }
                LastFrame = scored;
                return scored;
            }
            var evidence = AcousticEvidence.Collect(_recognizer, _candidates);
            var hit = _streaming.Push(phonemes, evidence);

            // Scored again for reporting: the streak only tells us
            // whether the leader held, not who led or by how much.
            var best = _matcher.BestMatch(phonemes, _candidates, evidence);

            var frame = new FrameScore
            {
                Text = text ?? string.Empty,
                Phonemes = phonemes ?? new List<string>(),
                Best = best,
                Streak = _streaming.Streak,
                RequiredFrames = _streaming.RequiredFrames(best.TargetId),
                Confirmed = hit != null
            };
            double energy = 0.0;
            foreach (float sample in audioWindow16kMono)
            {
                double absolute = Math.Abs((double)sample);
                frame.AudioPeak = Math.Max(frame.AudioPeak, absolute);
                energy += (double)sample * sample;
            }
            frame.AudioRms = audioWindow16kMono.Length == 0 ? 0.0
                : Math.Sqrt(energy / audioWindow16kMono.Length);

            if (hit != null)
            {
                frame.Best = hit.Result;
                frame.Feedback = hit.Feedback;
                LastConfirmation = hit.Feedback;
                End();
            }

            _audioSeconds = elapsedAudioSeconds;
            LastFrame = frame;
            return frame;
        }

        /// <summary>
        /// Score whatever is in the buffer right now. Exactly
        /// <c>Push(buffer.Snapshot())</c>, and manual in the same way:
        /// it scores on being asked, not on new audio, so calling it on a
        /// timer makes the frame count follow the timer. It is kept for
        /// callers who already drive their own clock and want that.
        ///
        /// Prefer <see cref="TryPush"/> when the buffer has a hop.
        /// </summary>
        public FrameScore Push()
        {
            if (_buffer == null)
            {
                throw new InvalidOperationException(
                    "no AudioWindowBuffer was given to the constructor; "
                    + "pass the window to Push(float[]) instead");
            }

            return Push(_buffer.Snapshot());
        }

        /// <summary>
        /// Score the next whole hop of new audio, if one has arrived.
        ///
        /// Returns false - and runs no inference - when the buffer holds
        /// only part of a hop, so a caller may poll as often as it likes
        /// without changing what gets scored. Drain in a loop; the frames
        /// come out in the order the audio did:
        ///
        ///     while (session.TryPush(out var frame)) { ... }
        ///
        /// False once the session has confirmed or ended, which stops the
        /// loop on the confirming frame rather than scoring audio that
        /// arrived in the same chunk after the answer was already in.
        ///
        /// Requires a buffer built with a hop; throws otherwise rather
        /// than quietly behaving like the polling API.
        /// </summary>
        public bool TryPush(out FrameScore frame)
        {
            frame = null;
            if (_buffer == null)
            {
                throw new InvalidOperationException(
                    "no AudioWindowBuffer was given to the constructor; "
                    + "pass the window to Push(float[]) instead");
            }

            if (_candidates == null)
            {
                // Never begun is a caller mistake and reads the same as
                // "nothing to score" unless it is said out loud.
                throw new InvalidOperationException(
                    "call Begin() before TryPush()");
            }

            if (!IsActive)
            {
                return false;
            }

            HopFrame hop;
            try
            {
                if (!_buffer.TryReadFrame(out hop))
                {
                    return false;
                }
            }
            catch
            {
                // Frames were lost, so the streak has stopped meaning
                // "consecutive windows of audio" and a confirmation built
                // on it would be unreproducible. Throw the session away
                // and make the caller Begin() again - clearing the buffer
                // is not enough to make the old streak trustworthy.
                Invalidate();
                throw;
            }

            // Read before scoring: confirming ends the session, which
            // clears the buffer the frame came from.
            long endSample = hop.EndSampleIndex;
            try
            {
                frame = Push(hop.Window, endSample / (double)AudioWindowBuffer.TargetSampleRate);
            }
            catch
            {
                // The hop is spent either way. A caller that catches this
                // and carries on would otherwise join the frame before it
                // to the frame after it and call them consecutive, so the
                // streak goes with the frame that failed.
                frame = null;
                Invalidate();
                throw;
            }

            frame.EndSampleIndex = endSample;
            return true;
        }

        private Answer FindAnswer(string text)
        {
            return _catalog.Find(text);
        }
    }
}
