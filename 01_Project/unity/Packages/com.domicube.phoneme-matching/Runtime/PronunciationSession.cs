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

        /// <summary>Positive checks for the leader. Web v4 may bridge one uncertain check.</summary>
        public int Streak;

        /// <summary>The answer is confirmed and the session has ended.</summary>
        public bool Confirmed;

        /// <summary>Confirmation snapshot, or null before confirmation.</summary>
        public PronunciationFeedback Feedback;

        /// <summary>Web v4 diagnostics when the orthographic policy is selected.</summary>
        public OrthographicWordResult WordDecision;
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
        private readonly OrthographicWordPolicy _wordPolicy;
        private double _audioSeconds;

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
            AudioWindowBuffer buffer = null)
        {
            _buffer = buffer;
            _matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
            _catalog = catalog ?? throw new ArgumentNullException(nameof(catalog));
            _recognizer = recognizer
                ?? throw new ArgumentNullException(nameof(recognizer));
            if (consecutive < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(consecutive));
            }

            _consecutive = consecutive;
        }

        private PronunciationSession(ConfusionMatrix matrix, IPhonemeRecognizer recognizer,
            double threshold, int confirmations, AudioWindowBuffer buffer)
        {
            _matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
            _recognizer = recognizer ?? throw new ArgumentNullException(nameof(recognizer));
            _buffer = buffer;
            _consecutive = confirmations;
            _wordPolicy = new OrthographicWordPolicy(matrix, threshold, confirmations);
        }

        /// <summary>Current web v4 policy. Targets are converted at runtime;
        /// a prebuilt targets.json is not required. Existing constructor stays legacy.</summary>
        public static PronunciationSession ForOrthographicWord(ConfusionMatrix matrix,
            IPhonemeRecognizer recognizer, double threshold = .85, int confirmations = 2,
            AudioWindowBuffer buffer = null)
            => new PronunciationSession(matrix, recognizer, threshold, confirmations, buffer);

        public bool UsesOrthographicWordPolicy => _wordPolicy != null;

        /// <summary>True between <see cref="Begin"/> and the confirmation
        /// (or <see cref="End"/>).</summary>
        public bool IsActive { get; private set; }

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

        public FrameScore LastFrame { get; private set; }

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
                _matcher, _candidates, _consecutive);

            // Whatever is still in the window belongs to the last
            // question - usually the answer just confirmed, which would
            // otherwise be scored again against the new word.
            _buffer?.Reset();
            IsActive = true;
        }

        /// <summary>Stop without a confirmation.</summary>
        public void End()
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

        /// <summary>Pass the capture timestamp of the end of this window, relative
        /// to Begin. Use this overload when inference or capture can be delayed.</summary>
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

            double fraction = 0;
            if (_wordPolicy != null)
            {
                if (audioWindow16kMono.Length != OrthographicWordPolicy.WindowSamples)
                    throw new ArgumentException("Web v4 requires 2.5 seconds of 16 kHz mono audio (40000 samples).");
                fraction = OrthographicWordPolicy.VoiceFraction(audioWindow16kMono);
            }
            _recognizer.RecognizeWithText(
                audioWindow16kMono, out string text, out var phonemes);

            if (_wordPolicy != null)
            {
                var decision = _wordPolicy.Push(text, elapsedAudioSeconds, fraction);
                var scored = new FrameScore { Text = text ?? string.Empty,
                    Phonemes = phonemes ?? new List<string>(),
                    Best = _wordPolicy.ToMatchResult(decision), Streak = decision.Streak,
                    Confirmed = decision.Confirmed, WordDecision = decision };
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

            var hit = _streaming.Push(phonemes);

            // Scored again for reporting: the streak only tells us
            // whether the leader held, not who led or by how much.
            var best = _matcher.BestMatch(phonemes, _candidates);

            var frame = new FrameScore
            {
                Text = text ?? string.Empty,
                Phonemes = phonemes ?? new List<string>(),
                Best = best,
                Streak = _streaming.Streak,
                Confirmed = hit != null
            };

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
        /// Score the buffer this session was constructed with. Same as
        /// <c>Push(buffer.Snapshot())</c>.
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

        private Answer FindAnswer(string text)
        {
            return _catalog.Find(text);
        }
    }
}
