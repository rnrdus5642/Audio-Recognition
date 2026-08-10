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

        /// <summary>Best candidate for this window, with alignment.</summary>
        public MatchResult Best;

        /// <summary>Consecutive windows the leader has held.</summary>
        public int Streak;

        /// <summary>The answer is confirmed and the session has ended.</summary>
        public bool Confirmed;
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
        /// question that takes any of a set. Any of them confirms, and
        /// <see cref="FrameScore.Best"/> says which.
        /// </summary>
        public void Begin(IReadOnlyList<string> targetWords)
        {
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
        }

        /// <summary>
        /// Score one window of 16 kHz mono audio.
        ///
        /// The window must be the length the acoustic model expects - a
        /// statically exported graph accepts exactly one size.
        /// </summary>
        public FrameScore Push(float[] audioWindow16kMono)
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

            _recognizer.RecognizeWithText(
                audioWindow16kMono, out string text, out var phonemes);

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
                End();
            }

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
