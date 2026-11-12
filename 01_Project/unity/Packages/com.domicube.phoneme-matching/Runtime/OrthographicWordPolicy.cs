using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using DomiCube.PhonemeMatching.Korean;

namespace DomiCube.PhonemeMatching
{
    [Serializable]
    public sealed class WordCandidateScore
    {
        public string Target;
        public double Similarity;
        public string MatchedText;
        public string ComparisonText;
        public string Normalization;
        public string BlockedBy;
    }

    [Serializable]
    public sealed class OrthographicWordResult
    {
        public string Policy = OrthographicWordPolicy.Version;
        public string Text;
        public string Candidate;
        public string MatchedTarget;
        public double Similarity;
        public bool Passed;
        public bool Confirmed;
        public bool Ambiguous;
        public bool ConfirmationWaiting;
        public int Streak;
        public int Frames;
        public double ElapsedSeconds;
        public double VoiceFraction;
        public string MatchedText;
        public string ComparisonText;
        public string Normalization;
        public string BlockedBy;
        public string DecisionReason;
        public List<string> TiedTargets = new List<string>();
        public List<WordCandidateScore> CandidateScores = new List<WordCandidateScore>();
    }

    /// <summary>
    /// Deterministic port of the web's orthographic-word-preview-v4 policy.
    /// Recognised spelling in, candidate scores and confirmation out. Never
    /// rewrites the raw text, supplies targets to ASR, or loads an acoustic model.
    /// </summary>
    public sealed class OrthographicWordPolicy
    {
        public const string Version = "orthographic-word-preview-v4";
        public const double WindowSeconds = 2.5;
        public const double HopSeconds = 0.5;
        public const int SampleRate = 16000;
        public const int WindowSamples = 40000;
        private readonly Matcher _matcher;
        private readonly Dictionary<string, string> _families = new Dictionary<string, string>();
        private readonly Dictionary<string, List<string>> _ipa = new Dictionary<string, List<string>>();
        private readonly List<string> _targets = new List<string>();
        private readonly List<double> _evidence = new List<double>();
        private string _previous;
        private int _gaps, _frames;
        private double _elapsed;
        public double Threshold { get; }
        public int Confirmations { get; }
        public bool IsActive { get; private set; }
        public IReadOnlyList<string> Targets => _targets.AsReadOnly();

        public OrthographicWordPolicy(ConfusionMatrix matrix, double threshold = 0.85, int confirmations = 2)
        {
            if (double.IsNaN(threshold) || threshold < .7 || threshold > 1)
                throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be 0.7 to 1.");
            if (confirmations < 2 || confirmations > 4)
                throw new ArgumentOutOfRangeException(nameof(confirmations), "Confirmations must be 2 to 4.");
            _matcher = new Matcher(matrix, MatchMode.Exact);
            Threshold = threshold;
            Confirmations = confirmations;
            foreach (var family in OrthographicActionFamilies.Words)
                for (int i = 1; i < family.Length; i++)
                    _families.Add(SpellingKey(family[i]), family[0]);
        }

        public void Begin(IReadOnlyList<string> targets)
        {
            End();
            if (targets == null || targets.Count < 1 || targets.Count > 20)
                throw new ArgumentException("Provide 1 to 20 target words.", nameof(targets));
            var normalized = new List<string>();
            foreach (var value in targets)
            {
                string word = (value ?? "").Normalize(NormalizationForm.FormC).Trim()
                    .TrimEnd('?', '？').Trim();
                if (word.Length < 1 || word.Length > 20 || !HangulOnly(word))
                    throw new ArgumentException("Each target must contain 1 to 20 Hangul syllables; trailing ? is allowed.", nameof(targets));
                if (!normalized.Contains(word)) normalized.Add(word);
            }
            _targets.Clear();
            _targets.AddRange(normalized);
            _ipa.Clear();
            _elapsed = 0;
            _frames = 0;
            IsActive = true;
        }

        public void End()
        {
            IsActive = false;
            _evidence.Clear();
            _previous = null;
            _gaps = 0;
        }

        public OrthographicWordResult Push(string text, double elapsedSeconds, double voiceFraction)
        {
            if (!IsActive) throw new InvalidOperationException("Call Begin before Push.");
            if (!Finite(elapsedSeconds) || elapsedSeconds < .48 ||
                (_frames > 0 && elapsedSeconds - _elapsed < .45))
                throw new ArgumentOutOfRangeException(nameof(elapsedSeconds), "Use increasing audio capture timestamps, at least 0.45s apart.");
            if (!Finite(voiceFraction) || voiceFraction < 0 || voiceFraction > 1)
                throw new ArgumentOutOfRangeException(nameof(voiceFraction));
            text = text ?? "";
            var spans = ComparisonSpans(text);
            var scores = new List<WordCandidateScore>();
            foreach (var target in _targets)
            {
                var item = new WordCandidateScore { Target = target };
                if (voiceFraction >= .04)
                {
                    foreach (var span in spans)
                    {
                        _matcher.ScoreAgainst(Phonemes(span.Compared), Phonemes(target),
                            out double score, out _, out _);
                        if (score > item.Similarity)
                        {
                            item.Similarity = score;
                            item.MatchedText = span.Raw;
                            item.ComparisonText = span.Compared;
                            item.Normalization = span.Normalization;
                        }
                    }
                }
                if (_families.TryGetValue(SpellingKey(item.ComparisonText ?? ""), out var actual) &&
                    _families.TryGetValue(SpellingKey(target), out var expected) && actual != expected)
                    item.BlockedBy = Compact(item.ComparisonText);
                // Stable insertion sort preserves Python's target order on ties.
                int insert = scores.Count;
                while (insert > 0 && scores[insert - 1].Similarity < item.Similarity) insert--;
                scores.Insert(insert, item);
            }
            var top = scores[0];
            var leaders = new List<string>();
            foreach (var item in scores)
                if (Math.Abs(item.Similarity - top.Similarity) <= 1e-9) leaders.Add(item.Target);
            bool ambiguous = top.Similarity >= Threshold && leaders.Count > 1;
            string candidate = top.Similarity > 0 && leaders.Count == 1 ? leaders[0] : null;
            string blocked = candidate == null ? null : top.BlockedBy;
            bool passed = candidate != null && blocked == null && top.Similarity >= Threshold;

            if (elapsedSeconds - _elapsed > 1.25) _evidence.Clear();
            _evidence.RemoveAll(t => elapsedSeconds - t > (Confirmations + .5) * .5);
            if (_evidence.Count == 0) { _previous = null; _gaps = 0; }
            bool uncertain = string.IsNullOrWhiteSpace(text) || ContainsJamo(text) ||
                (candidate == _previous && top.Similarity >= Math.Max(.7, Threshold - .05));
            bool waiting = false;
            if (passed)
            {
                if (candidate != _previous) { _evidence.Clear(); _gaps = 0; }
                _evidence.Add(elapsedSeconds);
                _previous = candidate;
            }
            else if (_evidence.Count > 0 && uncertain && blocked == null && !ambiguous && _gaps == 0)
            {
                _gaps++;
                waiting = true;
            }
            else { _evidence.Clear(); _previous = null; _gaps = 0; }
            if (_evidence.Count == 0) { _previous = null; _gaps = 0; waiting = false; }
            bool confirmed = passed && _evidence.Count >= Confirmations;
            var result = new OrthographicWordResult
            {
                Text = text, Candidate = candidate, Similarity = top.Similarity,
                Passed = passed, Confirmed = confirmed, Ambiguous = ambiguous,
                ConfirmationWaiting = waiting, Streak = _evidence.Count,
                Frames = ++_frames, ElapsedSeconds = elapsedSeconds, VoiceFraction = voiceFraction,
                MatchedTarget = confirmed ? _previous : null,
                MatchedText = candidate == null ? null : top.MatchedText,
                ComparisonText = candidate == null ? null : top.ComparisonText,
                Normalization = candidate == null ? null : top.Normalization, BlockedBy = blocked,
                DecisionReason = blocked != null ? "different_action" : ambiguous ? "ambiguous" :
                    waiting ? "uncertain_check" : passed ? "confirming" : voiceFraction < .04 ? "no_voice" : "below_threshold",
                TiedTargets = ambiguous ? leaders : new List<string>(), CandidateScores = scores
            };
            _elapsed = elapsedSeconds;
            if (confirmed) IsActive = false;
            return result;
        }

        public MatchResult ToMatchResult(OrthographicWordResult result)
        {
            var heard = Phonemes(result.ComparisonText ?? "");
            var target = Phonemes(result.Candidate ?? "");
            double distance = _matcher.ScoreAgainst(heard, target, out _, out _, out _);
            return new MatchResult
            {
                TargetId = result.Candidate, TargetText = result.Candidate,
                Score = result.Similarity, Passed = result.Passed, Threshold = Threshold,
                Distance = distance, UserPhonemes = new List<string>(heard),
                TargetPhonemes = new List<string>(target), WindowStart = 0, WindowEnd = heard.Count
            };
        }

        public static double VoiceFraction(float[] audio)
        {
            if (audio == null) throw new ArgumentNullException(nameof(audio));
            int total = audio.Length / 320, voiced = 0;
            for (int i = 0; i < total; i++)
            {
                double energy = 0;
                for (int j = 0; j < 320; j++)
                {
                    double sample = audio[i * 320 + j];
                    if (!Finite(sample) || Math.Abs(sample) > 1.01)
                        throw new ArgumentException("Audio samples must be finite PCM values in [-1, 1].", nameof(audio));
                    energy += sample * sample;
                }
                if ((float)Math.Sqrt(energy / 320) >= .006f) voiced++;
            }
            return total == 0 ? 0 : voiced / (double)total;
        }

        private sealed class Span
        {
            public string Raw, Compared, Normalization;
        }

        private List<Span> ComparisonSpans(string text)
        {
            var tokens = Regex.Split(text.Normalize(NormalizationForm.FormC).Trim(), @"\s+");
            var result = new List<Span>();
            for (int start = tokens.Length - 1; start >= 0; start--)
            {
                if (!HangulOnly(tokens[start])) break;
                string raw = string.Join(" ", tokens, start, tokens.Length - start);
                string joined = Compact(raw);
                result.Add(new Span { Raw = raw, Compared = joined });
                for (int count = 2; count <= 4; count++)
                {
                    int width = joined.Length / count;
                    if (joined.Length % count != 0 || width < 2) continue;
                    string root = joined.Substring(0, width);
                    if (!_targets.Contains(root)) continue;
                    var repeated = new StringBuilder();
                    for (int n = 0; n < count; n++) repeated.Append(root);
                    if (repeated.ToString() != joined) continue;
                    result.Add(new Span { Raw = raw, Compared = root, Normalization = "repeated_word" });
                    break;
                }
            }
            return result;
        }

        private List<string> Phonemes(string text)
        {
            if (_ipa.TryGetValue(text, out var result)) return result;
            if (_ipa.Count >= 256) _ipa.Clear();
            result = JamoIpa.ToPhonemes(Rules.Apply(text));
            _ipa[text] = result;
            return result;
        }

        private static bool HangulOnly(string text)
        {
            if (text.Length == 0) return false;
            foreach (char ch in text) if (ch < '가' || ch > '힣') return false;
            return true;
        }
        private static bool ContainsJamo(string text)
        {
            foreach (char ch in text.Normalize(NormalizationForm.FormC))
                if (ch >= '\u1100' && ch <= '\u11ff') return true;
            return false;
        }
        private static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
        private static string Compact(string text) => Regex.Replace(text.Normalize(NormalizationForm.FormC), @"\s+", "");
        private static string SpellingKey(string text) => Regex.Replace(Compact(text), "거에요$", "거예요");
    }
}
