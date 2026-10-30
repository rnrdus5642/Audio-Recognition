using System;
using System.Collections.Generic;

namespace DomiCube.PhonemeMatching
{
    /// <summary>Optional support from the same inference, never a rewritten transcript.</summary>
    public interface IAcousticEvidenceRecognizer
    {
        AcousticEvidence GetAcousticEvidence(string targetText, string contrastText);
    }

    [Serializable]
    public sealed class AcousticKeywordPolicy
    {
        public double MinScore;
        public double MinLogProbability;
        public double MaxLogGap;
        public string ContrastText;
        public double MaxContrastLogGap;
        public int MaxExtraPhonemes;

        public void Validate(string target)
        {
            if (!Finite(MinScore) || MinScore <= 0 || MinScore > 1
                || !Finite(MinLogProbability) || MinLogProbability < -100 || MinLogProbability >= 0
                || !Finite(MaxLogGap) || MaxLogGap < 0 || MaxLogGap > 100
                || !Finite(MaxContrastLogGap) || MaxContrastLogGap < 0 || MaxContrastLogGap > 100
                || string.IsNullOrWhiteSpace(ContrastText) || ContrastText == target
                || MaxExtraPhonemes < 0 || MaxExtraPhonemes > 6)
                throw new ArgumentException("acoustic_keyword_policy requires finite limits, contrast_text and max_extra_phonemes in [0, 6]");
        }

        internal static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }

    /// <summary>Natural-log CTC sequence probabilities, not an accuracy percentage.</summary>
    public sealed class AcousticEvidence
    {
        public string TargetText;
        public string ContrastText;
        public double LogProbability;
        public double GreedyLogProbability;
        public double ContrastLogProbability;

        public bool Permits(Answer answer, int phonemeCount, double score)
        {
            var p = answer.AcousticKeywordPolicy;
            if (p == null || TargetText != answer.Text || ContrastText != p.ContrastText) return false;
            return Valid(LogProbability) && Valid(GreedyLogProbability) && Valid(ContrastLogProbability)
                && phonemeCount >= Math.Max(1, answer.Phonemes.Count - 1)
                && phonemeCount <= answer.Phonemes.Count + p.MaxExtraPhonemes
                && score >= p.MinScore && LogProbability >= p.MinLogProbability
                && GreedyLogProbability - LogProbability <= p.MaxLogGap
                && ContrastLogProbability - LogProbability <= p.MaxContrastLogGap;
        }

        private static bool Valid(double value) => AcousticKeywordPolicy.Finite(value) && value <= 1e-5;

        public static Dictionary<string, AcousticEvidence> Collect(IPhonemeRecognizer recognizer, IReadOnlyList<Answer> candidates)
        {
            if (!(recognizer is IAcousticEvidenceRecognizer source)) return null;
            var result = new Dictionary<string, AcousticEvidence>();
            foreach (var answer in candidates)
                if (answer.AcousticKeywordPolicy != null)
                    result[answer.Text] = source.GetAcousticEvidence(answer.Text, answer.AcousticKeywordPolicy.ContrastText);
            return result;
        }
    }

    /// <summary>CTC forward sum over all alignments of an exact sequence.</summary>
    public sealed class CtcSequenceScorer
    {
        private readonly double[] _logProbabilities;
        private readonly int _frames;
        private readonly CtcVocabulary _vocab;

        public CtcSequenceScorer(float[] logits, int frames, int classes, CtcVocabulary vocab)
        {
            if (vocab == null || classes != vocab.Size || vocab.BlankId < 0 || vocab.BlankId >= classes
                || logits == null || frames < 0 || (long)frames * classes != logits.Length)
                throw new ArgumentException("CTC logits and vocabulary dimensions must agree.");
            _frames = frames;
            _vocab = vocab;
            _logProbabilities = new double[logits.Length];
            for (int f = 0; f < frames; f++)
            {
                int offset = f * classes;
                double maximum = double.NegativeInfinity;
                for (int c = 0; c < classes; c++)
                {
                    if (!AcousticKeywordPolicy.Finite(logits[offset + c]))
                        throw new ArgumentException("CTC logits must be finite.");
                    maximum = Math.Max(maximum, logits[offset + c]);
                }
                double sum = 0;
                for (int c = 0; c < classes; c++) sum += Math.Exp(logits[offset + c] - maximum);
                double normalizer = maximum + Math.Log(sum);
                for (int c = 0; c < classes; c++) _logProbabilities[offset + c] = logits[offset + c] - normalizer;
            }
        }

        public double Score(string text)
        {
            if (text == null) return double.NegativeInfinity;
            var labels = new List<int> { _vocab.BlankId };
            foreach (char letter in text)
            {
                int token = letter == ' ' ? _vocab.WordDelimiterId : Array.IndexOf(_vocab.Tokens, letter.ToString());
                if (token < 0 || token == _vocab.BlankId || token == _vocab.UnkId) return double.NegativeInfinity;
                labels.Add(token);
                labels.Add(_vocab.BlankId);
            }
            var previous = new double[labels.Count];
            for (int i = 1; i < previous.Length; i++) previous[i] = double.NegativeInfinity;
            var current = new double[labels.Count];
            for (int f = 0; f < _frames; f++)
            {
                for (int i = 0; i < labels.Count; i++)
                {
                    double value = previous[i];
                    if (i > 0) value = Add(value, previous[i - 1]);
                    if (i > 1 && labels[i] != _vocab.BlankId && labels[i] != labels[i - 2])
                        value = Add(value, previous[i - 2]);
                    current[i] = value + _logProbabilities[f * _vocab.Size + labels[i]];
                }
                var swap = previous;
                previous = current;
                current = swap;
            }
            return labels.Count == 1 ? previous[0] : Add(previous[labels.Count - 1], previous[labels.Count - 2]);
        }

        private static double Add(double a, double b)
        {
            if (double.IsNegativeInfinity(a)) return b;
            if (double.IsNegativeInfinity(b)) return a;
            double larger = Math.Max(a, b);
            return larger + Math.Log(1 + Math.Exp(Math.Min(a, b) - larger));
        }
    }
}
