using System;
using System.Collections.Generic;

namespace DomiCube.PhonemeMatching
{
    /// <summary>
    /// Scoring overrides for continuous listening. Batch and streaming
    /// genuinely need different values: batch keeps a lenient coverage
    /// threshold so a partly recognised word still passes, and that same
    /// leniency lets fragments of other words clear the bar when the user
    /// is talking freely.
    /// </summary>
    public sealed class StreamingProfile
    {
        public double? SkipCost;
        public double? Coverage;
        public double? ContextMult;
    }

    /// <summary>
    /// Phoneme-level confusion costs, mirroring
    /// python/runtime/matching/confusion_matrix.py.
    ///
    /// All costs are in [0, 1] where 0 means "no penalty" and 1 means
    /// "as bad as an unrelated phoneme". Substitutions are commutative:
    /// the key 'a|b' covers both directions.
    /// </summary>
    public sealed class ConfusionMatrix
    {
        private readonly Dictionary<string, double> _substitutions;
        private readonly Dictionary<string, double> _deletions;
        private readonly Dictionary<string, double> _insertions;

        public string MatrixId { get; }
        public string Language { get; }
        public string Version { get; }
        public double DefaultSubstitution { get; }
        public double DefaultDeletion { get; }
        public double DefaultInsertion { get; }

        /// <summary>
        /// Per-phoneme cost of skipping user output outside the matched
        /// window (substring mode). Must stay well below
        /// <see cref="DefaultInsertion"/> so surrounding noise is still
        /// cheap to ignore. Zero makes skipping free, which makes the
        /// score non-decreasing in utterance length - a long enough
        /// utterance then clears any threshold.
        /// </summary>
        public double SkipCost { get; }

        public StreamingProfile Streaming { get; }

        public ConfusionMatrix(
            string matrixId,
            string language,
            string version,
            Dictionary<string, double> substitutions,
            Dictionary<string, double> deletions,
            Dictionary<string, double> insertions,
            double defaultSubstitution,
            double defaultDeletion,
            double defaultInsertion,
            double skipCost,
            StreamingProfile streaming)
        {
            _substitutions = substitutions
                ?? new Dictionary<string, double>();
            _deletions = deletions ?? new Dictionary<string, double>();
            _insertions = insertions ?? new Dictionary<string, double>();
            MatrixId = matrixId;
            Language = language;
            Version = version;
            DefaultSubstitution = defaultSubstitution;
            DefaultDeletion = defaultDeletion;
            DefaultInsertion = defaultInsertion;
            SkipCost = skipCost;
            Streaming = streaming ?? new StreamingProfile();
        }

        /// <summary>Canonical key for a commutative substitution pair.</summary>
        public static string PairKey(string a, string b)
        {
            return string.CompareOrdinal(a, b) <= 0
                ? a + "|" + b
                : b + "|" + a;
        }

        /// <summary>Cost of substituting <paramref name="a"/> with
        /// <paramref name="b"/>.</summary>
        public double SubCost(string a, string b)
        {
            if (string.Equals(a, b, StringComparison.Ordinal))
            {
                return 0.0;
            }

            return _substitutions.TryGetValue(PairKey(a, b), out var cost)
                ? cost
                : DefaultSubstitution;
        }

        /// <summary>Cost of the target having a phoneme the user did not
        /// produce.</summary>
        public double DelCost(string phoneme)
        {
            return _deletions.TryGetValue(phoneme, out var cost)
                ? cost
                : DefaultDeletion;
        }

        /// <summary>Cost of the user producing a phoneme the target does
        /// not have.</summary>
        public double InsCost(string phoneme)
        {
            return _insertions.TryGetValue(phoneme, out var cost)
                ? cost
                : DefaultInsertion;
        }
    }
}
