using System;
using System.Collections.Generic;
using DomiCube.PhonemeMatching.Korean;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

namespace DomiCube.PhonemeMatching
{
    /// <summary>
    /// One scored IPA comparison. Indices are zero-based; a missing side
    /// is -1. These are phoneme indices, NOT character indices or times.
    /// </summary>
    [Serializable]
    [JsonObject(NamingStrategyType = typeof(SnakeCaseNamingStrategy))]
    public sealed class PhonemeComparison
    {
        // Stable JSON values: match, sub, del, ins.
        public string Operation;
        public string TargetPhoneme;
        public string UserPhoneme;
        public int TargetIndex;
        public int UserIndex;
    }

    /// <summary>
    /// Display-neutral snapshot of the confirming frame, suitable for a
    /// future Unity UI. It reuses the scoring alignment, not a new ASR or
    /// edit-distance pass. Newtonsoft JSON uses the same snake_case
    /// contract as the web UI. Fields/lists are copied from MatchResult.
    /// </summary>
    [Serializable]
    [JsonObject(NamingStrategyType = typeof(SnakeCaseNamingStrategy))]
    public sealed class PronunciationFeedback
    {
        public int SchemaVersion = 2;
        public string TargetId;
        public string TargetText;
        public double Score;
        public double Distance;
        public double Threshold;
        public bool Passed;
        public int Frames;
        public int Streak;
        public List<string> TargetPhonemes;
        public List<string> UserPhonemes;
        public int WindowStart;
        public int WindowEnd;
        public bool AlignmentAvailable;
        public List<PhonemeComparison> Alignment = new List<PhonemeComparison>();
        public int MatchCount;
        public int SubstitutionCount;
        public int DeletionCount;
        public int InsertionCount;
        public KoreanFeedback KoreanFeedback = new KoreanFeedback();

        /// <summary>
        /// Preserve full-input indices even when context limiting moved
        /// the matching window. Phonemes outside the window are NOT
        /// reported as inserted sounds in the answer.
        /// </summary>
        public static PronunciationFeedback FromMatch(
            MatchResult result, int frames, int streak)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            var feedback = new PronunciationFeedback
            {
                TargetId = result.TargetId,
                TargetText = result.TargetText,
                Score = result.Score,
                Distance = result.Distance,
                Threshold = result.Threshold,
                Passed = result.Passed,
                Frames = frames,
                Streak = streak,
                TargetPhonemes = new List<string>(
                    result.TargetPhonemes ?? new List<string>()),
                UserPhonemes = new List<string>(
                    result.UserPhonemes ?? new List<string>()),
                WindowStart = result.WindowStart,
                WindowEnd = result.WindowEnd
            };
            int userIndex = result.WindowStart, targetIndex = 0;
            foreach (var step in result.Alignment ?? new List<AlignStep>())
            {
                string operation;
                switch (step.Op)
                {
                    case AlignOp.Match:
                        operation = "match";
                        feedback.MatchCount++;
                        break;
                    case AlignOp.Substitute:
                        operation = "sub";
                        feedback.SubstitutionCount++;
                        break;
                    case AlignOp.Delete:
                        operation = "del";
                        feedback.DeletionCount++;
                        break;
                    case AlignOp.Insert:
                        operation = "ins";
                        feedback.InsertionCount++;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(step.Op));
                }
                feedback.Alignment.Add(new PhonemeComparison
                {
                    Operation = operation,
                    TargetPhoneme = step.TargetPhoneme,
                    UserPhoneme = step.UserPhoneme,
                    TargetIndex = step.Op == AlignOp.Insert ? -1 : targetIndex,
                    UserIndex = step.Op == AlignOp.Delete ? -1 : userIndex
                });
                if (step.Op != AlignOp.Insert) targetIndex++;
                if (step.Op != AlignOp.Delete) userIndex++;
            }
            // Exact mode currently emits no trace. Missing alignment
            // must not be displayed as "no pronunciation differences".
            feedback.AlignmentAvailable = feedback.Alignment.Count > 0;
            feedback.KoreanFeedback = KoreanFeedbackBuilder.Build(feedback);
            return feedback;
        }
    }

    /// <summary>Korean UI-ready explanations; the raw alignment remains authoritative.</summary>
    [Serializable]
    [JsonObject(NamingStrategyType = typeof(SnakeCaseNamingStrategy))]
    public sealed class KoreanFeedback
    {
        public bool SyllableMappingAvailable;
        public string Summary = "";
        public List<PronunciationHint> Items = new List<PronunciationHint>();
    }

    [Serializable]
    [JsonObject(NamingStrategyType = typeof(SnakeCaseNamingStrategy))]
    public sealed class PronunciationHint
    {
        // sub / del / ins / mixed; a hint may group several differences.
        public string Kind;
        // Zero-based ordinal of Hangul syllables ONLY; -1 when not localized.
        public int SyllableIndex;
        public string TargetSyllable;
        // Approximate sound, NOT an ASR transcript. Empty if not reconstructable.
        public string HeardSyllable;
        public string Message;
        public List<int> AlignmentIndices;
    }
}
