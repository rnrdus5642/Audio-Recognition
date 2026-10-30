using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;

namespace DomiCube.PhonemeMatching
{
    /// <summary>Directional ASR observations. Missing pairs are unknown, not measured zeroes.</summary>
    public sealed class ModelErrorProfile
    {
        public string MatrixId { get; }
        public int Utterances { get; }
        private readonly Dictionary<string, (int Count, int Total)> _substitutions;
        private readonly Dictionary<string, (int Count, int Total)> _deletions;

        private ModelErrorProfile(JObject data)
        {
            MatrixId = (string)data["matrix_id"];
            Utterances = (int?)data["utterances"] ?? 0;
            if (string.IsNullOrWhiteSpace(MatrixId) || MatrixId == "unknown" || Utterances < 0)
                throw new ArgumentException("Invalid model error profile identity.");
            _substitutions = Read(data["substitutions"] as JObject);
            _deletions = Read(data["deletions"] as JObject);
        }

        public static ModelErrorProfile FromJson(string json) => new ModelErrorProfile(JObject.Parse(json));

        private static Dictionary<string, (int Count, int Total)> Read(JObject data)
        {
            var result = new Dictionary<string, (int Count, int Total)>();
            if (data == null) return result;
            foreach (var pair in data.Properties())
            {
                if (pair.Name.StartsWith("_", StringComparison.Ordinal)) continue;
                var values = pair.Value as JArray;
                if (values == null || values.Count != 2 || values.Any(v => v.Type != JTokenType.Integer))
                    throw new ArgumentException("Model error observations must be [count, total].");
                int count = (int)values[0], total = (int)values[1];
                if (count < 0 || total < 0 || count > total)
                    throw new ArgumentException("Invalid model error observation counts.");
                result.Add(pair.Name, (count, total));
            }
            return result;
        }

        internal bool TryStatistics(PhonemeComparison step, out int count, out int total)
        {
            count = total = 0;
            (int Count, int Total) values = default;
            bool found = step.Operation == "sub"
                ? _substitutions.TryGetValue(step.TargetPhoneme + "|" + step.UserPhoneme, out values)
                : step.Operation == "del" && _deletions.TryGetValue(step.TargetPhoneme, out values);
            if (!found) return false;
            count = values.Count;
            total = values.Total;
            return true;
        }
    }

    [Serializable, JsonObject(NamingStrategyType = typeof(SnakeCaseNamingStrategy))]
    public sealed class WithheldDifference
    {
        public int AlignmentIndex;
        public string Reason;
        public int Observations;
        public int Total;
        public double? ModelErrorRate;
    }

    [Serializable, JsonObject(NamingStrategyType = typeof(SnakeCaseNamingStrategy))]
    public sealed class FeedbackReview
    {
        public string MatrixId = "";
        public string ProfileMatrixId = "";
        public int ReferenceUtterances;
        // unavailable / matched / withheld / partial / review
        public string Status = "unavailable";
        public List<WithheldDifference> Withheld = new List<WithheldDifference>();
    }

    internal static class ModelFeedback
    {
        internal static void Apply(PronunciationFeedback feedback, string matrixId, ModelErrorProfile errors)
        {
            bool matches = !string.IsNullOrEmpty(matrixId) && matrixId != "unknown"
                && errors != null && errors.MatrixId == matrixId;
            var review = new FeedbackReview
            {
                MatrixId = matrixId ?? "", ProfileMatrixId = errors?.MatrixId ?? "",
                ReferenceUtterances = matches ? errors.Utterances : 0
            };
            for (int i = 0; i < feedback.Alignment.Count; i++)
            {
                var step = feedback.Alignment[i];
                if (step.Operation == "match") continue;
                int count = 0, total = 0;
                bool known = matches && errors.TryStatistics(step, out count, out total);
                string reason = feedback.AcousticRecovery ? "acoustic_keyword_recovery"
                    : !matches ? "missing_or_mismatched_profile"
                    : step.Operation == "ins" ? "unmeasured_insertion"
                    : !known ? "unmeasured_difference"
                    : total <= 0 || count < 2 ? "insufficient_observations"
                    : count / (double)total >= .02 ? "known_model_confusion" : "";
                if (reason.Length > 0)
                    review.Withheld.Add(new WithheldDifference
                    {
                        AlignmentIndex = i, Reason = reason, Observations = count, Total = total,
                        ModelErrorRate = total > 0 ? (double?)count / total : null
                    });
            }

            var muted = new HashSet<int>(review.Withheld.Select(item => item.AlignmentIndex));
            var raw = feedback.RawKoreanFeedback;
            var guide = new KoreanFeedback
            {
                SyllableMappingAvailable = raw.SyllableMappingAvailable, Summary = raw.Summary
            };
            foreach (var hint in raw.Items)
            {
                if (hint.AlignmentIndices.Any(muted.Contains)) continue;
                guide.Items.Add(new PronunciationHint
                {
                    Kind = hint.Kind, SyllableIndex = hint.SyllableIndex, TargetSyllable = hint.TargetSyllable,
                    HeardSyllable = hint.HeardSyllable, Message = hint.Message,
                    AlignmentIndices = new List<int>(hint.AlignmentIndices)
                });
            }
            if (!feedback.AlignmentAvailable) review.Status = "unavailable";
            else if (review.Withheld.Count > 0)
            {
                review.Status = guide.Items.Count > 0 ? "partial" : "withheld";
                // Presentation wording does not change the internal uncertainty record.
                guide.Summary = guide.Items.Count > 0
                    ? "아래 부분만 조금 더 또렷하게 말해 보세요."
                    : "발음에 문제 없어요.";
            }
            else review.Status = guide.Items.Count > 0 ? "review" : "matched";
            feedback.KoreanFeedback = guide;
            feedback.FeedbackReview = review;
        }
    }
}
