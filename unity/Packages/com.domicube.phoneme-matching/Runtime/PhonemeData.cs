using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

namespace DomiCube.PhonemeMatching
{
    /// <summary>One accepted answer, as built by the offline pipeline.</summary>
    public sealed class Answer
    {
        public string Id;
        public string Text;
        public List<string> Phonemes = new List<string>();

        /// <summary>
        /// Pass mark for this word. Derived from phoneme count at build
        /// time: short words carry less information and are more prone to
        /// chance matches, so they are held to a stricter score.
        /// </summary>
        public double Threshold = 0.6;
    }

    /// <summary>A group of answers that compete with each other.</summary>
    public sealed class Segment
    {
        public string Id;
        public string Language = "ko";
        public List<Answer> Answers = new List<Answer>();
    }

    /// <summary>
    /// Parsed targets.json. The runtime never runs G2P: the phoneme
    /// sequences here were produced offline with the full phonological
    /// rule set, which is why device code only needs the jamo tables.
    /// </summary>
    public sealed class TargetCatalog
    {
        public string Version;
        public string ConfusionMatrixId;
        public List<Segment> Segments = new List<Segment>();

        public Segment FindSegment(string segmentId)
        {
            return Segments.Find(s => s.Id == segmentId);
        }

        /// <summary>The answers competing alongside the given word.</summary>
        public List<Answer> SegmentOf(string answerIdOrText)
        {
            foreach (var seg in Segments)
            {
                if (seg.Answers.Exists(
                        a => a.Id == answerIdOrText || a.Text == answerIdOrText))
                {
                    return seg.Answers;
                }
            }

            return null;
        }
    }

    /// <summary>Readers for the JSON produced by the offline pipeline.</summary>
    public static class PhonemeData
    {
        /// <summary>
        /// Keys beginning with '_' are authoring comments, not data.
        /// </summary>
        private static bool IsComment(string key)
        {
            return string.IsNullOrEmpty(key) || key[0] == '_';
        }

        public static ConfusionMatrix LoadMatrix(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                throw new ArgumentException("matrix json is empty", nameof(json));
            }

            var root = JObject.Parse(json);

            var subs = new Dictionary<string, double>();
            foreach (var kv in Flat(root["substitutions"] as JObject))
            {
                // Normalise 'b|a' to 'a|b' so lookups are direction-free.
                var parts = kv.Key.Split('|');
                if (parts.Length != 2)
                {
                    continue;
                }

                subs[ConfusionMatrix.PairKey(parts[0], parts[1])] = kv.Value;
            }

            StreamingProfile streaming = null;
            if (root["streaming_profile"] is JObject sp)
            {
                streaming = new StreamingProfile
                {
                    SkipCost = (double?)sp["skip_cost"],
                    Coverage = (double?)sp["coverage"],
                    ContextMult = (double?)sp["context_mult"]
                };
            }

            return new ConfusionMatrix(
                matrixId: (string)root["matrix_id"] ?? "unknown",
                language: (string)root["language"] ?? "unknown",
                version: (string)root["version"] ?? "0.0.0",
                substitutions: subs,
                deletions: Flat(root["deletions"] as JObject),
                insertions: Flat(root["insertions"] as JObject),
                defaultSubstitution: (double?)root["default_substitution"] ?? 0.8,
                defaultDeletion: (double?)root["default_deletion"] ?? 0.6,
                defaultInsertion: (double?)root["default_insertion"] ?? 0.6,
                skipCost: (double?)root["skip_cost"] ?? 0.1,
                streaming: streaming);
        }

        private static Dictionary<string, double> Flat(JObject obj)
        {
            var result = new Dictionary<string, double>();
            if (obj == null)
            {
                return result;
            }

            foreach (var prop in obj.Properties())
            {
                if (IsComment(prop.Name))
                {
                    continue;
                }

                result[prop.Name] = (double)prop.Value;
            }

            return result;
        }

        public static TargetCatalog LoadTargets(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                throw new ArgumentException("targets json is empty", nameof(json));
            }

            var root = JObject.Parse(json);
            var catalog = new TargetCatalog
            {
                Version = (string)root["version"],
                ConfusionMatrixId = (string)root["confusion_matrix_id"]
            };

            if (root["segments"] is JArray segments)
            {
                foreach (var s in segments)
                {
                    var seg = new Segment
                    {
                        Id = (string)s["id"],
                        Language = (string)s["language"] ?? "ko"
                    };

                    if (s["answers"] is JArray answers)
                    {
                        foreach (var a in answers)
                        {
                            var answer = new Answer
                            {
                                Id = (string)a["id"],
                                Text = (string)a["text"],
                                Threshold = (double?)a["threshold"] ?? 0.6
                            };

                            if (a["phonemes"] is JArray phonemes)
                            {
                                foreach (var p in phonemes)
                                {
                                    answer.Phonemes.Add((string)p);
                                }
                            }

                            seg.Answers.Add(answer);
                        }
                    }

                    catalog.Segments.Add(seg);
                }
            }

            return catalog;
        }

        /// <summary>
        /// Reads the CTC token table exported alongside the ONNX model.
        /// </summary>
        public static CtcVocabulary LoadCtcVocabulary(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                throw new ArgumentException("vocab json is empty", nameof(json));
            }

            var root = JObject.Parse(json);
            var tokens = new List<string>();
            if (root["tokens"] is JArray array)
            {
                foreach (var t in array)
                {
                    tokens.Add((string)t);
                }
            }

            // The exporter records the model's output width separately.
            // If it disagrees with the table, ids no longer line up with
            // logits and every decode would be silently wrong.
            var declared = (int?)root["vocab_size"];
            if (declared.HasValue && declared.Value != tokens.Count)
            {
                throw new ArgumentException(
                    $"vocab_size is {declared.Value} but {tokens.Count} "
                    + "tokens are listed");
            }

            return new CtcVocabulary
            {
                Tokens = tokens.ToArray(),
                BlankId = (int?)root["blank_id"] ?? -1,
                UnkId = (int?)root["unk_id"] ?? -1,
                WordDelimiterId = (int?)root["word_delimiter_id"] ?? -1,
                Normalize = (bool?)root["normalize"] ?? true,
                NormalizeEpsilon = (float?)root["normalize_epsilon"] ?? 1e-7f,
                SamplingRate = (int?)root["sampling_rate"] ?? 16000
            };
        }
    }
}
