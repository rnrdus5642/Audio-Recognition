using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

namespace DomiCube.PhonemeMatching
{
    /// <summary>One accepted answer, as built by the offline pipeline.</summary>
    /// <summary>
    /// What licensed a word to confirm on one exact frame: the matrix,
    /// the streaming profile, the streak length and the exact word it was
    /// measured for. The runtime checks itself against this rather than
    /// against whatever matrix it happens to be holding, so an approval
    /// cannot travel to a different pronunciation, threshold or matrix.
    /// </summary>
    [Serializable]
    public sealed class ExactSingleFrameCalibration
    {
        public string MatrixId;
        public int Consecutive;
        public string Text;
        public double Threshold;
        public List<string> Phonemes = new List<string>();
        public double? SkipCost;
        public double? Coverage;
        public double? ContextMult;
    }

    /// <summary>
    /// One action an answer has to be told apart from, as the offline
    /// build embedded it. Never an answer itself: nothing here can be
    /// confirmed, only compared against what was asked for.
    /// </summary>
    [Serializable]
    public sealed class ContrastiveAlternative
    {
        public string Text;
        public string ActionGroup;
        public List<string> Phonemes = new List<string>();
    }

    /// <summary>A configured object prefix that may precede the answer.</summary>
    [Serializable]
    public sealed class ContextPrefix
    {
        public string Text;
        public List<string> Phonemes = new List<string>();
    }

    /// <summary>An opt-in lower threshold requiring short, whole-word evidence.</summary>
    [Serializable]
    public sealed class ShortUtterancePolicy
    {
        public double Threshold;
        public double MinWholeScore;
        public int MaxExtraPhonemes;
    }

    public sealed class Answer
    {
        public string Id;
        public string Text;
        public List<string> Phonemes = new List<string>();

        /// <summary>
        /// Which action this answer asks for, or null. Two answers
        /// sharing a group are two ways of naming the same action and
        /// never compete with each other.
        /// </summary>
        public string ActionGroup;

        /// <summary>
        /// The other actions this answer must beat to be believed, or
        /// null. Absent in older data, and absent means no veto - the
        /// answer is judged exactly as it always was.
        /// </summary>
        public List<ContrastiveAlternative> ContrastiveAlternatives;

        /// <summary>
        /// Pass mark for this word. Derived from phoneme count at build
        /// time: short words carry less information and are more prone to
        /// chance matches, so they are held to a stricter score.
        /// </summary>
        public double Threshold = 0.6;

        /// <summary>Per-word override (1..6); null uses the session default.</summary>
        public int? ConfirmationFrames;
        /// <summary>Optional whole-utterance floor in addition to the substring score.</summary>
        public double MinWholeScore;
        /// <summary>How much better another action must score to veto this answer.</summary>
        public double ContrastMargin;
        /// <summary>Optional phrases; a prefix must match before the answer without overlap.</summary>
        public List<ContextPrefix> ContextPrefixes;
        public ShortUtterancePolicy ShortUtterancePolicy;
        public AcousticKeywordPolicy AcousticKeywordPolicy;

        /// <summary>
        /// Whether one frame scoring exactly 1.0 may confirm this word on
        /// its own, without a second consecutive frame.
        ///
        /// Off unless the build calibrated it: a word only earns this
        /// when the resulting false-accept count stayed inside the same
        /// budget as the ordinary rule. Absent in older data, which is
        /// why the default is false and the strict two-frame rule stands.
        /// </summary>
        public bool AllowExactSingleFrame;

        /// <summary>The snapshot behind <see cref="AllowExactSingleFrame"/>,
        /// or null. Null means no shortcut, whatever the flag says.</summary>
        public ExactSingleFrameCalibration ExactSingleFrameCalibration;
    }

    public static class AnswerPolicy
    {
        public static void Validate(Answer answer)
        {
            answer.AcousticKeywordPolicy?.Validate(answer.Text);
            if (answer.ConfirmationFrames.HasValue
                && (answer.ConfirmationFrames.Value < 1 || answer.ConfirmationFrames.Value > 6))
                throw new ArgumentException($"'{answer.Text}' confirmation_frames must be an integer in [1, 6]");
            ValidateScore(answer.MinWholeScore, "min_whole_score", answer.Text);
            ValidateScore(answer.ContrastMargin, "contrast_margin", answer.Text);
            var shortPolicy = answer.ShortUtterancePolicy;
            if (shortPolicy != null)
            {
                ValidateScore(shortPolicy.Threshold, "short_utterance_policy threshold", answer.Text);
                ValidateScore(shortPolicy.MinWholeScore, "short_utterance_policy min_whole_score", answer.Text);
                if (shortPolicy.Threshold <= 0 || shortPolicy.MinWholeScore <= 0
                    || shortPolicy.MaxExtraPhonemes < 0 || shortPolicy.MaxExtraPhonemes > 6)
                    throw new ArgumentException("short_utterance_policy requires positive scores and max_extra_phonemes in [0, 6]");
            }
            if (answer.ContextPrefixes != null)
                foreach (var context in answer.ContextPrefixes)
                    if (context == null || string.IsNullOrWhiteSpace(context.Text)
                        || context.Phonemes == null || context.Phonemes.Count == 0
                        || context.Phonemes.Exists(string.IsNullOrWhiteSpace))
                        throw new ArgumentException($"'{answer.Text}' context_prefixes requires text and phonemes");
        }

        private static void ValidateScore(double value, string key, string word)
        {
            if (double.IsNaN(value) || double.IsInfinity(value) || value < 0.0 || value > 1.0)
                throw new ArgumentException($"'{word}' {key} must be a finite number in [0, 1]");
        }
    }

    /// <summary>
    /// What action-contrast metadata has to look like, checked the same
    /// way whether it arrived from JSON or was set in code.
    ///
    /// Mirrors `check_action_group` / `validate_contrast_metadata` in
    /// python/runtime/matching/matcher.py, so one bad catalog cannot load
    /// in the web tool and throw here - or, worse, load in both and
    /// silently protect nothing.
    /// </summary>
    public static class ActionContrast
    {
        public const string Rebuild =
            "rebuild targets.json with python -m python.build.build_targets";

        /// <summary>
        /// The group, or null when there is none. Absent is legacy data;
        /// empty or whitespaced is a name that matches nothing, so it is
        /// refused rather than read as absent.
        /// </summary>
        public static string CheckGroup(string value, string owner)
        {
            if (value == null)
            {
                return null;
            }

            if (value.Length == 0 || HasWhitespace(value))
            {
                throw new ArgumentException(
                    $"'{owner}' has an action_group that is not a name: "
                    + $"'{value}'; " + Rebuild);
            }

            return value;
        }

        private static bool HasWhitespace(string value)
        {
            foreach (char c in value)
            {
                if (char.IsWhiteSpace(c))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Refuse an answer whose contrast metadata cannot be honoured.
        /// An empty list is allowed - the build found nothing to embed -
        /// but alternatives without a group name nothing to be told apart
        /// from, and would leave the anchor undefined.
        /// </summary>
        public static void Validate(Answer answer)
        {
            string owner = answer.Text ?? answer.Id;
            CheckGroup(answer.ActionGroup, owner);
            var alternatives = answer.ContrastiveAlternatives;
            if (alternatives == null)
            {
                return;
            }

            if (answer.ActionGroup == null)
            {
                throw new ArgumentException(
                    $"'{owner}' ships contrastive alternatives but no "
                    + "action_group, so nothing says which action it is; "
                    + Rebuild);
            }

            foreach (var alternative in alternatives)
            {
                if (alternative == null || string.IsNullOrEmpty(alternative.Text)
                    || alternative.Phonemes == null
                    || alternative.Phonemes.Count == 0
                    || alternative.Phonemes.Exists(string.IsNullOrEmpty))
                {
                    throw new ArgumentException(
                        $"'{owner}' has a contrastive alternative missing "
                        + "text or phonemes; " + Rebuild);
                }

                if (CheckGroup(alternative.ActionGroup, owner) == null)
                {
                    throw new ArgumentException(
                        $"'{owner}' has a contrastive alternative with no "
                        + "action_group; " + Rebuild);
                }
            }
        }
    }

    /// <summary>
    /// Parsed targets.json. The runtime never runs G2P: the phoneme
    /// sequences here were produced offline with the full phonological
    /// rule set, which is why device code only needs the jamo tables.
    ///
    /// A flat list. Words used to be grouped into segments that competed
    /// with each other; measured on the golden set it changed neither
    /// positives nor negative rejection, so the grouping is gone and a
    /// question is scored against the word that was asked for.
    /// </summary>
    public sealed class TargetCatalog
    {
        public string Version;
        public string ConfusionMatrixId;
        public List<Answer> Answers = new List<Answer>();

        /// <summary>Look a word up by id or by text; null if absent.</summary>
        public Answer Find(string answerIdOrText)
        {
            return Answers.Find(
                a => a.Id == answerIdOrText || a.Text == answerIdOrText);
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

        /// <param name="matrixId">
        /// Which matrix this JSON is, for a file that does not say. The
        /// child matrix carries "matrix_id", so the one-argument call
        /// names itself and the exact-single-frame calibration can be
        /// checked against the matrix identity. Older files have no id;
        /// left null they load as "unknown", which identifies nothing and
        /// gets no calibrated shortcut. The file's own id wins - naming a
        /// matrix here cannot rename one that already said what it is.
        /// </param>
        public static ConfusionMatrix LoadMatrix(string json, string matrixId = null)
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
                matrixId: (string)root["matrix_id"] ?? matrixId ?? "unknown",
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

            if (!(root["answers"] is JArray answers))
            {
                throw new ArgumentException(
                    "targets.json has no 'answers' list. Rebuild it with "
                    + "python -m python.build.build_targets");
            }

            foreach (var a in answers)
            {
                var answer = new Answer
                {
                    Id = (string)a["id"],
                    Text = (string)a["text"],
                    Threshold = (double?)a["threshold"] ?? 0.6,
                    ConfirmationFrames = ReadConfirmationFrames(a["confirmation_frames"]),
                    MinWholeScore = ReadPolicyScore(a["min_whole_score"], "min_whole_score"),
                    ContrastMargin = ReadPolicyScore(a["contrast_margin"], "contrast_margin"),
                    ContextPrefixes = ReadContextPrefixes(a["context_prefixes"]),
                    ShortUtterancePolicy = ReadShortUtterancePolicy(a["short_utterance_policy"]),
                    AcousticKeywordPolicy = ReadAcousticKeywordPolicy(a["acoustic_keyword_policy"]),
                    // Absent in older data, and absent means the strict
                    // two-frame rule. Only a real JSON boolean counts: a
                    // string "false" must not read as approval.
                    AllowExactSingleFrame =
                        a["allow_exact_single_frame"] is JValue flag
                        && flag.Type == JTokenType.Boolean
                        && (bool)flag,
                    ExactSingleFrameCalibration =
                        ReadCalibration(a["exact_single_frame_calibration"]),
                    ActionGroup = ReadGroup(a["action_group"], (string)a["text"]),
                    ContrastiveAlternatives = ReadAlternatives(
                        a["contrastive_alternatives"], (string)a["text"])
                };

                if (a["phonemes"] is JArray phonemes)
                {
                    foreach (var p in phonemes)
                    {
                        answer.Phonemes.Add((string)p);
                    }
                }

                // The same semantic check the scorer runs, so a catalog
                // that would throw when a child speaks throws when it is
                // loaded instead.
                ActionContrast.Validate(answer);
                AnswerPolicy.Validate(answer);
                catalog.Answers.Add(answer);
            }

            return catalog;
        }

        private static AcousticKeywordPolicy ReadAcousticKeywordPolicy(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null) return null;
            if (!(token is JObject obj) || !Number(obj["min_score"]).HasValue
                || !Number(obj["min_log_probability"]).HasValue || !Number(obj["max_log_gap"]).HasValue
                || !Number(obj["max_contrast_log_gap"]).HasValue || !Count(obj["max_extra_phonemes"]).HasValue
                || Text(obj["contrast_text"]) == null)
                throw new ArgumentException("acoustic_keyword_policy is incomplete or has invalid types");
            return new AcousticKeywordPolicy
            {
                MinScore = Number(obj["min_score"]).Value,
                MinLogProbability = Number(obj["min_log_probability"]).Value,
                MaxLogGap = Number(obj["max_log_gap"]).Value,
                MaxContrastLogGap = Number(obj["max_contrast_log_gap"]).Value,
                ContrastText = Text(obj["contrast_text"]),
                MaxExtraPhonemes = Count(obj["max_extra_phonemes"]).Value
            };
        }

        private static ShortUtterancePolicy ReadShortUtterancePolicy(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null) return null;
            if (!(token is JObject obj) || !Number(obj["threshold"]).HasValue
                || !Number(obj["min_whole_score"]).HasValue || !Count(obj["max_extra_phonemes"]).HasValue)
                throw new ArgumentException("short_utterance_policy requires threshold, min_whole_score and max_extra_phonemes");
            return new ShortUtterancePolicy
            {
                Threshold = Number(obj["threshold"]).Value,
                MinWholeScore = Number(obj["min_whole_score"]).Value,
                MaxExtraPhonemes = Count(obj["max_extra_phonemes"]).Value
            };
        }

        private static List<ContextPrefix> ReadContextPrefixes(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null) return null;
            if (!(token is JArray array))
                throw new ArgumentException("context_prefixes must be a list");
            var contexts = new List<ContextPrefix>(array.Count);
            foreach (var entry in array)
            {
                if (!(entry is JObject obj))
                    throw new ArgumentException("context_prefixes entries must be objects");
                contexts.Add(new ContextPrefix
                {
                    Text = Text(obj["text"]),
                    Phonemes = Strings(obj["phonemes"])
                });
            }
            return contexts;
        }

        private static int? ReadConfirmationFrames(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null) return null;
            if (token.Type != JTokenType.Integer || !int.TryParse(token.ToString(), out int frames)
                || frames < 1 || frames > 6)
                throw new ArgumentException("confirmation_frames must be an integer in [1, 6]");
            return frames;
        }

        private static double ReadPolicyScore(JToken token, string key)
        {
            if (token == null) return 0.0;
            double? value = Number(token);
            if (!value.HasValue)
                throw new ArgumentException(key + " must be a finite number in [0, 1]");
            return value.Value;
        }

        /// <summary>
        /// Parse the calibration snapshot, or null.
        ///
        /// Every field the runtime compares itself against has to be
        /// present and of the right JSON type. A snapshot missing one, or
        /// carrying a string where a number belongs, describes nothing
        /// checkable, so it reads as null - no shortcut - rather than
        /// throwing on the cast or quietly coercing the value. Python
        /// refuses the same file for the same reasons.
        /// </summary>
        private static ExactSingleFrameCalibration ReadCalibration(JToken token)
        {
            if (!(token is JObject snapshot))
            {
                return null;
            }

            var profile = snapshot["streaming_profile"] as JObject;
            if (profile == null)
            {
                return null;
            }

            string matrixId = Text(snapshot["matrix_id"]);
            string text = Text(snapshot["text"]);
            int? consecutive = Count(snapshot["consecutive"]);
            double? threshold = Number(snapshot["threshold"]);
            double? skipCost = Number(profile["skip_cost"]);
            double? coverage = Number(profile["coverage"]);
            double? contextMult = Number(profile["context_mult"]);
            List<string> phonemes = Strings(snapshot["phonemes"]);

            if (!ConfusionMatrix.NamesAMatrix(matrixId)
                || text == null || phonemes == null
                || !consecutive.HasValue || !threshold.HasValue
                || !skipCost.HasValue || !coverage.HasValue
                || !contextMult.HasValue)
            {
                return null;
            }

            return new ExactSingleFrameCalibration
            {
                MatrixId = matrixId,
                Consecutive = consecutive.Value,
                Text = text,
                Threshold = threshold.Value,
                Phonemes = phonemes,
                SkipCost = skipCost,
                Coverage = coverage,
                ContextMult = contextMult,
            };
        }

        /// <summary>A JSON string, or null for anything else.</summary>
        private static string Text(JToken token)
        {
            return token is JValue value && value.Type == JTokenType.String
                ? (string)value.Value
                : null;
        }

        /// <summary>
        /// The answer's action group, or null when it has none.
        ///
        /// Absent is legacy data and means no contrast set. Present but
        /// not a name is a build that went wrong, and defaulting it
        /// would drop the protection the field exists for, so it throws -
        /// exactly as the Python reader does on the same file.
        /// </summary>
        private static string ReadGroup(JToken token, string owner)
        {
            if (token == null || token.Type == JTokenType.Null)
            {
                return null;
            }

            string value = Text(token);
            if (value == null)
            {
                throw new ArgumentException(
                    $"'{owner}' has an action_group that is not a string; "
                    + ActionContrast.Rebuild);
            }

            return ActionContrast.CheckGroup(value, owner);
        }

        /// <summary>The embedded contrast set, or null when there is none.</summary>
        private static List<ContrastiveAlternative> ReadAlternatives(
            JToken token, string owner)
        {
            if (token == null || token.Type == JTokenType.Null)
            {
                return null;
            }

            var array = token as JArray;
            if (array == null)
            {
                throw new ArgumentException(
                    $"'{owner}' has contrastive_alternatives that are not a "
                    + "list; " + ActionContrast.Rebuild);
            }

            var items = new List<ContrastiveAlternative>(array.Count);
            foreach (var entry in array)
            {
                var obj = entry as JObject;
                string text = obj == null ? null : Text(obj["text"]);
                string group = obj == null ? null : Text(obj["action_group"]);
                List<string> phonemes = obj == null ? null : Strings(obj["phonemes"]);
                if (string.IsNullOrEmpty(text) || string.IsNullOrEmpty(group)
                    || phonemes == null || phonemes.Exists(string.IsNullOrEmpty))
                {
                    throw new ArgumentException(
                        $"'{owner}' has a contrastive alternative missing text, "
                        + "phonemes or action_group; " + ActionContrast.Rebuild);
                }

                items.Add(new ContrastiveAlternative
                {
                    Text = text,
                    ActionGroup = ActionContrast.CheckGroup(group, owner),
                    Phonemes = phonemes,
                });
            }

            return items;
        }

        /// <summary>
        /// A finite JSON number, or null. true and "0.8" are not numbers:
        /// coercing them would invent a calibration nobody measured.
        /// </summary>
        private static double? Number(JToken token)
        {
            if (!(token is JValue value))
            {
                return null;
            }

            double parsed;
            if (value.Value is double asDouble) { parsed = asDouble; }
            else if (value.Value is long asLong) { parsed = asLong; }
            else if (value.Value is int asInt) { parsed = asInt; }
            else if (value.Value is decimal asDecimal) { parsed = (double)asDecimal; }
            else { return null; }

            return double.IsNaN(parsed) || double.IsInfinity(parsed)
                ? (double?)null
                : parsed;
        }

        /// <summary>A JSON integer that fits an int, or null.</summary>
        private static int? Count(JToken token)
        {
            if (!(token is JValue value) || value.Type != JTokenType.Integer
                || !(value.Value is long number)
                || number < int.MinValue || number > int.MaxValue)
            {
                return null;
            }

            return (int)number;
        }

        /// <summary>A non-empty array of JSON strings, or null.</summary>
        private static List<string> Strings(JToken token)
        {
            if (!(token is JArray array) || array.Count == 0)
            {
                return null;
            }

            var items = new List<string>(array.Count);
            foreach (var item in array)
            {
                string value = Text(item);
                if (value == null)
                {
                    return null;
                }

                items.Add(value);
            }

            return items;
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
