using System;
using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace DomiCube.PhonemeMatching
{
    /// <summary>Local diagnostic export; records model output, never a child's pronunciation verdict.</summary>
    public static class RecognitionDiagnostics
    {
        public static string ToJson(IReadOnlyList<string> targets, IReadOnlyList<FrameScore> frames)
        {
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (frames == null) throw new ArgumentNullException(nameof(frames));
            var rows = new JArray();
            foreach (var frame in frames)
            {
                if (frame == null) continue;
                var best = frame.Best;
                rows.Add(new JObject
                {
                    ["end_sample"] = frame.EndSampleIndex,
                    ["recognized_text"] = frame.Text,
                    ["phonemes"] = new JArray(frame.Phonemes ?? new List<string>()),
                    ["audio_rms"] = frame.AudioRms,
                    ["audio_peak"] = frame.AudioPeak,
                    ["decision_reason"] = frame.DecisionReason,
                    ["target"] = best?.TargetText,
                    ["score"] = best?.Score,
                    ["threshold"] = best?.Threshold,
                    ["short_utterance_policy"] = best?.UsedShortUtterancePolicy,
                    ["acoustic_keyword_policy"] = best?.AcousticEvidence != null,
                    ["acoustic_log_probability"] = best?.AcousticEvidence?.LogProbability,
                    ["acoustic_greedy_log_probability"] = best?.AcousticEvidence?.GreedyLogProbability,
                    ["acoustic_contrast_log_probability"] = best?.AcousticEvidence?.ContrastLogProbability,
                    ["whole_score"] = best?.AcceptedWholeScore,
                    ["other_action"] = best?.CompetitorText,
                    ["other_action_score"] = best?.CompetitorScore,
                    ["streak"] = frame.Streak,
                    ["required_frames"] = frame.RequiredFrames,
                    ["confirmed"] = frame.Confirmed
                });
                if (frame.WordDecision != null)
                    ((JObject)rows[rows.Count - 1])["web_word_policy"] = JObject.FromObject(frame.WordDecision);
            }
            return new JObject
            {
                ["schema_version"] = 1,
                ["created_utc"] = DateTime.UtcNow.ToString("o"),
                ["targets"] = new JArray(targets),
                ["note"] = "Model recognition and decision diagnostics; no audio recording or pronunciation diagnosis.",
                ["frames"] = rows
            }.ToString(Formatting.Indented);
        }
    }
}
