using System;
using System.Collections.Generic;
using System.Linq;

namespace DomiCube.PhonemeMatching.Korean
{
    /// <summary>
    /// Explains the existing scoring trace, without ASR, G2P or rescoring.
    /// Mirror of python/runtime/matching/korean_feedback.py. Written syllables
    /// are localized only when their canonical IPA exactly matches the target.
    /// </summary>
    internal static class KoreanFeedbackBuilder
    {
        private static readonly string[] InitialNames =
        {
            "기역", "쌍기역", "니은", "디귿", "쌍디귿", "리을", "미음", "비읍", "쌍비읍",
            "시옷", "쌍시옷", "이응", "지읒", "쌍지읒", "치읓", "키읔", "티읕", "피읖", "히읗"
        };
        private static readonly Dictionary<string, string> SoundLabels = BuildSoundLabels();
        // Representative coda sounds, NOT inferred orthographic spellings.
        private static readonly Dictionary<string, string[]> SimpleCodas =
            "ㄱㄴㄷㄹㅁㅂㅇ".ToDictionary(ch => ch.ToString(), ch => JamoIpa.Coda[ch.ToString()]);

        private static string Compose(string initial, string medial, string final)
        {
            return ((char)(JamoIpa.HangulBase
                + (Array.IndexOf(JamoIpa.Initials, initial) * 21
                    + Array.IndexOf(JamoIpa.Medials, medial)) * 28
                + Array.IndexOf(JamoIpa.Finals, final))).ToString();
        }

        private static Dictionary<string, string> BuildSoundLabels()
        {
            var labels = new Dictionary<string, string>();
            for (int i = 0; i < JamoIpa.Initials.Length; i++)
            {
                var tokens = JamoIpa.Onset[JamoIpa.Initials[i]];
                if (tokens.Length > 0) labels[tokens[0]] = InitialNames[i] + " 소리";
            }
            foreach (var pair in JamoIpa.Nucleus)
                if (pair.Value.Length == 1)
                    labels[pair.Value[0]] = $"‘{Compose("ㅇ", pair.Key, "")}’의 모음 소리";
            labels["j"] = "‘야’에서 앞에 붙는 소리";
            labels["w"] = "‘와’에서 앞에 붙는 소리";
            labels["ɰ"] = "‘의’에서 앞에 붙는 소리";
            labels["k̚"] = "기역 계열 받침 소리";
            labels["t̚"] = "디귿 계열 받침 소리";
            labels["p̚"] = "비읍 계열 받침 소리";
            labels["l"] = "리을 소리";
            labels["ŋ"] = "받침 이응 소리";
            return labels;
        }

        private static string Sound(string phoneme)
        {
            return SoundLabels.TryGetValue(phoneme, out var label)
                ? label : "한글로 옮기기 어려운 소리";
        }

        private static string Describe(PhonemeComparison step)
        {
            switch (step.Operation)
            {
                case "sub":
                    return $"{Sound(step.TargetPhoneme)}가 {Sound(step.UserPhoneme)}로 인식됐어요.";
                case "del":
                    return $"{Sound(step.TargetPhoneme)}가 인식 결과에서 빠졌어요.";
                default:
                    return $"정답에 없는 {Sound(step.UserPhoneme)}가 추가로 인식됐어요.";
            }
        }

        private static string Reverse(List<string> tokens,
            Dictionary<string, string[]> table, string original)
        {
            if (table.TryGetValue(original, out var unchanged) && unchanged.SequenceEqual(tokens))
                return original;
            var candidates = table.Where(pair => pair.Value.SequenceEqual(tokens))
                .Select(pair => pair.Key).ToList();
            return candidates.Count == 1 ? candidates[0] : null;
        }

        private static string HeardSyllable(string[] jamo, List<string>[] parts)
        {
            var initial = Reverse(parts[0], JamoIpa.Onset, jamo[0]);
            var medial = Reverse(parts[1], JamoIpa.Nucleus, jamo[1]);
            string final;
            if (parts[2].Count == 0) final = "";
            else if (JamoIpa.Coda.TryGetValue(jamo[2], out var unchanged)
                && unchanged.SequenceEqual(parts[2])) final = jamo[2];
            else final = Reverse(parts[2], SimpleCodas, jamo[2]);
            if (initial == null || medial == null || final == null) return "";
            return Compose(initial, medial, final);
        }

        public static KoreanFeedback Build(PronunciationFeedback feedback)
        {
            var guide = new KoreanFeedback();
            if (!feedback.AlignmentAvailable)
            {
                guide.Summary = "발음 비교 정보가 없어 자세히 설명할 수 없어요.";
                return guide;
            }
            var syllables = new List<(string Text, string[] Jamo)>();
            var owners = new List<(int Syllable, int Part)>();
            var canonical = new List<string>();
            foreach (char ch in feedback.TargetText ?? "")
            {
                if (!JamoIpa.TryDecompose(ch, out var initial, out var medial, out var final))
                    continue;
                var parts = new[]
                {
                    JamoIpa.Onset[initial], JamoIpa.Nucleus[medial],
                    final.Length == 0 ? Array.Empty<string>() : JamoIpa.Coda[final]
                };
                for (int part = 0; part < parts.Length; part++)
                    foreach (var token in parts[part])
                    {
                        canonical.Add(token);
                        owners.Add((syllables.Count, part));
                    }
                syllables.Add((ch.ToString(), new[] { initial, medial, final }));
            }
            var targetSteps = feedback.Alignment.Where(step => step.Operation != "ins").ToList();
            guide.SyllableMappingAvailable = syllables.Count > 0
                && canonical.SequenceEqual(feedback.TargetPhonemes)
                && targetSteps.Select(step => step.TargetPhoneme).SequenceEqual(canonical)
                && targetSteps.Select(step => step.TargetIndex).SequenceEqual(Enumerable.Range(0, canonical.Count));
            var differences = Enumerable.Range(0, feedback.Alignment.Count)
                .Where(i => feedback.Alignment[i].Operation != "match").ToList();
            if (differences.Count == 0)
            {
                guide.Summary = "비교한 구간에서 정답과 다른 소리가 발견되지 않았어요.";
                return guide;
            }
            guide.Summary = "정답과 다르게 인식된 부분이에요.";
            if (!guide.SyllableMappingAvailable)
            {
                guide.Summary += " 글자 위치를 정확히 연결하기 어려워 소리만 안내해요.";
                foreach (int i in differences)
                    guide.Items.Add(new PronunciationHint
                    {
                        Kind = feedback.Alignment[i].Operation, SyllableIndex = -1,
                        TargetSyllable = "", HeardSyllable = "",
                        Message = Describe(feedback.Alignment[i]), AlignmentIndices = new List<int> { i }
                    });
                return guide;
            }

            // Insertions have no target position. Never force them into a
            // syllable, or reconstruct their neighbors as complete sounds.
            var blocked = new HashSet<int>();
            for (int i = 0; i < feedback.Alignment.Count; i++)
            {
                if (feedback.Alignment[i].Operation != "ins") continue;
                foreach (int direction in new[] { -1, 1 })
                    for (int j = i + direction; j >= 0 && j < feedback.Alignment.Count; j += direction)
                        if (feedback.Alignment[j].TargetIndex >= 0)
                        {
                            blocked.Add(owners[feedback.Alignment[j].TargetIndex].Syllable);
                            break;
                        }
            }
            var heardParts = syllables.Select(_ => new[]
                { new List<string>(), new List<string>(), new List<string>() }).ToList();
            var groups = new Dictionary<int, List<int>>();
            for (int i = 0; i < feedback.Alignment.Count; i++)
            {
                var step = feedback.Alignment[i];
                if (step.Operation == "ins")
                {
                    guide.Items.Add(new PronunciationHint
                    {
                        Kind = "ins", SyllableIndex = -1, TargetSyllable = "", HeardSyllable = "",
                        Message = Describe(step), AlignmentIndices = new List<int> { i }
                    });
                    continue;
                }
                var owner = owners[step.TargetIndex];
                if (step.Operation != "del") heardParts[owner.Syllable][owner.Part].Add(step.UserPhoneme);
                if (step.Operation != "match")
                {
                    if (!groups.TryGetValue(owner.Syllable, out var group))
                        groups[owner.Syllable] = group = new List<int>();
                    group.Add(i);
                }
            }
            foreach (var group in groups)
            {
                int owner = group.Key;
                var indices = group.Value;
                var syllable = syllables[owner];
                var operations = indices.Select(i => feedback.Alignment[i].Operation).Distinct().ToList();
                string kind = operations.Count == 1 ? operations[0] : "mixed";
                string subject = $"‘{feedback.TargetText}’의 ‘{syllable.Text}’";
                string heard = blocked.Contains(owner) ? "" : HeardSyllable(syllable.Jamo, heardParts[owner]);
                string message;
                if (heard.Length > 0 && heard != syllable.Text)
                {
                    string particle = syllable.Jamo[2].Length > 0 ? "이" : "가";
                    message = $"{subject}{particle} ‘{heard}’에 가까운 소리로 인식됐어요.";
                }
                else if (heardParts[owner].All(part => part.Count == 0) && !blocked.Contains(owner))
                    message = $"{subject} 소리가 인식 결과에서 빠졌어요.";
                else
                {
                    heard = "";
                    message = subject + ": " + string.Join(" ", indices.Select(i => Describe(feedback.Alignment[i])));
                }
                guide.Items.Add(new PronunciationHint
                {
                    Kind = kind, SyllableIndex = owner, TargetSyllable = syllable.Text,
                    HeardSyllable = heard, Message = message, AlignmentIndices = new List<int>(indices)
                });
            }
            guide.Items.Sort((a, b) => a.AlignmentIndices[0].CompareTo(b.AlignmentIndices[0]));
            return guide;
        }
    }
}
