using System.Collections.Generic;
using System.Text;

namespace DomiCube.PhonemeMatching.Korean
{
    /// <summary>
    /// Hangul -> IPA, mirroring python/build/g2p/ko/jamo_ipa.py.
    ///
    /// This is the RUNTIME path and it deliberately applies no
    /// phonological rules. The build pipeline runs g2pkk offline when it
    /// produces targets.json; doing the same on device would mean porting
    /// a mecab-backed rule engine. Measured against the golden set, the
    /// rules change 3 of 18 target words and cost 1.4 points of negative
    /// rejection (91.7% -> 90.3%) with positives unchanged, because the
    /// three differences (r/l, k/k-unreleased, k/k-tense) are pairs the
    /// confusion matrix already treats as near-free.
    /// </summary>
    public static class JamoIpa
    {
        public const int HangulBase = 0xAC00;
        public const int HangulEnd = 0xD7A3;

        private const int MedialCount = 21;
        private const int FinalCount = 28;

        private static readonly string[] Initials =
        {
            "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
            "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
        };

        private static readonly string[] Medials =
        {
            "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
            "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
        };

        // Index 0 = no final consonant.
        private static readonly string[] Finals =
        {
            "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
            "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
            "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
        };

        /// <summary>Onset jamo -> IPA. ㅇ is silent in onset position.</summary>
        private static readonly Dictionary<string, string[]> Onset =
            new Dictionary<string, string[]>
            {
                { "ㄱ", new[] { "k" } },
                { "ㄲ", new[] { "k͈" } },
                { "ㅋ", new[] { "kʰ" } },
                { "ㄷ", new[] { "t" } },
                { "ㄸ", new[] { "t͈" } },
                { "ㅌ", new[] { "tʰ" } },
                { "ㅂ", new[] { "p" } },
                { "ㅃ", new[] { "p͈" } },
                { "ㅍ", new[] { "pʰ" } },
                { "ㅈ", new[] { "tɕ" } },
                { "ㅉ", new[] { "tɕ͈" } },
                { "ㅊ", new[] { "tɕʰ" } },
                { "ㅅ", new[] { "s" } },
                { "ㅆ", new[] { "s͈" } },
                { "ㅎ", new[] { "h" } },
                { "ㄴ", new[] { "n" } },
                { "ㅁ", new[] { "m" } },
                { "ㄹ", new[] { "ɾ" } },
                { "ㅇ", new string[0] }
            };

        /// <summary>
        /// Nucleus jamo -> IPA. Diphthongs split into glide + vowel so the
        /// matcher can tolerate glide deletion, which is common in child
        /// speech.
        /// </summary>
        private static readonly Dictionary<string, string[]> Nucleus =
            new Dictionary<string, string[]>
            {
                { "ㅏ", new[] { "a" } },
                { "ㅓ", new[] { "ʌ" } },
                { "ㅗ", new[] { "o" } },
                { "ㅜ", new[] { "u" } },
                { "ㅡ", new[] { "ɯ" } },
                { "ㅣ", new[] { "i" } },
                { "ㅐ", new[] { "ɛ" } },
                { "ㅔ", new[] { "e" } },
                { "ㅑ", new[] { "j", "a" } },
                { "ㅕ", new[] { "j", "ʌ" } },
                { "ㅛ", new[] { "j", "o" } },
                { "ㅠ", new[] { "j", "u" } },
                { "ㅒ", new[] { "j", "ɛ" } },
                { "ㅖ", new[] { "j", "e" } },
                { "ㅘ", new[] { "w", "a" } },
                { "ㅙ", new[] { "w", "ɛ" } },
                { "ㅝ", new[] { "w", "ʌ" } },
                { "ㅞ", new[] { "w", "e" } },
                { "ㅚ", new[] { "w", "e" } },
                { "ㅟ", new[] { "w", "i" } },
                { "ㅢ", new[] { "ɰ", "i" } }
            };

        /// <summary>Coda jamo -> IPA. Korean coda stops are unreleased.</summary>
        private static readonly Dictionary<string, string[]> Coda =
            new Dictionary<string, string[]>
            {
                { "ㄱ", new[] { "k̚" } },
                { "ㄲ", new[] { "k̚" } },
                { "ㅋ", new[] { "k̚" } },
                { "ㄴ", new[] { "n" } },
                { "ㄷ", new[] { "t̚" } },
                { "ㅅ", new[] { "t̚" } },
                { "ㅆ", new[] { "t̚" } },
                { "ㅈ", new[] { "t̚" } },
                { "ㅊ", new[] { "t̚" } },
                { "ㅌ", new[] { "t̚" } },
                { "ㅎ", new[] { "t̚" } },
                { "ㄹ", new[] { "l" } },
                { "ㅁ", new[] { "m" } },
                { "ㅂ", new[] { "p̚" } },
                { "ㅍ", new[] { "p̚" } },
                { "ㅇ", new[] { "ŋ" } },
                // Clusters, reduced per 자음군 단순화.
                { "ㄳ", new[] { "k̚" } },
                { "ㄵ", new[] { "n" } },
                { "ㄶ", new[] { "n" } },
                { "ㄺ", new[] { "k̚" } },
                { "ㄻ", new[] { "m" } },
                { "ㄼ", new[] { "l" } },
                { "ㄽ", new[] { "l" } },
                { "ㄾ", new[] { "l" } },
                { "ㄿ", new[] { "p̚" } },
                { "ㅀ", new[] { "l" } },
                { "ㅄ", new[] { "p̚" } }
            };

        /// <summary>
        /// Decompose one Hangul syllable into (initial, medial, final).
        /// Final is empty when absent. Returns false for non-Hangul.
        /// </summary>
        public static bool TryDecompose(
            char syllable, out string initial, out string medial,
            out string final)
        {
            initial = medial = final = string.Empty;
            int code = syllable;
            if (code < HangulBase || code > HangulEnd)
            {
                return false;
            }

            int offset = code - HangulBase;
            initial = Initials[offset / (MedialCount * FinalCount)];
            medial = Medials[offset / FinalCount % MedialCount];
            final = Finals[offset % FinalCount];
            return true;
        }

        /// <summary>
        /// Convert Hangul text to an IPA phoneme list. Non-Hangul
        /// characters (spaces, punctuation, latin, digits) are skipped.
        /// </summary>
        public static List<string> ToPhonemes(string text)
        {
            var result = new List<string>();
            if (string.IsNullOrEmpty(text))
            {
                return result;
            }

            foreach (char ch in text)
            {
                if (!TryDecompose(ch, out var initial, out var medial,
                        out var final))
                {
                    continue;
                }

                if (Onset.TryGetValue(initial, out var onset))
                {
                    result.AddRange(onset);
                }

                if (Nucleus.TryGetValue(medial, out var nucleus))
                {
                    result.AddRange(nucleus);
                }

                if (final.Length > 0 && Coda.TryGetValue(final, out var coda))
                {
                    result.AddRange(coda);
                }
            }

            return result;
        }

        /// <summary>Keep only Hangul syllables and spaces.</summary>
        public static string Sanitize(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return string.Empty;
            }

            var sb = new StringBuilder(text.Length);
            foreach (char ch in text)
            {
                if ((ch >= HangulBase && ch <= HangulEnd) || char.IsWhiteSpace(ch))
                {
                    sb.Append(ch);
                }
            }

            return sb.ToString().Trim();
        }
    }
}
