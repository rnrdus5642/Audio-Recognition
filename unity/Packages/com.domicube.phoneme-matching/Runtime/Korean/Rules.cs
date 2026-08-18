using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace DomiCube.PhonemeMatching.Korean
{
    /// <summary>
    /// Korean phonological rules: written form to the form it is actually
    /// spoken as. 먹어요 -> 머거요, 학교 -> 학꾜, 쳐 -> 처.
    /// </summary>
    /// <remarks>
    /// The answers carry surface forms because the build applies these
    /// rules to them, so the user's speech has to pass through the same
    /// thing or the two sides compare different spellings of the same
    /// sound. Until this existed the device skipped the step, and a
    /// perfectly spoken 쳐 scored 0.700 against a threshold of 0.850 -
    /// unreachable.
    ///
    /// This is g2pkk minus its morphological analyser. That analyser
    /// needs a 112 MB dictionary, and it earns nothing here: the five
    /// rules depending on it fire on 관형형 ㄹ 뒤 된소리 and 용언 어간
    /// 뒤 경음화, which need a word to carry an ending, and the product
    /// asks for single words. Measured identical on all 29 curriculum
    /// words and on 99.5% of single-word corpus utterances.
    ///
    /// Verified against the Python implementation by ParityTests over
    /// 807 inputs covering every entry of the rule table.
    /// </remarks>
    public static class Rules
    {
        private const int SyllableBase = 0xAC00;
        private const int MedialCount = 21;
        private const int FinalCount = 28;
        private const int InitialBase = 0x1100;
        private const int MedialBase = 0x1161;
        private const int FinalBase = 0x11A7;

        private static readonly Regex SilentOnset = new Regex(
            "(^|[^ᄀ-ᄒ])([ᅡ-ᅵ])");
        private static readonly Regex Jyeo = new Regex("([ᄌᄍᄎ])ᅧ");
        // Spelled out rather than a ᄀ-ᄒ range: ᄋ sits inside that range
        // and must not match, or 의 turns into 이 and every genitive in
        // the corpus shifts.
        private static readonly Regex ConsonantUi = new Regex(
            "([ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄌᄍᄎᄏᄐᄑᄒ])ᅴ");
        private static readonly Regex[] Compiled = BuildTable();

        private static Regex[] BuildTable()
        {
            int n = RulesData.Table.GetLength(0);
            var out_ = new Regex[n];
            for (int i = 0; i < n; i++)
            {
                out_[i] = new Regex(RulesData.Table[i, 0]);
            }

            return out_;
        }

        /// <summary>
        /// The surface form of <paramref name="text"/>. Input that is not
        /// Hangul passes through untouched.
        /// </summary>
        public static string Apply(string text)
        {
            if (string.IsNullOrEmpty(text) || text.Trim().Length == 0)
            {
                return text;
            }

            string s = Decompose(text);
            s = Jyeo.Replace(s, m => m.Groups[1].Value + "ᅥ");
            s = ConsonantUi.Replace(s, m => m.Groups[1].Value + "ᅵ");
            s = JamoNames(s);
            s = Balb(s);
            s = Palatalize(s);

            for (int i = 0; i < Compiled.Length; i++)
            {
                s = Compiled[i].Replace(s, RulesData.Table[i, 1]);
            }

            for (int i = 0; i < RulesData.Link.GetLength(0); i++)
            {
                s = s.Replace(RulesData.Link[i, 0], RulesData.Link[i, 1]);
            }

            return Compose(s);
        }

        /// <summary>Hangul syllables to conjoining jamo.</summary>
        private static string Decompose(string text)
        {
            var sb = new StringBuilder(text.Length * 3);
            foreach (char ch in text)
            {
                int code = ch - SyllableBase;
                if (code >= 0 && code < 11172)
                {
                    int initial = code / (MedialCount * FinalCount);
                    int rest = code % (MedialCount * FinalCount);
                    int medial = rest / FinalCount;
                    int final = rest % FinalCount;
                    sb.Append((char)(InitialBase + initial));
                    sb.Append((char)(MedialBase + medial));
                    if (final != 0)
                    {
                        sb.Append((char)(FinalBase + final));
                    }
                }
                else
                {
                    sb.Append(ch);
                }
            }

            return sb.ToString();
        }

        /// <summary>Jamo back to syllables, supplying ㅇ where needed.</summary>
        private static string Compose(string letters)
        {
            letters = SilentOnset.Replace(
                letters, m => m.Groups[1].Value + "ᄋ" + m.Groups[2].Value);

            var sb = new StringBuilder(letters.Length);
            int i = 0;
            while (i < letters.Length)
            {
                char ch = letters[i];
                if (ch >= 0x1100 && ch <= 0x1112 && i + 1 < letters.Length
                    && letters[i + 1] >= 0x1161 && letters[i + 1] <= 0x1175)
                {
                    int initial = ch - InitialBase;
                    int medial = letters[i + 1] - MedialBase;
                    int final = 0;
                    int step = 2;
                    if (i + 2 < letters.Length && letters[i + 2] >= 0x11A8
                        && letters[i + 2] <= 0x11C2)
                    {
                        final = letters[i + 2] - FinalBase;
                        step = 3;
                    }

                    sb.Append((char)(SyllableBase
                        + (initial * MedialCount + medial) * FinalCount
                        + final));
                    i += step;
                }
                else
                {
                    sb.Append(ch);
                    i++;
                }
            }

            return sb.ToString();
        }

        /// <summary>16  letter names: 디귿이 -> 디그시.</summary>
        private static string JamoNames(string s)
        {
            s = Regex.Replace(s, "([그])ᆮᄋ",
                m => m.Groups[1].Value + "ᄉ");
            s = Regex.Replace(s, "([으])[ᆽᆾᇀᇂ]ᄋ",
                m => m.Groups[1].Value + "ᄉ");
            s = Regex.Replace(s, "([으])[ᆿ]ᄋ",
                m => m.Groups[1].Value + "ᄀ");
            s = Regex.Replace(s, "([으])[ᇁ]ᄋ",
                m => m.Groups[1].Value + "ᄇ");
            return s;
        }

        /// <summary>10.1  밟- and 넓죽/넓둥 keep ㅂ.</summary>
        private static string Balb(string s)
        {
            s = Regex.Replace(s, "(바)ᆲ($|[^ᄋᄒ])",
                m => m.Groups[1].Value + "ᆸ" + m.Groups[2].Value);
            s = Regex.Replace(s,
                "(너)ᆲ([ᄌᄍ]ᅮ|[ᄃᄄ]ᅮ)",
                m => m.Groups[1].Value + "ᆸ" + m.Groups[2].Value);
            return s;
        }

        /// <summary>17  구개음화.</summary>
        private static string Palatalize(string s)
        {
            s = Regex.Replace(s, "ᆮᄋ([ᅵᅧ])",
                m => "ᄌ" + m.Groups[1].Value);
            s = Regex.Replace(s, "ᇀᄋ([ᅵᅧ])",
                m => "ᄎ" + m.Groups[1].Value);
            s = Regex.Replace(s, "ᆴᄋ([ᅵᅧ])",
                m => "ᆯᄎ" + m.Groups[1].Value);
            s = Regex.Replace(s, "ᆮᄒ([ᅵ])",
                m => "ᄎ" + m.Groups[1].Value);
            return s;
        }
    }
}
