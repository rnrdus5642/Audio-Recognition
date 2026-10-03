using System;
using System.Collections.Generic;
using System.Text;

namespace DomiCube.PhonemeMatching
{
    /// <summary>
    /// The token table a CTC model emits, as exported by
    /// python/tools/export_ctc_vocab.py.
    ///
    /// Sized to the model's output width, which is smaller than the
    /// tokenizer's vocabulary: the tokenizer carries sentence markers the
    /// acoustic model can never produce, and including them would shift
    /// every id.
    /// </summary>
    public sealed class CtcVocabulary
    {
        public string[] Tokens = new string[0];

        /// <summary>The CTC blank. Emitted constantly; never decoded.</summary>
        public int BlankId = -1;

        public int UnkId = -1;

        /// <summary>Decoded as a space rather than as its own symbol.</summary>
        public int WordDelimiterId = -1;

        /// <summary>Whether the model expects zero-mean unit-variance input.</summary>
        public bool Normalize = true;

        public float NormalizeEpsilon = 1e-7f;

        public int SamplingRate = 16000;

        public int Size { get { return Tokens.Length; } }
    }

    /// <summary>
    /// Greedy CTC decoding: argmax per frame, collapse repeats, drop
    /// blanks. Mirrors what <c>Wav2Vec2Processor.batch_decode</c> does in
    /// Python, which is where every threshold in this project was tuned.
    ///
    /// No beam search and no language model, deliberately: an LM would
    /// "correct" a mispronunciation into the word it resembles, which is
    /// exactly the judgement this system exists to make.
    ///
    /// Verified against Python in
    /// <c>Tests/Runtime/ctc_vectors.json</c>.
    /// </summary>
    public static class CtcDecoder
    {
        /// <summary>
        /// Pick the highest-scoring token per frame.
        ///
        /// <paramref name="logits"/> is a flat row-major
        /// [frames, classes] block, the shape Sentis hands back.
        /// </summary>
        public static List<int> ArgMax(float[] logits, int frames, int classes)
        {
            if (logits == null)
            {
                throw new ArgumentNullException("logits");
            }

            if (frames < 0 || classes <= 0
                || (long)frames * classes > logits.Length)
            {
                throw new ArgumentException(
                    string.Format(
                        "logits hold {0} values, not enough for {1}x{2}",
                        logits.Length, frames, classes));
            }

            var ids = new List<int>(frames);
            for (int f = 0; f < frames; f++)
            {
                int offset = f * classes;
                int best = 0;
                float bestValue = logits[offset];
                for (int c = 1; c < classes; c++)
                {
                    float v = logits[offset + c];
                    if (v > bestValue)
                    {
                        bestValue = v;
                        best = c;
                    }
                }

                ids.Add(best);
            }

            return ids;
        }

        /// <summary>
        /// Token ids to text. Repeated ids collapse to one, blanks are
        /// dropped, and the word delimiter becomes a space.
        ///
        /// Order matters: repeats collapse BEFORE blanks are removed, so
        /// a blank between two identical tokens preserves both - that is
        /// how CTC represents a genuine double letter.
        /// </summary>
        public static string Decode(IList<int> ids, CtcVocabulary vocab)
        {
            if (vocab == null)
            {
                throw new ArgumentNullException("vocab");
            }

            if (ids == null || ids.Count == 0)
            {
                return string.Empty;
            }

            var sb = new StringBuilder(ids.Count);
            int previous = -1;

            for (int i = 0; i < ids.Count; i++)
            {
                int id = ids[i];
                if (id < 0 || id >= vocab.Size)
                {
                    // A model/vocabulary mismatch would otherwise decode
                    // into plausible-looking nonsense, so say it now.
                    throw new ArgumentOutOfRangeException(
                        "ids",
                        string.Format(
                            "token id {0} outside vocabulary of {1}. "
                            + "Regenerate with "
                            + "python -m python.tools.export_ctc_vocab",
                            id, vocab.Size));
                }

                if (id != previous && id != vocab.BlankId)
                {
                    sb.Append(id == vocab.WordDelimiterId
                        ? " "
                        : vocab.Tokens[id]);
                }

                previous = id;
            }

            return Sanitize(sb.ToString());
        }

        /// <summary>Logits straight to text.</summary>
        public static string Decode(
            float[] logits, int frames, int classes, CtcVocabulary vocab)
        {
            return Decode(ArgMax(logits, frames, classes), vocab);
        }

        /// <summary>
        /// Keep Hangul syllables and spacing, drop everything else.
        ///
        /// The vocabulary's placeholder tokens ([UNK], [PAD]) decode to
        /// their literal spelling; dropping them here keeps that out of
        /// the G2P stage, which expects Hangul.
        /// </summary>
        public static string Sanitize(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return string.Empty;
            }

            var sb = new StringBuilder(text.Length);
            foreach (char c in text)
            {
                if ((c >= Korean.JamoIpa.HangulBase
                        && c <= Korean.JamoIpa.HangulEnd)
                    || c == ' ' || c == '\t' || c == '\n' || c == '\r')
                {
                    sb.Append(c);
                }
            }

            return sb.ToString().Trim();
        }
    }
}
