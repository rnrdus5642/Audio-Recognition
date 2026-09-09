using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    /// <summary>
    /// Checks the greedy CTC decoder against Python's
    /// <c>Wav2Vec2Processor.batch_decode</c>
    /// (python/tools/export_ctc_vocab.py).
    ///
    /// Most cases carry the token ids the shipping ONNX graph really
    /// emitted for the golden clips, so the awkward parts of CTC output -
    /// long blank runs, repeats spanning frames - are covered by what the
    /// model does rather than by what seemed worth writing down.
    ///
    /// Regenerate after changing the model or the vocabulary:
    ///     python -m python.tools.export_ctc_vocab
    /// </summary>
    [TestFixture]
    public sealed class CtcDecoderTests
    {
        private static JObject _vectors;
        private static CtcVocabulary _vocab;

        [OneTimeSetUp]
        public void LoadFixtures()
        {
            _vectors = JObject.Parse(RepoFiles.Read(
                "unity/Packages/com.domicube.phoneme-matching/Tests/Runtime/"
                + "ctc_vectors.json"));
            _vocab = PhonemeData.LoadCtcVocabulary(RepoFiles.Read(
                "unity/Assets/StreamingAssets/wav2vec2_ko_vocab.json"));
        }

        [Test]
        public void VocabularyMatchesTheExportedModel()
        {
            Assert.That(_vocab.Size, Is.EqualTo((int)_vectors["vocab_size"]),
                "vocabulary and vectors came from different exports");
            // True of the export that ships today, and worth noticing if
            // it changes - but it is a fact about this vocabulary, not a
            // rule the decoder relies on. A vocabulary with a syllable
            // appended after the blank decodes just as well; see
            // AppendedTokenCtcDecoderTests.
            Assert.That(_vocab.BlankId, Is.EqualTo(_vocab.Size - 1),
                "the shipped export puts blank last");
            Assert.That(_vocab.Normalize, Is.True,
                "recognizer normalizes input on the strength of this flag");
        }

        [Test]
        public void DecodeMatchesPython()
        {
            var cases = (JArray)_vectors["cases"];
            Assert.That(cases.Count, Is.GreaterThan(0), "no vectors");

            foreach (var c in cases)
            {
                var ids = new List<int>();
                foreach (var id in (JArray)c["ids"])
                {
                    ids.Add((int)id);
                }

                Assert.That(CtcDecoder.Decode(ids, _vocab),
                    Is.EqualTo((string)c["text"]),
                    $"case '{(string)c["id"]}'");
            }
        }

        [Test]
        public void DecodedTextProducesPythonPhonemes()
        {
            foreach (var c in (JArray)_vectors["cases"])
            {
                var expected = new List<string>();
                foreach (var p in (JArray)c["phonemes"])
                {
                    expected.Add((string)p);
                }

                var ids = new List<int>();
                foreach (var id in (JArray)c["ids"])
                {
                    ids.Add((int)id);
                }

                var phonemes = Korean.JamoIpa.ToPhonemes(
                    CtcDecoder.Decode(ids, _vocab));
                Assert.That(phonemes, Is.EqualTo(expected),
                    $"case '{(string)c["id"]}'");
            }
        }

        [Test]
        public void ArgMaxPicksThePerFrameWinner()
        {
            // 3 frames x 4 classes, winners 2, 0, 3.
            var logits = new[]
            {
                0.1f, 0.2f, 0.9f, 0.3f,
                1.5f, 0.2f, 0.9f, 0.3f,
                0.1f, 0.2f, 0.9f, 1.1f
            };

            Assert.That(CtcDecoder.ArgMax(logits, 3, 4),
                Is.EqualTo(new[] { 2, 0, 3 }));
        }

        [Test]
        public void RejectsIdsOutsideTheVocabulary()
        {
            // The failure this guards against is a stale vocabulary file
            // next to a newer model, which would otherwise decode into
            // convincing nonsense.
            Assert.Throws<System.ArgumentOutOfRangeException>(
                () => CtcDecoder.Decode(new List<int> { _vocab.Size }, _vocab));
        }

        [Test]
        public void VocabularyFileIsPopulated()
        {
            Assert.That(_vocab.Size, Is.GreaterThan(1000),
                "vocabulary looks truncated");
            Assert.That(_vocab.WordDelimiterId, Is.Not.EqualTo(_vocab.BlankId));
            Assert.That(string.IsNullOrEmpty(_vocab.Tokens[0]), Is.False);
        }
    }

    /// <summary>
    /// The decoder against a vocabulary that has grown a syllable.
    ///
    /// Widening a CTC head appends to the end, so a syllable added after
    /// the fact sits past the blank and the blank stops being the last
    /// token. Nothing in the decoder should care, and this is where that
    /// is checked - deliberately against a hand-built vocabulary rather
    /// than the shipped one, because the shipped model still emits 1205
    /// classes and its export must keep saying so.
    /// </summary>
    [TestFixture]
    public sealed class AppendedTokenCtcDecoderTests
    {
        private const int Ga = 0;
        private const int Yo = 1;
        private const int Delimiter = 2;
        private const int Unk = 3;
        private const int Blank = 4;
        private const int Nwa = 5;

        private static CtcVocabulary Widened()
        {
            return new CtcVocabulary
            {
                Tokens = new[] { "가", "요", "|", "[UNK]", "[PAD]", "놔" },
                BlankId = Blank,
                UnkId = Unk,
                WordDelimiterId = Delimiter,
                Normalize = true,
                SamplingRate = 16000
            };
        }

        [Test]
        public void TheBlankIsNoLongerLast()
        {
            var vocab = Widened();
            Assert.That(vocab.BlankId, Is.Not.EqualTo(vocab.Size - 1));
            Assert.That(vocab.Tokens[vocab.Size - 1], Is.EqualTo("놔"));
        }

        [Test]
        public void AnAppendedSyllableDecodesLikeAnyOther()
        {
            Assert.That(
                CtcDecoder.Decode(new List<int> { Nwa, Yo }, Widened()),
                Is.EqualTo("놔요"));
        }

        [Test]
        public void RepeatsOfAnAppendedSyllableCollapse()
        {
            Assert.That(
                CtcDecoder.Decode(new List<int> { Nwa, Nwa, Nwa }, Widened()),
                Is.EqualTo("놔"));
        }

        [Test]
        public void ABlankBetweenTwoAppendedSyllablesKeepsBoth()
        {
            // The only way CTC can say the same syllable twice.
            Assert.That(
                CtcDecoder.Decode(new List<int> { Nwa, Blank, Nwa }, Widened()),
                Is.EqualTo("놔놔"));
        }

        [Test]
        public void BlanksAroundAnAppendedSyllableAreStillDropped()
        {
            Assert.That(
                CtcDecoder.Decode(
                    new List<int> { Blank, Blank, Nwa, Yo, Blank }, Widened()),
                Is.EqualTo("놔요"));
        }

        [Test]
        public void TheDelimiterStillBecomesASpace()
        {
            Assert.That(
                CtcDecoder.Decode(
                    new List<int> { Nwa, Delimiter, Ga }, Widened()),
                Is.EqualTo("놔 가"));
        }

        [Test]
        public void ArgMaxFindsAnAppendedTokenInTheWidenedFrame()
        {
            // Two frames of six classes; the appended column wins the
            // second one, which is the column a 1205-wide vocabulary
            // would not have had.
            var logits = new[]
            {
                0.1f, 0.2f, 0.0f, 0.0f, 0.9f, 0.3f,
                0.1f, 0.2f, 0.0f, 0.0f, 0.4f, 1.7f
            };

            var ids = CtcDecoder.ArgMax(logits, 2, 6);

            Assert.That(ids, Is.EqualTo(new[] { Blank, Nwa }));
            Assert.That(CtcDecoder.Decode(ids, Widened()), Is.EqualTo("놔"));
        }

        [Test]
        public void AnIdPastTheWidenedVocabularyIsStillRejected()
        {
            Assert.Throws<System.ArgumentOutOfRangeException>(
                () => CtcDecoder.Decode(new List<int> { 6 }, Widened()));
        }
    }

    /// <summary>Locates repo files regardless of the test runner's cwd.</summary>
    internal static class RepoFiles
    {
        /// <summary>
        /// Walk up from the running assembly, so the same tests work
        /// under `dotnet test` and the Unity Test Runner.
        /// </summary>
        public static string Read(string relative)
        {
            var starts = new[]
            {
                System.AppContext.BaseDirectory,
                Directory.GetCurrentDirectory()
            };

            foreach (var start in starts)
            {
                var dir = new DirectoryInfo(start);
                while (dir != null)
                {
                    var candidate = Path.Combine(
                        dir.FullName,
                        relative.Replace('/', Path.DirectorySeparatorChar));
                    if (File.Exists(candidate))
                    {
                        return File.ReadAllText(candidate);
                    }

                    dir = dir.Parent;
                }
            }

            throw new FileNotFoundException(
                $"could not locate '{relative}' above {starts[0]} or "
                + $"{starts[1]}");
        }
    }
}
