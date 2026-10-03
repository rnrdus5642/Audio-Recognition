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
            Assert.That(_vocab.BlankId, Is.EqualTo(_vocab.Size - 1),
                "blank is expected last; ArgMax indexes assume nothing else");
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
