#if UNITY_EDITOR && UNITY_INCLUDE_TESTS
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor.PackageManager;

namespace DomiCube.PhonemeMatching.Tests
{
    public sealed class OrthographicWordPolicyTests
    {
        private string _package;
        private ConfusionMatrix _matrix;

        [OneTimeSetUp]
        public void LoadInstalledPackage()
        {
            var package = PackageInfo.FindForAssembly(typeof(OrthographicWordPolicy).Assembly);
            Assert.That(package, Is.Not.Null);
            _package = package.resolvedPath;
            _matrix = PhonemeData.LoadMatrix(File.ReadAllText(Path.Combine(_package,
                "Runtime/Resources/DomiCube/PhonemeMatching/MusicWebV4/ko_child_v2.json")));
        }

        [Test]
        public void EveryWebReplayMatchesScoresReasonsAndConfirmation()
        {
            JObject fixture;
            using (var stream = File.OpenRead(Path.Combine(_package, "Tests/Runtime/orthographic_word_vectors.json.gz")))
            using (var gzip = new GZipStream(stream, CompressionMode.Decompress))
            using (var reader = new StreamReader(gzip))
                fixture = JObject.Parse(reader.ReadToEnd());
            Assert.That((string)fixture["policy"], Is.EqualTo(OrthographicWordPolicy.Version));
            using (var sha = SHA256.Create())
            {
                var hash = sha.ComputeHash(File.ReadAllBytes(Path.Combine(_package, "Runtime/Resources/DomiCube/PhonemeMatching/MusicWebV4/ko_child_v2.json")));
                Assert.That(BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant(),
                    Is.EqualTo((string)fixture["matrix_sha256"]), "Packaged matrix must match the web matrix.");
            }
            var cases = (JArray)fixture["cases"];
            Assert.That(cases.Count, Is.EqualTo(11556), "Do not silently drop replay cases.");
            var policy = new OrthographicWordPolicy(_matrix);
            int frames = 0, accepted = 0;
            foreach (var trial in cases)
            {
                policy.Begin(trial["targets"].ToObject<string[]>());
                foreach (var check in trial["checks"])
                {
                    string context = (string)trial["id"] + " @ " + check["seconds"] + " text=" + check["text"];
                    var actual = policy.Push((string)check["text"], (double)check["seconds"], (double)check["voice_fraction"]);
                    var expected = check["expected"];
                    Assert.That(actual.Text, Is.EqualTo((string)check["text"]), context + " raw text");
                    Assert.That(actual.Candidate, Is.EqualTo((string)expected["candidate"]), context + " candidate");
                    Assert.That(actual.Similarity, Is.EqualTo((double)expected["similarity"]).Within(1e-9), context + " similarity");
                    Assert.That(actual.Passed, Is.EqualTo((bool)expected["passed"]), context + " passed");
                    Assert.That(actual.Confirmed, Is.EqualTo((bool)expected["confirmed"]), context + " confirmed");
                    Assert.That(actual.Ambiguous, Is.EqualTo((bool)expected["ambiguous"]), context + " ambiguous");
                    Assert.That(actual.MatchedText, Is.EqualTo((string)expected["matched_text"]), context + " matched text");
                    Assert.That(actual.ComparisonText, Is.EqualTo((string)expected["comparison_text"]), context + " comparison text");
                    Assert.That(actual.Normalization, Is.EqualTo((string)expected["normalization"]), context + " normalization");
                    Assert.That(actual.BlockedBy, Is.EqualTo((string)expected["blocked_by"]), context + " blocked");
                    Assert.That(actual.DecisionReason, Is.EqualTo((string)expected["decision_reason"]), context + " reason");
                    Assert.That(actual.ConfirmationWaiting, Is.EqualTo((bool)expected["confirmation_waiting"]), context + " waiting");
                    Assert.That(actual.Streak, Is.EqualTo((int)expected["streak"]), context + " evidence count");
                    Assert.That(actual.MatchedTarget, Is.EqualTo(actual.Confirmed ? actual.Candidate : null), context + " accepted target");
                    var scores = (JArray)expected["candidate_scores"];
                    Assert.That(actual.CandidateScores.Count, Is.EqualTo(scores.Count), context);
                    for (int i = 0; i < scores.Count; i++)
                    {
                        var item = actual.CandidateScores[i];
                        var score = scores[i];
                        Assert.That(item.Target, Is.EqualTo((string)score["target"]), context + " target order");
                        Assert.That(item.Similarity, Is.EqualTo((double)score["similarity"]).Within(1e-9), context + " target score");
                        Assert.That(item.MatchedText, Is.EqualTo((string)score["matched_text"]), context + " target span");
                        Assert.That(item.ComparisonText, Is.EqualTo((string)score["comparison_text"]), context + " target comparison");
                        Assert.That(item.Normalization, Is.EqualTo((string)score["normalization"]), context + " target normalization");
                        Assert.That(item.BlockedBy, Is.EqualTo((string)score["blocked_by"]), context + " target guard");
                    }
                    frames++;
                    if (actual.Confirmed) accepted++;
                }
            }
            Assert.That(frames, Is.EqualTo((int)fixture["frames"]));
            TestContext.WriteLine($"WEB_PARITY cases={cases.Count} frames={frames} accepted={accepted} policy={OrthographicWordPolicy.Version}");
        }

        [TestCase(2)]
        [TestCase(3)]
        [TestCase(4)]
        public void ConfigurableEvidenceCountAllowsOneUncertainFrame(int required)
        {
            var policy = new OrthographicWordPolicy(_matrix, confirmations: required);
            policy.Begin(new[] { "사과?", "사과？" });
            Assert.That(policy.Targets.Count, Is.EqualTo(1));
            Assert.That(policy.Push("사과", .5, 1).Confirmed, Is.False);
            Assert.That(policy.Push("", 1, 0).ConfirmationWaiting, Is.True);
            OrthographicWordResult result = null;
            for (int n = 1; n < required; n++)
            {
                result = policy.Push("사과", 1 + n * .5, 1);
                Assert.That(result.Confirmed, Is.EqualTo(n == required - 1));
            }
            Assert.That(result.Streak, Is.EqualTo(required));
            Assert.That(result.MatchedTarget, Is.EqualTo("사과"));
            Assert.That(policy.IsActive, Is.False);
            Assert.Throws<InvalidOperationException>(() => policy.Push("사과", 10, 1));
        }

        [Test]
        public void DelayedAudioCannotConfirmWithStaleEvidenceAndRestartClearsState()
        {
            var policy = new OrthographicWordPolicy(_matrix);
            policy.Begin(new[] { "사과" });
            policy.Push("사과", .5, 1);
            Assert.That(policy.Push("사과", 3, 1).Streak, Is.EqualTo(1));
            Assert.That(policy.Push("사과", 3.5, 1).Confirmed, Is.True);
            policy.Begin(new[] { "배" });
            Assert.That(policy.Push("배", .5, 1).Streak, Is.EqualTo(1));
            Assert.Throws<ArgumentOutOfRangeException>(() => policy.Push("배", .5, 1));
            Assert.That(policy.Push("배", 1, 1).Confirmed, Is.True);
        }

        [Test]
        public void TwoUncertainFramesAndChangedCandidateResetEvidence()
        {
            var policy = new OrthographicWordPolicy(_matrix);
            policy.Begin(new[] { "사과", "책" });
            policy.Push("사과", .5, 1);
            policy.Push("", 1, 0);
            Assert.That(policy.Push("", 1.5, 0).Streak, Is.Zero);
            Assert.That(policy.Push("사과", 2, 1).Streak, Is.EqualTo(1));
            Assert.That(policy.Push("책", 2.5, 1).Streak, Is.EqualTo(1));
            Assert.That(policy.Push("책", 3, 1).MatchedTarget, Is.EqualTo("책"));
        }

        [Test]
        public void SessionPreservesRawTextUsesOneInferenceAndReportsThroughExistingFeedback()
        {
            var recognizer = new TextRecognizer { Text = "음 사과" };
            var session = PronunciationSession.ForOrthographicWord(_matrix, recognizer);
            var audio = Enumerable.Repeat(.03f, 40000).ToArray();
            session.Begin(new[] { "사과?", "책" });
            var first = session.Push(audio, .5);
            Assert.That(first.Text, Is.EqualTo("음 사과"));
            Assert.That(first.WordDecision.MatchedText, Is.EqualTo("사과"));
            Assert.That(first.Confirmed, Is.False);
            var second = session.Push(audio, 1);
            Assert.That(recognizer.Calls, Is.EqualTo(2));
            Assert.That(second.Confirmed, Is.True);
            Assert.That(second.Feedback.TargetText, Is.EqualTo("사과"));
            Assert.That(second.Feedback.Score, Is.EqualTo(1));
            Assert.That(session.LastFrame, Is.SameAs(second));
            Assert.That(session.LastConfirmation, Is.SameAs(second.Feedback));
            Assert.That(session.IsActive, Is.False);
            session.Begin("책");
            Assert.That(session.LastFrame, Is.Null);
            Assert.That(session.LastConfirmation, Is.Null);
            recognizer.Text = "책";
            Assert.That(session.Push(new float[40000], .5).WordDecision.DecisionReason, Is.EqualTo("no_voice"));
            Assert.That(session.Push(audio, 1).Streak, Is.EqualTo(1));
        }

        [Test]
        public void EnergyGateUsesTwentyMillisecondFramesAtTheWebThreshold()
        {
            var audio = new float[40000];
            for (int i = 0; i < 320 * 5; i++) audio[i] = .006f;
            Assert.That(OrthographicWordPolicy.VoiceFraction(audio), Is.EqualTo(.04));
            for (int i = 0; i < 320 * 5; i++) audio[i] = .00599f;
            Assert.That(OrthographicWordPolicy.VoiceFraction(audio), Is.Zero);
        }

        [Test]
        public void WrongWindowOrInvalidPcmDoesNotReachTheModel()
        {
            var recognizer = new TextRecognizer();
            var session = PronunciationSession.ForOrthographicWord(_matrix, recognizer);
            session.Begin("사과");
            Assert.Throws<ArgumentException>(() => session.Push(new float[8000]));
            var invalid = new float[40000];
            invalid[123] = float.NaN;
            Assert.Throws<ArgumentException>(() => session.Push(invalid));
            Assert.That(recognizer.Calls, Is.Zero);
        }

        private sealed class TextRecognizer : IPhonemeRecognizer
        {
            public string Text = "사과";
            public int Calls;
            public string Language => "ko";
            public List<string> Recognize(float[] audio) => throw new Exception("Must use single-pass RecognizeWithText.");
            public void RecognizeWithText(float[] audio, out string text, out List<string> phonemes)
            {
                Calls++;
                text = Text;
                // Postprocessing must derive the selected text span, not use this whole-window IPA list.
                phonemes = new List<string>();
            }
        }
    }
}
#endif
