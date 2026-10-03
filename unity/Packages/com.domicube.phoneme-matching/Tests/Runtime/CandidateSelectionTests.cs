using System.Collections.Generic;
using Newtonsoft.Json;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    /// <summary>Accept-any selection using the actual child-model UPM data.</summary>
    public sealed class CandidateSelectionTests
    {
        private const string Data =
            "unity/Packages/com.domicube.phoneme-matching/Samples~/KoreanData/";
        private static readonly string[] Utterance = { "p", "a", "ŋ", "a", "tɕ", "i" };
        private ConfusionMatrix _matrix;
        private Matcher _matcher;
        private TargetCatalog _catalog;
        private Answer _puppy;
        private Answer _bread;

        [SetUp]
        public void LoadPackageData()
        {
            _matrix = PhonemeData.LoadMatrix(RepoFiles.Read(Data + "ko_child_v2.json"));
            _matcher = Matcher.ForStreaming(_matrix);
            _catalog = PhonemeData.LoadTargets(RepoFiles.Read(Data + "targets_child.json"));
            _puppy = _catalog.Find("puppy");
            _bread = _catalog.Find("bread");
        }

        [TestCase(false)]
        [TestCase(true)]
        public void PassingAnswerIsNotHiddenByHigherFailingScore(bool reverse)
        {
            var passing = _matcher.BestMatch(Utterance, new[] { _puppy });
            var failing = _matcher.BestMatch(Utterance, new[] { _bread });
            Assert.That(passing.Passed, Is.True);
            Assert.That(failing.Passed, Is.False);
            Assert.That(passing.Score, Is.EqualTo(0.8991666666666667).Within(1e-9));
            Assert.That(failing.Score, Is.EqualTo(0.9283333333333333).Within(1e-9));
            Assert.That(failing.Score, Is.GreaterThan(passing.Score));
            var candidates = reverse ? new[] { _bread, _puppy } : new[] { _puppy, _bread };
            AssertSameResult(_matcher.BestMatch(Utterance, candidates), passing);
        }

        [TestCase(0.8, 0.975, "puppy", true)]
        [TestCase(0.8, 0.9, "bread", true)]
        [TestCase(0.95, 0.9, "bread", true)]
        [TestCase(0.95, 0.975, "bread", false)]
        public void SelectionPolicy(double puppyThreshold, double breadThreshold,
            string expected, bool passed)
        {
            _puppy.Threshold = puppyThreshold;
            _bread.Threshold = breadThreshold;
            foreach (var candidates in new[]
                     { new[] { _bread, _puppy }, new[] { _puppy, _bread } })
            {
                var result = _matcher.BestMatch(Utterance, candidates);
                AssertSameResult(result,
                    _matcher.BestMatch(Utterance, new[] { _catalog.Find(expected) }));
                Assert.That(result.Passed, Is.EqualTo(passed));
            }
        }

        [TestCase(0.8, 0.8, "first", true)]
        [TestCase(0.975, 0.975, "first", false)]
        [TestCase(0.975, 0.8, "second", true)]
        [TestCase(0.8, 0.975, "first", true)]
        public void TiesKeepInputOrderOnlyWithinSamePassStatus(
            double firstThreshold, double secondThreshold, string expected, bool passed)
        {
            var candidates = new[]
            {
                new Answer { Id = "first", Text = _bread.Text,
                    Phonemes = _bread.Phonemes, Threshold = firstThreshold },
                new Answer { Id = "second", Text = _bread.Text,
                    Phonemes = _bread.Phonemes, Threshold = secondThreshold }
            };
            var result = _matcher.BestMatch(Utterance, candidates);
            Assert.That(result.TargetId, Is.EqualTo(expected));
            Assert.That(result.Passed, Is.EqualTo(passed));
        }

        [TestCase("batch")]
        [TestCase("streaming")]
        [TestCase("exact")]
        public void ThresholdEqualityPassesInEveryProfile(string profile)
        {
            var matcher = profile == "streaming" ? _matcher
                : new Matcher(_matrix, mode: profile == "exact" ? MatchMode.Exact : MatchMode.Substring);
            matcher.ScoreAgainst(Utterance, _puppy.Phonemes, out double puppyScore, out _, out _);
            matcher.ScoreAgainst(Utterance, _bread.Phonemes, out double breadScore, out _, out _);
            var lower = puppyScore < breadScore ? _puppy : _bread;
            var higher = puppyScore < breadScore ? _bread : _puppy;
            lower.Threshold = System.Math.Min(puppyScore, breadScore);
            higher.Threshold = 1.0;
            Assert.That(System.Math.Max(puppyScore, breadScore),
                Is.GreaterThan(lower.Threshold).And.LessThan(1.0));
            var result = matcher.BestMatch(Utterance, new[] { higher, lower });
            AssertSameResult(result, matcher.BestMatch(Utterance, new[] { lower }));
            Assert.That(result.Passed, Is.True);
        }

        [Test]
        public void OnlySuppliedAnswersAreAccepted()
        {
            var result = _matcher.BestMatch(Utterance, new[] { _bread });
            Assert.That(result.TargetId, Is.EqualTo("bread"));
            Assert.That(result.Passed, Is.False);
            Assert.That(_matcher.BestMatch(Utterance, new Answer[0]).Passed, Is.False);
        }

        [TestCase(null, "puppy", 2)]
        [TestCase("silence", "puppy", 4)]
        [TestCase("bread", "bread", 3)]
        public void TwoFramesRequireSamePassingAnswerAndItsFeedback(
            string interruption, string expectedId, int frameCount)
        {
            var sm = new StreamingMatcher(_matcher, new[] { _bread, _puppy });
            Assert.That(sm.Consecutive, Is.EqualTo(2));
            Assert.That(sm.Push(Utterance), Is.Null);
            Assert.That(sm.Streak, Is.EqualTo(1));
            IReadOnlyList<string> final = Utterance;
            if (interruption == "silence")
            {
                Assert.That(sm.Push(new string[0]), Is.Null);
                Assert.That(sm.Streak, Is.EqualTo(0));
                Assert.That(sm.Push(Utterance), Is.Null);
                Assert.That(sm.Streak, Is.EqualTo(1));
            }
            else if (interruption == "bread")
            {
                final = _bread.Phonemes;
                Assert.That(sm.Push(final), Is.Null);
                Assert.That(sm.Streak, Is.EqualTo(1));
            }

            var hit = sm.Push(final);
            Assert.That(hit, Is.Not.Null);
            Assert.That(hit.Result.TargetId, Is.EqualTo(expectedId));
            Assert.That(hit.Frames, Is.EqualTo(frameCount));
            Assert.That(hit.Streak, Is.EqualTo(2));
            var expected = _matcher.BestMatch(final, new[] { _catalog.Find(expectedId) });
            Assert.That(JsonConvert.SerializeObject(hit.Feedback), Is.EqualTo(
                JsonConvert.SerializeObject(PronunciationFeedback.FromMatch(expected, frameCount, 2))));
            sm.Reset();
            Assert.That(sm.Streak, Is.EqualTo(0));
            Assert.That(sm.Push(final), Is.Null);
        }

        private sealed class Recognizer : IPhonemeRecognizer
        {
            public string Language => "ko";
            public int Calls;
            public List<string> Recognize(float[] audio) => new List<string>(Utterance);
            public void RecognizeWithText(float[] audio, out string text, out List<string> phonemes)
            {
                Calls++;
                text = "방아지";
                phonemes = Recognize(audio);
            }
        }

        [Test]
        public void SessionReportsSelectedAnswerAndConfirmationWithoutExtraRecognition()
        {
            var recognizer = new Recognizer();
            var session = new PronunciationSession(_matrix, _catalog, recognizer);
            session.Begin(new[] { "빵", "강아지" });
            var expected = _matcher.BestMatch(Utterance, new[] { _puppy });
            var first = session.Push(new float[40000]);
            AssertSameResult(first.Best, expected);
            Assert.That(first.Confirmed, Is.False);
            Assert.That(first.Streak, Is.EqualTo(1));
            Assert.That(first.Feedback, Is.Null);

            var second = session.Push(new float[40000]);
            AssertSameResult(second.Best, expected);
            Assert.That(second.Confirmed, Is.True);
            Assert.That(second.Feedback, Is.SameAs(session.LastConfirmation));
            Assert.That(JsonConvert.SerializeObject(second.Feedback), Is.EqualTo(
                JsonConvert.SerializeObject(PronunciationFeedback.FromMatch(expected, 2, 2))));
            Assert.That(session.IsActive, Is.False);
            Assert.That(recognizer.Calls, Is.EqualTo(2));
            session.Begin("빵");
            Assert.That(session.LastConfirmation, Is.Null);
            var breadOnly = session.Push(new float[40000]);
            Assert.That(breadOnly.Best.TargetId, Is.EqualTo("bread"));
            Assert.That(breadOnly.Best.Passed, Is.False);
            Assert.That(breadOnly.Streak, Is.EqualTo(0));
            Assert.That(breadOnly.Confirmed, Is.False);
        }

        private static void AssertSameResult(MatchResult actual, MatchResult expected)
        {
            // Includes the selected word's threshold, distance, IPA and alignment.
            Assert.That(JsonConvert.SerializeObject(actual),
                Is.EqualTo(JsonConvert.SerializeObject(expected)));
        }
    }
}
