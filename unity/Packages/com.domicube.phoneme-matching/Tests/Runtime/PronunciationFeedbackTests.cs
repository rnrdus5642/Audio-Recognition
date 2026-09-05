using System;
using System.Collections.Generic;
using System.Linq;
using DomiCube.PhonemeMatching.Korean;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    [TestFixture]
    public sealed class PronunciationFeedbackTests
    {
        [Test]
        public void PayloadMatchesSharedWebContractVectors()
        {
            var cases = JArray.Parse(RepoFiles.Read(
                "unity/Packages/com.domicube.phoneme-matching/Tests/Runtime/feedback_vectors.json"));
            foreach (var item in cases)
            {
                var input = item["result"];
                var result = new MatchResult
                {
                    TargetId = (string)input["target_id"],
                    TargetText = (string)input["target_text"],
                    TargetPhonemes = input["target_phonemes"].ToObject<List<string>>(),
                    UserPhonemes = input["user_phonemes"].ToObject<List<string>>(),
                    Score = (double)input["score"],
                    Distance = (double)input["distance"],
                    Threshold = (double)input["threshold"],
                    Passed = (bool)input["passed"],
                    WindowStart = (int)input["window_start"],
                    WindowEnd = (int)input["window_end"]
                };
                var operations = new Dictionary<string, AlignOp>
                {
                    ["match"] = AlignOp.Match, ["sub"] = AlignOp.Substitute,
                    ["del"] = AlignOp.Delete, ["ins"] = AlignOp.Insert
                };
                foreach (var step in (JArray)input["alignment"])
                {
                    result.Alignment.Add(new AlignStep(
                        (string)step[0], (string)step[1], operations[(string)step[2]]));
                }
                var actual = PronunciationFeedback.FromMatch(
                    result, (int)item["frames"], (int)item["streak"]);
                var expected = item["expected"].ToObject<PronunciationFeedback>();
                Assert.That(JsonConvert.SerializeObject(actual),
                    Is.EqualTo(JsonConvert.SerializeObject(expected)), (string)item["name"]);
                Assert.That(JObject.FromObject(actual).Properties().Select(p => p.Name),
                    Is.EquivalentTo(((JObject)item["expected"]).Properties().Select(p => p.Name)));
            }
        }

        private static StreamingMatcher Streaming(int consecutive = 2)
        {
            var matrix = PhonemeData.LoadMatrix(RepoFiles.Read(
                "shared/confusion_matrices/ko_child_v1.json"));
            var candidate = new Answer
            {
                Id = "apple", Text = "사과", Threshold = 0.65,
                Phonemes = new List<string> { "s", "a", "k", "w", "a" }
            };
            return new StreamingMatcher(Matcher.ForStreaming(matrix),
                new List<Answer> { candidate }, consecutive);
        }

        [Test]
        public void KoreanHintsMatchSharedWebVectorsAndExplainEveryDifferenceOnce()
        {
            var cases = JArray.Parse(RepoFiles.Read(
                "unity/Packages/com.domicube.phoneme-matching/Tests/Runtime/korean_feedback_vectors.json"));
            var operations = new Dictionary<string, AlignOp>
            {
                ["match"] = AlignOp.Match, ["sub"] = AlignOp.Substitute,
                ["del"] = AlignOp.Delete, ["ins"] = AlignOp.Insert
            };
            foreach (var item in cases)
            {
                var alignment = (JArray)item["alignment"];
                var result = new MatchResult
                {
                    TargetId = "test", TargetText = (string)item["target_text"],
                    TargetPhonemes = item["target_phonemes"].ToObject<List<string>>(),
                    UserPhonemes = alignment.Where(s => (string)s[2] != "del").Select(s => (string)s[0]).ToList(),
                    Score = 0.9, Distance = 0.5, Threshold = 0.65, Passed = true
                };
                result.WindowEnd = result.UserPhonemes.Count;
                foreach (var step in alignment)
                    result.Alignment.Add(new AlignStep(
                        (string)step[0], (string)step[1], operations[(string)step[2]]));
                var feedback = PronunciationFeedback.FromMatch(result, 2, 2);
                var expected = item["expected"].ToObject<KoreanFeedback>();
                Assert.That(JsonConvert.SerializeObject(feedback.KoreanFeedback),
                    Is.EqualTo(JsonConvert.SerializeObject(expected)), (string)item["name"]);
                var explained = feedback.KoreanFeedback.Items.SelectMany(hint => hint.AlignmentIndices).OrderBy(i => i);
                var differences = Enumerable.Range(0, alignment.Count).Where(i => (string)alignment[i][2] != "match");
                Assert.That(explained, Is.EqualTo(differences), (string)item["name"]);
                Assert.That(feedback.SchemaVersion, Is.EqualTo(2));
            }
        }

        [Test]
        public void EveryHangulSyllableReusesTheCanonicalMapping()
        {
            for (int code = JamoIpa.HangulBase; code <= JamoIpa.HangulEnd; code++)
            {
                string text = ((char)code).ToString();
                var tokens = JamoIpa.ToPhonemes(text);
                var result = new MatchResult
                {
                    TargetText = text, TargetPhonemes = tokens, UserPhonemes = tokens,
                    Alignment = tokens.Select(token => new AlignStep(token, token, AlignOp.Match)).ToList()
                };
                var guide = PronunciationFeedback.FromMatch(result, 1, 1).KoreanFeedback;
                Assert.That(guide.SyllableMappingAvailable, Is.True, text);
                Assert.That(guide.Items, Is.Empty, text);
            }
        }

        [Test]
        public void ConfirmationSnapshotsTheLastFrameWithoutAveragingOrAliasing()
        {
            var sm = Streaming();
            Assert.That(sm.Push(new[] { "s", "a", "k", "w", "a" }), Is.Null);
            var hit = sm.Push(new[] { "t", "a", "k", "w", "a" });
            Assert.That(hit, Is.Not.Null);
            var feedback = hit.Feedback;
            Assert.That(feedback.Frames, Is.EqualTo(2));
            Assert.That(feedback.Streak, Is.EqualTo(2));
            Assert.That(feedback.SubstitutionCount, Is.EqualTo(1));
            Assert.That(feedback.Score, Is.EqualTo(hit.Result.Score).And.LessThan(1.0));
            Assert.That(feedback.Threshold, Is.EqualTo(0.65));
            hit.Result.UserPhonemes[0] = "x";
            hit.Result.TargetPhonemes[0] = "x";
            hit.Result.Alignment.Clear();
            sm.Reset();
            Assert.That(feedback.UserPhonemes[0], Is.EqualTo("t"));
            Assert.That(feedback.TargetPhonemes[0], Is.EqualTo("s"));
            Assert.That(feedback.Alignment.Count, Is.EqualTo(5));
            Assert.That(feedback.KoreanFeedback.Items[0].HeardSyllable, Is.EqualTo("다"));
        }

        [Test]
        public void ContextLimitingKeepsAbsoluteIndicesAndDoesNotReportPrefixAsInsertions()
        {
            var sm = Streaming(1);
            var user = new List<string>(Enumerable.Repeat("z", 40));
            user.AddRange(new[] { "s", "a", "k", "w", "a" });
            var hit = sm.Push(user);
            Assert.That(hit, Is.Not.Null);
            Assert.That(hit.Feedback.WindowStart, Is.EqualTo(40));
            Assert.That(hit.Feedback.Alignment[0].UserIndex, Is.EqualTo(40));
            Assert.That(hit.Feedback.InsertionCount, Is.Zero);
        }

        [Test]
        public void RejectedFrameDoesNotProduceConfirmation()
        {
            Assert.That(Streaming(1).Push(Array.Empty<string>()), Is.Null);
        }
    }
}
