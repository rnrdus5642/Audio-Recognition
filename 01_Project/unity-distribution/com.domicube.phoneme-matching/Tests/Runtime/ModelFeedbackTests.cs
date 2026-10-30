using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    public sealed class ModelFeedbackTests
    {
        [Test]
        public void SharedModelGuidanceVectorsKeepScoringAndRawDifferences()
        {
            var cases = JArray.Parse(RepoFiles.Read("unity/Packages/com.domicube.phoneme-matching/Tests/Runtime/model_feedback_vectors.json"));
            var errors = ModelErrorProfile.FromJson(RepoFiles.Read("shared/reference/reference_errors.json"));
            var operations = new Dictionary<string, AlignOp>
            {
                ["match"] = AlignOp.Match, ["sub"] = AlignOp.Substitute,
                ["del"] = AlignOp.Delete, ["ins"] = AlignOp.Insert
            };
            foreach (var item in cases)
            {
                string name = (string)item["name"];
                var alignment = (JArray)item["alignment"];
                var result = new MatchResult
                {
                    TargetText = (string)item["word"], TargetId = "fixture", Score = .9, Threshold = .7,
                    Passed = true, Distance = .2,
                    TargetPhonemes = alignment.Where(s => (string)s[2] != "ins").Select(s => (string)s[1]).ToList(),
                    UserPhonemes = alignment.Where(s => (string)s[2] != "del").Select(s => (string)s[0]).ToList(),
                    Alignment = alignment.Select(s => new AlignStep((string)s[0], (string)s[1], operations[(string)s[2]])).ToList()
                };
                var profile = (bool)item["profile"] ? errors : null;
                var feedback = PronunciationFeedback.FromMatch(result, 2, 2, (string)item["matrix"], profile);
                Assert.That(feedback.FeedbackReview.Status, Is.EqualTo((string)item["status"]), name);
                Assert.That(feedback.FeedbackReview.Withheld.Select(w => w.AlignmentIndex),
                    Is.EqualTo(item["withheld"].Select(w => (int)w[0])), name);
                Assert.That(feedback.FeedbackReview.Withheld.Select(w => w.Reason),
                    Is.EqualTo(item["withheld"].Select(w => (string)w[1])), name);
                Assert.That(feedback.KoreanFeedback.Items.Select(h => h.SyllableIndex),
                    Is.EqualTo(item["hint_syllables"].ToObject<int[]>()), name);
                Assert.That(feedback.Score, Is.EqualTo(.9));
                Assert.That(feedback.Threshold, Is.EqualTo(.7));
                Assert.That(feedback.Passed, Is.True);
                Assert.That(feedback.Alignment.Count, Is.EqualTo(alignment.Count));
                if ((string)item["status"] == "withheld")
                {
                    Assert.That(feedback.KoreanFeedback.Summary, Is.EqualTo("발음에 문제 없어요."), name);
                    Assert.That(feedback.RawKoreanFeedback.Items, Is.Not.Empty, name);
                }
                else if ((string)item["status"] == "partial")
                    Assert.That(feedback.KoreanFeedback.Summary, Is.EqualTo("아래 부분만 조금 더 또렷하게 말해 보세요."), name);
                string before = JsonConvert.SerializeObject(feedback);
                feedback.ApplyModelErrors((string)item["matrix"], profile);
                Assert.That(JsonConvert.SerializeObject(feedback), Is.EqualTo(before));
            }
        }

        [Test]
        public void NwaStillConfirmsWithModelGuidanceEnabled()
        {
            const string folder = "unity/Packages/com.domicube.phoneme-matching/Samples~/KoreanData/";
            var matrix = PhonemeData.LoadMatrix(RepoFiles.Read(folder + "ko_child_v2.json"));
            var catalog = PhonemeData.LoadTargets(RepoFiles.Read(folder + "targets_music_demo.json"));
            var errors = ModelErrorProfile.FromJson(RepoFiles.Read("shared/reference/reference_errors.json"));
            var baseline = new StreamingMatcher(Matcher.ForStreaming(matrix), new[] { catalog.Find("놔") });
            var reviewed = new StreamingMatcher(Matcher.ForStreaming(matrix), new[] { catalog.Find("놔") }, feedbackErrors: errors);
            Assert.That(reviewed.Push(new[] { "w", "a" }), Is.Null);
            Assert.That(baseline.Push(new[] { "w", "a" }), Is.Null);
            var old = baseline.Push(new[] { "w", "a" });
            var hit = reviewed.Push(new[] { "w", "a" });
            Assert.That(hit, Is.Not.Null);
            Assert.That(hit.Result.Score, Is.EqualTo(old.Result.Score));
            Assert.That(hit.Streak, Is.EqualTo(old.Streak));
            Assert.That(hit.Feedback.DeletionCount, Is.EqualTo(1));
            Assert.That(hit.Feedback.KoreanFeedback.Items, Is.Empty);
            Assert.That(hit.Feedback.FeedbackReview.Status, Is.EqualTo("withheld"));
        }
    }
}
