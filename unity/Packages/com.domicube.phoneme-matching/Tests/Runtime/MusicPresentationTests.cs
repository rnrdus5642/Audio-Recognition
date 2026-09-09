using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    public sealed class MusicPresentationTests
    {
        private const string Data = "unity/Packages/com.domicube.phoneme-matching/Samples~/KoreanData/";
        private Matcher _matcher;
        private TargetCatalog _catalog;

        [SetUp]
        public void Load()
        {
            _matcher = Matcher.ForStreaming(PhonemeData.LoadMatrix(RepoFiles.Read(Data + "ko_child_v2.json")));
            _catalog = PhonemeData.LoadTargets(RepoFiles.Read(Data + "targets_music_demo.json"));
        }

        [Test]
        public void AllTwelveWordsUseTheirOwnConfirmationCount()
        {
            Assert.That(_catalog.Answers.Count, Is.EqualTo(12));
            foreach (var a in _catalog.Answers)
            {
                var stream = new StreamingMatcher(_matcher, new[] { a }, consecutive: 5);
                int required = a.ConfirmationFrames.Value;
                for (int i = 1; i < required; i++) Assert.That(stream.Push(a.Phonemes), Is.Null, a.Text);
                var hit = stream.Push(a.Phonemes);
                Assert.That(hit, Is.Not.Null, a.Text);
                Assert.That(hit.Streak, Is.EqualTo(required));
                Assert.That(hit.RequiredFrames, Is.EqualTo(required));
                stream.Reset();
                Assert.That(stream.Streak, Is.Zero);
            }
        }

        [Test]
        public void EachStageAcceptsOnlyItsExpressions()
        {
            var root = JObject.Parse(RepoFiles.Read(Data + "targets_music_demo.json"));
            foreach (var stage in root["presentation_stages"])
            {
                var answers = stage["answers"].Select(t => _catalog.Find((string)t)).ToArray();
                foreach (var other in _catalog.Answers)
                {
                    var stream = new StreamingMatcher(_matcher, answers);
                    StreamingHit hit = null;
                    for (int i = 0; i < 6; i++)
                    {
                        hit = stream.Push(other.Phonemes);
                        if (hit != null) break;
                    }
                    Assert.That(hit != null, Is.EqualTo(answers.Any(a => a.Text == other.Text)),
                        $"stage {stage["id"]}, word {other.Text}");
                }
            }
        }

        [TestCase("실로폰", "s i l ɾ o pʰ o n")]
        [TestCase("탬버린", "tʰ ɛ m p ʌ ɾ i n")]
        [TestCase("북", "p u k̚")]
        [TestCase("내려와", "n ɛ ɾ j ʌ w a")]
        [TestCase("잡아먹어", "tɕ a p a m ʌ k ʌ")]
        public void RemovedNamesAndSimilarDifferentActionsAreComparisonOnly(string word, string ipa)
        {
            Assert.That(_catalog.Find(word), Is.Null);
            var root = JObject.Parse(RepoFiles.Read(Data + "targets_music_demo.json"));
            foreach (var stage in root["presentation_stages"])
            {
                var answers = stage["answers"].Select(t => _catalog.Find((string)t)).ToArray();
                var stream = new StreamingMatcher(_matcher, answers);
                for (int i = 0; i < 6; i++)
                    Assert.That(stream.Push(ipa.Split(' ')), Is.Null, $"stage {stage["id"]}, word {word}");
            }
        }

        [Test]
        public void LiftMarginAllowsCloseEvidenceButNotAnotherAction()
        {
            var a = _catalog.Find("들어");
            var heard = new[] { "k", "ɯ", "ɾ", "j", "ʌ" }; // 그려
            Assert.That(_matcher.BestMatch(heard, new[] { a }).Passed, Is.True);
            a.ContrastMargin = 0;
            Assert.That(_matcher.BestMatch(heard, new[] { a }).CompetitorRejected, Is.True);
            a.ContrastMargin = 0.08;
            Assert.That(_matcher.BestMatch(_catalog.Find("흔들어").Phonemes, new[] { a }).Passed, Is.False);
        }

        [Test]
        public void WholeFloorBlocksAFragmentWithoutHidingAnotherValidCandidate()
        {
            var input = Enumerable.Repeat(new[] { "k", "u" }, 9).SelectMany(x => x).ToList();
            var target = new List<string> { "s", "a", "k", "w", "a" };
            input.AddRange(target);
            var a = new Answer { Id = "a", Text = "사과", Phonemes = target,
                Threshold = .5, MinWholeScore = .8 };
            var result = _matcher.BestMatch(input, new[] { a });
            Assert.That(result.Score, Is.GreaterThan(a.Threshold));
            Assert.That(result.WholeRejected, Is.True);
            Assert.That(result.Passed, Is.False);
            var b = new Answer { Id = "b", Text = "전체", Phonemes = input, Threshold = .5 };
            result = _matcher.BestMatch(input, new[] { a, b });
            Assert.That(result.Passed, Is.True);
            Assert.That(result.TargetId, Is.EqualTo("b"));
        }

        [TestCase("confirmation_frames", "0")]
        [TestCase("confirmation_frames", "7")]
        [TestCase("confirmation_frames", "true")]
        [TestCase("confirmation_frames", "1.0")]
        [TestCase("confirmation_frames", "\"2\"")]
        [TestCase("min_whole_score", "null")]
        [TestCase("min_whole_score", "-0.1")]
        [TestCase("contrast_margin", "\"0.08\"")]
        [TestCase("contrast_margin", "1.1")]
        public void InvalidJsonPolicyIsRefused(string key, string value)
        {
            string json = "{\"answers\":[{\"id\":\"a\",\"text\":\"a\",\"phonemes\":[\"a\"],\""
                + key + "\":" + value + "}]}";
            var error = Assert.Throws<ArgumentException>(() => PhonemeData.LoadTargets(json));
            Assert.That(error.Message, Does.Contain(key));
        }

        [Test]
        public void InvalidCodePolicyIsRefused()
        {
            var a = _catalog.Find("쳐");
            a.MinWholeScore = double.NaN;
            Assert.Throws<ArgumentException>(() => new StreamingMatcher(_matcher, new[] { a }));
            Assert.Throws<ArgumentException>(() => _matcher.BestMatch(a.Phonemes, new[] { a }));
        }

        [Test]
        public void ThreeFramesCannotBorrowLegacyExactSingleApproval()
        {
            var a = _catalog.Find("북채");
            a.ConfirmationFrames = 3;
            a.AllowExactSingleFrame = true;
            a.ExactSingleFrameCalibration = new ExactSingleFrameCalibration
            {
                MatrixId = "ko_child_v2", Text = a.Text, Phonemes = a.Phonemes,
                Threshold = a.Threshold, Consecutive = 2,
                SkipCost = .005, Coverage = .8, ContextMult = 8
            };
            var stream = new StreamingMatcher(_matcher, new[] { a });
            Assert.That(stream.Push(a.Phonemes), Is.Null);
            Assert.That(stream.Push(a.Phonemes), Is.Null);
            Assert.That(stream.Push(a.Phonemes).Streak, Is.EqualTo(3));
        }

        [Test]
        public void PhraseContextRequiresSeparatePrefixAndVerbEvidence()
        {
            var a = new Answer { Id = "hit", Text = "쳐", Threshold = .69, MinWholeScore = .65,
                Phonemes = _catalog.Find("쳐").Phonemes };
            var prefix = "s i l ɾ o pʰ o n tɕʰ ɛ ɾ ɯ l".Split(' ');
            var phrase = prefix.Concat(a.Phonemes).ToList();
            Assert.That(_matcher.BestMatch(phrase, new[] { a }).WholeRejected, Is.True);
            a.ContextPrefixes = new List<ContextPrefix> { new ContextPrefix
                { Text = "실로폰채를", Phonemes = prefix.ToList() } };
            Assert.That(_matcher.BestMatch(phrase, new[] { a }).Passed, Is.True);
            var baseline = new Answer { Id = a.Id, Text = a.Text, Phonemes = a.Phonemes,
                Threshold = a.Threshold, MinWholeScore = a.MinWholeScore };
            foreach (var user in new[] { prefix.Take(10).ToArray(), prefix })
            {
                Assert.That(_matcher.AcceptedWholeScore(user, new[] { a }, a), Is.EqualTo(
                    _matcher.AcceptedWholeScore(user, new[] { baseline }, baseline)).Within(1e-9));
                Assert.That(_matcher.BestMatch(user, new[] { a }).Passed, Is.False);
            }
            foreach (var word in new[] { "들어", "흔들어" })
                Assert.That(_matcher.BestMatch(prefix.Concat(_catalog.Find(word).Phonemes).ToList(),
                    new[] { a }).Passed, Is.False);
        }

        [TestCase("false")]
        [TestCase("{}")]
        [TestCase("[null]")]
        [TestCase("[{}]")]
        [TestCase("[{\"text\":\"a\",\"phonemes\":[]}]")]
        [TestCase("[{\"text\":\"a\",\"phonemes\":[1]}]")]
        public void InvalidContextJsonIsRefused(string contexts)
        {
            string json = "{\"answers\":[{\"id\":\"a\",\"text\":\"a\",\"phonemes\":[\"a\"],\"context_prefixes\":"
                + contexts + "}]}";
            Assert.That(Assert.Throws<ArgumentException>(() => PhonemeData.LoadTargets(json)).Message,
                Does.Contain("context_prefixes"));
        }

        [Test]
        public void ShortPolicyPreservesLongFragmentAndRivalProtection()
        {
            var a = new Answer { Id = "play", Text = "연주해", Threshold = .90, ConfirmationFrames = 2,
                Phonemes = _catalog.Find("연주해").Phonemes };
            var user = "j ʌ n s i m h ɛ".Split(' ');
            Assert.That(_matcher.BestMatch(user, new[] { a }).Passed, Is.False);
            a.ShortUtterancePolicy = new ShortUtterancePolicy
                { Threshold = .75, MinWholeScore = .75, MaxExtraPhonemes = 2 };
            var result = _matcher.BestMatch(user, new[] { a });
            Assert.That(result.Passed, Is.True);
            Assert.That(result.UsedShortUtterancePolicy, Is.True);
            Assert.That(result.Threshold, Is.EqualTo(.75));
            var stream = new StreamingMatcher(_matcher, new[] { a });
            Assert.That(stream.Push(user), Is.Null);
            Assert.That(stream.Push(user), Is.Not.Null);
            var longInput = Enumerable.Repeat(new[] { "k", "u" }, 9).SelectMany(x => x).Concat(user).ToList();
            Assert.That(_matcher.BestMatch(longInput, new[] { a }).Passed, Is.False);
            Assert.That(_matcher.BestMatch(new[] { "h", "ɛ" }, new[] { a }).Passed, Is.False);
            a.ActionGroup = "play";
            a.ContrastiveAlternatives = new List<ContrastiveAlternative>
                { new ContrastiveAlternative { Text = "rival", Phonemes = user.ToList(), ActionGroup = "other" } };
            Assert.That(_matcher.BestMatch(user, new[] { a }).CompetitorRejected, Is.True);
        }

        [TestCase("false")]
        [TestCase("[]")]
        [TestCase("{}")]
        [TestCase("{\"threshold\":true,\"min_whole_score\":0.7,\"max_extra_phonemes\":2}")]
        [TestCase("{\"threshold\":0.7,\"min_whole_score\":0,\"max_extra_phonemes\":2}")]
        [TestCase("{\"threshold\":0.7,\"min_whole_score\":0.7,\"max_extra_phonemes\":2.0}")]
        [TestCase("{\"threshold\":0.7,\"min_whole_score\":0.7,\"max_extra_phonemes\":7}")]
        public void InvalidShortPolicyJsonIsRefused(string policy)
        {
            string json = "{\"answers\":[{\"id\":\"a\",\"text\":\"a\",\"phonemes\":[\"a\"],\"short_utterance_policy\":"
                + policy + "}]}";
            Assert.That(Assert.Throws<ArgumentException>(() => PhonemeData.LoadTargets(json)).Message,
                Does.Contain("short_utterance_policy"));
        }

        [Test]
        public void ConfiguredObjectCommandsConfirmInEachApplicableStage()
        {
            var root = JObject.Parse(RepoFiles.Read(Data + "targets_music_demo.json"));
            foreach (var answer in _catalog.Answers.Where(a => a.ContextPrefixes != null))
                foreach (var prefix in answer.ContextPrefixes)
                    foreach (var ending in new[] { new string[0], new[] { "j", "o" } })
                        foreach (var stage in root["presentation_stages"])
                        {
                            var candidates = stage["answers"].Select(t => _catalog.Find((string)t)).ToArray();
                            if (!candidates.Contains(answer)) continue;
                            var user = prefix.Phonemes.Concat(answer.Phonemes).Concat(ending).ToList();
                            var stream = new StreamingMatcher(_matcher, candidates);
                            Assert.That(stream.Push(user), Is.Null);
                            Assert.That(stream.Push(user), Is.Not.Null, $"stage {stage["id"]}, {prefix.Text} {answer.Text}");
                        }
        }
    }
}
