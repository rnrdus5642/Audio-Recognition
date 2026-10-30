using System.Collections.Generic;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    /// <summary>
    /// Telling one music action from another, against the shipped child
    /// data. 들기's phonemes are the tail of 흔들기's, so a window allowed
    /// to start anywhere reads a child who said 흔들기 as a perfect 들기.
    /// The catalog carries the other actions and the runtime uses them
    /// only to refuse a pass - never to rank, confirm or rescore.
    ///
    /// Same cases as python/tests/test_action_contrast.py, same numbers.
    /// </summary>
    public sealed class ActionContrastTests
    {
        private const string Data =
            "unity/Packages/com.domicube.phoneme-matching/Samples~/KoreanData/";
        private ConfusionMatrix _matrix;
        private Matcher _matcher;
        private TargetCatalog _catalog;

        [SetUp]
        public void LoadPackageData()
        {
            _matrix = PhonemeData.LoadMatrix(RepoFiles.Read(Data + "ko_child_v2.json"));
            _matcher = Matcher.ForStreaming(_matrix);
            _catalog = PhonemeData.LoadTargets(RepoFiles.Read(Data + "targets_child.json"));
        }

        private List<Answer> Answers(params string[] words)
        {
            var answers = new List<Answer>(words.Length);
            foreach (var word in words)
            {
                var answer = _catalog.Find(word);
                Assert.That(answer, Is.Not.Null, word + " is not in the catalog");
                answers.Add(answer);
            }

            return answers;
        }

        [Test]
        public void TheKnownCollisionIsRefusedAndSaysWhichAction()
        {
            var result = _matcher.BestMatch(
                _catalog.Find("흔들기").Phonemes, Answers("들어", "들기"));

            Assert.That(result.TargetText, Is.EqualTo("들기"));
            Assert.That(result.Score, Is.EqualTo(0.997).Within(1e-9));
            Assert.That(result.Score,
                Is.GreaterThanOrEqualTo(_catalog.Find("들기").Threshold));
            Assert.That(result.Passed, Is.False);
            Assert.That(result.CompetitorRejected, Is.True);
            Assert.That(result.CompetitorReason,
                Is.EqualTo(Matcher.OtherActionScoredHigher));
            Assert.That(result.CompetitorText, Is.EqualTo("흔들기"));
            Assert.That(result.CompetitorScore.Value, Is.EqualTo(1.0).Within(1e-9));
            Assert.That(result.AcceptedWholeScore.Value,
                Is.EqualTo(0.64).Within(1e-9));
        }

        [Test]
        public void TheReverseCollisionIsStillRefusedByThePassMark()
        {
            var result = _matcher.BestMatch(
                _catalog.Find("들기").Phonemes, Answers("흔들어", "흔들기"));

            Assert.That(result.Score, Is.EqualTo(0.64453125).Within(1e-9));
            Assert.That(result.Passed, Is.False);
            Assert.That(result.CompetitorRejected, Is.False);
        }

        [Test]
        public void AnActionTheQuestionAlsoAcceptsIsNotACompetitor()
        {
            var result = _matcher.BestMatch(
                _catalog.Find("흔들기").Phonemes,
                Answers("들어", "들기", "흔들어", "흔들기"));

            Assert.That(result.TargetText, Is.EqualTo("흔들기"));
            Assert.That(result.Passed, Is.True);
            Assert.That(result.CompetitorRejected, Is.False);
        }

        [Test]
        public void AnUnrequestedAliasOfTheWinnersOwnActionDefendsIt()
        {
            // The lesson asked for 두드려; the child said 두드리기 with a
            // stray sound in front. Same instruction, only one form asked
            // for: measured against the requested form alone the winner is
            // worth 0.671, which 흔들기 (0.713) beats. The embedded alias
            // is worth 0.925 and is what the action is really worth here.
            var spoken = new List<string> { "h" };
            spoken.AddRange(_catalog.Find("두드리기").Phonemes);
            var asked = Answers("두드려");

            var result = _matcher.BestMatch(spoken, asked);

            Assert.That(result.Score, Is.EqualTo(0.8692857142857142).Within(1e-12));
            Assert.That(_matcher.WholeScore(spoken, _catalog.Find("두드려").Phonemes),
                Is.EqualTo(0.6714285714285715).Within(1e-12));
            Assert.That(_matcher.WholeScore(spoken, _catalog.Find("흔들기").Phonemes),
                Is.EqualTo(0.7125).Within(1e-12));
            Assert.That(_matcher.WholeScore(spoken, _catalog.Find("두드리기").Phonemes),
                Is.EqualTo(0.925).Within(1e-12));
            Assert.That(result.TargetText, Is.EqualTo("두드려"));
            Assert.That(result.Passed, Is.True);
            Assert.That(result.CompetitorRejected, Is.False);
        }

        [Test]
        public void WithoutTheEmbeddedAliasThatCaseWouldBeRefused()
        {
            var spoken = new List<string> { "h" };
            spoken.AddRange(_catalog.Find("두드리기").Phonemes);
            var stripped = _catalog.Find("두드려");
            stripped.ContrastiveAlternatives =
                stripped.ContrastiveAlternatives.FindAll(
                    a => a.ActionGroup != "drum");

            var result = _matcher.BestMatch(spoken, new[] { stripped });

            Assert.That(result.Passed, Is.False);
            Assert.That(result.CompetitorText, Is.EqualTo("흔들기"));
        }

        [Test]
        public void EveryGroupedAnswerCarriesEveryGroupedAlternative()
        {
            var groups = new HashSet<string>();
            int grouped = 0;
            foreach (var answer in _catalog.Answers)
            {
                if (answer.ActionGroup == null)
                {
                    Assert.That(answer.ContrastiveAlternatives, Is.Null);
                    continue;
                }

                grouped++;
                groups.Add(answer.ActionGroup);
                Assert.That(answer.ContrastiveAlternatives.Count, Is.EqualTo(10));
                Assert.That(answer.ContrastiveAlternatives.FindAll(
                        a => a.ActionGroup == answer.ActionGroup).Count,
                    Is.EqualTo(2), "its own action, both forms");
            }

            Assert.That(grouped, Is.EqualTo(10));
            Assert.That(groups, Is.EquivalentTo(new[]
                { "lift", "strike", "put_down", "shake", "drum" }));
        }

        [Test]
        public void ATieDoesNotVetoButStrictlyBetterDoes()
        {
            // Phonemes the matrix has never seen cost the same default
            // insertion, so answer and rival are worth exactly the same.
            var spoken = new[] { "qq", "rr" };
            var answer = new Answer
            {
                Id = "tie", Text = "들기", Threshold = 0.9,
                ActionGroup = "lift",
                Phonemes = new List<string> { "qq" },
                ContrastiveAlternatives = new List<ContrastiveAlternative>
                {
                    new ContrastiveAlternative
                    {
                        Text = "흔들기", ActionGroup = "shake",
                        Phonemes = new List<string> { "rr" },
                    },
                },
            };

            var tied = _matcher.BestMatch(spoken, new[] { answer });
            Assert.That(_matcher.WholeScore(spoken, new[] { "qq" }),
                Is.EqualTo(_matcher.WholeScore(spoken, new[] { "rr" })));
            Assert.That(tied.Passed, Is.True);
            Assert.That(tied.CompetitorRejected, Is.False);

            answer.ContrastiveAlternatives[0].Phonemes =
                new List<string> { "qq", "rr" };
            var beaten = _matcher.BestMatch(spoken, new[] { answer });
            Assert.That(beaten.Passed, Is.False);
            Assert.That(beaten.CompetitorRejected, Is.True);
        }

        [Test]
        public void AVetoedFrameNeitherStreaksNorConfirmsOnOneFrame()
        {
            var lift = Answers("들어", "들기");
            var streaming = new StreamingMatcher(_matcher, lift);
            var wrong = _catalog.Find("흔들기").Phonemes;

            Assert.That(streaming.Push(wrong), Is.Null);
            Assert.That(streaming.Streak, Is.EqualTo(0));
            Assert.That(streaming.Push(wrong), Is.Null);
            Assert.That(streaming.Streak, Is.EqualTo(0));

            streaming.Reset();
            var right = _catalog.Find("들기").Phonemes;
            Assert.That(streaming.Push(right), Is.Null);
            var hit = streaming.Push(right);
            Assert.That(hit, Is.Not.Null);
            Assert.That(hit.Result.TargetText, Is.EqualTo("들기"));
        }

        [Test]
        public void MalformedContrastMetadataIsRefusedNotScored()
        {
            const string json = "{\"answers\":[{\"id\":\"a\",\"text\":\"들기\","
                + "\"phonemes\":[\"t\"],\"action_group\":\"lift\","
                + "\"contrastive_alternatives\":[{\"text\":\"흔들기\"}]}]}";

            Assert.Throws<System.ArgumentException>(
                () => PhonemeData.LoadTargets(json));
        }

        [TestCase("false")]
        [TestCase("0")]
        [TestCase("\"\"")]
        [TestCase("{}")]
        [TestCase("\"흔들기\"")]
        public void AContainerThatIsNotAListIsRefusedNotIgnored(string container)
        {
            string json = "{\"answers\":[{\"id\":\"a\",\"text\":\"들기\","
                + "\"phonemes\":[\"t\"],\"action_group\":\"lift\","
                + "\"contrastive_alternatives\":" + container + "}]}";

            Assert.Throws<System.ArgumentException>(
                () => PhonemeData.LoadTargets(json));
        }

        [Test]
        public void AnEmptyListIsAllowedAndVetoesNothing()
        {
            const string json = "{\"answers\":[{\"id\":\"a\",\"text\":\"들기\","
                + "\"phonemes\":[\"t\"],\"threshold\":0.5,"
                + "\"action_group\":\"lift\","
                + "\"contrastive_alternatives\":[]}]}";

            var loaded = PhonemeData.LoadTargets(json);
            var result = _matcher.BestMatch(
                new[] { "t" }, loaded.Answers);

            Assert.That(result.Passed, Is.True);
            Assert.That(result.CompetitorRejected, Is.False);
        }

        [TestCase("\"\"")]
        [TestCase("\" lift\"")]
        [TestCase("\"put down\"")]
        [TestCase("7")]
        public void AGroupThatIsNotANameIsRefusedByTheLoader(string group)
        {
            string json = "{\"answers\":[{\"id\":\"a\",\"text\":\"들기\","
                + "\"phonemes\":[\"t\"],\"action_group\":" + group + "}]}";

            Assert.Throws<System.ArgumentException>(
                () => PhonemeData.LoadTargets(json));
        }

        [Test]
        public void MetadataSetInCodeIsRefusedWhenItIsScored()
        {
            // Public mutable fields: a caller can build what no loader
            // would accept, and the scorer must not silently drop the
            // protection because of it.
            var answer = new Answer
            {
                Id = "a", Text = "들기", Threshold = 0.5,
                ActionGroup = "li ft",
                Phonemes = new List<string> { "t" },
            };

            Assert.Throws<System.ArgumentException>(
                () => _matcher.BestMatch(new[] { "t" }, new[] { answer }));

            answer.ActionGroup = null;
            answer.ContrastiveAlternatives = new List<ContrastiveAlternative>
            {
                new ContrastiveAlternative
                {
                    Text = "흔들기", ActionGroup = "shake",
                    Phonemes = new List<string> { "h" },
                },
            };
            Assert.Throws<System.ArgumentException>(
                () => _matcher.BestMatch(new[] { "t" }, new[] { answer }));
        }

        [Test]
        public void AnAnswerThatShipsNoContrastsIsUntouched()
        {
            var apple = _catalog.Find("사과");
            Assert.That(apple.ContrastiveAlternatives, Is.Null);
            Assert.That(apple.ActionGroup, Is.Null);

            var result = _matcher.BestMatch(apple.Phonemes, new[] { apple });

            Assert.That(result.Passed, Is.True);
            Assert.That(result.CompetitorRejected, Is.False);
            Assert.That(result.CompetitorText, Is.Null);
        }

        /// <summary>
        /// A word licensed to confirm on one exact frame, and the matcher
        /// the licence was written for. The narrow context limit is what
        /// makes a 흔들기 frame score exactly 1.0 against 들기: the scored
        /// slice is the last five phonemes, which are 들기's, while the
        /// whole utterance is still 흔들기's.
        /// </summary>
        private Matcher Licensed(out Answer answer)
        {
            var profile = _matrix.Streaming;
            var scorer = new Matcher(
                _matrix, MatchMode.Substring, profile.SkipCost,
                profile.Coverage ?? 0.5, 0.4);
            answer = _catalog.Find("들기");
            answer.AllowExactSingleFrame = true;
            answer.ExactSingleFrameCalibration = new ExactSingleFrameCalibration
            {
                MatrixId = _matrix.MatrixId,
                Consecutive = 2,
                Text = answer.Text,
                Threshold = answer.Threshold,
                Phonemes = new List<string>(answer.Phonemes),
                SkipCost = scorer.SkipCost,
                Coverage = scorer.Coverage,
                ContextMult = 0.4,
            };
            return scorer;
        }

        [Test]
        public void TheVetoPrecedesTheOneExactFrameShortcut()
        {
            var scorer = Licensed(out var answer);
            var spoken = _catalog.Find("흔들기").Phonemes;
            var alternatives = answer.ContrastiveAlternatives;

            answer.ContrastiveAlternatives = null;
            var shortcut = new StreamingMatcher(scorer, new[] { answer })
                .Push(spoken);
            Assert.That(shortcut, Is.Not.Null);
            Assert.That(shortcut.Streak, Is.EqualTo(1));
            Assert.That(shortcut.Result.Score, Is.EqualTo(1.0));

            answer.ContrastiveAlternatives = alternatives;
            var guarded = new StreamingMatcher(scorer, new[] { answer });
            Assert.That(guarded.Push(spoken), Is.Null);
            Assert.That(guarded.Streak, Is.EqualTo(0));

            var refused = scorer.BestMatch(spoken, new[] { answer });
            Assert.That(refused.Score, Is.EqualTo(1.0),
                "the score it earned is left alone");
            Assert.That(refused.Passed, Is.False);
            Assert.That(refused.CompetitorText, Is.EqualTo("흔들기"));
        }

        [TestCase("들어", "들기")]
        [TestCase("쳐", "치기")]
        [TestCase("놓아", "놓기")]
        [TestCase("흔들어", "흔들기")]
        [TestCase("두드려", "두드리기")]
        public void EveryActionStillConfirmsItself(string plain, string gerund)
        {
            foreach (var spoken in new[] { plain, gerund })
            {
                var result = _matcher.BestMatch(
                    _catalog.Find(spoken).Phonemes, Answers(plain, gerund));

                Assert.That(result.TargetText, Is.EqualTo(spoken));
                Assert.That(result.Score, Is.EqualTo(1.0).Within(1e-9));
                Assert.That(result.Passed, Is.True);
                Assert.That(result.CompetitorRejected, Is.False);
            }
        }

        [TestCase("들어", "들기")]
        [TestCase("쳐", "치기")]
        [TestCase("놓아", "놓기")]
        [TestCase("흔들어", "흔들기")]
        [TestCase("두드려", "두드리기")]
        public void EveryPairTheLessonAsksForResolves(string plain, string gerund)
        {
            var session = new PronunciationSession(
                _matrix, _catalog, new SilentRecognizer());

            Assert.DoesNotThrow(() => session.Begin(new[] { plain, gerund }));
            Assert.That(session.Candidates.Count, Is.EqualTo(2));
            session.End();
        }

        private sealed class SilentRecognizer : IPhonemeRecognizer
        {
            public string Language => "ko";

            public List<string> Recognize(float[] audio16kMono)
            {
                return new List<string>();
            }

            public void RecognizeWithText(
                float[] audio16kMono, out string text, out List<string> phonemes)
            {
                text = string.Empty;
                phonemes = new List<string>();
            }
        }
    }
}
