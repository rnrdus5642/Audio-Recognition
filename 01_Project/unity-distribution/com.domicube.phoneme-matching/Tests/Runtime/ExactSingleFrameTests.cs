using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    /// <summary>
    /// One exact frame may confirm, but only where calibration said so.
    /// Mirrors python/tests/test_exact_single_frame.py.
    /// </summary>
    public sealed class ExactSingleFrameTests
    {
        private const string Data = "shared/";
        private static readonly List<string> Bread =
            new List<string> { "p͈", "a", "ŋ" };

        private static ConfusionMatrix Matrix()
        {
            return PhonemeData.LoadMatrix(
                RepoFiles.Read(Data + "confusion_matrices/ko_child_v2.json"));
        }

        private static TargetCatalog Catalog()
        {
            return PhonemeData.LoadTargets(
                RepoFiles.Read(Data + "targets_child.json"));
        }

        private static Answer Word(string text)
        {
            return Catalog().Answers.First(a => a.Text == text);
        }

        private static Answer Unflagged(string text)
        {
            var answer = Word(text);
            return new Answer
            {
                Id = answer.Id, Text = answer.Text,
                Phonemes = answer.Phonemes, Threshold = answer.Threshold,
                AllowExactSingleFrame = false,
            };
        }

        /// <summary>
        /// The shipped matrix with its id taken out - what an older file
        /// that never carried one looks like to the loader.
        /// </summary>
        private static string Anonymous()
        {
            var root = Newtonsoft.Json.Linq.JObject.Parse(
                RepoFiles.Read(Data + "confusion_matrices/ko_child_v2.json"));
            root.Remove("matrix_id");
            return root.ToString();
        }

        [Test]
        public void TheShippedMatrixNamesItselfSoOneArgumentLoadingWorks()
        {
            Assert.That(Matrix().MatrixId, Is.EqualTo("ko_child_v2"),
                "LoadMatrix(json) must name the packaged child matrix");
            var sm = new StreamingMatcher(
                Matcher.ForStreaming(Matrix()), new[] { Word("빵") });
            var hit = sm.Push(Bread);
            Assert.That(hit, Is.Not.Null,
                "the packaged data needs no extra argument from the caller");
            Assert.That(hit.Streak, Is.EqualTo(1));
        }

        [Test]
        public void AMatrixThatNamesNothingGetsNoShortcut()
        {
            var anonymous = PhonemeData.LoadMatrix(Anonymous());
            Assert.That(anonymous.MatrixId, Is.EqualTo("unknown"));
            var sm = new StreamingMatcher(
                Matcher.ForStreaming(anonymous), new[] { Word("빵") });
            Assert.That(sm.Push(Bread), Is.Null,
                "an unnamed matrix must not be assumed to be the right one");
            Assert.That(sm.Push(Bread), Is.Not.Null,
                "the ordinary two-frame rule still applies");
        }

        [Test]
        public void NamingAnUnnamedMatrixThroughTheOverloadRestoresIt()
        {
            var named = PhonemeData.LoadMatrix(Anonymous(), "ko_child_v2");
            Assert.That(named.MatrixId, Is.EqualTo("ko_child_v2"));
            var sm = new StreamingMatcher(
                Matcher.ForStreaming(named), new[] { Word("빵") });
            Assert.That(sm.Push(Bread), Is.Not.Null);
        }

        [Test]
        public void TheFilesOwnIdWinsOverTheCallers()
        {
            var matrix = PhonemeData.LoadMatrix(
                RepoFiles.Read(Data + "confusion_matrices/ko_child_v2.json"),
                "ko_child_v1");
            Assert.That(matrix.MatrixId, Is.EqualTo("ko_child_v2"),
                "a file that says what it is cannot be renamed by a caller");
        }

        [Test]
        public void TheShippedCatalogCalibratedTwentyEightOfThirtyFive()
        {
            var answers = Catalog().Answers;
            Assert.That(answers.Count, Is.EqualTo(35));
            // The six action words added for the music lesson were derived
            // at the ordinary pass mark only, so they confirm on two frames.
            Assert.That(answers.Count(a => a.AllowExactSingleFrame),
                Is.EqualTo(28));
            foreach (var word in new[]
                     { "들기", "치기", "놓아", "놓기", "흔들기", "두드리기" })
            {
                Assert.That(Word(word).AllowExactSingleFrame, Is.False, word);
            }

            Assert.That(Word("와요").AllowExactSingleFrame, Is.False);
            Assert.That(Word("책").Threshold, Is.EqualTo(0.826));
        }

        [Test]
        public void OneExactFrameConfirmsWithAGenuineStreakOfOne()
        {
            var sm = new StreamingMatcher(
                Matcher.ForStreaming(Matrix()), new[] { Word("빵") });
            var hit = sm.Push(Bread);

            Assert.That(hit, Is.Not.Null);
            Assert.That(hit.Result.Score, Is.EqualTo(1.0));
            Assert.That(hit.Streak, Is.EqualTo(1));
            Assert.That(sm.Streak, Is.EqualTo(1));
            Assert.That(hit.Frames, Is.EqualTo(1));
            Assert.That(hit.Feedback.Streak, Is.EqualTo(1));
            Assert.That(hit.Feedback.Frames, Is.EqualTo(1));
        }

        [Test]
        public void WithoutTheFlagOneExactFrameIsNotEnough()
        {
            var sm = new StreamingMatcher(
                Matcher.ForStreaming(Matrix()), new[] { Unflagged("빵") });
            Assert.That(sm.Push(Bread), Is.Null);
            Assert.That(sm.Streak, Is.EqualTo(1));
            Assert.That(sm.Push(Bread), Is.Not.Null);
        }

        [Test]
        public void ANearPerfectFrameDoesNotQualify()
        {
            var matcher = Matcher.ForStreaming(Matrix());
            var answer = Word("빵");
            var nearly = new List<string>(Bread) { "i" };
            var best = matcher.BestMatch(nearly, new[] { answer });
            Assert.That(best.Score, Is.GreaterThanOrEqualTo(answer.Threshold));
            Assert.That(best.Score, Is.LessThan(1.0));

            var sm = new StreamingMatcher(matcher, new[] { answer });
            Assert.That(sm.Push(nearly), Is.Null, "only exactly 1.0 confirms alone");
            Assert.That(sm.Push(nearly), Is.Not.Null);
        }

        [Test]
        public void EmptyInputConfirmsNothing()
        {
            var sm = new StreamingMatcher(
                Matcher.ForStreaming(Matrix()), new[] { Word("빵") });
            Assert.That(sm.Push(new List<string>()), Is.Null);
            Assert.That(sm.Streak, Is.Zero);
        }

        [Test]
        public void OnlyTheCalibratedStreakLengthGetsTheEscape()
        {
            var one = new StreamingMatcher(
                Matcher.ForStreaming(Matrix()), new[] { Word("빵") }, 1);
            Assert.That(one.Push(Bread), Is.Not.Null);

            var three = new StreamingMatcher(
                Matcher.ForStreaming(Matrix()), new[] { Word("빵") }, 3);
            Assert.That(three.Push(Bread), Is.Null);
            Assert.That(three.Push(Bread), Is.Null);
            Assert.That(three.Push(Bread), Is.Not.Null);
        }

        [Test]
        public void AHandTunedProfileTurnsTheEscapeOff()
        {
            var matcher = new Matcher(Matrix(), MatchMode.Substring, 0.2, 0.5, 8.0);
            var sm = new StreamingMatcher(matcher, new[] { Word("빵") });
            Assert.That(sm.Push(Bread), Is.Null,
                "an uncalibrated profile must stay strict");
            Assert.That(sm.Push(Bread), Is.Not.Null);
        }

        [Test]
        public void AbsentFlagLoadsAsFalse()
        {
            var catalog = PhonemeData.LoadTargets(
                RepoFiles.Read(Data + "targets.json"));
            Assert.That(catalog.Answers.Any(a => a.AllowExactSingleFrame),
                Is.False, "adult targets carry no calibrated escape");
        }

        [Test]
        public void ASharedIdCannotBorrowAnotherAnswersApproval()
        {
            var bread = Word("빵");
            var impostor = new Answer
            {
                Id = bread.Id, Text = "강아지",
                Phonemes = Word("강아지").Phonemes,
                Threshold = Word("강아지").Threshold,
                AllowExactSingleFrame = false,
            };
            var sm = new StreamingMatcher(Matcher.ForStreaming(Matrix()),
                                          new[] { bread, impostor });
            Assert.That(sm.Push(Bread), Is.Null,
                "a duplicate id must void the shortcut");
            Assert.That(sm.Push(Bread), Is.Not.Null);
        }

        [Test]
        public void ApprovalIsReReadAndCannotGoStale()
        {
            var answer = Word("빵");
            var sm = new StreamingMatcher(Matcher.ForStreaming(Matrix()),
                                          new[] { answer });
            answer.AllowExactSingleFrame = false;
            Assert.That(sm.Push(Bread), Is.Null,
                "approval was cached when the matcher was built");

            answer.AllowExactSingleFrame = true;
            sm.Reset();
            Assert.That(sm.Push(Bread), Is.Not.Null);
        }

        [Test]
        public void ExactModeDoesNotGetTheEscape()
        {
            var matrix = Matrix();
            var profile = matrix.Streaming;
            var exact = new Matcher(matrix, MatchMode.Exact,
                profile.SkipCost.Value, profile.Coverage.Value,
                profile.ContextMult);
            var sm = new StreamingMatcher(exact, new[] { Word("빵") });
            Assert.That(sm.Push(Bread), Is.Null);
            Assert.That(sm.Push(Bread), Is.Not.Null);
        }


        [Test]
        public void AMissingSnapshotGrantsNothing()
        {
            var answer = Word("빵");
            answer.ExactSingleFrameCalibration = null;
            var sm = new StreamingMatcher(Matcher.ForStreaming(Matrix()),
                                          new[] { answer });
            Assert.That(sm.Push(Bread), Is.Null);
            Assert.That(sm.Push(Bread), Is.Not.Null);
        }

        [TestCase("threshold")]
        [TestCase("text")]
        [TestCase("phonemes")]
        [TestCase("consecutive")]
        [TestCase("matrixId")]
        public void ASnapshotThatNoLongerDescribesTheAnswerGrantsNothing(string field)
        {
            var answer = Word("빵");
            var snapshot = answer.ExactSingleFrameCalibration;
            switch (field)
            {
                case "threshold": snapshot.Threshold = 0.5; break;
                case "text": snapshot.Text = "다른말"; break;
                case "phonemes": snapshot.Phonemes = new List<string> { "k" }; break;
                case "consecutive": snapshot.Consecutive = 3; break;
                case "matrixId": snapshot.MatrixId = "ko_child_v1"; break;
            }

            var matcher = new Matcher(
                PhonemeData.LoadMatrix(
                    RepoFiles.Read(Data + "confusion_matrices/ko_child_v2.json"),
                    "ko_child_v2"),
                MatchMode.Substring, 0.005, 0.8, 8.0);
            var sm = new StreamingMatcher(matcher, new[] { answer });
            Assert.That(sm.Push(Bread), Is.Null, field + " still granted it");
        }

        [Test]
        public void EditingTheAnswerAfterCalibrationGrantsNothing()
        {
            var answer = Word("빵");
            answer.Threshold = 0.5;
            var sm = new StreamingMatcher(Matcher.ForStreaming(Matrix()),
                                          new[] { answer });
            Assert.That(sm.Push(Bread), Is.Null);
        }

        [Test]
        public void AnotherMatrixWithItsOwnValidProfileGrantsNothing()
        {
            var other = PhonemeData.LoadMatrix(
                RepoFiles.Read(Data + "confusion_matrices/ko_child_v1.json"),
                "ko_child_v1");
            var sm = new StreamingMatcher(Matcher.ForStreaming(other),
                                          new[] { Word("빵") });
            Assert.That(sm.Push(Bread), Is.Null,
                "approval must not travel to another matrix");
        }

        [TestCase("\"true\"")]
        [TestCase("\"false\"")]
        [TestCase("1")]
        [TestCase("0")]
        public void AFlagThatIsNotARealBooleanGrantsNothing(string literal)
        {
            var json = "{\"answers\":[{\"id\":\"x\",\"text\":\"빵\","
                       + "\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],\"threshold\":0.975,"
                       + "\"allow_exact_single_frame\":" + literal + "}]}";
            var catalog = PhonemeData.LoadTargets(json);
            Assert.That(catalog.Answers[0].AllowExactSingleFrame, Is.False);
        }

        [Test]
        public void ARealCandidateSwitchResetsTheStreak()
        {
            var bread = Unflagged("빵");
            var puppy = Unflagged("강아지");
            var sm = new StreamingMatcher(Matcher.ForStreaming(Matrix()),
                                          new[] { bread, puppy });
            Assert.That(sm.Push(bread.Phonemes), Is.Null);
            Assert.That(sm.Streak, Is.EqualTo(1));
            Assert.That(sm.Push(puppy.Phonemes), Is.Null);
            Assert.That(sm.Streak, Is.EqualTo(1));
            Assert.That(sm.Push(puppy.Phonemes), Is.Not.Null);
        }

        /// <summary>An answer whose snapshot is whatever JSON is given.</summary>
        private static Answer Inline(string snapshot)
        {
            string json = "{\"answers\":[{\"id\":\"bread\",\"text\":\"빵\","
                + "\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],\"threshold\":0.975,"
                + "\"allow_exact_single_frame\":true,"
                + "\"exact_single_frame_calibration\":" + snapshot + "}]}";
            return PhonemeData.LoadTargets(json).Answers[0];
        }

        private const string Snapshot =
            "{\"matrix_id\":\"ko_child_v2\",\"streaming_profile\":"
            + "{\"skip_cost\":0.005,\"coverage\":0.8,\"context_mult\":8.0},"
            + "\"consecutive\":2,\"text\":\"빵\","
            + "\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],\"threshold\":0.975}";

        [Test]
        public void TheControlSnapshotIsAcceptedSoTheRefusalsMeanSomething()
        {
            var answer = Inline(Snapshot);
            Assert.That(answer.ExactSingleFrameCalibration, Is.Not.Null);
            var sm = new StreamingMatcher(Matcher.ForStreaming(Matrix()),
                                          new[] { answer });
            Assert.That(sm.Push(Bread), Is.Not.Null);
        }

        [TestCase("skip_cost")]
        [TestCase("coverage")]
        [TestCase("context_mult")]
        public void ClearingAProfileValueAfterParsingGrantsNothing(string field)
        {
            // The snapshot fields are public and mutable, so the runtime
            // has to re-check presence itself. A missing context multiplier
            // must not read as the 0 an unset matcher reports.
            var answer = Word("빵");
            var sm = new StreamingMatcher(Matcher.ForStreaming(Matrix()),
                                          new[] { answer });
            var snapshot = answer.ExactSingleFrameCalibration;
            switch (field)
            {
                case "skip_cost": snapshot.SkipCost = null; break;
                case "coverage": snapshot.Coverage = null; break;
                case "context_mult": snapshot.ContextMult = null; break;
            }

            Assert.That(sm.Push(Bread), Is.Null, field + " still granted it");
            Assert.That(sm.Push(Bread), Is.Not.Null);
        }

        [TestCase(double.NaN)]
        [TestCase(double.PositiveInfinity)]
        public void AProfileValueThatIsNotARealNumberGrantsNothing(double value)
        {
            var answer = Word("빵");
            var sm = new StreamingMatcher(Matcher.ForStreaming(Matrix()),
                                          new[] { answer });
            answer.ExactSingleFrameCalibration.ContextMult = value;
            Assert.That(sm.Push(Bread), Is.Null);
        }

        [TestCase(" unknown ")]
        [TestCase("unknown")]
        [TestCase("   ")]
        [TestCase("")]
        [TestCase(null)]
        public void AnIdThatNamesNothingIsRefusedOnBothSides(string name)
        {
            // Both sides carry the same value: matching "unknown" against
            // "unknown" is not agreement about which matrix ran.
            var answer = Word("빵");
            answer.ExactSingleFrameCalibration.MatrixId = name;
            var matrix = PhonemeData.LoadMatrix(Anonymous(), name);
            var sm = new StreamingMatcher(Matcher.ForStreaming(matrix),
                                          new[] { answer });
            Assert.That(sm.Push(Bread), Is.Null);
            Assert.That(sm.Push(Bread), Is.Not.Null);
        }

        [Test]
        public void AWhitespaceWrappedUnknownDoesNotParseAsASnapshot()
        {
            var answer = Inline(Snapshot.Replace(
                "\"matrix_id\":\"ko_child_v2\"", "\"matrix_id\":\" unknown \""));
            Assert.That(answer.ExactSingleFrameCalibration, Is.Null);
        }

        [TestCase("{\"streaming_profile\":{\"skip_cost\":0.005,\"coverage\":0.8,"
                  + "\"context_mult\":8.0},\"consecutive\":2,\"text\":\"빵\","
                  + "\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],\"threshold\":0.975}")]
        [TestCase("{\"matrix_id\":\"\",\"streaming_profile\":{\"skip_cost\":0.005,"
                  + "\"coverage\":0.8,\"context_mult\":8.0},\"consecutive\":2,"
                  + "\"text\":\"빵\",\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],"
                  + "\"threshold\":0.975}")]
        [TestCase("{\"matrix_id\":2,\"streaming_profile\":{\"skip_cost\":0.005,"
                  + "\"coverage\":0.8,\"context_mult\":8.0},\"consecutive\":2,"
                  + "\"text\":\"빵\",\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],"
                  + "\"threshold\":0.975}")]
        [TestCase("{\"matrix_id\":\"ko_child_v2\",\"consecutive\":2,"
                  + "\"text\":\"빵\",\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],"
                  + "\"threshold\":0.975}")]
        [TestCase("{\"matrix_id\":\"ko_child_v2\",\"streaming_profile\":"
                  + "{\"skip_cost\":\"0.005\",\"coverage\":0.8,"
                  + "\"context_mult\":8.0},\"consecutive\":2,\"text\":\"빵\","
                  + "\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],\"threshold\":0.975}")]
        [TestCase("{\"matrix_id\":\"ko_child_v2\",\"streaming_profile\":"
                  + "{\"skip_cost\":true,\"coverage\":0.8,\"context_mult\":8.0},"
                  + "\"consecutive\":2,\"text\":\"빵\","
                  + "\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],\"threshold\":0.975}")]
        [TestCase("{\"matrix_id\":\"ko_child_v2\",\"streaming_profile\":"
                  + "{\"coverage\":0.8,\"context_mult\":8.0},\"consecutive\":2,"
                  + "\"text\":\"빵\",\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],"
                  + "\"threshold\":0.975}")]
        [TestCase("{\"matrix_id\":\"ko_child_v2\",\"streaming_profile\":"
                  + "{\"skip_cost\":0.005,\"coverage\":0.8,\"context_mult\":8.0},"
                  + "\"consecutive\":\"2\",\"text\":\"빵\","
                  + "\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],\"threshold\":0.975}")]
        [TestCase("{\"matrix_id\":\"ko_child_v2\",\"streaming_profile\":"
                  + "{\"skip_cost\":0.005,\"coverage\":0.8,\"context_mult\":8.0},"
                  + "\"consecutive\":2,\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],"
                  + "\"threshold\":0.975}")]
        [TestCase("{\"matrix_id\":\"ko_child_v2\",\"streaming_profile\":"
                  + "{\"skip_cost\":0.005,\"coverage\":0.8,\"context_mult\":8.0},"
                  + "\"consecutive\":2,\"text\":\"빵\",\"phonemes\":[],"
                  + "\"threshold\":0.975}")]
        [TestCase("{\"matrix_id\":\"ko_child_v2\",\"streaming_profile\":"
                  + "{\"skip_cost\":0.005,\"coverage\":0.8,\"context_mult\":8.0},"
                  + "\"consecutive\":2,\"text\":\"빵\",\"phonemes\":[\"p͈\",5],"
                  + "\"threshold\":0.975}")]
        [TestCase("{\"matrix_id\":\"ko_child_v2\",\"streaming_profile\":"
                  + "{\"skip_cost\":0.005,\"coverage\":0.8,\"context_mult\":8.0},"
                  + "\"consecutive\":2,\"text\":\"빵\","
                  + "\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],\"threshold\":\"0.975\"}")]
        [TestCase("{\"matrix_id\":\"ko_child_v2\",\"streaming_profile\":"
                  + "{\"skip_cost\":0.005,\"coverage\":0.8,\"context_mult\":8.0},"
                  + "\"consecutive\":2,\"text\":\"빵\","
                  + "\"phonemes\":[\"p͈\",\"a\",\"ŋ\"]}")]
        [TestCase("{\"matrix_id\":\"ko_child_v2\",\"streaming_profile\":[],"
                  + "\"consecutive\":2,\"text\":\"빵\","
                  + "\"phonemes\":[\"p͈\",\"a\",\"ŋ\"],\"threshold\":0.975}")]
        public void AMalformedSnapshotReadsAsNoneAndNeverThrows(string snapshot)
        {
            var answer = Inline(snapshot);
            Assert.That(answer.ExactSingleFrameCalibration, Is.Null,
                "a snapshot the runtime cannot check must not parse");
            var sm = new StreamingMatcher(Matcher.ForStreaming(Matrix()),
                                          new[] { answer });
            Assert.That(sm.Push(Bread), Is.Null);
            Assert.That(sm.Push(Bread), Is.Not.Null,
                "the ordinary two-frame rule still applies");
        }

        [TestCase("threshold")]
        [TestCase("phonemes")]
        [TestCase("text")]
        public void EditingTheAnswerAfterConstructionGrantsNothing(string field)
        {
            // The other test edits before the matcher is built; this one
            // edits an answer the matcher is already holding.
            var answer = Word("빵");
            var sm = new StreamingMatcher(Matcher.ForStreaming(Matrix()),
                                          new[] { answer });
            var frame = Bread;
            switch (field)
            {
                case "threshold": answer.Threshold = 0.5; break;
                case "text": answer.Text = "다른말"; break;
                case "phonemes":
                    frame = new List<string> { "p͈", "a" };
                    answer.Phonemes = frame;
                    break;
            }

            Assert.That(sm.Push(frame), Is.Null,
                "approval was cached when the matcher was built");
            Assert.That(sm.Push(frame), Is.Not.Null);
        }

        private sealed class Once : IPhonemeRecognizer
        {
            private readonly List<string> _frame;

            public Once(List<string> frame) { _frame = frame; }

            public int Calls;
            public string Language => "ko";

            public List<string> Recognize(float[] audio)
            {
                return new List<string>(_frame);
            }

            public void RecognizeWithText(
                float[] audio, out string text, out List<string> phonemes)
            {
                Calls++;
                text = string.Empty;
                phonemes = Recognize(audio);
            }
        }

        [Test]
        public void ASessionConfirmsOnceOnOneExactFrameAndIsThenOver()
        {
            var recognizer = new Once(Bread);
            var session = new PronunciationSession(
                Matrix(), Catalog(), recognizer);
            session.Begin("빵");

            var frame = session.Push(new float[40000]);
            Assert.That(frame.Confirmed, Is.True);
            // A genuine streak of 1, not a fabricated second frame.
            Assert.That(frame.Streak, Is.EqualTo(1));
            Assert.That(frame.Feedback.Streak, Is.EqualTo(1));
            Assert.That(frame.Feedback.Frames, Is.EqualTo(1));
            Assert.That(frame.Best.Score, Is.EqualTo(1.0));
            Assert.That(frame.Best.TargetId, Is.EqualTo(Word("빵").Id));
            Assert.That(frame.Best.Threshold, Is.EqualTo(Word("빵").Threshold));
            Assert.That(session.LastConfirmation, Is.SameAs(frame.Feedback));
            Assert.That(session.IsActive, Is.False, "the session is over");
            Assert.That(recognizer.Calls, Is.EqualTo(1),
                "one window was scored, not two");
            Assert.That(() => session.Push(new float[40000]),
                Throws.InstanceOf<InvalidOperationException>());
        }
    }
}
