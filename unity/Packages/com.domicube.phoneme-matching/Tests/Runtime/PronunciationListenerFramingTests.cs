#if UNITY_EDITOR && UNITY_INCLUDE_TESTS
using System.Collections.Generic;
using System.Reflection;
using DomiCube.PhonemeMatching.Unity;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace DomiCube.PhonemeMatching.Tests
{
    /// <summary>
    /// The listener drives the drain, so the rules about when scoring
    /// stops are its to keep. Driven here without a microphone: the
    /// session is handed a buffer the test fills itself.
    /// </summary>
    public sealed class PronunciationListenerFramingTests
    {
        private const int Hop = 8000;

        private sealed class Recognizer : IPhonemeRecognizer
        {
            public string Language => "ko";
            public int Calls;

            public List<string> Recognize(float[] audio)
            {
                return new List<string> { "s", "a", "k", "w", "a" };
            }

            public void RecognizeWithText(
                float[] audio, out string text, out List<string> phonemes)
            {
                Calls++;
                text = "사과";
                phonemes = Recognize(audio);
            }
        }

        private static readonly MethodInfo ScoreOneFrame =
            typeof(PronunciationListener).GetMethod(
                "ScoreOneFrame", BindingFlags.Instance | BindingFlags.NonPublic);

        private static readonly MethodInfo ConfirmMethod =
            typeof(PronunciationListener).GetMethod(
                "Confirm", BindingFlags.Instance | BindingFlags.NonPublic);

        private static readonly FieldInfo SessionField =
            typeof(PronunciationListener).GetField(
                "_session", BindingFlags.Instance | BindingFlags.NonPublic);

        private static readonly FieldInfo GenerationField =
            typeof(PronunciationListener).GetField(
                "_generation", BindingFlags.Instance | BindingFlags.NonPublic);

        [Test]
        public void DetailedFrameEventCanStopAndStillRetainsTheFailureEvidence()
        {
            var gameObject = new GameObject("Detailed frame diagnostic test");
            try
            {
                var listener = gameObject.AddComponent<PronunciationListener>();
                var recognizer = new Recognizer();
                var buffer = new AudioWindowBuffer(2.5f, .5f);
                var matrix = PhonemeData.LoadMatrix(RepoFiles.Read("shared/confusion_matrices/ko_child_v2.json"));
                var session = new PronunciationSession(matrix, Catalog(), recognizer, 2, buffer);
                session.Begin("사과");
                SessionField.SetValue(listener, session);
                buffer.Append(new float[2 * Hop]);
                int count = 0;
                listener.OnFrameScoredDetailed.AddListener(frame =>
                {
                    count++;
                    Assert.That(frame, Is.SameAs(listener.LastFrameScored));
                    Assert.That(frame.DecisionReason, Is.EqualTo("waiting_for_confirmation"));
                    listener.StopListening();
                });
                Assert.That(Drain(listener), Is.EqualTo(new[] { "Stopped" }));
                Assert.That(count, Is.EqualTo(1));
                Assert.That(recognizer.Calls, Is.EqualTo(1));
                Assert.That(listener.RecentDiagnosticFrames.Count, Is.EqualTo(1));
                Assert.That(listener.LastFrameScored.Text, Is.EqualTo("사과"));
            }
            finally { Object.DestroyImmediate(gameObject); }
        }

        private static TargetCatalog Catalog()
        {
            return new TargetCatalog
            {
                Answers = new List<Answer>
                {
                    new Answer
                    {
                        Id = "apple", Text = "사과", Threshold = 0.65,
                        Phonemes = new List<string> { "s", "a", "k", "w", "a" }
                    }
                }
            };
        }

        /// <summary>A confirmation snapshot, built without a microphone.</summary>
        private static PronunciationFeedback Feedback()
        {
            var result = new MatchResult
            {
                TargetId = "apple", TargetText = "사과", Score = 0.9,
                Distance = 0.0, Threshold = 0.65, Passed = true,
                UserPhonemes = new List<string> { "s", "a" },
                TargetPhonemes = new List<string> { "s", "a" },
                WindowEnd = 2,
                Alignment = new List<AlignStep>
                {
                    new AlignStep("s", "s", AlignOp.Match),
                    new AlignStep("a", "a", AlignOp.Match)
                }
            };
            return PronunciationFeedback.FromMatch(result, 2, 2);
        }

        /// <summary>Run the listener drain loop by hand, as ListenLoop does.</summary>
        private static List<string> Drain(PronunciationListener listener)
        {
            int generation = (int)GenerationField.GetValue(listener);
            var steps = new List<string>();
            while (true)
            {
                var step = ScoreOneFrame
                    .Invoke(listener, new object[] { generation }).ToString();
                steps.Add(step);
                if (step != "Scored")
                {
                    return steps;
                }
            }
        }

        [Test]
        public void StoppingFromAFrameEventPreventsTheNextRecognition()
        {
            var gameObject = new GameObject("Framing test (no microphone)");
            try
            {
                var listener = gameObject.AddComponent<PronunciationListener>();
                var recognizer = new Recognizer();
                var buffer = new AudioWindowBuffer(2.5f, 0.5f);
                var matrix = PhonemeData.LoadMatrix(
                    RepoFiles.Read("shared/confusion_matrices/ko_child_v2.json"));
                var session = new PronunciationSession(
                    matrix, Catalog(), recognizer, 2, buffer);
                session.Begin("사과");
                SessionField.SetValue(listener, session);

                // Two whole hops in one go: enough to confirm, if anything
                // scored them both before the first event was delivered.
                buffer.Append(new float[2 * Hop]);

                int events = 0;
                listener.OnFrameScored.AddListener((word, score, streak) =>
                {
                    events++;
                    listener.StopListening();
                });

                var steps = Drain(listener);

                Assert.That(steps, Is.EqualTo(new[] { "Stopped" }));
                Assert.That(events, Is.EqualTo(1));
                Assert.That(recognizer.Calls, Is.EqualTo(1),
                    "the second hop was recognised after the stop");
                Assert.That(listener.LastConfirmation, Is.Null);
                Assert.That(session.IsActive, Is.False);
            }
            finally
            {
                Object.DestroyImmediate(gameObject);
            }
        }

        [Test]
        public void ANormalConfirmationFiresDetailedThenLegacy()
        {
            var gameObject = new GameObject("Normal confirm test");
            try
            {
                var listener = gameObject.AddComponent<PronunciationListener>();
                var feedback = Feedback();
                var order = new List<string>();
                listener.OnConfirmedDetailed.AddListener(f =>
                {
                    Assert.That(listener.LastConfirmation, Is.SameAs(f));
                    order.Add("detailed");
                });
                listener.OnConfirmed.AddListener(
                    (word, score) => order.Add("legacy"));

                ConfirmMethod.Invoke(listener, new object[] { feedback });

                Assert.That(order, Is.EqualTo(new[] { "detailed", "legacy" }));
                Assert.That(listener.LastConfirmation, Is.SameAs(feedback));
            }
            finally
            {
                Object.DestroyImmediate(gameObject);
            }
        }

        [Test]
        public void ADetailedHandlerThatEndsTheAttemptStopsTheLegacyEvent()
        {
            var gameObject = new GameObject("Reentrant confirm test");
            try
            {
                var listener = gameObject.AddComponent<PronunciationListener>();
                var feedback = Feedback();

                var order = new List<string>();
                listener.OnConfirmedDetailed.AddListener(f =>
                {
                    order.Add("detailed");
                    // A lesson that moves straight on to the next word.
                    listener.StopListening();
                });
                listener.OnConfirmed.AddListener((word, score) => order.Add("legacy"));

                ConfirmMethod.Invoke(listener, new object[] { feedback });

                Assert.That(order, Is.EqualTo(new[] { "detailed" }),
                    "the old answer was delivered into the next attempt");
                Assert.That(listener.LastConfirmation, Is.SameAs(feedback));
            }
            finally
            {
                Object.DestroyImmediate(gameObject);
            }
        }

        [Test]
        public void ARestartThatCannotStartStillRetiresTheOldAttempt()
        {
            var gameObject = new GameObject("Failed restart test");
            try
            {
                var listener = gameObject.AddComponent<PronunciationListener>();
                listener.SetRecognizer(new Recognizer());
                listener.LoadData(
                    RepoFiles.Read("shared/confusion_matrices/ko_child_v2.json"),
                    RepoFiles.Read("shared/targets_child.json"));
                var feedback = Feedback();

                var order = new List<string>();
                listener.OnConfirmedDetailed.AddListener(f =>
                {
                    order.Add("detailed");
                    // The lesson moves on, but the next question has no
                    // word in it, so the attempt is turned away. The
                    // answer just confirmed must not be delivered into it.
                    listener.TargetWords = new string[0];
                    LogAssert.Expect(LogType.Error,
                        "[PronunciationListener] TargetWords가 비어 있습니다.");
                    listener.Listen();
                });
                listener.OnConfirmed.AddListener(
                    (word, score) => order.Add("legacy"));

                ConfirmMethod.Invoke(listener, new object[] { feedback });

                Assert.That(order, Is.EqualTo(new[] { "detailed" }),
                    "a failed restart left the old legacy event running");
                Assert.That(listener.LastConfirmation, Is.Null,
                    "a new attempt clears the old snapshot, even if it fails");
                Assert.That(listener.IsListening, Is.False);
            }
            finally
            {
                Object.DestroyImmediate(gameObject);
            }
        }
    }
}
#endif
