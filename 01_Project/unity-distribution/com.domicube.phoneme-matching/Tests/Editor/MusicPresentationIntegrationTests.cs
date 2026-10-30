using System;
using System.Collections.Generic;
using System.Reflection;
using DomiCube.PhonemeMatching.Unity;
using NUnit.Framework;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Editor.Tests
{
    public sealed class MusicPresentationIntegrationTests
    {
        private sealed class WaRecognizer : IPhonemeRecognizer
        {
            public string Language => "ko";
            public int Calls;
            public List<string> Recognize(float[] audio) => new List<string> { "w", "a" };
            public void RecognizeWithText(float[] audio, out string text, out List<string> phonemes)
            {
                Calls++;
                text = "와";
                phonemes = Recognize(audio);
            }
        }

        [Test]
        public void BundledEvidenceReachesSessionListenerAndLessonGuidance()
        {
            var profile = MusicPresentation.Load();
            Assert.That(profile.FeedbackErrors, Is.Not.Null);
            Assert.That(profile.FeedbackErrors.MatrixId, Is.EqualTo("ko_child_v2"));
            Assert.That(ModelFeedbackResources.Load("ko_child_v1"), Is.Null);
            var recognizer = new WaRecognizer();
            var session = profile.CreateSession(recognizer);
            session.Begin("놔");
            Assert.That(session.Push(new float[40000]).Confirmed, Is.False);
            var result = session.Push(new float[40000]);
            Assert.That(result.Confirmed, Is.True);
            Assert.That(result.Feedback.FeedbackReview.Status, Is.EqualTo("withheld"));
            Assert.That(result.Feedback.FeedbackReview.ReferenceUtterances, Is.EqualTo(697));
            Assert.That(result.Feedback.KoreanFeedback.Items, Is.Empty);
            Assert.That(recognizer.Calls, Is.EqualTo(2));
            var go = new GameObject("Music guidance integration test");
            try
            {
                var lesson = go.AddComponent<MusicPresentationLesson>();
                var flags = BindingFlags.Instance | BindingFlags.NonPublic;
                // EditMode does not run the scene lifecycle automatically.
                typeof(MusicPresentationLesson).GetMethod("Awake", flags).Invoke(lesson, null);
                MusicPresentation.ApplyTo(lesson.Listener, 3);
                var loaded = (ModelErrorProfile)typeof(PronunciationListener).GetField("_feedbackErrors", flags).GetValue(lesson.Listener);
                Assert.That(loaded.MatrixId, Is.EqualTo("ko_child_v2"));
                typeof(PronunciationListener).GetMethod("Confirm", flags).Invoke(lesson.Listener, new object[] { result.Feedback });
                Assert.That(lesson.LastFeedback, Is.SameAs(result.Feedback));
                Assert.That(lesson.Status, Is.EqualTo("정답: 놔"));
                Assert.That(lesson.LastFeedback.KoreanFeedback.Summary, Is.EqualTo("발음에 문제 없어요."));
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }

        [TestCase(1, "실로폰채", "들어", "잡아")]
        [TestCase(2, "쳐", "때려", "연주해")]
        [TestCase(3, "놓아", "놔", "내려놔")]
        [TestCase(4, "들어", "잡아")]
        [TestCase(5, "쳐", "때려")]
        [TestCase(6, "흔들어")]
        [TestCase(7, "놓아", "놔", "내려놔")]
        [TestCase(8, "북채", "들어", "잡아")]
        [TestCase(9, "두드려", "때려", "쳐")]
        public void ApplySwitchesExistingListenerToOnlyTheSelectedStage(int stage, params string[] expected)
        {
            var go = new GameObject("Music profile integration test");
            try
            {
                var listener = go.AddComponent<PronunciationListener>();
                listener.TargetWords = new[] { "사과" };
                listener.Consecutive = 5;
                listener.WindowSeconds = 1f;
                listener.HopSeconds = .1f;
                int callbacks = 0;
                listener.OnConfirmed.AddListener((word, score) => callbacks++);
                MusicPresentation.ApplyTo(listener, stage);
                Assert.That(listener.TargetWords, Is.EqualTo(expected));
                Assert.That(listener.WindowSeconds, Is.EqualTo(2.5f));
                Assert.That(listener.HopSeconds, Is.EqualTo(.5f));
                Assert.That(listener.Consecutive, Is.EqualTo(2));
                Assert.That(listener.IsListening, Is.False);
                listener.OnConfirmed.Invoke(expected[0], 1f);
                Assert.That(callbacks, Is.EqualTo(1));
                Assert.Throws<ArgumentOutOfRangeException>(() => MusicPresentation.ApplyTo(listener, 10));
                Assert.That(listener.TargetWords, Is.EqualTo(expected));
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }

        [Test]
        public void EachLoadIsIndependentAndLessonDoesNotStartTheMicrophoneOnImport()
        {
            var original = MusicPresentation.Load();
            original.Catalog.Find("북채").ConfirmationFrames = 1;
            var fresh = MusicPresentation.Load();
            Assert.That(fresh.Catalog.Find("북채").ConfirmationFrames, Is.EqualTo(2));
            Assert.That(fresh.Stages.Count, Is.EqualTo(9));
            var go = new GameObject("Music lesson integration test");
            try
            {
                var lesson = go.AddComponent<MusicPresentationLesson>();
                Assert.That(lesson.Listener, Is.Not.Null);
                Assert.That(lesson.IsPrepared, Is.False);
                Assert.That(lesson.Listener.IsListening, Is.False);
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }
    }
}
