#if UNITY_EDITOR && UNITY_INCLUDE_TESTS
using System.Collections;
using System.Reflection;
using DomiCube.PhonemeMatching.Unity;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace DomiCube.PhonemeMatching.Tests
{
    public sealed class AsyncListenerTests
    {
        private static readonly BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;

        private static PronunciationSession Attach(PronunciationListener listener,
            AsyncRecognitionSessionTests.DeferredRecognizer recognizer, AudioWindowBuffer buffer)
        {
            var matrix = PhonemeData.LoadMatrix(Resources.Load<TextAsset>(
                "DomiCube/PhonemeMatching/MusicWebV4/ko_child_v2").text);
            var session = PronunciationSession.ForOrthographicWord(matrix, recognizer, buffer: buffer);
            session.Begin("사과");
            listener.SetRecognizer(recognizer);
            typeof(PronunciationListener).GetField("_session", Private).SetValue(listener, session);
            return session;
        }

        private static string Tick(PronunciationListener listener)
        {
            int generation = (int)typeof(PronunciationListener).GetField("_generation", Private).GetValue(listener);
            return typeof(PronunciationListener).GetMethod("ScoreOneFrame", Private)
                .Invoke(listener, new object[] { generation }).ToString();
        }

        [UnityTest]
        public IEnumerator ExternalInputOwnsNoMicrophoneAndClearsOldFramesAndLateResults()
        {
            if (!Application.isPlaying) Assert.Ignore("Public ListenFrom coroutine is validated in PlayMode.");
            var go = new GameObject("External PCM listener");
            try
            {
                var listener = go.AddComponent<PronunciationListener>();
                var recognizer = new AsyncRecognitionSessionTests.DeferredRecognizer();
                var buffer = new AudioWindowBuffer(2.5f, .5f);
                listener.SetRecognizer(recognizer);
                typeof(PronunciationListener).GetField("_matrix", Private).SetValue(listener,
                    PhonemeData.LoadMatrix(Resources.Load<TextAsset>("DomiCube/PhonemeMatching/MusicWebV4/ko_child_v2").text));
                listener.UseOrthographicWordPolicy = true;
                listener.TargetWords = new[] { "사과" }; listener.WindowSeconds = 2.5f; listener.HopSeconds = .5f;
                listener.TimeoutSeconds = 10;
                listener.MicrophoneDevice = "THIS_DEVICE_MUST_NEVER_BE_OPENED";
                buffer.Append(AsyncRecognitionSessionTests.Speech());
                listener.ListenFrom(buffer);
                Assert.That(listener.IsListening, Is.True); Assert.That(buffer.PendingFrames, Is.Zero);
                Assert.That(typeof(PronunciationListener).GetField("_mic", Private).GetValue(listener), Is.Null);
                buffer.Append(AsyncRecognitionSessionTests.Speech());
                float deadline = Time.realtimeSinceStartup + 2;
                while (recognizer.Calls == 0 && Time.realtimeSinceStartup < deadline) yield return null;
                Assert.That(recognizer.Calls, Is.EqualTo(1));
                int confirmed = 0; listener.OnConfirmed.AddListener((_, __) => confirmed++);
                listener.StopListening(); Assert.That(recognizer.LastToken.IsCancellationRequested, Is.True);
                recognizer.Complete("사과"); yield return null;
                Assert.That(confirmed, Is.Zero); Assert.That(listener.LastFrameScored, Is.Null);
                buffer.Append(AsyncRecognitionSessionTests.Speech()); listener.ListenFrom(buffer);
                Assert.That(buffer.AppendedSamples, Is.Zero); Assert.That(buffer.PendingFrames, Is.Zero);
                Assert.Throws<System.ArgumentException>(() => listener.ListenFrom(new AudioWindowBuffer(1, .5f)));
                Assert.That(listener.IsListening, Is.False);
            }
            finally { Object.DestroyImmediate(go); }
        }

        [Test]
        public void PendingProducesNoEventsAndCompletedFramesCanCancelTheQuestion()
        {
            var go = new GameObject("Async listener events");
            try
            {
                var listener = go.AddComponent<PronunciationListener>();
                var recognizer = new AsyncRecognitionSessionTests.DeferredRecognizer();
                var buffer = new AudioWindowBuffer(2.5f, .5f);
                var session = Attach(listener, recognizer, buffer);
                buffer.Append(AsyncRecognitionSessionTests.Speech(16000));
                int texts = 0, frames = 0, confirmations = 0;
                listener.OnTextRecognized.AddListener(_ => texts++);
                listener.OnFrameScoredDetailed.AddListener(_ => { frames++; listener.StopListening(); });
                listener.OnConfirmed.AddListener((_, __) => confirmations++);
                Assert.That(Tick(listener), Is.EqualTo("Idle"));
                for (int i = 0; i < 10; i++) Assert.That(Tick(listener), Is.EqualTo("Idle"));
                Assert.That(texts + frames + confirmations, Is.Zero);
                recognizer.Complete("사과");
                Assert.That(Tick(listener), Is.EqualTo("Stopped"));
                Assert.That(texts, Is.EqualTo(1));
                Assert.That(frames, Is.EqualTo(1));
                Assert.That(confirmations, Is.Zero);
                Assert.That(session.IsActive, Is.False);
                Assert.That(recognizer.Calls, Is.EqualTo(1));
            }
            finally { Object.DestroyImmediate(go); }
        }

        [Test]
        public void ChangingRecognizerRetiresPendingWorkBeforeAcceptingItsResult()
        {
            var go = new GameObject("Async listener replace");
            try
            {
                var listener = go.AddComponent<PronunciationListener>();
                var recognizer = new AsyncRecognitionSessionTests.DeferredRecognizer();
                var buffer = new AudioWindowBuffer(2.5f, .5f);
                var session = Attach(listener, recognizer, buffer);
                buffer.Append(AsyncRecognitionSessionTests.Speech()); Tick(listener);
                listener.SetRecognizer(new AsyncRecognitionSessionTests.DeferredRecognizer());
                Assert.That(recognizer.LastToken.IsCancellationRequested, Is.True);
                recognizer.Complete("사과");
                Assert.That(session.IsActive, Is.False);
                Assert.That(listener.LastFrameScored, Is.Null);
            }
            finally { Object.DestroyImmediate(go); }
        }

        [UnityTest]
        public IEnumerator InferenceFailureStopsWithoutPretendingTheAnswerTimedOut()
        {
            var go = new GameObject("Async listener failure");
            try
            {
                var listener = go.AddComponent<PronunciationListener>();
                var recognizer = new AsyncRecognitionSessionTests.DeferredRecognizer();
                var buffer = new AudioWindowBuffer(2.5f, .5f);
                Attach(listener, recognizer, buffer);
                buffer.Append(AsyncRecognitionSessionTests.Speech());
                listener.TimeoutSeconds = 0;
                int timedOut = 0, confirmed = 0;
                listener.OnTimedOut.AddListener(() => timedOut++);
                listener.OnConfirmed.AddListener((_, __) => confirmed++);
                int generation = (int)typeof(PronunciationListener).GetField("_generation", Private).GetValue(listener);
                var loop = (IEnumerator)typeof(PronunciationListener).GetMethod("ListenLoop", Private)
                    .Invoke(listener, new object[] { generation });
                loop.MoveNext(); loop.MoveNext();
                Assert.That(recognizer.Calls, Is.EqualTo(1));
                recognizer.Fail();
                LogAssert.Expect(LogType.Exception, new System.Text.RegularExpressions.Regex("InvalidOperationException"));
                Assert.That(loop.MoveNext(), Is.False);
                Assert.That(loop.MoveNext(), Is.False);
                yield return null;
                Assert.That(timedOut, Is.Zero);
                Assert.That(confirmed, Is.Zero);
                Assert.That(listener.IsListening, Is.False);
                Assert.That(listener.LastFrameScored, Is.Null);
            }
            finally { Object.DestroyImmediate(go); }
        }

        [UnityTest]
        public IEnumerator TimeoutKeepsRunningWhileInferenceIsPendingAndDoesNotNeedNewAudio()
        {
            var go = new GameObject("Async listener timeout");
            try
            {
                var listener = go.AddComponent<PronunciationListener>();
                var recognizer = new AsyncRecognitionSessionTests.DeferredRecognizer();
                var buffer = new AudioWindowBuffer(2.5f, .5f);
                Attach(listener, recognizer, buffer);
                buffer.Append(AsyncRecognitionSessionTests.Speech());
                listener.TimeoutSeconds = .08f;
                int timedOut = 0, confirmed = 0;
                listener.OnTimedOut.AddListener(() => timedOut++);
                listener.OnConfirmed.AddListener((_, __) => confirmed++);
                int generation = (int)typeof(PronunciationListener).GetField("_generation", Private).GetValue(listener);
                var loop = (IEnumerator)typeof(PronunciationListener).GetMethod("ListenLoop", Private)
                    .Invoke(listener, new object[] { generation });
                float deadline = Time.realtimeSinceStartup + 2;
                // Drive the production loop without opening a hardware microphone.
                while (timedOut == 0 && Time.realtimeSinceStartup < deadline)
                {
                    loop.MoveNext();
                    yield return null;
                }
                Assert.That(timedOut, Is.EqualTo(1));
                Assert.That(recognizer.Calls, Is.EqualTo(1));
                Assert.That(recognizer.LastToken.IsCancellationRequested, Is.True);
                recognizer.Complete("사과");
                yield return null;
                Assert.That(confirmed, Is.Zero);
                Assert.That(listener.LastFrameScored, Is.Null);
            }
            finally { Object.DestroyImmediate(go); }
        }
    }
}
#endif
