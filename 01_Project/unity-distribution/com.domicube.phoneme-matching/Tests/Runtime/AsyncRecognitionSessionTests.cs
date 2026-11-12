using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    public sealed class AsyncRecognitionSessionTests
    {
        public sealed class DeferredRecognizer : IAsyncPhonemeRecognizer
        {
            public string Language => "ko";
            public bool IsRecognizing => Pending != null;
            public TaskCompletionSource<RecognitionResult> Pending;
            public CancellationToken LastToken;
            public int Calls;
            public List<string> Recognize(float[] audio) => throw new InvalidOperationException("Synchronous path used.");
            public void RecognizeWithText(float[] audio, out string text, out List<string> phonemes)
                => throw new InvalidOperationException("Synchronous path used.");
            public Task<RecognitionResult> RecognizeAsync(float[] audio, CancellationToken cancellation)
            {
                if (IsRecognizing) throw new InvalidOperationException("Concurrent inference.");
                Calls++;
                LastToken = cancellation;
                Pending = new TaskCompletionSource<RecognitionResult>();
                return Pending.Task;
            }
            // Deliberately complete even a cancelled request, like already submitted GPU work.
            public void Complete(string text)
            {
                var task = Pending;
                Pending = null;
                task.SetResult(new RecognitionResult(text, Korean.JamoIpa.ToPhonemes(Korean.Rules.Apply(text))));
            }
            public void Fail()
            {
                var task = Pending;
                Pending = null;
                task.SetException(new InvalidOperationException("GPU failed"));
            }
        }

        private sealed class ImmediateRecognizer : IPhonemeRecognizer
        {
            public string Text;
            public string Language => "ko";
            public List<string> Recognize(float[] audio) => Korean.JamoIpa.ToPhonemes(Korean.Rules.Apply(Text));
            public void RecognizeWithText(float[] audio, out string text, out List<string> phonemes)
            { text = Text; phonemes = Recognize(audio); }
        }

        private static ConfusionMatrix Matrix()
        {
#if UNITY_5_3_OR_NEWER
            return PhonemeData.LoadMatrix(UnityEngine.Resources.Load<UnityEngine.TextAsset>(
                "DomiCube/PhonemeMatching/MusicWebV4/ko_child_v2").text);
#else
            return PhonemeData.LoadMatrix(RepoFiles.Read("shared/confusion_matrices/ko_child_v2.json"));
#endif
        }
        public static float[] Speech(int samples = 8000) => Enumerable.Repeat(.1f, samples).ToArray();
        private static PronunciationSession Session(IPhonemeRecognizer recognizer, AudioWindowBuffer buffer)
        {
            var session = PronunciationSession.ForOrthographicWord(Matrix(), recognizer, buffer: buffer);
            session.Begin("사과");
            return session;
        }

        [Test]
        public void PendingPollsNeitherScoreSilenceNorStartAnotherInference()
        {
            var recognizer = new DeferredRecognizer();
            var buffer = new AudioWindowBuffer(2.5f, .5f);
            var session = Session(recognizer, buffer);
            Assert.That(session.TryAdvance(out _), Is.False);
            Assert.That(recognizer.Calls, Is.Zero);
            buffer.Append(Speech());
            Assert.That(session.TryAdvance(out _), Is.False);
            buffer.Append(Speech());
            for (int i = 0; i < 50; i++) Assert.That(session.TryAdvance(out _), Is.False);
            Assert.That(session.LastFrame, Is.Null);
            Assert.That(recognizer.Calls, Is.EqualTo(1));
            recognizer.Complete("사과");
            Assert.That(session.TryAdvance(out var first), Is.True);
            Assert.That(first.EndSampleIndex, Is.EqualTo(8000));
            Assert.That(first.WordDecision.ElapsedSeconds, Is.EqualTo(.5));
            Assert.That(first.Streak, Is.EqualTo(1));
            Assert.That(first.Confirmed, Is.False);
            Assert.That(session.TryAdvance(out _), Is.False);
            recognizer.Complete("사과");
            Assert.That(session.TryAdvance(out var second), Is.True);
            Assert.That(second.EndSampleIndex, Is.EqualTo(16000));
            Assert.That(second.WordDecision.ElapsedSeconds, Is.EqualTo(1));
            Assert.That(second.Confirmed, Is.True);
            Assert.That(session.IsActive, Is.False);
        }

        [Test]
        public void RestartDropsLateResultAndPreservesNewQuestionsQueuedAudio()
        {
            var recognizer = new DeferredRecognizer();
            var buffer = new AudioWindowBuffer(2.5f, .5f);
            var session = Session(recognizer, buffer);
            buffer.Append(Speech()); session.TryAdvance(out _);
            var oldToken = recognizer.LastToken;
            session.End(); session.Begin("배"); buffer.Append(Speech());
            Assert.That(oldToken.IsCancellationRequested, Is.True);
            Assert.That(session.TryAdvance(out _), Is.False);
            Assert.That(recognizer.Calls, Is.EqualTo(1));
            recognizer.Complete("사과");
            Assert.That(session.LastFrame, Is.Null);
            Assert.That(session.TryAdvance(out _), Is.False);
            recognizer.Complete("배");
            Assert.That(session.TryAdvance(out var frame), Is.True);
            Assert.That(frame.Text, Is.EqualTo("배"));
            Assert.That(frame.EndSampleIndex, Is.EqualTo(8000));
            Assert.That(frame.Streak, Is.EqualTo(1));
            Assert.That(session.LastConfirmation, Is.Null);
        }

        [Test]
        public void EndDropsAnAlreadyCompletedUnconsumedResult()
        {
            var recognizer = new DeferredRecognizer();
            var buffer = new AudioWindowBuffer(2.5f, .5f);
            var session = Session(recognizer, buffer);
            buffer.Append(Speech()); session.TryAdvance(out _);
            recognizer.Complete("사과"); session.End();
            Assert.That(session.TryAdvance(out _), Is.False);
            Assert.That(session.LastFrame, Is.Null);
            Assert.That(session.IsRecognitionPending, Is.False);
        }

        [Test]
        public void FailureInvalidatesStreakAndDoesNotScoreAnotherHop()
        {
            var recognizer = new DeferredRecognizer();
            var buffer = new AudioWindowBuffer(2.5f, .5f);
            var session = Session(recognizer, buffer);
            buffer.Append(Speech(24000)); session.TryAdvance(out _);
            recognizer.Complete("사과"); session.TryAdvance(out _);
            session.TryAdvance(out _); recognizer.Fail();
            Assert.Throws<InvalidOperationException>(() => session.TryAdvance(out _));
            Assert.That(session.IsActive, Is.False);
            Assert.That(session.LastConfirmation, Is.Null);
            Assert.That(session.TryAdvance(out _), Is.False);
            Assert.That(recognizer.Calls, Is.EqualTo(2));
        }

        [Test]
        public void QueueOverflowCancelsPendingRecognitionWithoutLosingTheFault()
        {
            var recognizer = new DeferredRecognizer();
            var buffer = new AudioWindowBuffer(2.5f, .5f, maxPendingFrames: 2);
            var session = Session(recognizer, buffer);
            buffer.Append(Speech()); session.TryAdvance(out _);
            buffer.Append(Speech(24000));
            Assert.Throws<InvalidOperationException>(() => session.TryAdvance(out _));
            Assert.That(buffer.Overflowed, Is.True);
            Assert.That(session.IsActive, Is.False);
            Assert.That(recognizer.LastToken.IsCancellationRequested, Is.True);
            recognizer.Complete("사과");
            Assert.That(session.LastConfirmation, Is.Null);
        }

        [Test]
        public void SyncCallsCannotConsumeAHopWhileAnAsyncHopIsPending()
        {
            var recognizer = new DeferredRecognizer();
            var buffer = new AudioWindowBuffer(2.5f, .5f);
            var session = Session(recognizer, buffer);
            buffer.Append(Speech(16000)); session.TryAdvance(out _);
            Assert.Throws<InvalidOperationException>(() => session.TryPush(out _));
            Assert.Throws<InvalidOperationException>(() => session.Push(Speech(40000)));
            recognizer.Complete("사과"); session.TryAdvance(out _);
            session.TryAdvance(out _); recognizer.Complete("사과");
            session.TryAdvance(out var frame);
            Assert.That(frame.EndSampleIndex, Is.EqualTo(16000));
            Assert.That(frame.Confirmed, Is.True);
        }

        [Test]
        public void InvalidPcmIsRejectedBeforeStartingAsyncInference()
        {
            var recognizer = new DeferredRecognizer();
            var buffer = new AudioWindowBuffer(2.5f, .5f);
            var session = Session(recognizer, buffer);
            var audio = Speech();
            audio[0] = float.NaN;
            buffer.Append(audio);
            Assert.Throws<ArgumentException>(() => session.TryAdvance(out _));
            Assert.That(recognizer.Calls, Is.Zero);
            Assert.That(session.IsActive, Is.False);
        }

        [Test]
        public void SyncCallsPreserveNewHopWhileCancelledWorkerDrains()
        {
            var recognizer = new DeferredRecognizer();
            var buffer = new AudioWindowBuffer(2.5f, .5f);
            var session = Session(recognizer, buffer);
            buffer.Append(Speech()); session.TryAdvance(out _);
            session.End(); session.Begin("배"); buffer.Append(Speech());
            Assert.That(session.TryPush(out _), Is.False);
            Assert.Throws<InvalidOperationException>(() => session.Push(Speech(40000)));
            Assert.That(session.IsActive, Is.True);
            recognizer.Complete("사과");
            Assert.That(session.TryAdvance(out _), Is.False);
            recognizer.Complete("배");
            Assert.That(session.TryAdvance(out var frame), Is.True);
            Assert.That(frame.EndSampleIndex, Is.EqualTo(8000));
            Assert.That(frame.Text, Is.EqualTo("배"));
        }

        [Test]
        public void PendingHopKeepsItsOwnAudioLevelsWhileNewAudioArrives()
        {
            var recognizer = new DeferredRecognizer();
            var buffer = new AudioWindowBuffer(2.5f, .5f);
            var session = Session(recognizer, buffer);
            var first = Speech();
            buffer.Append(first); session.TryAdvance(out _);
            var later = Speech(40000);
            for (int i = 0; i < later.Length; i++) later[i] *= .01f;
            buffer.Append(later);
            recognizer.Complete("사과");
            session.TryAdvance(out var frame);
            double power = 0, peak = 0;
            foreach (float value in first) { power += value * value; peak = Math.Max(peak, Math.Abs(value)); }
            Assert.That(frame.AudioRms, Is.EqualTo(Math.Sqrt(power / 40000)).Within(1e-6));
            Assert.That(frame.AudioPeak, Is.EqualTo(peak).Within(1e-6));
        }

        [Test]
        public void AsyncAndSyncPreserveWordContextAndConfirmationDecisions()
        {
            var delayed = new DeferredRecognizer();
            var immediate = new ImmediateRecognizer();
            var aBuffer = new AudioWindowBuffer(2.5f, .5f);
            var bBuffer = new AudioWindowBuffer(2.5f, .5f);
            var a = Session(delayed, aBuffer);
            var b = Session(immediate, bBuffer);
            foreach (string text in new[] { "포도", "", "이건 사과인 것 같아요", "사과를 먹어요" })
            {
                aBuffer.Append(Speech()); bBuffer.Append(Speech());
                a.TryAdvance(out _); delayed.Complete(text); a.TryAdvance(out var actual);
                immediate.Text = text; b.TryAdvance(out var expected);
                Assert.That(actual.Text, Is.EqualTo(expected.Text));
                Assert.That(actual.EndSampleIndex, Is.EqualTo(expected.EndSampleIndex));
                Assert.That(actual.Best.Score, Is.EqualTo(expected.Best.Score));
                Assert.That(actual.DecisionReason, Is.EqualTo(expected.DecisionReason));
                Assert.That(actual.Streak, Is.EqualTo(expected.Streak));
                Assert.That(actual.Confirmed, Is.EqualTo(expected.Confirmed));
            }
            Assert.That(a.LastConfirmation, Is.Not.Null);
        }
    }
}
