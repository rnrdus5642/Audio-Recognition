using System;
using System.Collections.Generic;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    /// <summary>
    /// A frame is one hop of new audio, never one call. The same PCM has
    /// to produce the same windows, in the same order, ending on the same
    /// sample, however the capture layer chopped it up.
    ///
    /// The guarantee is about identical 16 kHz mono PCM. Audio that needs
    /// resampling on the way in goes through a separate, unchanged stage
    /// and is not covered here.
    /// </summary>
    public sealed class HopFramingTests
    {
        private const int Rate = AudioWindowBuffer.TargetSampleRate;
        private const int Hop = 8000;

        /// <summary>
        /// Samples that differ by absolute position, so a window taken at
        /// the wrong moment cannot compare equal to the right one.
        /// </summary>
        private static float[] Ramp(int count, int offset = 0)
        {
            var samples = new float[count];
            for (int i = 0; i < count; i++)
            {
                samples[i] = (offset + i + 1) / 100000f;
            }

            return samples;
        }

        private static List<HopFrame> Drain(AudioWindowBuffer buffer)
        {
            var frames = new List<HopFrame>();
            HopFrame frame;
            while (buffer.TryReadFrame(out frame))
            {
                frames.Add(frame);
            }

            return frames;
        }

        private static List<HopFrame> FeedInChunks(
            float[] audio, int[] sizes, float windowSeconds = 2.5f,
            float hopSeconds = 0.5f)
        {
            var buffer = new AudioWindowBuffer(windowSeconds, hopSeconds);
            var frames = new List<HopFrame>();
            int offset = 0;
            int i = 0;
            while (offset < audio.Length)
            {
                int size = Math.Min(sizes[i % sizes.Length], audio.Length - offset);
                var chunk = new float[size];
                Array.Copy(audio, offset, chunk, 0, size);
                buffer.Append(chunk);
                frames.AddRange(Drain(buffer));
                offset += size;
                i++;
            }

            return frames;
        }

        private static void AssertSameFrames(
            List<HopFrame> expected, List<HopFrame> actual, string what)
        {
            Assert.That(actual.Count, Is.EqualTo(expected.Count), what);
            for (int i = 0; i < actual.Count; i++)
            {
                Assert.That(actual[i].EndSampleIndex,
                    Is.EqualTo(expected[i].EndSampleIndex), what);
                Assert.That(actual[i].Window,
                    Is.EqualTo(expected[i].Window), what + " window " + i);
            }
        }

        [Test]
        public void SplitAndMergedInputProduceTheSameFrames()
        {
            var audio = Ramp(Rate);
            var reference = FeedInChunks(audio, new[] { Rate });
            Assert.That(reference.Count, Is.EqualTo(2));
            Assert.That(reference[0].EndSampleIndex, Is.EqualTo(8000));
            Assert.That(reference[1].EndSampleIndex, Is.EqualTo(16000));

            AssertSameFrames(reference, FeedInChunks(audio, new[] { 8000 }), "8000x2");
            AssertSameFrames(reference, FeedInChunks(audio, new[] { 4000 }), "4000x4");
            AssertSameFrames(reference, FeedInChunks(audio, new[] { 1 }), "one at a time");
            AssertSameFrames(reference,
                FeedInChunks(audio, new[] { 1, 7999, 3, 5000, 4001 }), "irregular");
            AssertSameFrames(reference,
                FeedInChunks(audio, new[] { 12000, 3999, 1 }), "straddling boundaries");
        }

        [Test]
        public void AChunkLongerThanTheWindowKeepsEveryFrame()
        {
            // The buffer used to keep only the tail of an oversized chunk,
            // so three seconds arriving at once scored one window instead
            // of six. Merging chunks then changed the answer.
            var audio = Ramp(3 * Rate);
            var oneChunk = FeedInChunks(audio, new[] { audio.Length });
            var byHop = FeedInChunks(audio, new[] { Hop });

            Assert.That(oneChunk.Count, Is.EqualTo(6));
            AssertSameFrames(byHop, oneChunk, "3 s in one chunk");
        }

        [Test]
        public void APartialHopIsNotAFrame()
        {
            var buffer = new AudioWindowBuffer(2.5f, 0.5f);
            buffer.Append(Ramp(7999));
            Assert.That(Drain(buffer), Is.Empty);
            Assert.That(buffer.AppendedSamples, Is.EqualTo(7999));

            buffer.Append(Ramp(1, 7999));
            var frames = Drain(buffer);

            Assert.That(frames.Count, Is.EqualTo(1));
            Assert.That(frames[0].EndSampleIndex, Is.EqualTo(8000));
            Assert.That(frames[0].Window.Length, Is.EqualTo(40000));
            for (int i = 0; i < 32000; i++)
            {
                Assert.That(frames[0].Window[i], Is.EqualTo(0f));
            }

            var expected = Ramp(8000);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.That(frames[0].Window[32000 + i], Is.EqualTo(expected[i]));
            }
        }

        [Test]
        public void EmptyAndRepeatedPollsChangeNothing()
        {
            var buffer = new AudioWindowBuffer(2.5f, 0.5f);
            buffer.Append(null);
            buffer.Append(new float[0]);
            for (int i = 0; i < 5; i++)
            {
                Assert.That(Drain(buffer), Is.Empty);
            }

            Assert.That(buffer.AppendedSamples, Is.Zero);
        }

        [Test]
        public void IdenticalSamplesAreStillNewAudio()
        {
            // Silence is audio. Nothing may deduplicate by value.
            var buffer = new AudioWindowBuffer(2.5f, 0.5f);
            for (int i = 0; i < 4; i++)
            {
                buffer.Append(new float[Hop]);
            }

            var frames = Drain(buffer);
            Assert.That(frames.Count, Is.EqualTo(4));
            Assert.That(frames[3].EndSampleIndex, Is.EqualTo(4 * Hop));
        }

        [Test]
        public void FramingIsOffUnlessAskedFor()
        {
            var buffer = new AudioWindowBuffer(2.5f);
            buffer.Append(Ramp(Hop));

            Assert.That(buffer.HopSamples, Is.Zero);
            Assert.That(buffer.PendingFrames, Is.Zero);
            HopFrame frame;
            Assert.Throws<InvalidOperationException>(
                () => buffer.TryReadFrame(out frame));

            var snapshot = buffer.Snapshot();
            Assert.That(snapshot.Length, Is.EqualTo(40000));
            var expected = Ramp(Hop);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.That(snapshot[32000 + i], Is.EqualTo(expected[i]));
            }
        }

        [Test]
        public void AnUndrainedQueueFailsVisiblyRatherThanDroppingFrames()
        {
            var buffer = new AudioWindowBuffer(2.5f, 0.5f, 2);
            buffer.Append(Ramp(3 * Hop));

            Assert.That(buffer.Overflowed, Is.True);
            Assert.That(buffer.PendingFrames, Is.Zero);
            HopFrame frame;
            Assert.Throws<InvalidOperationException>(
                () => buffer.TryReadFrame(out frame));
            Assert.Throws<InvalidOperationException>(
                () => buffer.TryReadFrame(out frame));

            buffer.Reset();
            Assert.That(buffer.Overflowed, Is.False);
            Assert.That(buffer.TryReadFrame(out frame), Is.False);

            buffer.Append(Ramp(Hop));
            Assert.That(buffer.TryReadFrame(out frame), Is.True);
            Assert.That(frame.EndSampleIndex, Is.EqualTo(Hop));
        }

        [Test]
        public void ResetForgetsThePartialHopAsWellAsTheAudio()
        {
            var buffer = new AudioWindowBuffer(2.5f, 0.5f);
            buffer.Append(Ramp(4000));
            buffer.Reset();

            buffer.Append(Ramp(4000));
            Assert.That(Drain(buffer), Is.Empty, "boundary phase survived a reset");

            buffer.Append(Ramp(4000, 4000));
            var frames = Drain(buffer);
            Assert.That(frames.Count, Is.EqualTo(1));
            Assert.That(frames[0].EndSampleIndex, Is.EqualTo(8000));
        }

        [Test]
        public void AppendingWithoutReadingKeepsEveryWindowInOrder()
        {
            // Nobody drains between the two appends, so all four windows
            // have to be waiting afterwards - each one holding the audio
            // as it stood at its own boundary, oldest first.
            var audio = Ramp(4 * Hop);
            var buffer = new AudioWindowBuffer(2.5f, 0.5f);
            var first = new float[Hop];
            var rest = new float[3 * Hop];
            Array.Copy(audio, 0, first, 0, first.Length);
            Array.Copy(audio, Hop, rest, 0, rest.Length);

            buffer.Append(first);
            buffer.Append(rest);
            Assert.That(buffer.PendingFrames, Is.EqualTo(4));

            var frames = Drain(buffer);
            Assert.That(frames.Count, Is.EqualTo(4));
            for (int i = 0; i < frames.Count; i++)
            {
                int end = (i + 1) * Hop;
                Assert.That(frames[i].EndSampleIndex, Is.EqualTo(end));
                for (int s = 0; s < 40000; s++)
                {
                    int source = end - 40000 + s;
                    Assert.That(frames[i].Window[s],
                        Is.EqualTo(source < 0 ? 0f : audio[source]),
                        "frame " + i + " sample " + s);
                }
            }
        }

        [Test]
        public void ChunksThatStraddleQuarterHopsFrameTheSameWay()
        {
            // A hop the chunk size never lines up with, which is the
            // normal case once a device picks its own buffer length.
            var audio = Ramp(12000);
            var reference = FeedInChunks(audio, new[] { 12000 }, 1.0f, 0.25f);
            Assert.That(reference.Count, Is.EqualTo(3));
            AssertSameFrames(reference,
                FeedInChunks(audio, new[] { 1500 }, 1.0f, 0.25f), "1500 at a time");
            AssertSameFrames(reference,
                FeedInChunks(audio, new[] { 1500, 2500, 500, 7500 }, 1.0f, 0.25f),
                "mixed");
        }

        [Test]
        public void MoreAudioDoesNotReviveAFaultedQueue()
        {
            var buffer = new AudioWindowBuffer(2.5f, 0.5f, 2);
            buffer.Append(Ramp(3 * Hop));
            Assert.That(buffer.Overflowed, Is.True);

            // Feeding it more must not quietly start a fresh queue: the
            // frames in between are gone and nothing may pretend they are
            // consecutive with what comes next.
            buffer.Append(Ramp(2 * Hop, 3 * Hop));
            Assert.That(buffer.Overflowed, Is.True);
            Assert.That(buffer.PendingFrames, Is.Zero);
            HopFrame frame;
            Assert.Throws<InvalidOperationException>(
                () => buffer.TryReadFrame(out frame));

            buffer.Reset();
            Assert.That(buffer.TryReadFrame(out frame), Is.False);
        }

        [Test]
        public void ANonDefaultHopFramesOnItsOwnBoundaries()
        {
            var audio = Ramp(Rate);
            var reference = FeedInChunks(audio, new[] { Rate }, 1.0f, 0.25f);

            Assert.That(reference.Count, Is.EqualTo(4));
            Assert.That(reference[0].EndSampleIndex, Is.EqualTo(4000));
            Assert.That(reference[0].Window.Length, Is.EqualTo(16000));
            AssertSameFrames(reference,
                FeedInChunks(audio, new[] { 999, 1 }, 1.0f, 0.25f), "quarter hop");
        }

        [Test]
        public void ImpossibleConfigurationsAreRejected()
        {
            Assert.Throws<ArgumentOutOfRangeException>(
                () => new AudioWindowBuffer(2.5f, -0.1f));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => new AudioWindowBuffer(2.5f, 0.00001f));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => new AudioWindowBuffer(2.5f, 0.5f, 0));
        }
    }

    /// <summary>
    /// The session scores hops of captured audio, so how often the caller
    /// polls cannot change how many windows a child is judged on - which
    /// is what decides when the answer confirms.
    /// </summary>
    public sealed class HopFramedSessionTests
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

        private sealed class FailingRecognizer : IPhonemeRecognizer
        {
            public string Language => "ko";
            public int Calls;
            public int FailOn = 2;

            public List<string> Recognize(float[] audio)
            {
                return new List<string> { "s", "a", "k", "w", "a" };
            }

            public void RecognizeWithText(
                float[] audio, out string text, out List<string> phonemes)
            {
                Calls++;
                if (Calls == FailOn)
                {
                    throw new InvalidOperationException("the model gave up");
                }

                text = "사과";
                phonemes = Recognize(audio);
            }
        }

        private static PronunciationSession Session(
            AudioWindowBuffer buffer, IPhonemeRecognizer recognizer)
        {
            var matrix = PhonemeData.LoadMatrix(
                RepoFiles.Read("shared/confusion_matrices/ko_child_v2.json"));
            var catalog = new TargetCatalog
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
            return new PronunciationSession(
                matrix, catalog, recognizer, 2, buffer);
        }

        private static List<FrameScore> Drain(PronunciationSession session)
        {
            var frames = new List<FrameScore>();
            FrameScore frame;
            while (session.TryPush(out frame))
            {
                frames.Add(frame);
                if (frame.Confirmed)
                {
                    break;
                }
            }

            return frames;
        }

        [Test]
        public void PollingWithoutAWholeHopRunsNoInference()
        {
            var buffer = new AudioWindowBuffer(2.5f, 0.5f);
            var recognizer = new Recognizer();
            var session = Session(buffer, recognizer);
            session.Begin("사과");

            buffer.Append(new float[Hop - 1]);
            for (int i = 0; i < 5; i++)
            {
                Assert.That(Drain(session), Is.Empty);
            }

            Assert.That(recognizer.Calls, Is.Zero);
            Assert.That(session.IsActive, Is.True);

            buffer.Append(new float[1]);
            var frames = Drain(session);

            Assert.That(frames.Count, Is.EqualTo(1));
            Assert.That(recognizer.Calls, Is.EqualTo(1));
            Assert.That(frames[0].EndSampleIndex, Is.EqualTo(Hop));
            Assert.That(frames[0].Streak, Is.EqualTo(1));
            Assert.That(frames[0].Confirmed, Is.False);
        }

        [Test]
        public void OneBigChunkStopsAtTheConfirmingFrame()
        {
            var buffer = new AudioWindowBuffer(2.5f, 0.5f);
            var recognizer = new Recognizer();
            var session = Session(buffer, recognizer);
            session.Begin("사과");

            // Four hops in a single chunk: the answer is in after two.
            buffer.Append(new float[4 * Hop]);
            var frames = Drain(session);

            Assert.That(frames.Count, Is.EqualTo(2));
            Assert.That(recognizer.Calls, Is.EqualTo(2), "late frames were scored");
            Assert.That(frames[1].Confirmed, Is.True);
            Assert.That(frames[1].EndSampleIndex, Is.EqualTo(2 * Hop));
            Assert.That(frames[1].Feedback.Frames, Is.EqualTo(2));
            Assert.That(frames[1].Feedback.Streak, Is.EqualTo(2));
            Assert.That(session.IsActive, Is.False);

            Assert.That(Drain(session), Is.Empty);
            Assert.That(recognizer.Calls, Is.EqualTo(2));
        }

        [Test]
        public void SplitAndMergedInputConfirmOnTheSameSample()
        {
            foreach (var size in new[] { 16000, 8000, 4000, 1 })
            {
                var buffer = new AudioWindowBuffer(2.5f, 0.5f);
                var recognizer = new Recognizer();
                var session = Session(buffer, recognizer);
                session.Begin("사과");

                var frames = new List<FrameScore>();
                for (int sent = 0; sent < 16000 && session.IsActive; sent += size)
                {
                    buffer.Append(new float[size]);
                    frames.AddRange(Drain(session));
                }

                Assert.That(frames.Count, Is.EqualTo(2), "chunk " + size);
                Assert.That(frames[1].Confirmed, Is.True, "chunk " + size);
                Assert.That(frames[1].EndSampleIndex, Is.EqualTo(2 * Hop),
                    "chunk " + size);
                Assert.That(recognizer.Calls, Is.EqualTo(2), "chunk " + size);
            }
        }

        [Test]
        public void AFramingFaultInvalidatesTheSessionUntilItBeginsAgain()
        {
            var buffer = new AudioWindowBuffer(2.5f, 0.5f, 1);
            var recognizer = new Recognizer();
            var session = Session(buffer, recognizer);
            session.Begin("사과");

            buffer.Append(new float[Hop]);
            Assert.That(Drain(session).Count, Is.EqualTo(1));

            // Two hops with nobody reading between them: the second one
            // has nowhere to go, and frames must not be dropped quietly.
            buffer.Append(new float[2 * Hop]);
            FrameScore frame;
            Assert.Throws<InvalidOperationException>(
                () => session.TryPush(out frame));
            Assert.That(session.IsActive, Is.False, "streak survived a lost frame");

            // Clearing the buffer is not enough: the streak was built on
            // windows that are gone, so it must not carry on.
            buffer.Reset();
            buffer.Append(new float[Hop]);
            Assert.That(Drain(session), Is.Empty);
            Assert.That(session.LastConfirmation, Is.Null);
            Assert.That(recognizer.Calls, Is.EqualTo(1));

            // A fresh question works, from zero. One hop at a time, since
            // this buffer only has room for one unread frame.
            session.Begin("사과");
            var frames = new List<FrameScore>();
            for (int i = 0; i < 2; i++)
            {
                buffer.Append(new float[Hop]);
                frames.AddRange(Drain(session));
            }

            Assert.That(frames.Count, Is.EqualTo(2));
            Assert.That(frames[1].Confirmed, Is.True);
            Assert.That(frames[1].EndSampleIndex, Is.EqualTo(2 * Hop));
        }

        [Test]
        public void EndClearsTheQueuedFrameAndThePartialHop()
        {
            var buffer = new AudioWindowBuffer(2.5f, 0.5f);
            var recognizer = new Recognizer();
            var session = Session(buffer, recognizer);
            session.Begin("사과");

            buffer.Append(new float[12000]);
            Assert.That(buffer.PendingFrames, Is.EqualTo(1));

            session.End();
            Assert.That(buffer.PendingFrames, Is.Zero, "an unread frame survived");
            Assert.That(buffer.AppendedSamples, Is.Zero, "the partial hop survived");

            // 4000 left over plus 4000 new would have made a frame if the
            // phase had carried over. It must not.
            buffer.Append(new float[4000]);
            Assert.That(buffer.PendingFrames, Is.Zero);
            buffer.Append(new float[4000]);
            Assert.That(buffer.PendingFrames, Is.EqualTo(1));
        }

        [Test]
        public void EndKeepsTheLastConfirmation()
        {
            var buffer = new AudioWindowBuffer(2.5f, 0.5f);
            var session = Session(buffer, new Recognizer());
            session.Begin("사과");
            buffer.Append(new float[2 * Hop]);
            var frames = Drain(session);
            Assert.That(frames[1].Confirmed, Is.True);

            session.End();
            Assert.That(session.LastConfirmation, Is.SameAs(frames[1].Feedback));
        }

        [Test]
        public void AScoringFailureBreaksTheStreakRatherThanBridgingIt()
        {
            var buffer = new AudioWindowBuffer(2.5f, 0.5f);
            var recognizer = new FailingRecognizer();
            var session = Session(buffer, recognizer);
            session.Begin("사과");

            buffer.Append(new float[Hop]);
            Assert.That(Drain(session).Count, Is.EqualTo(1), "first hop");

            buffer.Append(new float[Hop]);
            FrameScore frame;
            Assert.Throws<InvalidOperationException>(
                () => session.TryPush(out frame));
            Assert.That(session.IsActive, Is.False,
                "the streak outlived the frame that failed");

            // A caller that swallows the exception and keeps feeding must
            // not be able to join the hop before the failure to the hop
            // after it and call the two consecutive.
            buffer.Append(new float[Hop]);
            Assert.That(Drain(session), Is.Empty);
            Assert.That(session.LastConfirmation, Is.Null);
            Assert.That(recognizer.Calls, Is.EqualTo(2));

            // Beginning again works, counting from zero.
            session.Begin("사과");
            var frames = new List<FrameScore>();
            for (int i = 0; i < 2; i++)
            {
                buffer.Append(new float[Hop]);
                frames.AddRange(Drain(session));
            }

            Assert.That(frames.Count, Is.EqualTo(2));
            Assert.That(frames[1].Confirmed, Is.True);
            Assert.That(frames[1].Feedback.Frames, Is.EqualTo(2));
            Assert.That(recognizer.Calls, Is.EqualTo(4));
        }

        [Test]
        public void DrainingNeedsABufferAndAQuestion()
        {
            var recognizer = new Recognizer();
            FrameScore frame;

            var unbuffered = Session(null, recognizer);
            unbuffered.Begin("사과");
            Assert.Throws<InvalidOperationException>(
                () => unbuffered.TryPush(out frame));

            var session = Session(new AudioWindowBuffer(2.5f, 0.5f), recognizer);
            Assert.Throws<InvalidOperationException>(
                () => session.TryPush(out frame));
            Assert.That(recognizer.Calls, Is.Zero);
        }
    }
}
