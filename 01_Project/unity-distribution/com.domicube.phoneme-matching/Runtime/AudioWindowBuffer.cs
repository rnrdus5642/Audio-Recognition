using System;
using System.Collections.Generic;

namespace DomiCube.PhonemeMatching
{
    /// <summary>
    /// One window, copied at the exact sample where a hop boundary was
    /// crossed. <see cref="EndSampleIndex"/> is how many 16 kHz samples
    /// had been appended at that moment, so it is always a multiple of
    /// the hop and names the frame no matter how the audio was chunked
    /// on the way in.
    /// </summary>
    public sealed class HopFrame
    {
        /// <summary>The trailing window as it stood at the boundary.</summary>
        public float[] Window;

        /// <summary>16 kHz samples appended when this window was taken.</summary>
        public long EndSampleIndex;
    }

    /// <summary>
    /// Keeps the most recent N seconds of audio at 16 kHz, fed from
    /// wherever the application already has it.
    ///
    /// Exists because an app that captures its own microphone - a VR
    /// title with voice chat, say - must not have this package open the
    /// same device a second time. Hand the samples over instead:
    ///
    ///     buffer.Append(myChunk, 48000);      // whenever audio arrives
    ///     // every hop:
    ///     var frame = session.Push(buffer.Snapshot());
    ///
    /// That polling shape scores whenever the caller happens to ask, so
    /// how often a child is judged follows the capture callback rather
    /// than the audio. Pass `hopSeconds` instead and the buffer queues
    /// one <see cref="HopFrame"/> per hop of *new samples*, which a
    /// consumer drains in order:
    ///
    ///     var buffer = new AudioWindowBuffer(2.5f, 0.5f);
    ///     var session = new PronunciationSession(m, c, r, 2, buffer);
    ///     buffer.Append(myChunk, 48000);
    ///     while (session.TryPush(out var frame)) { ... }
    ///
    /// Framing is off unless asked for, so a caller that only ever calls
    /// <see cref="Snapshot"/> never grows a queue nobody drains.
    ///
    /// <see cref="Unity.MicrophoneRollingBuffer"/> is this class plus a
    /// microphone, for callers that would rather not own one.
    ///
    /// Safe to append from the audio thread while scoring on the main
    /// one: Unity's OnAudioFilterRead is the usual way to tap audio and
    /// it does not run on the main thread. Both sides do nothing but copy
    /// a few thousand floats under the lock.
    /// </summary>
    public sealed class AudioWindowBuffer
    {
        public const int TargetSampleRate = 16000;

        /// <summary>
        /// Unread frames tolerated before the queue faults. 64 is 32 s at
        /// the default hop - far longer than any question, so reaching it
        /// means the consumer stopped draining rather than fell behind.
        /// </summary>
        public const int DefaultMaxPendingFrames = 64;

        private readonly object _gate = new object();
        private readonly float[] _window;
        private readonly int _hopSamples;
        private readonly int _maxPendingFrames;
        private readonly Queue<HopFrame> _frames = new Queue<HopFrame>();
        private int _filled;
        private long _appended;
        private long _nextBoundary;
        private bool _overflowed;

        /// <param name="windowSeconds">
        /// Must match the length the acoustic model expects - a
        /// statically exported graph accepts exactly one size.
        /// </param>
        /// <param name="hopSeconds">
        /// New audio per scored frame. 0 (the default) leaves framing off
        /// and the buffer behaves exactly as it always did. Fixed for the
        /// life of the buffer, so a session cannot change its hop halfway
        /// through a question.
        /// </param>
        /// <param name="maxPendingFrames">
        /// How many unread frames may queue up before the buffer faults.
        /// </param>
        public AudioWindowBuffer(
            float windowSeconds = 2.5f,
            float hopSeconds = 0f,
            int maxPendingFrames = DefaultMaxPendingFrames)
        {
            if (windowSeconds <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(windowSeconds));
            }

            if (hopSeconds < 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(hopSeconds));
            }

            if (maxPendingFrames < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(maxPendingFrames));
            }

            _hopSamples = hopSeconds <= 0f
                ? 0
                : (int)Math.Round(hopSeconds * TargetSampleRate);
            if (hopSeconds > 0f && _hopSamples < 1)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(hopSeconds), "hop rounds to zero samples");
            }

            _maxPendingFrames = maxPendingFrames;
            _nextBoundary = _hopSamples;
            WindowSeconds = windowSeconds;
            _window = new float[
                (int)Math.Ceiling(windowSeconds * TargetSampleRate)];
        }

        public float WindowSeconds { get; }

        /// <summary>Samples the window holds - what Snapshot returns.</summary>
        public int WindowSamples => _window.Length;

        /// <summary>New samples per frame; 0 when framing is off.</summary>
        public int HopSamples => _hopSamples;

        /// <summary>Unread frames allowed before the queue faults.</summary>
        public int MaxPendingFrames => _maxPendingFrames;

        /// <summary>16 kHz samples appended since the last Reset.</summary>
        public long AppendedSamples
        {
            get { lock (_gate) { return _appended; } }
        }

        /// <summary>Frames waiting to be read.</summary>
        public int PendingFrames
        {
            get { lock (_gate) { return _frames.Count; } }
        }

        /// <summary>
        /// True once more than <see cref="MaxPendingFrames"/> frames went
        /// unread. Latched until <see cref="Reset"/>.
        /// </summary>
        public bool Overflowed
        {
            get { lock (_gate) { return _overflowed; } }
        }

        /// <summary>True once real audio fills the whole window.</summary>
        public bool IsFull
        {
            get
            {
                lock (_gate)
                {
                    return _filled >= _window.Length;
                }
            }
        }

        /// <summary>
        /// Add mono samples in roughly [-1, 1]. Resampled to 16 kHz when
        /// <paramref name="sampleRate"/> differs. Interleaved stereo must
        /// be mixed down to mono first.
        /// </summary>
        public void Append(float[] samples, int sampleRate = TargetSampleRate)
        {
            if (samples == null || samples.Length == 0)
            {
                return;
            }

            if (sampleRate <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(sampleRate));
            }

            // Resampling allocates, so keep it outside the lock.
            var resampled = Resample(samples, sampleRate, TargetSampleRate);

            lock (_gate)
            {
                if (_hopSamples <= 0)
                {
                    WriteLocked(resampled, 0, resampled.Length);
                    _appended += resampled.Length;
                    return;
                }

                int offset = 0;
                while (offset < resampled.Length)
                {
                    // Never write past the next boundary. The frame taken
                    // there has to be the window as it stood at that
                    // sample, not the newest one - otherwise one chunk
                    // carrying three hops would score the same window
                    // three times, and merging chunks would change the
                    // answer, which is the whole bug this replaces.
                    int room = (int)Math.Min(
                        (long)(resampled.Length - offset),
                        _nextBoundary - _appended);
                    WriteLocked(resampled, offset, room);
                    offset += room;
                    _appended += room;
                    if (_appended >= _nextBoundary)
                    {
                        EnqueueLocked();
                        _nextBoundary += _hopSamples;
                    }
                }
            }
        }

        /// <summary>
        /// Shift the window left by `count` and write that many samples
        /// from `source`. Caller holds the lock.
        /// </summary>
        private void WriteLocked(float[] source, int offset, int count)
        {
            if (count <= 0)
            {
                return;
            }

            if (count >= _window.Length)
            {
                Array.Copy(source, offset + count - _window.Length,
                    _window, 0, _window.Length);
                _filled = _window.Length;
                return;
            }

            int keep = _window.Length - count;
            Array.Copy(_window, count, _window, 0, keep);
            Array.Copy(source, offset, _window, keep, count);
            _filled = Math.Min(_window.Length, _filled + count);
        }

        /// <summary>Queue the window at a boundary. Caller holds the lock.</summary>
        private void EnqueueLocked()
        {
            if (_overflowed)
            {
                return;
            }

            if (_frames.Count >= _maxPendingFrames)
            {
                // Dropping the oldest frame and carrying on would leave a
                // streak built on windows nobody scored - a confirmation
                // that cannot be reproduced from the audio. Fail where it
                // can be seen instead, and make the reader clear it.
                _overflowed = true;
                _frames.Clear();
                return;
            }

            _frames.Enqueue(new HopFrame
            {
                Window = SnapshotLocked(),
                EndSampleIndex = _appended
            });
        }

        /// <summary>
        /// Take the oldest queued frame. False when none is ready, which
        /// is the normal answer for a partial hop - the caller must not
        /// score anything then.
        /// </summary>
        public bool TryReadFrame(out HopFrame frame)
        {
            lock (_gate)
            {
                if (_hopSamples <= 0)
                {
                    throw new InvalidOperationException(
                        "hop framing is off; construct AudioWindowBuffer "
                        + "with hopSeconds > 0 to read frames, or keep "
                        + "using Snapshot()");
                }

                if (_overflowed)
                {
                    throw new InvalidOperationException(
                        $"{_maxPendingFrames} frames went unread and the "
                        + "queue was cleared; audio arrived faster than it "
                        + "was scored. Call Reset() before reading again");
                }

                if (_frames.Count == 0)
                {
                    frame = null;
                    return false;
                }

                frame = _frames.Dequeue();
                return true;
            }
        }

        /// <summary>
        /// Mix interleaved channels down to mono, then append. This is
        /// the shape OnAudioFilterRead hands over.
        /// </summary>
        public void AppendInterleaved(
            float[] samples, int channels, int sampleRate = TargetSampleRate)
        {
            if (samples == null || samples.Length == 0)
            {
                return;
            }

            if (channels <= 1)
            {
                Append(samples, sampleRate);
                return;
            }

            int frames = samples.Length / channels;
            var mono = new float[frames];
            for (int f = 0; f < frames; f++)
            {
                float sum = 0f;
                int offset = f * channels;
                for (int c = 0; c < channels; c++)
                {
                    sum += samples[offset + c];
                }

                mono[f] = sum / channels;
            }

            Append(mono, sampleRate);
        }

        /// <summary>
        /// The trailing window, freshly allocated so callers may keep it.
        ///
        /// Always <see cref="WindowSamples"/> long. Before enough audio
        /// has arrived the missing head is silence rather than a shorter
        /// array: a statically exported model accepts exactly one length,
        /// so a short first frame would throw instead of scoring.
        /// </summary>
        public float[] Snapshot()
        {
            lock (_gate)
            {
                return SnapshotLocked();
            }
        }

        private float[] SnapshotLocked()
        {
            var copy = new float[_window.Length];
            Array.Copy(_window, _window.Length - _filled,
                copy, _window.Length - _filled, _filled);
            return copy;
        }

        /// <summary>
        /// Forget everything captured so far: the window, any queued
        /// frames, the overflow fault and the boundary phase. The next
        /// frame ends at exactly one hop, so a new question is never cut
        /// on the leftovers of the last one.
        /// </summary>
        public void Reset()
        {
            lock (_gate)
            {
                Array.Clear(_window, 0, _window.Length);
                _filled = 0;
                _frames.Clear();
                _appended = 0;
                _nextBoundary = _hopSamples;
                _overflowed = false;
            }
        }

        /// <summary>
        /// Linear resample. Good enough here because capture is normally
        /// requested at 16 kHz already; this covers sources that refuse
        /// that rate.
        /// </summary>
        public static float[] Resample(float[] input, int from, int to)
        {
            if (from == to || input.Length == 0)
            {
                return input;
            }

            int count = Math.Max(1,
                (int)Math.Round(input.Length * (to / (double)from)));
            var output = new float[count];
            double step = (input.Length - 1) / (double)Math.Max(1, count - 1);

            for (int i = 0; i < count; i++)
            {
                double pos = i * step;
                int index = (int)pos;
                double frac = pos - index;
                output[i] = index + 1 < input.Length
                    ? (float)(input[index] * (1.0 - frac)
                              + input[index + 1] * frac)
                    : input[input.Length - 1];
            }

            return output;
        }
    }
}
