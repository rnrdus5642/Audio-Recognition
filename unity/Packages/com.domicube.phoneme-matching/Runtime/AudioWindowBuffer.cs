using System;

namespace DomiCube.PhonemeMatching
{
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

        private readonly object _gate = new object();
        private readonly float[] _window;
        private int _filled;

        /// <param name="windowSeconds">
        /// Must match the length the acoustic model expects - a
        /// statically exported graph accepts exactly one size.
        /// </param>
        public AudioWindowBuffer(float windowSeconds = 2.5f)
        {
            if (windowSeconds <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(windowSeconds));
            }

            WindowSeconds = windowSeconds;
            _window = new float[
                (int)Math.Ceiling(windowSeconds * TargetSampleRate)];
        }

        public float WindowSeconds { get; }

        /// <summary>Samples the window holds - what Snapshot returns.</summary>
        public int WindowSamples => _window.Length;

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
                if (resampled.Length >= _window.Length)
                {
                    Array.Copy(resampled, resampled.Length - _window.Length,
                        _window, 0, _window.Length);
                    _filled = _window.Length;
                    return;
                }

                // Shift the existing tail left, then write the new samples.
                int keep = _window.Length - resampled.Length;
                Array.Copy(_window, resampled.Length, _window, 0, keep);
                Array.Copy(resampled, 0, _window, keep, resampled.Length);
                _filled = Math.Min(_window.Length, _filled + resampled.Length);
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
            var copy = new float[_window.Length];
            lock (_gate)
            {
                Array.Copy(_window, _window.Length - _filled,
                    copy, _window.Length - _filled, _filled);
            }

            return copy;
        }

        /// <summary>Forget everything captured so far.</summary>
        public void Reset()
        {
            lock (_gate)
            {
                Array.Clear(_window, 0, _window.Length);
                _filled = 0;
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
