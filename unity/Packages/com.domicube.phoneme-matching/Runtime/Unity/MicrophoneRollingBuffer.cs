#if UNITY_5_3_OR_NEWER
using System;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Unity
{
    /// <summary>
    /// Captures the microphone and hands out the most recent
    /// <see cref="WindowSeconds"/> of audio at 16 kHz mono.
    ///
    /// Unity's Microphone writes into a looping AudioClip, so this tracks
    /// the write head and copies out only the trailing window. Callers
    /// re-recognise that whole window every hop rather than recognising
    /// each new chunk on its own: wav2vec2 is a context model and short
    /// isolated fragments come back garbled (measured on a 4.4s
    /// utterance, 0.5s chunks lost a quarter of the phonemes).
    /// </summary>
    public sealed class MicrophoneRollingBuffer : IDisposable
    {
        public const int TargetSampleRate = 16000;

        private readonly int _requestedSampleRate;
        private readonly int _clipSeconds;

        /// <summary>
        /// What the device actually opened at, which is not always what
        /// we asked for: a microphone that cannot do 16 kHz is opened at
        /// its own rate instead. Resampling with the requested rate then
        /// shifts the audio in time and the recogniser returns nonsense -
        /// silently, since the samples still look like speech.
        /// </summary>
        private int _deviceSampleRate;
        private AudioClip _clip;
        private string _device;
        private int _lastReadPosition;

        /// <summary>Seconds of audio handed to the recogniser.</summary>
        public float WindowSeconds { get; }

        /// <summary>Total seconds captured since <see cref="Start"/>.</summary>
        public float Elapsed { get; private set; }

        public bool IsRecording => _clip != null
            && Microphone.IsRecording(_device);

        private readonly float[] _window;
        private int _filled;

        public MicrophoneRollingBuffer(
            float windowSeconds = 2.5f,
            int deviceSampleRate = 16000,
            int clipSeconds = 10)
        {
            if (windowSeconds <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(windowSeconds));
            }

            WindowSeconds = windowSeconds;
            _requestedSampleRate = deviceSampleRate;
            _deviceSampleRate = deviceSampleRate;
            _clipSeconds = clipSeconds;
            _window = new float[Mathf.CeilToInt(
                windowSeconds * TargetSampleRate)];
        }

        /// <summary>Begin capture. Pass null for the default device.</summary>
        public void Start(string device = null)
        {
            Stop();

            if (Microphone.devices.Length == 0)
            {
                throw new InvalidOperationException(
                    "no microphone device available");
            }

            _device = device ?? Microphone.devices[0];
            _clip = Microphone.Start(
                _device, true, _clipSeconds, _requestedSampleRate);

            if (_clip == null)
            {
                throw new InvalidOperationException(
                    $"microphone '{_device}' could not be opened");
            }

            // Trust the clip over the request, so a device that refused
            // 16 kHz is resampled from the rate it really gave us.
            _deviceSampleRate = _clip.frequency > 0
                ? _clip.frequency
                : _requestedSampleRate;

            _lastReadPosition = 0;
            _filled = 0;
            Elapsed = 0f;
        }

        /// <summary>The rate the device opened at (see Start).</summary>
        public int DeviceSampleRate => _deviceSampleRate;

        public void Stop()
        {
            if (_clip == null)
            {
                return;
            }

            if (_device != null && Microphone.IsRecording(_device))
            {
                Microphone.End(_device);
            }

            UnityEngine.Object.Destroy(_clip);
            _clip = null;
        }

        /// <summary>
        /// Drain whatever the microphone captured since the last call into
        /// the rolling window. Returns false when nothing new arrived.
        /// </summary>
        public bool Pump()
        {
            if (_clip == null)
            {
                return false;
            }

            int position = Microphone.GetPosition(_device);
            int available = position - _lastReadPosition;
            if (available < 0)
            {
                // The looping clip wrapped around.
                available += _clip.samples;
            }

            if (available <= 0)
            {
                return false;
            }

            var raw = new float[available];
            // ReadRaw handles the wrap for us by reading from the offset.
            _clip.GetData(raw, _lastReadPosition % _clip.samples);
            _lastReadPosition = position % _clip.samples;

            var resampled = Resample(raw, _deviceSampleRate, TargetSampleRate);
            Append(resampled);
            Elapsed += available / (float)_deviceSampleRate;
            return true;
        }

        /// <summary>
        /// The trailing window as 16 kHz mono. The array is freshly
        /// allocated so callers may hold on to it.
        /// </summary>
        public float[] Snapshot()
        {
            var copy = new float[_filled];
            Array.Copy(_window, _window.Length - _filled, copy, 0, _filled);
            return copy;
        }

        private void Append(float[] samples)
        {
            if (samples.Length >= _window.Length)
            {
                Array.Copy(samples, samples.Length - _window.Length,
                    _window, 0, _window.Length);
                _filled = _window.Length;
                return;
            }

            // Shift the existing tail left, then write the new samples.
            int keep = _window.Length - samples.Length;
            Array.Copy(_window, samples.Length, _window, 0, keep);
            Array.Copy(samples, 0, _window, keep, samples.Length);
            _filled = Mathf.Min(_window.Length, _filled + samples.Length);
        }

        /// <summary>
        /// Linear resample. Good enough because the capture rate is
        /// requested as 16 kHz already; this only covers devices that
        /// refuse that rate.
        /// </summary>
        private static float[] Resample(float[] input, int from, int to)
        {
            if (from == to || input.Length == 0)
            {
                return input;
            }

            int count = Mathf.Max(1,
                Mathf.RoundToInt(input.Length * (to / (float)from)));
            var output = new float[count];
            float step = (input.Length - 1) / (float)Mathf.Max(1, count - 1);

            for (int i = 0; i < count; i++)
            {
                float pos = i * step;
                int idx = (int)pos;
                float frac = pos - idx;
                output[i] = idx + 1 < input.Length
                    ? Mathf.Lerp(input[idx], input[idx + 1], frac)
                    : input[input.Length - 1];
            }

            return output;
        }

        public void Dispose()
        {
            Stop();
        }
    }
}
#endif
