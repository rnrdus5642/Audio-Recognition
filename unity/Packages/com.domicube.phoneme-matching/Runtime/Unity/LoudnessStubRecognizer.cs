#if UNITY_5_3_OR_NEWER
using System.Collections.Generic;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Unity
{
    /// <summary>
    /// Stand-in recogniser for bringing the loop up before an acoustic
    /// model exists.
    ///
    /// Emits <see cref="Phonemes"/> whenever the window is louder than
    /// <see cref="RmsThreshold"/>, and nothing when it is quiet. That is
    /// enough to exercise everything around the model - microphone
    /// capture, the rolling window, the streak, the auto-stop - and to
    /// confirm the timing feels right, without waiting on ONNX export.
    ///
    /// It says nothing about recognition accuracy: any loud sound counts
    /// as the answer. Replace it with a real
    /// <see cref="IPhonemeRecognizer"/> before drawing conclusions about
    /// whether pronunciation matching works.
    /// </summary>
    public sealed class LoudnessStubRecognizer : IPhonemeRecognizer
    {
        /// <summary>What to emit while sound is present.</summary>
        public List<string> Phonemes { get; set; } =
            new List<string> { "s", "a", "k", "w", "a" };

        /// <summary>RMS above which the window counts as speech.</summary>
        public float RmsThreshold { get; set; } = 0.01f;

        public string Language => "ko";

        public List<string> Recognize(float[] audio16kMono)
        {
            RecognizeWithText(audio16kMono, out _, out var phonemes);
            return phonemes;
        }

        public void RecognizeWithText(
            float[] audio16kMono, out string text, out List<string> phonemes)
        {
            float rms = Rms(audio16kMono);
            if (rms < RmsThreshold)
            {
                text = string.Empty;
                phonemes = new List<string>();
                return;
            }

            text = $"(stub rms={rms:F3})";
            phonemes = new List<string>(Phonemes);
        }

        private static float Rms(float[] samples)
        {
            if (samples == null || samples.Length == 0)
            {
                return 0f;
            }

            double sum = 0.0;
            for (int i = 0; i < samples.Length; i++)
            {
                sum += (double)samples[i] * samples[i];
            }

            return Mathf.Sqrt((float)(sum / samples.Length));
        }
    }
}
#endif
