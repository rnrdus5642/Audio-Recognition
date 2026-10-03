#if UNITY_5_3_OR_NEWER
using System;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Unity
{
    /// <summary>
    /// wav2vec2 CTC through Sentis 1.x: audio -> logits -> greedy decode
    /// -> Hangul -> IPA.
    ///
    /// Same class name and same behaviour as the Sentis 2.x version next
    /// door; only one of the two ever compiles, decided by which package
    /// the project has. That way calling code is identical on Unity 2022
    /// and Unity 6, and only the `using` for ModelAsset differs.
    ///
    /// Sentis 1.2 is the newest build that runs on Unity 2022, and it
    /// cannot execute a graph with a dynamic time axis - every input
    /// length dies in the same Reshape. Export the model with the axis
    /// pinned to the window length:
    ///
    ///     python -m python.tools.export_onnx --static-samples 40000
    ///
    /// Measured on 2022.3 with that graph: 30 ms per 2.5 s window on
    /// GPUCompute, 390 ms on CPU, phoneme ids identical to Python across
    /// all 124 frames of the golden clips.
    /// </summary>
    public sealed class SentisPhonemeRecognizer : IPhonemeRecognizer, IDisposable
    {
        private readonly CtcVocabulary _vocab;
        private readonly IWorker _worker;
        private float[] _normalized = new float[0];
        private bool _disposed;

        public string Language => "ko";

        /// <summary>Samples per second the model was trained on.</summary>
        public int SampleRate => _vocab.SamplingRate;

        public SentisPhonemeRecognizer(ModelAsset modelAsset, CtcVocabulary vocab)
        {
            if (modelAsset == null)
            {
                throw new ArgumentNullException(nameof(modelAsset));
            }

            _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));
            _worker = WorkerFactory.CreateWorker(
                BackendType.GPUCompute, ModelLoader.Load(modelAsset));
        }

        /// <summary>
        /// Run one throwaway inference so shader compilation and the
        /// weight upload land here instead of mid-lesson. Pass the same
        /// window length the session will use - with a static graph, any
        /// other length throws.
        /// </summary>
        public void Warmup(float windowSeconds = 2.5f)
        {
            int samples = Mathf.Max(1, Mathf.CeilToInt(
                windowSeconds * _vocab.SamplingRate));
            Infer(new float[samples], out _, out _);
        }

        public List<string> Recognize(float[] audio16kMono)
        {
            RecognizeWithText(audio16kMono, out _, out var phonemes);
            return phonemes;
        }

        public void RecognizeWithText(
            float[] audio16kMono, out string text, out List<string> phonemes)
        {
            if (audio16kMono == null || audio16kMono.Length == 0)
            {
                text = string.Empty;
                phonemes = new List<string>();
                return;
            }

            Infer(audio16kMono, out var logits, out var shape);
            text = CtcDecoder.Decode(logits, shape.frames, shape.classes, _vocab);
            phonemes = text.Length == 0
                ? new List<string>()
                : Korean.JamoIpa.ToPhonemes(Korean.Rules.Apply(text));
        }

        private void Infer(
            float[] audio, out float[] logits, out (int frames, int classes) shape)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(SentisPhonemeRecognizer));
            }

            var input = Prepare(audio);

            using (var tensor = new TensorFloat(
                       new TensorShape(1, input.Length), input))
            {
                try
                {
                    _worker.Execute(tensor);
                }
                catch (Exception e)
                {
                    // Nearly always the window and the graph disagreeing
                    // about length, and the engine's own message says
                    // only "reshaped length does not match".
                    throw new InvalidOperationException(
                        $"inference failed on {input.Length} samples "
                        + $"({input.Length / (float)_vocab.SamplingRate:F2}s). "
                        + "A statically exported graph accepts exactly the "
                        + "length it was exported with - re-export with "
                        + "--static-samples matching the window.", e);
                }

                var output = _worker.PeekOutput() as TensorFloat;
                if (output == null)
                {
                    throw new InvalidOperationException(
                        "model produced no float output");
                }

                output.MakeReadable();
                shape = (output.shape[1], output.shape[2]);
                if (shape.classes != _vocab.Size)
                {
                    throw new InvalidOperationException(
                        $"model emits {shape.classes} classes but the "
                        + $"vocabulary has {_vocab.Size}. Regenerate it with "
                        + "python -m python.tools.export_ctc_vocab");
                }

                logits = output.ToReadOnlyArray();
            }
        }

        /// <summary>
        /// Zero mean, unit variance - what
        /// <c>Wav2Vec2FeatureExtractor</c> does before inference.
        /// Skipping it does not fail loudly; the model simply returns
        /// worse phonemes, which would read as a matching problem.
        ///
        /// The caller's array is left alone: it belongs to the rolling
        /// microphone buffer, which reuses it.
        /// </summary>
        private float[] Prepare(float[] audio)
        {
            if (!_vocab.Normalize)
            {
                return audio;
            }

            if (_normalized.Length != audio.Length)
            {
                _normalized = new float[audio.Length];
            }

            double sum = 0.0;
            for (int i = 0; i < audio.Length; i++)
            {
                sum += audio[i];
            }

            double mean = sum / audio.Length;

            double variance = 0.0;
            for (int i = 0; i < audio.Length; i++)
            {
                double d = audio[i] - mean;
                variance += d * d;
            }

            variance /= audio.Length;

            double scale = 1.0 / Math.Sqrt(variance + _vocab.NormalizeEpsilon);
            for (int i = 0; i < audio.Length; i++)
            {
                _normalized[i] = (float)((audio[i] - mean) * scale);
            }

            return _normalized;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _worker.Dispose();
        }
    }
}
#endif
