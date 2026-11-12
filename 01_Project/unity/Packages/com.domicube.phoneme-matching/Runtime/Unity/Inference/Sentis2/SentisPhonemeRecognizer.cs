#if UNITY_5_3_OR_NEWER
using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Unity
{
    /// <summary>
    /// Korean CTC through Inference Engine: audio -> logits -> greedy decode ->
    /// Hangul -> IPA. The Unity counterpart of
    /// python/runtime/recognizer/ko/asr.py.
    ///
    /// ModelAsset construction defaults to GPUCompute. FromFile also allows
    /// explicit CPU validation. Latency and memory depend on the model; the
    /// old wav2vec2 timings do not apply to the larger HuBERT model.
    /// Call <see cref="Warmup"/> while loading the scene, then measure frame
    /// pacing on the actual PC/VR device before choosing the capture hop.
    /// </summary>
    public sealed class SentisPhonemeRecognizer : IPhonemeRecognizer, IDisposable
    {
        private readonly CtcVocabulary _vocab;
        private readonly Worker _worker;
        private float[] _normalized = new float[0];
        private bool _disposed;

        public string Language => "ko";

        /// <summary>Samples per second the model was trained on.</summary>
        public int SampleRate => _vocab.SamplingRate;

        public SentisPhonemeRecognizer(ModelAsset modelAsset, CtcVocabulary vocab)
            : this(LoadAsset(modelAsset), vocab, BackendType.GPUCompute)
        {
        }

        private static Model LoadAsset(ModelAsset modelAsset)
        {
            if (modelAsset == null)
            {
                throw new ArgumentNullException(nameof(modelAsset));
            }
            return ModelLoader.Load(modelAsset);
        }

        private SentisPhonemeRecognizer(Model model, CtcVocabulary vocab, BackendType backend)
        {
            _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));
            _worker = new Worker(model, backend);
        }

        /// <summary>
        /// Load a .sentis file directly on Windows. The existing ModelAsset API
        /// keeps its GPU default; CPU is an explicit validation option here.
        /// </summary>
        public static SentisPhonemeRecognizer FromFile(string path, CtcVocabulary vocab,
            BackendType backend = BackendType.GPUCompute)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Model path is empty", nameof(path));
            if (vocab == null) throw new ArgumentNullException(nameof(vocab));
            return new SentisPhonemeRecognizer(ModelLoader.Load(path), vocab, backend);
        }

        /// <summary>
        /// Run one throwaway inference so the cost of compiling shaders
        /// and uploading weights lands here instead of mid-lesson.
        /// Pass the same window length the session will use.
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

            // Sentis schedules the whole graph and the readback blocks
            // until it finishes. This API is synchronous: model latency is
            // also a main-thread stall. Measure it on the target PC/VR device.
            using (var tensor = new Tensor<float>(
                       new TensorShape(1, input.Length), input))
            {
                try
                {
                    _worker.Schedule(tensor);
                }
                catch (Exception e)
                {
                    // A statically exported graph accepts only the length
                    // it was exported with, and the engine's own message
                    // says nothing about which length that is.
                    throw new InvalidOperationException(
                        $"inference failed on {input.Length} samples "
                        + $"({input.Length / (float)_vocab.SamplingRate:F2}s). "
                        + "If the model was exported with --static-samples, "
                        + "the window must match it exactly.", e);
                }

                var output = _worker.PeekOutput() as Tensor<float>;
                if (output == null)
                {
                    throw new InvalidOperationException(
                        "model produced no float output");
                }

                using (var cpu = output.ReadbackAndClone())
                {
                    shape = (cpu.shape[1], cpu.shape[2]);
                    if (shape.classes != _vocab.Size)
                    {
                        throw new InvalidOperationException(
                            $"model emits {shape.classes} classes but the "
                            + $"vocabulary has {_vocab.Size}. Regenerate it "
                            + "with python -m python.tools.export_ctc_vocab");
                    }

                    logits = cpu.DownloadToArray();
                }
            }
        }

        /// <summary>
        /// Apply zero-mean unit-variance input only when the vocabulary's
        /// model contract requests it. The orthographic HuBERT model expects
        /// raw audio and must bypass normalization.
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
