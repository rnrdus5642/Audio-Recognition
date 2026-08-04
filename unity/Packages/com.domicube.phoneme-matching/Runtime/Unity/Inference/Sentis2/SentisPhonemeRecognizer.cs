#if UNITY_5_3_OR_NEWER
using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Unity
{
    /// <summary>
    /// wav2vec2 CTC through Sentis: audio -> logits -> greedy decode ->
    /// Hangul -> IPA. The Unity counterpart of
    /// python/runtime/recognizer/ko/asr.py.
    ///
    /// Backend is GPUCompute and there is deliberately no CPU fallback.
    /// Measured on this graph with a 2.5 s window: GPU 22-32 ms, CPU
    /// 460 ms against a 500 ms hop budget - and the CPU path saturates
    /// six cores doing it, which is what actually breaks VR. A machine
    /// too weak for compute shaders cannot run PCVR either.
    ///
    /// The first inference pays ~1.9 s for shader compilation and for
    /// moving 1.2 GB of weights to the GPU. Call <see cref="Warmup"/>
    /// while the scene is loading; otherwise the child's first word is
    /// the thing that waits.
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
        {
            if (modelAsset == null)
            {
                throw new ArgumentNullException(nameof(modelAsset));
            }

            _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));
            _worker = new Worker(
                ModelLoader.Load(modelAsset), BackendType.GPUCompute);
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
                : Korean.JamoIpa.ToPhonemes(text);
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
            // until it finishes, so this costs a frame's worth of main
            // thread. At ~25 ms that is tolerable at a 0.5 s hop; if VR
            // frame pacing suffers, split it with ScheduleIterable.
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
        /// Zero mean, unit variance - what
        /// <c>Wav2Vec2FeatureExtractor</c> does before inference. Skipping
        /// it does not fail loudly; the model simply returns worse
        /// phonemes, which would read as a matching problem.
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
