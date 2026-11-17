#if UNITY_5_3_OR_NEWER
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Unity.Sentis;
using Unity.Profiling;
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
    public sealed class SentisPhonemeRecognizer : IAsyncPhonemeRecognizer, IAcousticEvidenceRecognizer, IDisposable
    {
        private readonly CtcVocabulary _vocab;
        private readonly IWorker _worker;
        private readonly Model _loadedModel;
        private static readonly ProfilerMarker LoadFileMarker = new ProfilerMarker("DomiCube.Model.LoadFile");
        private static readonly ProfilerMarker CreateWorkerMarker = new ProfilerMarker("DomiCube.Model.CreateWorker");
        private static readonly ProfilerMarker ScheduleMarker = new ProfilerMarker("DomiCube.Inference.Schedule");
        private float[] _normalized = new float[0];
        private float[] _inflightInput = new float[0];
        private bool _disposed;
        private bool _workerDisposed;
        private Exception _asyncFailure;
        private readonly int _ownerThread = Thread.CurrentThread.ManagedThreadId;
        public bool IsRecognizing { get; private set; }
        public bool IsFaulted => _asyncFailure != null;

        // A layer count also bounds GPU bursts; CPU scheduling time alone does not.
        // 64 layers required ~21 frames (~700 ms at 30 fps) for the shipped
        // graph, exceeding its 500 ms input hop. Keep both bounds, with enough
        // throughput at 30 fps. A single Sentis layer can exceed the time budget.
        private const int LayersPerFrame = 128;
        private const double ScheduleBudgetMilliseconds = 2.0;
        private float[] _lastLogits;
        private int _lastFrames, _lastClasses;
        private string _lastText = "";
        private CtcSequenceScorer _lastScorer;

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

        /// <summary>Load a verified external model without importing/copying it into a Unity project.</summary>
        public SentisPhonemeRecognizer(string modelPath, CtcVocabulary vocab)
        {
            _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));
            // Sentis Load(string) opens read/write exclusively. Use its stream API for shared read access.
            using (var input = new System.IO.FileStream(modelPath, System.IO.FileMode.Open,
                       System.IO.FileAccess.Read, System.IO.FileShare.Read))
                _worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, ModelLoader.Load(input));
        }

        private SentisPhonemeRecognizer(Model model, CtcVocabulary vocab)
        {
            _vocab = vocab;
            using (CreateWorkerMarker.Auto())
                _worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
            _loadedModel = model;
        }

        /// <summary>
        /// Sentis 1 external-file preparation. Only stream deserialization and
        /// managed weight pinning run in the background; worker/GPU operations
        /// and continuations stay on the calling Unity thread. Call during scene
        /// preparation and await before starting an answer's clock.
        /// </summary>
        public static async Task<SentisPhonemeRecognizer> CreatePreparedAsync(
            string modelPath, CtcVocabulary vocab, float windowSeconds = 2.5f,
            CancellationToken cancellation = default)
        {
            if (vocab == null) throw new ArgumentNullException(nameof(vocab));
            if (SynchronizationContext.Current == null)
                throw new InvalidOperationException("Model preparation requires Unity's synchronization context.");
            cancellation.ThrowIfCancellationRequested();
            var model = await Task.Run(() =>
            {
                // This overload never accesses ModelAsset/Resources or a GPU.
                using (LoadFileMarker.Auto())
                using (var input = new System.IO.FileStream(modelPath, System.IO.FileMode.Open,
                           System.IO.FileAccess.Read, System.IO.FileShare.Read))
                    return ModelLoader.Load(input);
            }, cancellation);
            SentisPhonemeRecognizer recognizer = null;
            try
            {
                cancellation.ThrowIfCancellationRequested();
                recognizer = new SentisPhonemeRecognizer(model, vocab);
                await NextEngineFrame();
                await recognizer.RecognizeAsync(new float[Mathf.Max(1,
                    Mathf.CeilToInt(windowSeconds * vocab.SamplingRate))], cancellation);
                cancellation.ThrowIfCancellationRequested();
                recognizer.ClearEvidence();
                return recognizer;
            }
            catch
            {
                recognizer?.Dispose();
                throw;
            }
            finally
            {
                // Constants can retain views into the pinned CPU weights.
                // Keep them alive until the worker drains and is disposed.
                if (recognizer == null) ReleaseModelWeights(model);
            }
        }

        private static void ReleaseModelWeights(Model model)
        {
            if (model == null) return;
            var released = new HashSet<NativeTensorArray>();
            foreach (var constant in model.constants)
                if (constant.weights != null && released.Add(constant.weights)) constant.weights.Dispose();
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
            RequireAvailable();
            ClearEvidence();
            if (audio16kMono == null || audio16kMono.Length == 0)
            {
                text = string.Empty;
                phonemes = new List<string>();
                return;
            }

            Infer(audio16kMono, out var logits, out var shape);
            var result = DecodeResult(logits, shape.frames, shape.classes);
            text = result.Text;
            phonemes = result.Phonemes;
        }

        private RecognitionResult DecodeResult(float[] logits, int frames, int classes)
        {
            if (classes != _vocab.Size)
                throw new InvalidOperationException($"model emits {classes} classes but the vocabulary has {_vocab.Size}");
            var text = CtcDecoder.Decode(logits, frames, classes, _vocab);
            _lastLogits = logits;
            _lastFrames = frames;
            _lastClasses = classes;
            _lastText = text;
            var phonemes = text.Length == 0
                ? new List<string>()
                : Korean.JamoIpa.ToPhonemes(Korean.Rules.Apply(text));
            return new RecognitionResult(text, phonemes);
        }

        /// <summary>
        /// Schedules bounded GPU work on Unity's thread and reads back only after
        /// completion. Never wrap Unity GPU calls in Task.Run. Cancellation drops
        /// the result while already submitted work drains with the same budget.
        /// </summary>
        public async Task<RecognitionResult> RecognizeAsync(float[] audio16kMono, CancellationToken cancellation)
        {
            RequireAvailable();
            cancellation.ThrowIfCancellationRequested();
            if (!SystemInfo.supportsAsyncGPUReadback)
                throw new NotSupportedException("This GPU/API does not support asynchronous readback. Use a supported graphics device/API.");
            if (SynchronizationContext.Current == null)
                throw new InvalidOperationException("Asynchronous recognition requires Unity's synchronization context.");
            ClearEvidence();
            if (audio16kMono == null || audio16kMono.Length == 0)
                return new RecognitionResult("", null);

            IsRecognizing = true;
            try
            {
                // The caller may recycle its audio buffer while the GPU is busy.
                var prepared = Prepare(audio16kMono);
                if (_inflightInput.Length != prepared.Length) _inflightInput = new float[prepared.Length];
                Array.Copy(prepared, _inflightInput, prepared.Length);
                var samples = _inflightInput;
                using (var tensor = new TensorFloat(new TensorShape(1, samples.Length), samples))
                {
                    var schedule = _worker.StartManualSchedule(tensor);
                    bool finished = false;
                    var budget = new System.Diagnostics.Stopwatch();
                    while (!finished)
                    {
                        budget.Restart();
                        using (ScheduleMarker.Auto())
                        {
                            for (int layer = 0; layer < LayersPerFrame && budget.Elapsed.TotalMilliseconds < ScheduleBudgetMilliseconds; layer++)
                            {
                                if (schedule.MoveNext()) continue;
                                finished = true;
                                break;
                            }
                        }
                        if (!finished) await NextEngineFrame();
                    }

                    var output = _worker.PeekOutput() as TensorFloat;
                    if (output == null) throw new InvalidOperationException("model produced no float output");
                    int frames = output.shape[1], classes = output.shape[2];
                    var readback = new TaskCompletionSource<float[]>(TaskCreationOptions.RunContinuationsAsynchronously);
                    output.AsyncReadbackRequest(success =>
                    {
                        try
                        {
                            if (!success) throw new InvalidOperationException("GPU readback failed.");
                            // Consume the readback in its callback: Unity's native
                            // request data has a limited lifetime. No GPU wait here.
                            output.MakeReadable();
                            readback.TrySetResult(output.ToReadOnlyArray());
                        }
                        catch (Exception error) { readback.TrySetException(error); }
                    });
                    var logits = await readback.Task;
                    cancellation.ThrowIfCancellationRequested();
                    if (_disposed) throw new OperationCanceledException("Recognizer was disposed during inference.");
                    return DecodeResult(logits, frames, classes);
                }
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception error)
            {
                // A partially failed graph must not silently score another question.
                _asyncFailure = error;
                ClearEvidence();
                throw;
            }
            finally
            {
                IsRecognizing = false;
                if (_disposed) ReleaseWorker();
            }
        }

        private static async Task NextEngineFrame()
        {
            int frame = Time.frameCount;
            bool playing = Application.isPlaying;
            // UnitySynchronizationContext keeps the continuation on Unity's thread.
            // Guard against a continuation slot later in this same Player frame.
            do { await Task.Yield(); }
            while (playing && Application.isPlaying && Time.frameCount == frame);
        }

        private void RequireAvailable()
        {
            if (Thread.CurrentThread.ManagedThreadId != _ownerThread)
                throw new InvalidOperationException("Call the recognizer on its Unity thread.");
            if (_disposed) throw new ObjectDisposedException(nameof(SentisPhonemeRecognizer));
            if (IsRecognizing) throw new InvalidOperationException("An inference already owns this worker.");
            if (_asyncFailure != null)
                throw new InvalidOperationException("Recreate the recognizer after a GPU inference failure.", _asyncFailure);
        }

        private void ClearEvidence()
        {
            _lastLogits = null;
            _lastText = "";
            _lastScorer = null;
        }

        public AcousticEvidence GetAcousticEvidence(string targetText, string contrastText)
        {
            if (_lastLogits == null || string.IsNullOrEmpty(_lastText)) return null;
            if (_lastScorer == null)
                _lastScorer = new CtcSequenceScorer(_lastLogits, _lastFrames, _lastClasses, _vocab);
            return new AcousticEvidence
            {
                TargetText = targetText, ContrastText = contrastText,
                LogProbability = _lastScorer.Score(targetText),
                GreedyLogProbability = _lastScorer.Score(_lastText),
                ContrastLogProbability = _lastScorer.Score(contrastText)
            };
        }

        private void Infer(
            float[] audio, out float[] logits, out (int frames, int classes) shape)
        {
            RequireAvailable();

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
            ClearEvidence();
            // A stopped question/scene must not release tensors still used by the
            // GPU. The independent async operation drains and releases in finally.
            if (!IsRecognizing) ReleaseWorker();
        }

        private void ReleaseWorker()
        {
            if (_workerDisposed) return;
            _worker.Dispose();
            ReleaseModelWeights(_loadedModel);
            _workerDisposed = true;
        }
    }
}
#endif
