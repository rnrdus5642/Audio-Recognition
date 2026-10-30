#if UNITY_5_3_OR_NEWER
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

namespace DomiCube.PhonemeMatching.Unity
{
    [Serializable]
    public sealed class AnswerConfirmedEvent : UnityEvent<string, float> { }

    [Serializable]
    public sealed class PronunciationConfirmedEvent : UnityEvent<PronunciationFeedback> { }

    [Serializable]
    public sealed class FrameScoredEvent : UnityEvent<string, float, int> { }

    [Serializable]
    public sealed class FrameScoredDetailedEvent : UnityEvent<FrameScore> { }

    [Serializable]
    public sealed class RecognizedTextEvent : UnityEvent<string> { }

    /// <summary>
    /// Drives the whole listening loop: microphone -> recogniser ->
    /// <see cref="StreamingMatcher"/> -> stop on confirmation.
    ///
    /// The recogniser is injected, so the loop can be exercised with a
    /// stub before any acoustic model exists.
    ///
    /// Wire <see cref="OnConfirmed"/> to whatever advances the lesson.
    /// </summary>
    public sealed class PronunciationListener : MonoBehaviour
    {
        [Header("Data (StreamingAssets 파일명)")]
        [Tooltip("confusion matrix JSON")]
        public string MatrixFileName = "ko_child_v1.json";

        [Tooltip("빌드 파이프라인이 만든 targets.json")]
        public string TargetsFileName = "targets.json";

        public bool UseOrthographicWordPolicy;
        [Range(.7f, 1f)] public float WordThreshold = .85f;
        public RecognizedTextEvent OnTextRecognized = new RecognizedTextEvent();
        public FrameScore LastFrame => LastFrameScored;

        public void ConfigureOrthographicWordPolicy(float threshold = .85f, int confirmations = 2)
        {
            if (float.IsNaN(threshold) || threshold < .7f || threshold > 1f || confirmations < 2 || confirmations > 4)
                throw new ArgumentOutOfRangeException("Invalid web v4 policy settings.");
            StopListening();
            UseOrthographicWordPolicy = true;
            WordThreshold = threshold;
            Consecutive = confirmations;
            WindowSeconds = 2.5f;
            HopSeconds = .5f;
            TimeoutSeconds = 0f;
            MatrixFileName = "ko_child_v2.json";
            _matrix = null;
            _catalog = null;
        }

        [Header("Listening")]
        [Tooltip("정답 단어. 여러 개를 넣으면 그중 하나만 맞아도 확정됩니다 "
                 + "(동의어용). 다만 오확정 확률이 대략 곱해지므로, "
                 + "필요한 만큼만 넣으세요.")]
        public string[] TargetWords = { "사과" };

        [Tooltip("ASR에 넣는 오디오 창 (초)")]
        [Range(1f, 4f)] public float WindowSeconds = 2.5f;

        [Tooltip("채점 주기 (초)")]
        [Range(0.2f, 1f)] public float HopSeconds = 0.5f;

        [Tooltip("확정에 필요한 연속 프레임 수. 1이면 무관한 발화도 통과합니다.")]
        [Range(1, 6)] public int Consecutive = 2;

        [Tooltip("이 시간이 지나면 스스로 멈춥니다 (0이면 무제한).")]
        public float TimeoutSeconds = 30f;

        [Tooltip("마이크 이름. 비우면 OS 기본 장치입니다. VR에서는 헤드셋 "
                 + "마이크를 지정해야 하는 경우가 많습니다 "
                 + "(Microphone.devices 로 이름 확인).")]
        public string MicrophoneDevice = "";

        [Header("Events")]
        public AnswerConfirmedEvent OnConfirmed = new AnswerConfirmedEvent();

        /// <summary>
        /// Detailed snapshot for pronunciation UI. Fired after capture
        /// stops, before the legacy OnConfirmed event can advance scenes.
        /// </summary>
        public PronunciationConfirmedEvent OnConfirmedDetailed =
            new PronunciationConfirmedEvent();

        /// <summary>
        /// Nothing was confirmed before <see cref="TimeoutSeconds"/> ran
        /// out. A lesson needs this to offer another try; without it the
        /// only way to notice is polling <see cref="IsListening"/>.
        /// </summary>
        public UnityEvent OnTimedOut = new UnityEvent();

        /// <summary>(단어, 점수, 연속 횟수) - 디버그 UI용.</summary>
        public FrameScoredEvent OnFrameScored = new FrameScoredEvent();

        /// <summary>Same inference, including text, rejection reason and input level.</summary>
        public FrameScoredDetailedEvent OnFrameScoredDetailed = new FrameScoredDetailedEvent();

        /// <summary>Last frame remains available after timeout; reset on Listen.</summary>
        public FrameScore LastFrameScored { get; private set; }
        private readonly List<FrameScore> _diagnosticFrames = new List<FrameScore>();
        private readonly List<string> _diagnosticTargets = new List<string>();
        public IReadOnlyList<FrameScore> RecentDiagnosticFrames => _diagnosticFrames.AsReadOnly();

        [ContextMenu("Save recognition diagnostics / 인식 진단 저장")]
        public void SaveRecognitionDiagnostics()
        {
            var directory = System.IO.Path.Combine(Application.persistentDataPath, "PhonemeMatchingDiagnostics");
            System.IO.Directory.CreateDirectory(directory);
            var path = System.IO.Path.Combine(directory, "recognition-" + DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff") + ".json");
            System.IO.File.WriteAllText(path, RecognitionDiagnostics.ToJson(_diagnosticTargets, _diagnosticFrames));
            Debug.Log("Recognition diagnostics saved: " + path, this);
        }

        private IPhonemeRecognizer _recognizer;
        private ConfusionMatrix _matrix;
        private ModelErrorProfile _feedbackErrors;
        private TargetCatalog _catalog;
        private PronunciationSession _session;
        private MicrophoneRollingBuffer _mic;
        private Coroutine _loop;

        // Bumped by every Listen/StopListening. A running loop carries the
        // value it started with, so a handler that stops or restarts the
        // component from inside a frame event cannot have the old loop
        // keep scoring, firing events or cleaning up after it.
        private int _generation;

        public bool IsListening => _loop != null;

        /// <summary>Last confirmed feedback; cleared on the next Listen attempt.</summary>
        public PronunciationFeedback LastConfirmation { get; private set; }

        /// <summary>
        /// The judging session, live only while listening. Use
        /// <see cref="PronunciationSession"/> directly instead of this
        /// component when the application owns its own audio capture.
        /// </summary>
        public PronunciationSession Session => _session;

        /// <summary>
        /// Supply the acoustic model. Call before <see cref="Listen"/>;
        /// without it the component does nothing.
        /// </summary>
        public void SetRecognizer(IPhonemeRecognizer recognizer)
        {
            _recognizer = recognizer;
        }

        /// <summary>
        /// Load matrix and targets from StreamingAssets. Separate from
        /// Awake so a caller can inject data another way instead.
        /// </summary>
        public void LoadData()
        {
            LoadData(ReadStreamingAsset(MatrixFileName), UseOrthographicWordPolicy ? null : ReadStreamingAsset(TargetsFileName));
        }

        public void LoadData(string matrixJson, string targetsJson, ModelErrorProfile feedbackErrors = null)
        {
            _matrix = PhonemeData.LoadMatrix(matrixJson, UseOrthographicWordPolicy ? "ko_child_v2" : null);
            _catalog = UseOrthographicWordPolicy ? null : PhonemeData.LoadTargets(targetsJson);
            _feedbackErrors = UseOrthographicWordPolicy ? null : feedbackErrors ?? ModelFeedbackResources.Load(_matrix.MatrixId);
        }

        private static string ReadStreamingAsset(string fileName)
        {
            var path = System.IO.Path.Combine(
                Application.streamingAssetsPath, fileName);
            if (!System.IO.File.Exists(path))
            {
                throw new System.IO.FileNotFoundException(
                    $"'{fileName}' 없음. shared/ 의 JSON을 "
                    + "Assets/StreamingAssets/ 로 복사하세요.", path);
            }

            return System.IO.File.ReadAllText(path);
        }

        /// <summary>Start listening for the configured target.</summary>
        public void Listen()
        {
            // Every attempt retires the one before it, whether or not
            // this one gets off the ground. A handler that calls Listen()
            // from inside a confirmation and is turned away at validation
            // must still leave the old attempt unable to fire the rest of
            // its events into the question it just tried to start.
            _generation++;
            LastConfirmation = null;
            LastFrameScored = null;
            _diagnosticFrames.Clear();
            _diagnosticTargets.Clear();
            if (_loop != null)
            {
                StopListening();
            }

            if (_recognizer == null)
            {
                Debug.LogError(
                    "[PronunciationListener] recognizer가 없습니다. "
                    + "SetRecognizer()를 먼저 호출하세요.");
                return;
            }

            if (_matrix == null || (!UseOrthographicWordPolicy && _catalog == null))
            {
                LoadData();
            }

            var words = new List<string>();
            if (TargetWords != null)
            {
                foreach (var w in TargetWords)
                {
                    if (!string.IsNullOrWhiteSpace(w))
                    {
                        words.Add(w.Trim());
                    }
                }
            }

            if (words.Count == 0)
            {
                Debug.LogError(
                    "[PronunciationListener] TargetWords가 비어 있습니다.");
                return;
            }
            _diagnosticTargets.AddRange(words);

            // The buffer is built first because the session drains it. Both
            // freeze the window and hop for this question: changing the
            // Inspector mid-answer must not move the frame boundaries.
            _mic = new MicrophoneRollingBuffer(
                WindowSeconds, hopSeconds: HopSeconds);
            try
            {
                if (UseOrthographicWordPolicy && (Math.Abs(WindowSeconds - 2.5f) > .001f || Math.Abs(HopSeconds - .5f) > .001f))
                    throw new ArgumentException("Web v4 requires a 2.5s window and 0.5s hop.");
                _session = UseOrthographicWordPolicy
                    ? PronunciationSession.ForOrthographicWord(_matrix, _recognizer, Math.Round(WordThreshold, 6), Consecutive, _mic.Buffer)
                    : new PronunciationSession(_matrix, _catalog, _recognizer, Consecutive, _mic.Buffer, _feedbackErrors);
                _session.Begin(words);
            }
            catch (ArgumentException e)
            {
                // An unknown word is an authoring mistake, but throwing
                // out of a UI callback would take the caller down with
                // it. Say it and stay put.
                Debug.LogError($"[PronunciationListener] {e.Message}");
                _session = null;
                _mic.Dispose();
                _mic = null;
                return;
            }

            try
            {
                _mic.Start(string.IsNullOrEmpty(MicrophoneDevice)
                    ? null
                    : MicrophoneDevice);
            }
            catch
            {
                // Do not leave the device or the session half-open when
                // the microphone refuses to start.
                _mic.Dispose();
                _mic = null;
                _session.End();
                _session = null;
                throw;
            }

            _loop = StartCoroutine(ListenLoop(_generation));
        }

        public void StopListening()
        {
            // Any loop still on the stack belongs to the previous attempt.
            _generation++;

            if (_loop != null)
            {
                StopCoroutine(_loop);
                _loop = null;
            }

            _session?.End();

            // Releases the device: nothing is captured between questions.
            _mic?.Dispose();
            _mic = null;
        }

        /// <summary>What one attempt at scoring a queued frame did.</summary>
        private enum Step
        {
            /// <summary>No whole hop was waiting; nothing ran.</summary>
            Idle,

            /// <summary>A frame was scored and reported; look for more.</summary>
            Scored,

            /// <summary>The answer is in and the confirmation has fired.</summary>
            Confirmed,

            /// <summary>A handler stopped or restarted us mid-drain.</summary>
            Stopped,

            /// <summary>Scoring threw; the loop is over.</summary>
            Failed
        }

        /// <summary>
        /// Score at most one queued frame, report it, and act on it.
        ///
        /// Deliberately one frame per call. The recogniser must not run
        /// again until the last frame has been through
        /// <see cref="OnFrameScored"/>, because a handler is allowed to
        /// stop the question there - and audio that arrived in the same
        /// pump would otherwise already have been scored, and could
        /// already have confirmed, before it got the chance.
        /// </summary>
        private Step ScoreOneFrame(int generation)
        {
            FrameScore frame;
            try
            {
                if (!_session.TryPush(out frame))
                {
                    return Step.Idle;
                }
            }
            catch (Exception e)
            {
                // An overflowing frame queue invalidates the session; it
                // must not take the scene down with it either.
                Debug.LogException(e);
                return Step.Failed;
            }

            LastFrameScored = frame;
            if (_diagnosticFrames.Count >= 120) _diagnosticFrames.RemoveAt(0);
            _diagnosticFrames.Add(frame);
            OnTextRecognized.Invoke(frame.Text);
            if (generation != _generation) return Step.Stopped;
            OnFrameScored.Invoke(
                frame.Best.TargetText ?? string.Empty,
                (float)frame.Best.Score,
                frame.Streak);

            if (generation != _generation)
            {
                return Step.Stopped;
            }

            OnFrameScoredDetailed.Invoke(frame);
            if (generation != _generation) return Step.Stopped;

            if (frame.Confirmed)
            {
                // IsActive is already false here - that is what
                // confirming means, not a reason to drop the event.
                Confirm(frame.Feedback);
                return Step.Confirmed;
            }

            return Step.Scored;
        }

        /// <summary>
        /// Poll the microphone every frame and score whole hops of new
        /// audio, in order.
        ///
        /// The loop used to wait one hop and score whatever was in the
        /// window, which made a frame "one coroutine tick" - a stalled
        /// frame or a device that delivers in bursts changed how many
        /// windows a child was judged on, and so changed when the answer
        /// confirmed. Now polling only decides how promptly a frame is
        /// noticed; <see cref="HopSeconds"/> of captured samples is what
        /// makes one.
        /// </summary>
        private IEnumerator ListenLoop(int generation)
        {
            float started = Time.realtimeSinceStartup;

            // Distinguishes "gave up" from "the recogniser threw", which
            // both end the loop but mean different things to the caller.
            bool timedOut = false;
            bool failed = false;

            while (true)
            {
                yield return null;

                if (generation != _generation)
                {
                    // Stopped or restarted from a callback; whoever did
                    // that owns the cleanup and the new session.
                    yield break;
                }

                bool pumped;
                try
                {
                    pumped = _mic != null && _mic.Pump();
                }
                catch (Exception e)
                {
                    // A dead microphone must not take the scene with it.
                    Debug.LogException(e);
                    pumped = false;
                    failed = true;
                }

                while (!failed && pumped)
                {
                    // One frame at a time, all the way through: score it,
                    // tell the listeners, and only then look for the next
                    // one. Scoring the whole queue first would run the
                    // recogniser on audio that arrived after a handler
                    // had already stopped us - and could confirm on it.
                    var step = ScoreOneFrame(generation);
                    if (step == Step.Idle)
                    {
                        break;
                    }

                    if (step == Step.Failed)
                    {
                        failed = true;
                        break;
                    }

                    if (step != Step.Scored)
                    {
                        // Confirmed, or a handler stopped or restarted
                        // us. Either way this loop is finished and the
                        // cleanup belongs to whoever did it.
                        yield break;
                    }
                }

                if (failed)
                {
                    break;
                }

                // Checked on every poll, so a silent room still times out.
                if (TimedOut(started))
                {
                    timedOut = true;
                    break;
                }
            }

            StopListening();

            if (timedOut)
            {
                OnTimedOut.Invoke();
            }
        }

        private void Confirm(PronunciationFeedback feedback)
        {
            // Stop capture before UI callbacks can advance scenes.
            StopListening();

            // Read after the stop, so this names the attempt that just
            // confirmed rather than one a handler goes on to start.
            int generation = _generation;
            LastConfirmation = feedback;
            OnConfirmedDetailed?.Invoke(feedback);

            if (generation != _generation)
            {
                // The detailed handler began or stopped another attempt.
                // Delivering the old answer to the legacy event now would
                // advance a question that has nothing to do with it.
                return;
            }

            OnConfirmed.Invoke(feedback.TargetText, (float)feedback.Score);
        }

        private bool TimedOut(float started)
        {
            return TimeoutSeconds > 0f
                && Time.realtimeSinceStartup - started > TimeoutSeconds;
        }

        private void OnDisable()
        {
            StopListening();
        }
    }
}
#endif
