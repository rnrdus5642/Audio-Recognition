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

        [Tooltip("원문 자모 모델에 웹사이트 v4 후처리를 사용합니다. 이 모드에서는 targets.json이 필요 없습니다.")]
        public bool UseOrthographicWordPolicy;

        [Range(.7f, 1f)] public float WordThreshold = .85f;

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

        /// <summary>Raw ASR text; never replaced by the selected answer.</summary>
        public RecognizedTextEvent OnTextRecognized = new RecognizedTextEvent();

        public FrameScore LastFrame { get; private set; }

        private IPhonemeRecognizer _recognizer;
        private ConfusionMatrix _matrix;
        private TargetCatalog _catalog;
        private PronunciationSession _session;
        private MicrophoneRollingBuffer _mic;
        private Coroutine _loop;

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

        /// <summary>Opt in to the current web policy and its capture defaults.
        /// Call before Listen. Existing scenes retain their legacy configuration.</summary>
        public void ConfigureOrthographicWordPolicy(float threshold = .85f, int confirmations = 2)
        {
            if (float.IsNaN(threshold) || threshold < .7f || threshold > 1f || confirmations < 2 || confirmations > 4)
                throw new ArgumentOutOfRangeException("Invalid web word policy settings.");
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

        /// <summary>
        /// Load matrix and targets from StreamingAssets. Separate from
        /// Awake so a caller can inject data another way instead.
        /// </summary>
        public void LoadData()
        {
            _matrix = PhonemeData.LoadMatrix(
                ReadStreamingAsset(MatrixFileName));
            _catalog = UseOrthographicWordPolicy ? null : PhonemeData.LoadTargets(
                ReadStreamingAsset(TargetsFileName));
        }

        public void LoadData(string matrixJson, string targetsJson)
        {
            _matrix = PhonemeData.LoadMatrix(matrixJson);
            _catalog = UseOrthographicWordPolicy ? null : PhonemeData.LoadTargets(targetsJson);
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
            LastConfirmation = null;
            LastFrame = null;
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
                _session = null;
                return;
            }

            try
            {
                if (UseOrthographicWordPolicy && (Math.Abs(WindowSeconds - 2.5f) > .001f || Math.Abs(HopSeconds - .5f) > .001f))
                    throw new ArgumentException("웹 v4 판정은 WindowSeconds=2.5, HopSeconds=0.5가 필요합니다.");
                // Round serialized float settings so 0.85f does not become
                // 0.8500000238 and reject the web's exact 0.85 boundary.
                _session = UseOrthographicWordPolicy
                    ? PronunciationSession.ForOrthographicWord(_matrix, _recognizer, Math.Round(WordThreshold, 6), Consecutive)
                    : new PronunciationSession(_matrix, _catalog, _recognizer, Consecutive);
                _session.Begin(words);
            }
            catch (ArgumentException e)
            {
                // An unknown word is an authoring mistake, but throwing
                // out of a UI callback would take the caller down with
                // it. Say it and stay put.
                Debug.LogError($"[PronunciationListener] {e.Message}");
                _session = null;
                return;
            }

            _mic = new MicrophoneRollingBuffer(WindowSeconds);
            _mic.Start(string.IsNullOrEmpty(MicrophoneDevice)
                ? null
                : MicrophoneDevice);
            _loop = StartCoroutine(ListenLoop());
        }

        public void StopListening()
        {
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

        private IEnumerator ListenLoop()
        {
            object wait = UseOrthographicWordPolicy
                ? (object)new WaitForSecondsRealtime(HopSeconds) : new WaitForSeconds(HopSeconds);
            float started = Time.realtimeSinceStartup;

            // Distinguishes "gave up" from "the recogniser threw", which
            // both end the loop but mean different things to the caller.
            bool timedOut = false;

            while (true)
            {
                yield return wait;

                if (_mic == null || !_mic.Pump())
                {
                    if (TimedOut(started))
                    {
                        timedOut = true;
                        break;
                    }

                    continue;
                }

                // Wait until the microphone has captured a complete new hop.
                // Some devices begin producing samples after the first tick.
                if (_session.UsesOrthographicWordPolicy &&
                    _mic.Elapsed - (_session.LastFrame?.WordDecision?.ElapsedSeconds ?? 0) < OrthographicWordPolicy.HopSeconds)
                    continue;

                FrameScore frame;
                var scoringSession = _session;
                try
                {
                    frame = scoringSession.Push(_mic.Snapshot(), _mic.Elapsed);
                }
                catch (Exception e)
                {
                    Debug.LogException(e);
                    break;
                }

                LastFrame = frame;
                OnTextRecognized?.Invoke(frame.Text);
                if (_mic == null || !ReferenceEquals(scoringSession, _session)) yield break;
                OnFrameScored.Invoke(
                    frame.Best.TargetText ?? string.Empty,
                    (float)frame.Best.Score,
                    frame.Streak);
                if (_mic == null || !ReferenceEquals(scoringSession, _session)) yield break;

                if (frame.Confirmed)
                {
                    Confirm(frame.Feedback);
                    yield break;
                }

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
            LastConfirmation = feedback;
            OnConfirmedDetailed?.Invoke(feedback);
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
