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
    public sealed class FrameScoredEvent : UnityEvent<string, float, int> { }

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

        [Header("Listening")]
        [Tooltip("정답 단어. 비우면 세그먼트 전체를 후보로 삼습니다.")]
        public string TargetText = "사과";

        [Tooltip("ASR에 넣는 오디오 창 (초)")]
        [Range(1f, 4f)] public float WindowSeconds = 2.5f;

        [Tooltip("채점 주기 (초)")]
        [Range(0.2f, 1f)] public float HopSeconds = 0.5f;

        [Tooltip("확정에 필요한 연속 프레임 수. 1이면 무관한 발화도 통과합니다.")]
        [Range(1, 6)] public int Consecutive = 3;

        [Tooltip("이 시간이 지나면 스스로 멈춥니다 (0이면 무제한).")]
        public float TimeoutSeconds = 30f;

        [Tooltip("마이크 이름. 비우면 OS 기본 장치입니다. VR에서는 헤드셋 "
                 + "마이크를 지정해야 하는 경우가 많습니다 "
                 + "(Microphone.devices 로 이름 확인).")]
        public string MicrophoneDevice = "";

        [Header("Events")]
        public AnswerConfirmedEvent OnConfirmed = new AnswerConfirmedEvent();

        /// <summary>
        /// Nothing was confirmed before <see cref="TimeoutSeconds"/> ran
        /// out. A lesson needs this to offer another try; without it the
        /// only way to notice is polling <see cref="IsListening"/>.
        /// </summary>
        public UnityEvent OnTimedOut = new UnityEvent();

        /// <summary>(단어, 점수, 연속 횟수) - 디버그 UI용.</summary>
        public FrameScoredEvent OnFrameScored = new FrameScoredEvent();

        private IPhonemeRecognizer _recognizer;
        private ConfusionMatrix _matrix;
        private TargetCatalog _catalog;
        private StreamingMatcher _streaming;
        private MicrophoneRollingBuffer _mic;
        private Coroutine _loop;

        public bool IsListening => _loop != null;

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
            _matrix = PhonemeData.LoadMatrix(
                ReadStreamingAsset(MatrixFileName));
            _catalog = PhonemeData.LoadTargets(
                ReadStreamingAsset(TargetsFileName));
        }

        public void LoadData(string matrixJson, string targetsJson)
        {
            _matrix = PhonemeData.LoadMatrix(matrixJson);
            _catalog = PhonemeData.LoadTargets(targetsJson);
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

            if (_matrix == null || _catalog == null)
            {
                LoadData();
            }

            var candidates = ResolveCandidates();
            if (candidates == null || candidates.Count == 0)
            {
                Debug.LogError(
                    $"[PronunciationListener] '{TargetText}' 를 "
                    + "targets.json 에서 찾지 못했습니다.");
                return;
            }

            _streaming = new StreamingMatcher(
                Matcher.ForStreaming(_matrix), candidates, Consecutive);
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

            _mic?.Dispose();
            _mic = null;
        }

        private List<Answer> ResolveCandidates()
        {
            if (string.IsNullOrEmpty(TargetText))
            {
                var all = new List<Answer>();
                foreach (var seg in _catalog.Segments)
                {
                    all.AddRange(seg.Answers);
                }

                return all;
            }

            // The whole segment competes, as it will in the lesson.
            return _catalog.SegmentOf(TargetText);
        }

        private IEnumerator ListenLoop()
        {
            var wait = new WaitForSeconds(HopSeconds);
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

                var window = _mic.Snapshot();
                List<string> phonemes;
                try
                {
                    phonemes = _recognizer.Recognize(window);
                }
                catch (Exception e)
                {
                    Debug.LogException(e);
                    break;
                }

                var hit = _streaming.Push(phonemes);
                ReportFrame(phonemes);

                if (hit != null)
                {
                    // This is the moment the lesson moves on.
                    StopListening();
                    OnConfirmed.Invoke(
                        hit.Result.TargetText, (float)hit.Result.Score);
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

        private bool TimedOut(float started)
        {
            return TimeoutSeconds > 0f
                && Time.realtimeSinceStartup - started > TimeoutSeconds;
        }

        private void ReportFrame(List<string> phonemes)
        {
            if (OnFrameScored == null)
            {
                return;
            }

            var best = _streaming.Matcher.BestMatch(
                phonemes, _streaming.Candidates);
            OnFrameScored.Invoke(
                best.TargetText ?? string.Empty,
                (float)best.Score,
                _streaming.Streak);
        }

        private void OnDisable()
        {
            StopListening();
        }
    }
}
#endif
