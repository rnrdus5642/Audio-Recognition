#if UNITY_5_3_OR_NEWER
using System;
using UnityEngine;
#if DOMICUBE_INFERENCE_ENGINE
using Unity.InferenceEngine;
#elif DOMICUBE_SENTIS_1
using Unity.Sentis;
#endif

namespace DomiCube.PhonemeMatching.Unity
{
    /// <summary>
    /// Ready-to-use lesson: packaged policies, child model loaded by path,
    /// nine stage choices and the existing listener's confirmation events.
    /// Microphone capture starts only when ListenStage is called.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(PronunciationListener))]
    [AddComponentMenu("Phoneme Matching/악기놀이 발표")]
    public sealed class MusicPresentationLesson : MonoBehaviour
    {
        [Range(1, 9)] public int Stage = 1;
        public bool ShowControls = true;
        public string Status { get; private set; } = "단계를 선택하고 준비를 누르세요";
        public PronunciationListener Listener => GetComponent<PronunciationListener>();
        public bool IsPrepared => _recognizer != null;
        public PronunciationFeedback LastFeedback { get; private set; }

        private IPhonemeRecognizer _recognizer;
        private MusicPresentationProfile _profile;

        private void Awake()
        {
            Listener.OnConfirmed.AddListener(Confirmed);
            Listener.OnConfirmedDetailed.AddListener(ShowFeedback);
            Listener.OnTimedOut.AddListener(TimedOut);
        }

        public bool Prepare()
        {
            if (IsPrepared) return true;
            try
            {
                _profile = MusicPresentation.Load();
                _profile.GetStage(Stage);
#if DOMICUBE_INFERENCE_ENGINE || DOMICUBE_SENTIS_1
                var model = Resources.Load<ModelAsset>(MusicPresentationProfile.ModelResourcePath);
                if (model == null)
                    throw new InvalidOperationException(
                        "아동 모델이 없습니다. Tools > Phoneme Matching > 악기놀이 발표 설정을 실행하세요.");
                var recognizer = new SentisPhonemeRecognizer(model, _profile.Vocabulary);
                try { recognizer.Warmup(MusicPresentationProfile.WindowSeconds); }
                catch { recognizer.Dispose(); throw; }
                _recognizer = recognizer;
                Listener.SetRecognizer(_recognizer);
                MusicPresentation.ApplyTo(Listener, Stage);
                Status = "준비 완료 · 시작을 누르세요";
                return true;
#else
                throw new InvalidOperationException(
                    "추론 엔진이 없습니다. Tools > Phoneme Matching > 악기놀이 발표 설정을 실행하세요.");
#endif
            }
            catch (Exception e)
            {
                (_recognizer as IDisposable)?.Dispose();
                _recognizer = null;
                Listener.SetRecognizer(null);
                Status = e.Message;
                Debug.LogError("[MusicPresentation] " + Status);
                return false;
            }
        }

        /// <summary>Use this from an existing lesson's stage-change callback.</summary>
        public void ListenStage(int stage)
        {
            var profile = _profile ?? MusicPresentation.Load();
            var selected = profile.GetStage(stage);
            Stage = stage;
            LastFeedback = null;
            if (!Prepare()) return;
            MusicPresentation.ApplyTo(Listener, stage);
            Listener.Listen();
            Status = Listener.IsListening
                ? selected.Title + " · 듣고 있어요"
                : "마이크를 시작하지 못했습니다. Console을 확인하세요.";
        }

        public void Stop()
        {
            Listener.StopListening();
            Status = "중지됨";
        }

        private void Confirmed(string word, float score) => Status = "정답: " + word;
        private void ShowFeedback(PronunciationFeedback feedback) => LastFeedback = feedback;
        private void TimedOut() => Status = "다시 말해볼까요?";
        private void OnDisable()
        {
            if (Listener != null) Listener.StopListening();
        }

        private void OnDestroy()
        {
            var listener = Listener;
            if (listener != null)
            {
                listener.StopListening();
                listener.OnConfirmed.RemoveListener(Confirmed);
                listener.OnConfirmedDetailed.RemoveListener(ShowFeedback);
                listener.OnTimedOut.RemoveListener(TimedOut);
                listener.SetRecognizer(null);
            }
            (_recognizer as IDisposable)?.Dispose();
            _recognizer = null;
        }

        private void OnGUI()
        {
            if (!ShowControls) return;
            GUILayout.BeginArea(new Rect(16, 16, 460, 440), GUI.skin.box);
            GUILayout.Label("악기놀이");
            GUILayout.Label(Status);
            if (LastFeedback != null)
            {
                GUILayout.Label(LastFeedback.KoreanFeedback.Summary);
                foreach (var hint in LastFeedback.KoreanFeedback.Items)
                    GUILayout.Label(hint.Message);
            }
            _profile = _profile ?? MusicPresentation.Load();
            string[] titles = new string[_profile.Stages.Count];
            for (int i = 0; i < titles.Length; i++)
                titles[i] = $"{i + 1}. {_profile.Stages[i].Title}";
            int next = GUILayout.SelectionGrid(Mathf.Clamp(Stage - 1, 0, 8), titles, 2) + 1;
            if (next != Stage)
            {
                Stop();
                Stage = next;
                LastFeedback = null;
                Status = _profile.GetStage(Stage).Title;
            }
            GUILayout.Space(10);
            if (!IsPrepared && GUILayout.Button("준비")) Prepare();
            if (IsPrepared && !Listener.IsListening && GUILayout.Button("시작 (마이크 켜기)"))
                ListenStage(Stage);
            if (Listener.IsListening && GUILayout.Button("중지")) Stop();
            GUILayout.EndArea();
        }
    }
}
#endif
