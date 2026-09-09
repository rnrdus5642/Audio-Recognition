using System.Collections.Generic;
using System.Linq;
using DomiCube.PhonemeMatching.Unity;
using UnityEditor;
using UnityEngine;

#if DOMICUBE_INFERENCE_ENGINE
using Unity.InferenceEngine;
#elif DOMICUBE_SENTIS_1
using Unity.Sentis;
#endif

namespace DomiCube.PhonemeMatching.Editor
{
    /// <summary>
    /// Speak into the microphone and see whether the word is confirmed.
    ///
    /// Exercises the whole chain - capture, rolling window, recognition,
    /// matching, streak, stop - which no fixture can cover, and does it
    /// the same way on Unity 2022 and Unity 6 so a project can be
    /// checked without building a scene first.
    ///
    /// Ordinary profiles use StreamingAssets; the music profile uses
    /// package Resources. Both need the corresponding imported .onnx model.
    ///
    /// Tools > Phoneme Matching > 발음 테스트 (마이크)
    /// </summary>
    public sealed class PronunciationTestWindow : EditorWindow
    {
        private const string ModelKey = "DomiCube.PhonemeMatching.TestModel";
        private const string DataChoiceKey = "DomiCube.PhonemeMatching.TestData.";
        private const double Hop = 0.5;
        private static readonly string[] DataChoices =
            { "선택하세요", "기본 모델용", "아동 파인튜닝 모델용", "악기놀이 발표용 (12단어)" };

        private string _word = "사과";
        private string _status = "시작을 누르세요";
        private float _timeout = 20f;
        private readonly List<string> _frames = new List<string>();
        private Vector2 _scroll;
        private int _deviceIndex;

#if DOMICUBE_INFERENCE_ENGINE || DOMICUBE_SENTIS_1
        private ModelAsset _model;
        private ModelAsset _preparedModel;
        private TestDataChoice _dataChoice;
        private PronunciationTestSettings _preparedSettings;
        private SentisPhonemeRecognizer _recognizer;
        private PronunciationSession _session;
        private MicrophoneRollingBuffer _mic;
        private ConfusionMatrix _matrix;
        private TargetCatalog _catalog;
        private MusicPresentationProfile _musicProfile;
        private int _musicStage = 1;
        private double _started;
#endif

        [MenuItem("Tools/Phoneme Matching/발음 테스트 (마이크)", false, 40)]
        public static void Open()
        {
            GetWindow<PronunciationTestWindow>("발음 테스트").minSize =
                new Vector2(420, 420);
        }

        public static void OpenMusicDemo()
        {
            Open();
#if DOMICUBE_INFERENCE_ENGINE || DOMICUBE_SENTIS_1
            ConfigureMusicDemo();
#endif
        }

#if !DOMICUBE_INFERENCE_ENGINE && !DOMICUBE_SENTIS_1
        private void OnGUI()
        {
            EditorGUILayout.HelpBox(
                "추론 엔진 패키지가 없습니다.\n"
                + "Unity 2022: com.unity.sentis 1.2.0-exp.2\n"
                + "Unity 6: com.unity.ai.inference 2.6.1\n"
                + "manifest.json 에 추가하세요.",
                MessageType.Warning);
        }
#else
        private void OnDisable()
        {
            ClearPrepared();
        }

        private static void ConfigureMusicDemo()
        {
            var window = GetWindow<PronunciationTestWindow>();
            window.ClearPrepared();
            window._musicProfile = MusicPresentation.Load();
            window._model = Resources.Load<ModelAsset>(MusicPresentationProfile.ModelResourcePath);
            window._dataChoice = TestDataChoice.MusicDemo;
            window._musicStage = 1;
            string guid = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(window._model));
            EditorPrefs.SetString(ModelKey, guid);
            if (!string.IsNullOrEmpty(guid))
                EditorPrefs.SetInt(DataChoiceKey + guid, (int)TestDataChoice.MusicDemo);
            window._status = "악기놀이 · 단계를 선택하고 시작을 누르세요";
            window.Repaint();
        }

        private void ClearPrepared()
        {
            Stop();
            _recognizer?.Dispose();
            _recognizer = null;
            _session = null;
            _matrix = null;
            _catalog = null;
            _preparedModel = null;
            _preparedSettings = null;
            _frames.Clear();
        }

        private bool Prepare()
        {
            if (_model == null)
            {
                ClearPrepared();
                _status = "모델(.onnx)을 지정하세요";
                return false;
            }

            var settings = PronunciationTestSettings.Resolve(
                AssetDatabase.GetAssetPath(_model), _dataChoice);
            if (settings == null)
            {
                ClearPrepared();
                _status = "모델 종류를 자동으로 알 수 없습니다. 채점 설정을 선택하세요.";
                return false;
            }
            if (_recognizer != null && _catalog != null
                && _preparedModel == _model && _preparedSettings == settings)
                return true;

            ClearPrepared();
            try
            {
                settings.Load(Application.streamingAssetsPath,
                    out var matrix, out var catalog, out var vocab);
                _recognizer = new SentisPhonemeRecognizer(_model, vocab);

                _status = "모델 준비 중... (첫 실행 ~2초)";
                Repaint();
                _recognizer.Warmup(2.5f);

                // The session is built per attempt in Listen(), because it
                // drains the microphone buffer and that buffer only exists
                // while a device is open.
                _matrix = matrix;
                _catalog = catalog;
                _preparedModel = _model;
                _preparedSettings = settings;
                return true;
            }
            catch (System.Exception e)
            {
                // Usually the StreamingAssets JSON is missing.
                _status = e.Message;
                ClearPrepared();
                return false;
            }
        }

        private void Listen()
        {
            if (!Prepare())
            {
                return;
            }

            var devices = Microphone.devices;
            if (devices.Length == 0)
            {
                _status = "마이크가 없습니다";
                return;
            }

            _frames.Clear();
            _mic = new MicrophoneRollingBuffer(2.5f, hopSeconds: (float)Hop);
            _session = new PronunciationSession(
                _matrix, _catalog, _recognizer, 2, _mic.Buffer, ModelFeedbackResources.Load(_matrix.MatrixId));

            try
            {
                var words = _preparedSettings == PronunciationTestSettings.MusicDemo && _musicStage > 0
                    ? (_musicProfile ?? (_musicProfile = MusicPresentation.Load())).GetStage(_musicStage).Answers.ToArray()
                    : new[] { _word };
                _session.Begin(words);
            }
            catch (System.Exception e)
            {
                _status = e.Message;
                _mic.Dispose();
                _mic = null;
                _session = null;
                return;
            }

            try
            {
                _mic.Start(devices[_deviceIndex % devices.Length]);
            }
            catch (System.Exception e)
            {
                _status = e.Message;
                Stop();
                return;
            }

            _started = EditorApplication.timeSinceStartup;
            _status = _preparedSettings == PronunciationTestSettings.MusicDemo && _musicStage > 0
                ? _musicProfile.GetStage(_musicStage).Title + " · 듣는 중"
                : $"듣는 중 — '{_word}' 라고 말해보세요";

            EditorApplication.update -= Tick;
            EditorApplication.update += Tick;
        }

        private void Stop()
        {
            EditorApplication.update -= Tick;
            _session?.End();
            _mic?.Dispose();
            _mic = null;
        }

        private void Tick()
        {
            if (_mic == null || _session == null || !_session.IsActive)
            {
                return;
            }

            // Editor ticks are irregular, so score captured hops rather
            // than "whatever is in the window when the editor got round
            // to us" - the frame numbers then mean the same thing here as
            // they do in the game and in the Python tool.
            bool confirmed = false;
            try
            {
                _mic.Pump();

                FrameScore frame;
                while (_session.TryPush(out frame))
                {
                    double at = frame.EndSampleIndex
                        / (double)AudioWindowBuffer.TargetSampleRate;
                    _frames.Insert(0, string.Format(
                        "{0,5:F1}s  {1}/{5}  {2,-6} {3:F3}  '{4}'",
                        at, frame.Streak,
                        string.IsNullOrEmpty(frame.Best.TargetText)
                            ? "—" : frame.Best.TargetText,
                        frame.Best.Score, frame.Text, frame.RequiredFrames));

                    if (frame.Confirmed)
                    {
                        _status = $"확정: {frame.Best.TargetText} "
                            + $"({frame.Best.Score:F3}, {at:F1}초)";
                        confirmed = true;
                        break;
                    }
                }
            }
            catch (System.Exception e)
            {
                _status = "실패: " + e.Message;
                Stop();
                Repaint();
                return;
            }

            double now = EditorApplication.timeSinceStartup;
            if (confirmed)
            {
                Stop();
            }
            else if (_timeout > 0f && now - _started > _timeout)
            {
                _status = "시간 초과 — 확정되지 않았습니다";
                Stop();
            }

            Repaint();
        }

        private void OnEnable()
        {
            string guid = EditorPrefs.GetString(ModelKey, "");
            if (!string.IsNullOrEmpty(guid))
            {
                _model = AssetDatabase.LoadAssetAtPath<ModelAsset>(
                    AssetDatabase.GUIDToAssetPath(guid));
            }
            RestoreDataChoice(guid);
        }

        private void RestoreDataChoice(string guid)
        {
            int choice = string.IsNullOrEmpty(guid) ? 0
                : EditorPrefs.GetInt(DataChoiceKey + guid, 0);
            _dataChoice = choice >= 0 && choice < DataChoices.Length
                ? (TestDataChoice)choice : TestDataChoice.Unspecified;
        }

        private void OnGUI()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField(_status, EditorStyles.boldLabel);
            if (_session?.LastConfirmation != null)
            {
                var feedback = _session.LastConfirmation;
                EditorGUILayout.HelpBox(feedback.TargetText + " · " + feedback.KoreanFeedback.Summary, MessageType.Info);
                foreach (var hint in feedback.KoreanFeedback.Items)
                    EditorGUILayout.LabelField(hint.Message, EditorStyles.wordWrappedLabel);
            }
            EditorGUILayout.Space();

            using (new EditorGUI.DisabledScope(_mic != null))
            {
                var picked = (ModelAsset)EditorGUILayout.ObjectField(
                    "모델 (.onnx)", _model, typeof(ModelAsset), false);
                if (picked != _model)
                {
                    ClearPrepared();
                    _model = picked;
                    string guid = picked == null
                        ? ""
                        : AssetDatabase.AssetPathToGUID(
                            AssetDatabase.GetAssetPath(picked));
                    EditorPrefs.SetString(ModelKey, guid);
                    RestoreDataChoice(guid);
                    _status = "모델이 변경되었습니다. 시작을 누르세요";
                }

                string modelPath = AssetDatabase.GetAssetPath(_model);
                if (_model != null)
                {
                    var choice = (TestDataChoice)EditorGUILayout.Popup(
                        "채점 설정 (직접 지정)", (int)_dataChoice, DataChoices);
                    if (choice != _dataChoice)
                    {
                        ClearPrepared();
                        _dataChoice = choice;
                        EditorPrefs.SetInt(DataChoiceKey + AssetDatabase.AssetPathToGUID(modelPath),
                            (int)choice);
                        _status = "채점 설정이 변경되었습니다. 시작을 누르세요";
                    }
                }
                var settings = _model == null ? null
                    : PronunciationTestSettings.Resolve(modelPath, _dataChoice);
                if (settings != null)
                {
                    EditorGUILayout.HelpBox(
                        $"채점 설정: {settings.Label}\n"
                        + $"{settings.MatrixFile} + {settings.TargetsFile}\n"
                        + $"공통 어휘: {PronunciationTestSettings.VocabularyFile}"
                        + (settings == PronunciationTestSettings.MusicDemo
                            ? "\n패키지에 포함된 발표 설정이 적용됩니다." : ""), MessageType.Info);
                }
                else if (_model != null)
                {
                    EditorGUILayout.HelpBox(
                        "이름을 바꾼 모델이나 별도 모델은 종류를 자동 판별할 수 없습니다.\n"
                        + "위에서 모델에 맞는 채점 설정을 선택하세요.", MessageType.Warning);
                }

                if (settings == PronunciationTestSettings.MusicDemo)
                {
                    _musicProfile = _musicProfile ?? MusicPresentation.Load();
                    var stages = new[] { "직접 단어 입력" }.Concat(
                        _musicProfile.Stages.Select(s => $"{s.Id}. {s.Title}")).ToArray();
                    _musicStage = EditorGUILayout.Popup("악기놀이 단계", _musicStage, stages);
                    if (_musicStage > 0)
                        EditorGUILayout.LabelField("정답 표현",
                            string.Join(" / ", _musicProfile.GetStage(_musicStage).Answers));
                }
                if (settings != PronunciationTestSettings.MusicDemo || _musicStage == 0)
                    _word = EditorGUILayout.TextField("정답 단어", _word);

                var devices = Microphone.devices;
                if (devices.Length > 0)
                {
                    _deviceIndex = EditorGUILayout.Popup(
                        "마이크", _deviceIndex % devices.Length, devices);
                }

                _timeout = EditorGUILayout.Slider("타임아웃 (초)", _timeout, 5f, 60f);
            }

            EditorGUILayout.Space();
            if (_mic == null)
            {
                if (GUILayout.Button("시작 (마이크 켜기)", GUILayout.Height(30)))
                {
                    Listen();
                }
            }
            else if (GUILayout.Button("중지", GUILayout.Height(30)))
            {
                _status = "중지됨";
                Stop();
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField(
                "시각   연속  선택후보 점수   ASR", EditorStyles.miniLabel);

            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            foreach (var line in _frames)
            {
                EditorGUILayout.LabelField(line, EditorStyles.miniLabel);
            }

            EditorGUILayout.EndScrollView();
        }
#endif
    }
}
