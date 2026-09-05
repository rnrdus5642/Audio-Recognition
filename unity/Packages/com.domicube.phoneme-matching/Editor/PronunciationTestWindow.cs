using System.Collections.Generic;
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
    /// Needs the model's matrix/targets and shared vocabulary JSON in
    /// StreamingAssets, plus an imported .onnx model; see the package README.
    ///
    /// Tools > Phoneme Matching > 발음 테스트 (마이크)
    /// </summary>
    public sealed class PronunciationTestWindow : EditorWindow
    {
        private const string ModelKey = "DomiCube.PhonemeMatching.TestModel";
        private const string DataChoiceKey = "DomiCube.PhonemeMatching.TestData.";
        private const double Hop = 0.5;
        private static readonly string[] DataChoices =
            { "선택하세요", "기본 모델용", "아동 파인튜닝 모델용" };

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
        private double _nextScore;
        private double _started;
#endif

        [MenuItem("Tools/Phoneme Matching/발음 테스트 (마이크)", false, 40)]
        public static void Open()
        {
            GetWindow<PronunciationTestWindow>("발음 테스트").minSize =
                new Vector2(420, 420);
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

        private void ClearPrepared()
        {
            Stop();
            _recognizer?.Dispose();
            _recognizer = null;
            _session = null;
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
            if (_recognizer != null && _session != null
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

                _session = new PronunciationSession(matrix, catalog, _recognizer);
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

            try
            {
                _session.Begin(_word);
            }
            catch (System.Exception e)
            {
                _status = e.Message;
                return;
            }

            _frames.Clear();
            _mic = new MicrophoneRollingBuffer(2.5f);
            _mic.Start(devices[_deviceIndex % devices.Length]);

            _started = EditorApplication.timeSinceStartup;
            _nextScore = _started + Hop;
            _status = $"듣는 중 — '{_word}' 라고 말해보세요";

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

            _mic.Pump();

            double now = EditorApplication.timeSinceStartup;
            if (now < _nextScore)
            {
                return;
            }

            _nextScore = now + Hop;

            FrameScore frame;
            try
            {
                frame = _session.Push(_mic.Snapshot());
            }
            catch (System.Exception e)
            {
                _status = "실패: " + e.Message;
                Stop();
                Repaint();
                return;
            }

            _frames.Insert(0, string.Format(
                "{0,5:F1}s  {1}/{5}  {2,-6} {3:F3}  '{4}'",
                now - _started, frame.Streak,
                string.IsNullOrEmpty(frame.Best.TargetText)
                    ? "—" : frame.Best.TargetText,
                frame.Best.Score, frame.Text, _session.Consecutive));

            if (frame.Confirmed)
            {
                _status = $"확정: {frame.Best.TargetText} "
                    + $"({frame.Best.Score:F3}, {now - _started:F1}초)";
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
                if (_model != null && PronunciationTestSettings.FromModelPath(modelPath) == null)
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
                        + $"공통 어휘: {PronunciationTestSettings.VocabularyFile}", MessageType.Info);
                }
                else if (_model != null)
                {
                    EditorGUILayout.HelpBox(
                        "이름을 바꾼 모델이나 별도 모델은 종류를 자동 판별할 수 없습니다.\n"
                        + "위에서 모델에 맞는 채점 설정을 선택하세요.", MessageType.Warning);
                }

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
