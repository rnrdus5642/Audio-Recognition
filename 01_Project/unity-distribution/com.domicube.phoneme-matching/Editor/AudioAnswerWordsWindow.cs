using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using DomiCube.PhonemeMatching.Unity;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Editor
{
    public static class AudioAnswerWords
    {
        public static string Build(string original, string[] stageWords)
        {
            var doc = JObject.Parse(original);
            var stages = (JArray)doc["presentation_stages"];
            if (stageWords.Length != stages.Count) throw new ArgumentException("단계 개수가 일치하지 않습니다.");
            var previous = ((JArray)doc["answers"]).ToDictionary(a => (string)a["text"], a => a);
            var answers = new Dictionary<string, JToken>();
            for (int i = 0; i < stages.Count; i++)
            {
                var display = stageWords[i].Split(new[] { ',', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(w => w.Normalize(NormalizationForm.FormC).Trim()).Where(w => w.Length > 0).ToArray();
                var words = display.Select(w => w.TrimEnd('?', '？').Trim()).ToArray();
                if (words.Length < 1 || words.Length > 20 || words.Distinct().Count() != words.Length
                    || words.Any(w => w.Length < 1 || w.Length > 20 || w.Any(c => c < '가' || c > '힣')))
                    throw new ArgumentException($"{i + 1}단계: 한글 1~20글자의 표현을 중복 없이 1~20개 입력하세요. 끝의 물음표는 사용할 수 있습니다.");
                stages[i]["answers"] = new JArray(words);
                stages[i]["display_answers"] = new JArray(display);
                foreach (var word in words)
                    if (!answers.ContainsKey(word)) answers[word] = previous.TryGetValue(word, out var old)
                        ? old.DeepClone() : new JObject { ["id"] = word, ["text"] = word, ["language"] = "ko",
                            ["threshold"] = .85, ["confirmation_frames"] = 2 };
            }
            doc["answers"] = new JArray(answers.Values);
            string result = doc.ToString(Formatting.Indented) + "\n";
            AudioAnswerContent.Validate(result);
            return result;
        }
    }

    public sealed class AudioAnswerWordsDraft : ScriptableObject
    {
        public string[] words;
    }

    public sealed class AudioAnswerWordsWindow : EditorWindow
    {
        [SerializeField] AudioAnswerWordsDraft _draft;
        [SerializeField] string _original, _path;
        [SerializeField] string[] _titles;
        string _error;
        Vector2 _scroll;
        [MenuItem("Tools/오디오 정답 처리 기능/정답 단어 설정", false, 20)]
        public static void Open()
        {
            var window = GetWindow<AudioAnswerWordsWindow>("정답 단어 설정");
            window.minSize = new Vector2(560, 430);
            window.Show();
        }
        void OnEnable()
        {
            if (_draft == null)
            {
                _draft = CreateInstance<AudioAnswerWordsDraft>();
                _draft.hideFlags = HideFlags.HideAndDontSave;
            }
            Undo.undoRedoPerformed += OnUndo;
            if (_draft.words == null || _original == null) Reload();
            RefreshDirty();
        }
        void OnDisable()
        {
            Undo.undoRedoPerformed -= OnUndo;
        }
        void OnDestroy()
        {
            if (_draft != null) DestroyImmediate(_draft);
        }
        void OnUndo() { GUI.FocusControl(null); RefreshDirty(); Repaint(); }
        void RefreshDirty()
        {
            if (_draft == null || _draft.words == null || string.IsNullOrEmpty(_original)) { hasUnsavedChanges = false; return; }
            var stages = (JArray)JObject.Parse(_original)["presentation_stages"];
            var saved = stages.Select(s => string.Join(", ", (s["display_answers"] ?? s["answers"]).Values<string>()));
            hasUnsavedChanges = !saved.SequenceEqual(_draft.words);
            saveChangesMessage = "저장하지 않은 정답 단어 변경이 있습니다.";
        }
        public override void SaveChanges() { Save(); }
        public override void DiscardChanges() { Reload(); base.DiscardChanges(); }
        void Reload()
        {
            try
            {
                var asset = Resources.Load<TextAsset>(AudioAnswerContent.TargetsResourcePath);
                string path = AssetDatabase.GetAssetPath(asset);
                if (string.IsNullOrEmpty(path) || !path.StartsWith("Assets/", StringComparison.Ordinal))
                    throw new InvalidOperationException("프로젝트 Assets 폴더의 정답 목록을 연결해야 합니다.");
                string original = File.ReadAllText(path, Encoding.UTF8);
                var stages = (JArray)JObject.Parse(original)["presentation_stages"];
                string[] titles = stages.Select(s => s["id"] + " · " + (string)s["title"]).ToArray();
                string[] words = stages.Select(s => string.Join(", ", (s["display_answers"] ?? s["answers"]).Values<string>())).ToArray();
                _path = path;
                _original = original;
                _titles = titles;
                _draft.words = words;
                hasUnsavedChanges = false;
                _error = "";
            }
            catch (Exception e) { _error = e.Message; }
        }
        void Save()
        {
            try
            {
                string result = AudioAnswerWords.Build(_original, _draft.words);
                if (File.ReadAllText(_path, Encoding.UTF8) != _original)
                    throw new InvalidOperationException("파일이 외부에서 변경됐습니다. 다시 읽은 뒤 수정하세요.");
                string temp = _path + ".audioanswer.tmp";
                try
                {
                    File.WriteAllText(temp, result, new UTF8Encoding(false));
                    File.Replace(temp, _path, null);
                }
                finally { if (File.Exists(temp)) File.Delete(temp); }
                AssetDatabase.ImportAsset(_path, ImportAssetOptions.ForceSynchronousImport);
                _original = result;
                RefreshDirty();
                _error = "저장 완료. 다음 게임 시작 또는 다음 음성 테스트부터 적용됩니다.";
            }
            catch (Exception e) { _error = e.Message; }
        }
        void OnGUI()
        {
            GUILayout.Label("단계별 정답 단어", EditorStyles.boldLabel);
            GUILayout.Label("표현은 쉼표 또는 줄바꿈으로 구분합니다. 모델을 다시 학습할 필요는 없습니다.", EditorStyles.wordWrappedLabel);
            if (!string.IsNullOrEmpty(_error)) EditorGUILayout.HelpBox(_error, MessageType.Info);
            using (new EditorGUI.DisabledScope(EditorApplication.isPlayingOrWillChangePlaymode))
            {
                _scroll = EditorGUILayout.BeginScrollView(_scroll);
                if (_draft.words != null)
                    for (int i = 0; i < _draft.words.Length; i++)
                    {
                        GUILayout.Label(_titles[i], EditorStyles.boldLabel);
                        string value = EditorGUILayout.TextArea(_draft.words[i], GUILayout.MinHeight(45));
                        if (value != _draft.words[i])
                        {
                            Undo.RecordObject(_draft, "정답 단어 수정");
                            _draft.words[i] = value;
                            RefreshDirty();
                        }
                    }
                EditorGUILayout.EndScrollView();
                using (new EditorGUILayout.HorizontalScope())
                {
                    if (GUILayout.Button("파일 다시 읽기") && (!hasUnsavedChanges || EditorUtility.DisplayDialog(
                            "정답 다시 읽기", "저장하지 않은 변경을 버리고 파일을 다시 읽을까요?", "다시 읽기", "취소"))) Reload();
                    using (new EditorGUI.DisabledScope(_draft.words == null))
                        if (GUILayout.Button("정답 단어 저장")) Save();
                }
            }
        }
    }
}
