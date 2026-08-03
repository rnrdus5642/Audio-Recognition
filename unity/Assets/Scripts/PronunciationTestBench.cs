using System.Collections;
using System.Collections.Generic;
using System.Text;
using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// In-editor counterpart of the Python web UI's 발음 테스트 tab.
///
/// Shows every number the web page shows - ASR Hangul, IPA, per-candidate
/// score against its own threshold, the matched window and the phoneme
/// alignment - so the two can be put side by side and compared on the
/// same word. That is how this port is checked for drift: the automated
/// vectors pin the matcher, this pins the whole chain including the
/// acoustic model, which no fixture can capture.
///
/// Candidates come from targets.json. Unity cannot run g2pkk, so a word
/// that was never built has no IPA here - unlike the web page, which
/// runs G2P on the fly.
/// </summary>
public sealed class PronunciationTestBench : MonoBehaviour
{
    [Header("Model")]
    [Tooltip("Assets/Models/wav2vec2_ko.onnx")]
    public ModelAsset Model;

    [Header("StreamingAssets")]
    public string MatrixFileName = "ko_child_v1.json";
    public string TargetsFileName = "targets.json";
    public string VocabFileName = "wav2vec2_ko_vocab.json";

    [Header("Recording")]
    [Range(1f, 6f)] public float RecordSeconds = 3f;

    private ConfusionMatrix _matrix;
    private TargetCatalog _catalog;
    private SentisPhonemeRecognizer _recognizer;

    private string _targetInput = "사과\n바나나\n빵";
    private int _deviceIndex;
    private bool _streamingProfile;
    private bool _busy;

    private string _status = "준비 중...";
    private string _hangul = string.Empty;
    private List<string> _phonemes = new List<string>();
    private long _inferenceMs;
    private readonly List<Row> _rows = new List<Row>();
    private MatchResult _best;
    private readonly List<string> _history = new List<string>();
    private Vector2 _scroll;

    private struct Row
    {
        public string Text;
        public double Score;
        public double Threshold;
        public bool Passed;
    }

    private IEnumerator Start()
    {
        _matrix = PhonemeData.LoadMatrix(ReadStreamingAsset(MatrixFileName));
        _catalog = PhonemeData.LoadTargets(ReadStreamingAsset(TargetsFileName));

        if (Model == null)
        {
            _status = "ModelAsset 이 비어 있습니다 (인스펙터에 넣으세요)";
            yield break;
        }

        var vocab = PhonemeData.LoadCtcVocabulary(
            ReadStreamingAsset(VocabFileName));
        _recognizer = new SentisPhonemeRecognizer(Model, vocab);

        _status = "모델 준비 중...";
        yield return null;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        _recognizer.Warmup(RecordSeconds);
        _status = $"준비 완료 (워밍업 {sw.ElapsedMilliseconds}ms) — 녹음을 누르세요";
    }

    private void OnDestroy()
    {
        if (_recognizer != null)
        {
            _recognizer.Dispose();
            _recognizer = null;
        }
    }

    private static string ReadStreamingAsset(string fileName)
    {
        var path = System.IO.Path.Combine(
            Application.streamingAssetsPath, fileName);
        return System.IO.File.ReadAllText(path);
    }

    private IEnumerator RecordAndScore()
    {
        _busy = true;
        _rows.Clear();
        _best = null;
        _hangul = string.Empty;
        _phonemes = new List<string>();

        var candidates = ResolveCandidates(out var missing);
        if (candidates.Count == 0)
        {
            _status = missing.Count > 0
                ? $"targets.json 에 없는 단어: {string.Join(", ", missing)}"
                : "정답 단어를 입력하세요";
            _busy = false;
            yield break;
        }

        var device = Microphone.devices.Length > 0
            ? Microphone.devices[_deviceIndex % Microphone.devices.Length]
            : null;

        // Same capture path the lesson uses, so what is measured here is
        // what ships.
        var mic = new MicrophoneRollingBuffer(RecordSeconds);
        mic.Start(device);

        float until = Time.realtimeSinceStartup + RecordSeconds;
        while (Time.realtimeSinceStartup < until)
        {
            mic.Pump();
            _status = $"녹음 중... {until - Time.realtimeSinceStartup:F1}s";
            yield return null;
        }

        mic.Pump();
        var window = mic.Snapshot();
        mic.Dispose();

        var sw = System.Diagnostics.Stopwatch.StartNew();
        _recognizer.RecognizeWithText(window, out _hangul, out _phonemes);
        _inferenceMs = sw.ElapsedMilliseconds;

        var matcher = _streamingProfile
            ? Matcher.ForStreaming(_matrix)
            : new Matcher(_matrix);

        foreach (var c in candidates)
        {
            matcher.ScoreAgainst(
                _phonemes, c.Phonemes, out double score, out _, out _);
            _rows.Add(new Row
            {
                Text = c.Text,
                Score = score,
                Threshold = c.Threshold,
                Passed = score >= c.Threshold
            });
        }

        _best = matcher.BestMatch(_phonemes, candidates);
        _rows.Sort((a, b) => b.Score.CompareTo(a.Score));

        _status = _best.Passed
            ? $"통과: {_best.TargetText} ({_best.Score:F3})"
            : $"실패 (최고 {_best.TargetText} {_best.Score:F3})";

        _history.Insert(0,
            $"{(_best.Passed ? "O" : "X")} '{_hangul}' -> {_best.TargetText} "
            + $"{_best.Score:F3}");
        if (_history.Count > 8)
        {
            _history.RemoveAt(_history.Count - 1);
        }

        if (missing.Count > 0)
        {
            _status += $"  (없는 단어 무시: {string.Join(", ", missing)})";
        }

        _busy = false;
    }

    private List<Answer> ResolveCandidates(out List<string> missing)
    {
        var result = new List<Answer>();
        missing = new List<string>();

        foreach (var line in _targetInput.Split('\n'))
        {
            var word = line.Trim();
            if (word.Length == 0)
            {
                continue;
            }

            var found = FindAnswer(word);
            if (found == null)
            {
                missing.Add(word);
            }
            else
            {
                result.Add(found);
            }
        }

        return result;
    }

    private Answer FindAnswer(string text)
    {
        foreach (var seg in _catalog.Segments)
        {
            var hit = seg.Answers.Find(a => a.Text == text || a.Id == text);
            if (hit != null)
            {
                return hit;
            }
        }

        return null;
    }

    private static string FormatAlignment(MatchResult result)
    {
        if (result == null || result.Alignment == null
            || result.Alignment.Count == 0)
        {
            return "(없음)";
        }

        var sb = new StringBuilder();
        foreach (var step in result.Alignment)
        {
            switch (step.Op)
            {
                case AlignOp.Match:
                    sb.Append(step.UserPhoneme);
                    break;
                case AlignOp.Substitute:
                    sb.Append(step.UserPhoneme).Append('→')
                      .Append(step.TargetPhoneme);
                    break;
                case AlignOp.Insert:
                    sb.Append('+').Append(step.UserPhoneme);
                    break;
                default:
                    sb.Append('-').Append(step.TargetPhoneme);
                    break;
            }

            sb.Append("  ");
        }

        return sb.ToString().TrimEnd();
    }

    private void OnGUI()
    {
        const int pad = 12;
        var label = new GUIStyle(GUI.skin.label) { fontSize = 14 };
        var small = new GUIStyle(GUI.skin.label) { fontSize = 12 };
        var head = new GUIStyle(GUI.skin.label)
        {
            fontSize = 16,
            fontStyle = FontStyle.Bold
        };

        GUILayout.BeginArea(new Rect(pad, pad, 380, Screen.height - pad * 2),
            GUI.skin.box);
        GUILayout.Label("설정", head);

        GUILayout.Label("정답 단어 (한 줄에 하나)", small);
        _targetInput = GUILayout.TextArea(_targetInput, GUILayout.Height(80));

        GUILayout.Space(6);
        var devices = Microphone.devices;
        GUILayout.Label(devices.Length == 0
            ? "마이크 없음"
            : $"마이크: {devices[_deviceIndex % devices.Length]}", small);
        if (devices.Length > 1 && GUILayout.Button("마이크 바꾸기"))
        {
            _deviceIndex = (_deviceIndex + 1) % devices.Length;
        }

        GUILayout.Space(6);
        GUILayout.Label($"녹음 길이 {RecordSeconds:F1}s", small);
        RecordSeconds = Mathf.Round(
            GUILayout.HorizontalSlider(RecordSeconds, 1f, 6f) * 10f) / 10f;

        _streamingProfile = GUILayout.Toggle(_streamingProfile,
            "스트리밍 프로파일로 채점");
        GUILayout.Label(_streamingProfile
            ? "skip 0.05 · 커버리지 0.8 · 문맥 제한 있음"
            : "skip 0.15 · 커버리지 0.5 · 문맥 제한 없음", small);

        GUILayout.Space(10);
        GUI.enabled = !_busy && _recognizer != null;
        if (GUILayout.Button("녹음 & 채점", GUILayout.Height(36)))
        {
            StartCoroutine(RecordAndScore());
        }

        GUI.enabled = true;
        GUILayout.Space(10);
        GUILayout.Label("세션 기록", head);
        foreach (var h in _history)
        {
            GUILayout.Label(h, small);
        }

        GUILayout.EndArea();

        GUILayout.BeginArea(
            new Rect(pad * 2 + 380, pad, Screen.width - 380 - pad * 3,
                Screen.height - pad * 2), GUI.skin.box);

        GUILayout.Label(_status, head);
        GUILayout.Space(8);

        _scroll = GUILayout.BeginScrollView(_scroll);

        GUILayout.Label($"ASR 한글:  {(_hangul.Length == 0 ? "(없음)" : _hangul)}",
            label);
        GUILayout.Label(
            $"IPA ({_phonemes.Count}):  [{string.Join(" ", _phonemes)}]", label);
        GUILayout.Label($"인식 시간:  {_inferenceMs} ms", small);

        GUILayout.Space(10);
        GUILayout.Label("후보별 점수", head);
        foreach (var row in _rows)
        {
            GUILayout.Label(
                $"{(row.Passed ? "O" : "X")}  {row.Text}    "
                + $"{row.Score:F3}  /  임계값 {row.Threshold:F2}", label);
        }

        if (_best != null)
        {
            GUILayout.Space(10);
            GUILayout.Label("정렬 (최고 후보)", head);
            GUILayout.Label(FormatAlignment(_best), label);
            GUILayout.Label(
                $"매칭 윈도우: 사용자 음소 [{_best.WindowStart}, "
                + $"{_best.WindowEnd})  ·  거리 {_best.Distance:F3}", small);
            GUILayout.Label(
                $"정답 IPA: [{string.Join(" ", _best.TargetPhonemes)}]", small);
        }

        GUILayout.EndScrollView();
        GUILayout.EndArea();
    }
}
