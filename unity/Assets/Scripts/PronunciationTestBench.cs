using System.Collections;
using System.Collections.Generic;
using System.Text;
using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// In-editor counterpart of the Python web UI's 실시간 tab - the flow the
/// VR lesson actually uses.
///
/// The microphone stays open and every hop the most recent window is
/// re-recognised and scored; an answer that wins <see cref="Consecutive"/>
/// frames in a row stops the recording. What matters for tuning is not
/// the verdict but the frame trail behind it: which candidate led, what
/// it scored against its own threshold, and which phonemes were in
/// context at the time. All of that is on screen, so this scene and the
/// web page can be given the same word and read side by side.
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

    [Header("Listening")]
    [Tooltip("ASR에 넣는 오디오 창 (초)")]
    public float WindowSeconds = 2.5f;

    [Tooltip("채점 주기 (초). 웹 실시간 탭과 같은 0.5초 고정.")]
    public float HopSeconds = 0.5f;

    [Tooltip("확정에 필요한 연속 프레임 수")]
    [Range(1, 6)] public int Consecutive = 2;

    private ConfusionMatrix _matrix;
    private TargetCatalog _catalog;
    private SentisPhonemeRecognizer _recognizer;
    private Texture2D _pixel;

    private string _targetInput = "사과\n바나나\n빵";
    private int _deviceIndex;
    private bool _listening;
    private bool _ready;

    private string _status = "준비 중...";
    private readonly List<Frame> _frames = new List<Frame>();
    private List<Answer> _candidates = new List<Answer>();
    private string _scoredIpa = string.Empty;
    private string _profileLine = string.Empty;
    private int _deviceRate;
    private Vector2 _scroll;

    /// <summary>One scoring event, as the frame table shows it.</summary>
    private struct Frame
    {
        public float Time;
        public int Streak;
        public string Word;
        public double Score;
        public double Threshold;
        public long Ms;
        public string Hangul;
    }

    private void Awake()
    {
        // IMGUI draws straight over the camera; a skybox behind small
        // text is unreadable.
        _pixel = new Texture2D(1, 1) { hideFlags = HideFlags.HideAndDontSave };
        _pixel.SetPixel(0, 0, Color.white);
        _pixel.Apply();
    }

    private IEnumerator Start()
    {
        _matrix = PhonemeData.LoadMatrix(ReadStreamingAsset(MatrixFileName));
        _catalog = PhonemeData.LoadTargets(ReadStreamingAsset(TargetsFileName));

        var profile = Matcher.ForStreaming(_matrix);
        _profileLine = $"skip {profile.SkipCost:F2} · 커버리지 "
            + $"{profile.Coverage:F2} · 문맥 {profile.ContextMult:F1}×정답길이";

        if (Model == null)
        {
            _status = "ModelAsset 이 비어 있습니다 (인스펙터에 넣으세요)";
            yield break;
        }

        var vocab = PhonemeData.LoadCtcVocabulary(
            ReadStreamingAsset(VocabFileName));
        _recognizer = new SentisPhonemeRecognizer(Model, vocab);

        // Shader compilation and the weight upload land on the first
        // inference. Pay it here so the first spoken word does not.
        _status = "모델 준비 중...";
        yield return null;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        _recognizer.Warmup(WindowSeconds);
        _ready = true;
        _status = $"준비 완료 (워밍업 {sw.ElapsedMilliseconds}ms) — 시작을 누르세요";
    }

    private void OnDestroy()
    {
        _listening = false;

        if (_recognizer != null)
        {
            _recognizer.Dispose();
            _recognizer = null;
        }

        if (_pixel != null)
        {
            Destroy(_pixel);
            _pixel = null;
        }
    }

    private static string ReadStreamingAsset(string fileName)
    {
        var path = System.IO.Path.Combine(
            Application.streamingAssetsPath, fileName);
        return System.IO.File.ReadAllText(path);
    }

    private IEnumerator ListenLoop()
    {
        _frames.Clear();
        _scoredIpa = string.Empty;

        _candidates = ResolveCandidates(out var missing);
        if (_candidates.Count == 0)
        {
            _status = missing.Count > 0
                ? $"targets.json 에 없는 단어: {string.Join(", ", missing)}"
                : "정답 단어를 입력하세요";
            _listening = false;
            yield break;
        }

        var matcher = Matcher.ForStreaming(_matrix);
        var streaming = new StreamingMatcher(matcher, _candidates, Consecutive);

        var device = Microphone.devices.Length > 0
            ? Microphone.devices[_deviceIndex % Microphone.devices.Length]
            : null;
        var mic = new MicrophoneRollingBuffer(WindowSeconds);
        try
        {
            mic.Start(device);
        }
        catch (System.Exception e)
        {
            // Another app holding the device is the usual cause, and it
            // would otherwise look like a silent room.
            _status = $"마이크를 열 수 없습니다: {e.Message}";
            _listening = false;
            mic.Dispose();
            yield break;
        }

        _deviceRate = mic.DeviceSampleRate;

        var wait = new WaitForSeconds(HopSeconds);
        float started = Time.realtimeSinceStartup;
        var sw = new System.Diagnostics.Stopwatch();

        if (missing.Count > 0)
        {
            _status = $"듣는 중 (없는 단어 무시: {string.Join(", ", missing)})";
        }
        else
        {
            _status = "듣는 중 — 말해보세요";
        }

        while (_listening)
        {
            yield return wait;

            if (!mic.Pump())
            {
                continue;
            }

            // Re-recognise the whole recent window, never a fresh chunk:
            // wav2vec2 is a context model and mangles short fragments.
            sw.Restart();
            _recognizer.RecognizeWithText(
                mic.Snapshot(), out var hangul, out var phonemes);
            sw.Stop();

            var hit = streaming.Push(phonemes);
            var best = matcher.BestMatch(phonemes, _candidates);

            _frames.Add(new Frame
            {
                Time = Time.realtimeSinceStartup - started,
                Streak = streaming.Streak,
                Word = best.TargetText,
                Score = best.Score,
                Threshold = ThresholdOf(best.TargetId),
                Ms = sw.ElapsedMilliseconds,
                Hangul = hangul
            });

            _scoredIpa = DescribeScored(matcher, phonemes, best);

            if (hit != null)
            {
                _status = $"확정: {hit.Result.TargetText} "
                    + $"({hit.Result.Score:F3}, {_frames.Count}프레임)";
                _listening = false;
                break;
            }
        }

        mic.Dispose();

        if (_status.StartsWith("듣는 중"))
        {
            _status = $"중지 ({_frames.Count}프레임, 확정 없음)";
        }
    }

    private double ThresholdOf(string targetId)
    {
        var hit = _candidates.Find(a => a.Id == targetId);
        return hit == null ? 0.0 : hit.Threshold;
    }

    /// <summary>
    /// The phonemes that were actually scored this frame, with the
    /// matched window marked.
    ///
    /// Streaming bounds how far back the matcher looks, so what is on
    /// screen is not the whole recognition - showing the full list would
    /// suggest phonemes counted that never did.
    /// </summary>
    private static string DescribeScored(
        Matcher matcher, List<string> phonemes, MatchResult best)
    {
        if (phonemes.Count == 0)
        {
            return "(무음)";
        }

        int start = 0;
        if (best.TargetPhonemes != null)
        {
            matcher.ContextSlice(phonemes, best.TargetPhonemes, out start, out _);
        }

        var sb = new StringBuilder();
        for (int i = start; i < phonemes.Count; i++)
        {
            bool inWindow = i >= best.WindowStart && i < best.WindowEnd;
            sb.Append(inWindow ? $"<b>{phonemes[i]}</b>" : phonemes[i])
              .Append(' ');
        }

        if (start > 0)
        {
            sb.Insert(0, $"…({start}개 문맥 밖) ");
        }

        return sb.ToString().TrimEnd();
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
        return _catalog.Find(text);
    }

    // ------------------------------------------------------------------
    // GUI
    // ------------------------------------------------------------------

    private void OnGUI()
    {
        const int pad = 12;

        GUI.color = new Color(0.16f, 0.16f, 0.17f);
        GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), _pixel);
        GUI.color = Color.white;

        var label = new GUIStyle(GUI.skin.label) { fontSize = 14 };
        var small = new GUIStyle(GUI.skin.label) { fontSize = 12 };
        var mono = new GUIStyle(GUI.skin.label)
        {
            fontSize = 14,
            richText = true,
            wordWrap = true
        };
        var head = new GUIStyle(GUI.skin.label)
        {
            fontSize = 16,
            fontStyle = FontStyle.Bold
        };

        DrawSettings(new Rect(pad, pad, 330, Screen.height - pad * 2),
            head, small);
        DrawResults(
            new Rect(pad * 2 + 330, pad, Screen.width - 330 - pad * 3,
                Screen.height - pad * 2),
            head, label, small, mono);
    }

    private void DrawSettings(Rect area, GUIStyle head, GUIStyle small)
    {
        GUILayout.BeginArea(area, GUI.skin.box);
        GUILayout.Label("설정", head);

        GUILayout.Label("정답 단어 (한 줄에 하나)", small);
        GUI.enabled = !_listening;
        _targetInput = GUILayout.TextArea(_targetInput, GUILayout.Height(70));

        GUILayout.Space(8);
        GUILayout.Label("마이크", small);
        DrawDeviceList(small);

        GUI.enabled = _ready;
        GUILayout.Space(10);
        if (!_listening)
        {
            if (GUILayout.Button("시작", GUILayout.Height(36)))
            {
                _listening = true;
                StartCoroutine(ListenLoop());
            }
        }
        else if (GUILayout.Button("중지", GUILayout.Height(36)))
        {
            _listening = false;
        }

        GUI.enabled = true;
        GUILayout.Space(8);
        GUILayout.Label(
            $"창 {WindowSeconds:F1}s · hop {HopSeconds:F1}s · 확정 {Consecutive}회",
            small);
        GUILayout.Label(
            $"확정까지 최소 {Consecutive * HopSeconds:F1}s — 정답을 말한 뒤에도"
            + " 그만큼 더 들어야 합니다.", small);

        GUILayout.Space(10);
        GUILayout.Label("스트리밍 프로파일", head);
        GUILayout.Label(_profileLine, small);
        GUILayout.EndArea();
    }

    /// <summary>
    /// Every connected microphone, one row each, always visible - a VR
    /// machine typically has several and the headset one is rarely the
    /// system default.
    /// </summary>
    private void DrawDeviceList(GUIStyle small)
    {
        var devices = Microphone.devices;
        if (devices.Length == 0)
        {
            GUILayout.Label("연결된 마이크가 없습니다", small);
            return;
        }

        if (_deviceIndex >= devices.Length)
        {
            _deviceIndex = 0;
        }

        for (int i = 0; i < devices.Length; i++)
        {
            bool selected = i == _deviceIndex;
            if (GUILayout.Toggle(selected, devices[i], GUI.skin.button)
                && !selected)
            {
                _deviceIndex = i;
            }

            Microphone.GetDeviceCaps(devices[i], out int min, out int max);
            // 0/0 means the driver imposes no limit. What the device
            // advertises and what Unity hands back are not the same
            // thing - the Oculus headset mic reports 48 kHz only and
            // still returns a 16 kHz clip - so report the claim here and
            // the fact below, without predicting one from the other.
            GUILayout.Label(
                min == 0 && max == 0
                    ? "   장치 보고: 제한 없음"
                    : $"   장치 보고: {min}~{max} Hz",
                small);
        }

        if (_deviceRate > 0)
        {
            GUILayout.Label(
                _deviceRate == MicrophoneRollingBuffer.TargetSampleRate
                    ? $"   실제 개방: {_deviceRate} Hz"
                    : $"   실제 개방: {_deviceRate} Hz → 16 kHz 리샘플링",
                small);
        }
    }

    private void DrawResults(
        Rect area, GUIStyle head, GUIStyle label, GUIStyle small, GUIStyle mono)
    {
        GUILayout.BeginArea(area, GUI.skin.box);
        GUILayout.Label(_status, head);

        GUILayout.Space(6);
        GUILayout.Label(ThresholdLine(), small);

        // Score over time. Thresholds are listed above rather than drawn
        // as lines: the bar moves with whichever candidate leads that
        // frame, so a "threshold line" would be several lines crossing.
        var chart = GUILayoutUtility.GetRect(10, 110, GUILayout.ExpandWidth(true));
        DrawChart(chart);

        GUILayout.Space(4);
        GUILayout.Label("채점된 IPA (굵게 = 매칭 윈도우)", small);
        GUILayout.Label(
            _scoredIpa.Length == 0 ? "(아직 없음)" : _scoredIpa, mono);

        GUILayout.Space(8);
        GUILayout.Label("프레임 상세", head);
        GUILayout.Label("  시각    연속   최고 후보    점수    임계값   ms   ASR",
            small);

        _scroll = GUILayout.BeginScrollView(_scroll);
        for (int i = _frames.Count - 1; i >= 0; i--)
        {
            var f = _frames[i];
            GUILayout.Label(
                $"{f.Time,6:F1}s  {f.Streak}/{Consecutive}   "
                + $"{(string.IsNullOrEmpty(f.Word) ? "—" : f.Word),-6}  "
                + $"{f.Score:F3}   {f.Threshold:F2}   {f.Ms,4}  "
                + $"'{f.Hangul}'", label);
        }

        GUILayout.EndScrollView();
        GUILayout.EndArea();
    }

    private string ThresholdLine()
    {
        if (_candidates.Count == 0)
        {
            return "후보 임계값: (시작하면 표시됩니다)";
        }

        var sb = new StringBuilder("후보 임계값:  ");
        for (int i = 0; i < _candidates.Count; i++)
        {
            var c = _candidates[i];
            sb.Append($"{c.Text} {c.Threshold:F2} ({c.Phonemes.Count}음소)");
            if (i < _candidates.Count - 1)
            {
                sb.Append("  ·  ");
            }
        }

        return sb.ToString();
    }

    private void DrawChart(Rect area)
    {
        GUI.color = new Color(0.11f, 0.11f, 0.12f);
        GUI.DrawTexture(area, _pixel);
        GUI.color = Color.white;

        if (_frames.Count == 0)
        {
            return;
        }

        // Keep the most recent frames visible rather than squeezing a
        // long session into the same width.
        const int maxBars = 60;
        int first = Mathf.Max(0, _frames.Count - maxBars);
        int count = _frames.Count - first;
        float barWidth = Mathf.Max(2f, (area.width - 4f) / maxBars);

        for (int i = 0; i < count; i++)
        {
            var f = _frames[first + i];
            float h = Mathf.Clamp01((float)f.Score) * (area.height - 6f);
            var bar = new Rect(
                area.x + 2f + i * barWidth,
                area.yMax - 3f - h,
                Mathf.Max(1f, barWidth - 1f),
                h);

            GUI.color = f.Score >= f.Threshold
                ? new Color(0.35f, 0.75f, 0.45f)
                : new Color(0.45f, 0.5f, 0.6f);
            GUI.DrawTexture(bar, _pixel);
        }

        GUI.color = Color.white;
        GUI.Label(new Rect(area.x + 4f, area.y + 2f, 120f, 18f), "점수 1.0");
        GUI.Label(new Rect(area.x + 4f, area.yMax - 20f, 120f, 18f), "0.0");
    }
}
