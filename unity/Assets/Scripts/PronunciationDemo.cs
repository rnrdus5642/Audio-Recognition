using System.Collections;
using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// Bring-up scene: microphone -> rolling window -> wav2vec2 -> matching
/// -> confirm -> auto-stop.
///
/// Assign the ONNX model in the inspector. Without it the scene falls
/// back to <see cref="LoudnessStubRecognizer"/>, which treats any loud
/// sound as the answer - useful for checking the plumbing, useless for
/// judging pronunciation, so the banner says which one is running.
///
/// Put this on an empty GameObject and press Play. Space starts
/// listening.
/// </summary>
[RequireComponent(typeof(PronunciationListener))]
public sealed class PronunciationDemo : MonoBehaviour
{
    [Tooltip("Assets/Models/wav2vec2_ko.onnx. 비우면 stub으로 동작합니다.")]
    public ModelAsset Model;

    [Tooltip("StreamingAssets 안의 CTC 어휘 파일")]
    public string VocabFileName = "wav2vec2_ko_vocab.json";

    [Tooltip("자동으로 듣기 시작")]
    public bool ListenOnStart = true;

    private PronunciationListener _listener;
    private SentisPhonemeRecognizer _recognizer;
    private string _status = "대기 중 — Space를 눌러 시작";
    private string _lastFrame = string.Empty;
    private string _engine = "stub — 소리 크기만 봅니다 (발음 판정 아님)";

    private void Awake()
    {
        _listener = GetComponent<PronunciationListener>();
        _listener.SetRecognizer(CreateRecognizer());

        _listener.OnConfirmed.AddListener((word, score) =>
        {
            _status = $"확정: {word}  (점수 {score:F3})";
            Debug.Log($"[Demo] {_status}");
        });

        _listener.OnTimedOut.AddListener(() =>
        {
            _status = "못 알아들었어요 — Space를 눌러 다시";
            Debug.Log("[Demo] 타임아웃");
        });

        _listener.OnFrameScored.AddListener((word, score, streak) =>
        {
            _lastFrame =
                $"{(string.IsNullOrEmpty(word) ? "—" : word)}  "
                + $"{score:F3}  연속 {streak}/{_listener.Consecutive}";
        });
    }

    /// <summary>
    /// Sentis when a model is assigned, stub otherwise.
    /// </summary>
    private IPhonemeRecognizer CreateRecognizer()
    {
        if (Model == null)
        {
            Debug.LogWarning(
                "[Demo] ModelAsset이 없어 stub으로 실행합니다. "
                + "Assets/Models/wav2vec2_ko.onnx 를 인스펙터에 넣으세요.");
            return new LoudnessStubRecognizer();
        }

        var path = System.IO.Path.Combine(
            Application.streamingAssetsPath, VocabFileName);
        var vocab = PhonemeData.LoadCtcVocabulary(
            System.IO.File.ReadAllText(path));

        _recognizer = new SentisPhonemeRecognizer(Model, vocab);
        _engine = $"wav2vec2 + Sentis GPU ({vocab.Size} 토큰)";
        return _recognizer;
    }

    private IEnumerator Start()
    {
        if (Microphone.devices.Length == 0)
        {
            _status = "마이크를 찾을 수 없습니다";
            Debug.LogError("[Demo] " + _status);
            yield break;
        }

        Debug.Log($"[Demo] 마이크: {Microphone.devices[0]}");

        if (_recognizer != null)
        {
            // Shader compilation and a 1.2 GB weight upload land on the
            // first inference. Pay it here rather than on the child's
            // first word.
            _status = "모델 준비 중...";
            yield return null;   // let the banner draw before we block

            var sw = System.Diagnostics.Stopwatch.StartNew();
            _recognizer.Warmup(_listener.WindowSeconds);
            Debug.Log($"[Demo] 워밍업 {sw.ElapsedMilliseconds}ms");
        }

        if (ListenOnStart)
        {
            StartListening();
        }
        else
        {
            _status = "대기 중 — Space를 눌러 시작";
        }
    }

    private void OnDestroy()
    {
        if (_recognizer != null)
        {
            _recognizer.Dispose();
            _recognizer = null;
        }
    }

    private void Update()
    {
        // This project has the Input System package active, where the
        // legacy Input class throws on every access. Keep both paths so
        // the scene also opens in a default project.
#if ENABLE_INPUT_SYSTEM
        var keyboard = UnityEngine.InputSystem.Keyboard.current;
        bool pressed = keyboard != null && keyboard.spaceKey.wasPressedThisFrame;
#else
        bool pressed = Input.GetKeyDown(KeyCode.Space);
#endif

        if (pressed && !_listener.IsListening)
        {
            StartListening();
        }
    }

    private void StartListening()
    {
        _status = $"듣는 중 — '{_listener.TargetText}' 라고 말해보세요";
        _lastFrame = string.Empty;
        _listener.Listen();
    }

    private void OnGUI()
    {
        const int pad = 16;
        GUI.Box(new Rect(pad, pad, 560, 132), string.Empty);

        var style = new GUIStyle(GUI.skin.label) { fontSize = 18 };
        GUI.Label(new Rect(pad + 12, pad + 10, 540, 30), _status, style);

        var small = new GUIStyle(GUI.skin.label) { fontSize = 14 };
        GUI.Label(new Rect(pad + 12, pad + 44, 540, 24),
            $"프레임: {_lastFrame}", small);
        GUI.Label(new Rect(pad + 12, pad + 68, 540, 24),
            $"창 {_listener.WindowSeconds:F1}s · hop {_listener.HopSeconds:F1}s"
            + $" · 확정 {_listener.Consecutive}회"
            + $" · 예상 지연 {_listener.Consecutive * _listener.HopSeconds:F1}s",
            small);
        GUI.Label(new Rect(pad + 12, pad + 92, 540, 24), _engine, small);
    }
}
