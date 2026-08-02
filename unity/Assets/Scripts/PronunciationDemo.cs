using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using UnityEngine;

/// <summary>
/// Bring-up scene: verifies microphone -> rolling window -> matching ->
/// confirm -> auto-stop, using a stub recogniser.
///
/// Any loud sound counts as the answer here, so this proves the plumbing
/// and the timing, NOT that pronunciation matching works. Swap in a real
/// IPhonemeRecognizer for that.
///
/// Put this on an empty GameObject and press Play. Space starts
/// listening.
/// </summary>
[RequireComponent(typeof(PronunciationListener))]
public sealed class PronunciationDemo : MonoBehaviour
{
    [Tooltip("자동으로 듣기 시작")]
    public bool ListenOnStart = true;

    private PronunciationListener _listener;
    private string _status = "대기 중 — Space를 눌러 시작";
    private string _lastFrame = string.Empty;

    private void Awake()
    {
        _listener = GetComponent<PronunciationListener>();
        _listener.SetRecognizer(new LoudnessStubRecognizer());

        _listener.OnConfirmed.AddListener((word, score) =>
        {
            _status = $"확정: {word}  (점수 {score:F3})";
            Debug.Log($"[Demo] {_status}");
        });

        _listener.OnFrameScored.AddListener((word, score, streak) =>
        {
            _lastFrame =
                $"{(string.IsNullOrEmpty(word) ? "—" : word)}  "
                + $"{score:F3}  연속 {streak}/{_listener.Consecutive}";
        });
    }

    private void Start()
    {
        if (Microphone.devices.Length == 0)
        {
            _status = "마이크를 찾을 수 없습니다";
            Debug.LogError("[Demo] " + _status);
            return;
        }

        Debug.Log($"[Demo] 마이크: {Microphone.devices[0]}");

        if (ListenOnStart)
        {
            StartListening();
        }
    }

    private void Update()
    {
        // Old input manager: works without extra setup in a fresh
        // project. Swap for the Input System if the project moves to it.
        if (Input.GetKeyDown(KeyCode.Space) && !_listener.IsListening)
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
        GUI.Label(new Rect(pad + 12, pad + 92, 540, 24),
            "stub recognizer — 소리 크기만 봅니다 (발음 판정 아님)", small);
    }
}
