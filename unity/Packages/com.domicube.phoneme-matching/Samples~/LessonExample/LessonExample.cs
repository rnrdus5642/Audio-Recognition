using System.Collections;
using System.Collections.Generic;
using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using UnityEngine;

// ModelAsset lives in whichever inference package the project has. The
// assembly definition next to this file turns the installed one into a
// symbol, so the same source compiles on Unity 2022 and Unity 6.
#if DOMICUBE_INFERENCE_ENGINE
using Unity.InferenceEngine;
#elif DOMICUBE_SENTIS_1
using Unity.Sentis;
#endif

/// <summary>
/// A lesson that asks for one word at a time - README 의 방법 1, where
/// the package owns the microphone.
///
/// If the app already captures audio (voice chat, an SDK callback), do
/// NOT use this: opening the same device twice conflicts. Use
/// PronunciationSession with AudioWindowBuffer instead - README 방법
/// 2~4 - and hand the samples over.
///
/// Copy this into your project and replace the GUI with your own. What
/// matters is the order of operations, not the presentation:
///
///   1. build the recogniser once and warm it up while the scene loads
///   2. set the target word, then open the microphone
///   3. react to OnConfirmed / OnTimedOut
///   4. close the microphone when the lesson pauses
///
/// The microphone is NOT held open between questions. Listen() opens it,
/// and it closes on confirmation, on timeout, or on StopListening().
/// </summary>
[RequireComponent(typeof(PronunciationListener))]
public sealed class LessonExample : MonoBehaviour
{
#if DOMICUBE_INFERENCE_ENGINE || DOMICUBE_SENTIS_1
    [Tooltip("Assets 아래 둔 wav2vec2_ko.onnx")]
    public ModelAsset Model;
#endif

    [Tooltip("StreamingAssets 안의 CTC 어휘 파일")]
    public string VocabFileName = "wav2vec2_ko_vocab.json";

    [Tooltip("출제할 단어. targets.json 에 있는 것만 됩니다.")]
    public string[] Words = { "사과", "엄마", "토끼" };

    private PronunciationListener _listener;
    private SentisPhonemeRecognizer _recognizer;
    private int _index;
    private string _status = "준비 중...";
    private string _live = string.Empty;

    private void Awake()
    {
        _listener = GetComponent<PronunciationListener>();

#if DOMICUBE_INFERENCE_ENGINE || DOMICUBE_SENTIS_1
        var vocabPath = System.IO.Path.Combine(
            Application.streamingAssetsPath, VocabFileName);
        var vocab = PhonemeData.LoadCtcVocabulary(
            System.IO.File.ReadAllText(vocabPath));

        _recognizer = new SentisPhonemeRecognizer(Model, vocab);
        _listener.SetRecognizer(_recognizer);
#else
        // No inference package installed: the matching layer still works,
        // so the loop can be exercised with any IPhonemeRecognizer.
        _listener.SetRecognizer(new LoudnessStubRecognizer());
#endif

        // The word was heard. This is where the lesson moves on.
        _listener.OnConfirmed.AddListener((word, score) =>
        {
            _status = $"잘했어요! ({word}, {score:F2})";
            _index++;
            if (_index < Words.Length)
            {
                Invoke(nameof(AskCurrent), 1.5f);
            }
            else
            {
                _status = "모두 완료!";
            }
        });

        // Nothing was confirmed in time. There is no "wrong answer"
        // event: the matcher never claims the child said something else,
        // it only fails to recognise the target, which is the same thing
        // from the lesson's point of view.
        _listener.OnTimedOut.AddListener(() =>
        {
            _status = $"'{Words[_index]}' 다시 해볼까요?";
        });

        // Every hop, for a progress bar or a live meter. Optional.
        _listener.OnFrameScored.AddListener((word, score, streak) =>
        {
            _live = $"{word} {score:F2} ({streak}/{_listener.Consecutive})";
        });
    }

    private IEnumerator Start()
    {
        if (_recognizer != null)
        {
            // ~2 s of shader compilation and weight upload. Do it here,
            // not on the child's first word.
            _status = "준비 중...";
            yield return null;
            _recognizer.Warmup(_listener.WindowSeconds);
        }

        _status = "시작을 누르세요";
    }

    private void OnDestroy()
    {
        _recognizer?.Dispose();
    }

    /// <summary>Ask the current word and open the microphone.</summary>
    public void AskCurrent()
    {
        if (_index >= Words.Length)
        {
            return;
        }

        _listener.TargetWords = new[] { Words[_index] };
        _status = $"'{Words[_index]}' 라고 말해보세요";
        _live = string.Empty;
        _listener.Listen();
    }

    /// <summary>Close the microphone without waiting for a timeout.</summary>
    public void Stop()
    {
        _listener.StopListening();
        _status = "중지됨";
    }

    private void OnGUI()
    {
        GUILayout.BeginArea(new Rect(16, 16, 420, 200), GUI.skin.box);
        GUILayout.Label(_status, new GUIStyle(GUI.skin.label) { fontSize = 18 });
        GUILayout.Label(_live);
        GUILayout.Space(8);

        if (!_listener.IsListening && GUILayout.Button("시작 (마이크 켜기)"))
        {
            AskCurrent();
        }

        if (_listener.IsListening && GUILayout.Button("중지 (마이크 끄기)"))
        {
            Stop();
        }

        GUILayout.Label(_listener.IsListening
            ? "마이크: 켜짐"
            : "마이크: 꺼짐");
        GUILayout.EndArea();
    }
}
