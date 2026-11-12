using System;
using System.Collections;
using System.IO;
using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// Unity 6 integration of the current web word policy. No UI or server required.
/// Attach to an object, assign the HuBERT ModelAsset, and connect the listener's
/// OnTextRecognized / OnConfirmed events to the game's existing UI and progression.
/// Call StartListening after OnReady; call Begin with new candidates each step.
/// </summary>
[RequireComponent(typeof(PronunciationListener))]
public sealed class WebWordExample : MonoBehaviour
{
    public ModelAsset Model;
    public string[] Candidates = { "들어", "들어요", "들까?" };
    [Range(.7f, 1f)] public float Threshold = .85f;
    [Range(2, 4)] public int Confirmations = 2;
    public UnityEvent OnReady = new UnityEvent();

    public bool IsReady { get; private set; }
    public PronunciationListener Listener { get; private set; }
    private SentisPhonemeRecognizer _recognizer;

    private IEnumerator Start()
    {
        Listener = GetComponent<PronunciationListener>();
        // Allow the application's loading UI to render before synchronous warmup.
        yield return null;
        try
        {
            Listener.ConfigureOrthographicWordPolicy(Threshold, Confirmations);
            Listener.LoadData();
            var vocabulary = PhonemeData.LoadCtcVocabulary(File.ReadAllText(Path.Combine(
                Application.streamingAssetsPath, "hubert_ko_child_jamo_vocab.json")));
            _recognizer = new SentisPhonemeRecognizer(Model, vocabulary);
            _recognizer.Warmup(2.5f);
            Listener.SetRecognizer(_recognizer);
            IsReady = true;
        }
        catch (Exception error)
        {
            _recognizer?.Dispose();
            _recognizer = null;
            Debug.LogException(error, this);
        }
        if (IsReady) OnReady.Invoke();
    }

    /// <summary>Starts one question using the current Candidates. No overall timeout.</summary>
    public void StartListening()
    {
        if (!IsReady || !isActiveAndEnabled)
        {
            Debug.LogWarning("모델 준비가 끝난 뒤 활성화된 WebWordExample에서 시작하세요.", this);
            return;
        }
        Listener.TargetWords = Candidates == null ? null : (string[])Candidates.Clone();
        Listener.Listen();
    }

    /// <summary>Switches to the next game's candidate set, discarding previous audio/evidence.</summary>
    public void Begin(string[] candidates)
    {
        Candidates = candidates == null ? null : (string[])candidates.Clone();
        StartListening();
    }

    public void StopListening() => Listener?.StopListening();
    private void OnDisable() => StopListening();
    private void OnDestroy()
    {
        StopListening();
        if (Listener != null) Listener.SetRecognizer(null);
        _recognizer?.Dispose();
        IsReady = false;
    }
}
