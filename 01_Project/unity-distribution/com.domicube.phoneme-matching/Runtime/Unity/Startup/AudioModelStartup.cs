using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;

namespace DomiCube.PhonemeMatching.Unity
{
    [Serializable]
    public sealed class AudioAnswerLaunchConfiguration
    {
        public const string ResourcePath = "CRC/AudioAnswerLaunch";
        public string gameScene;
    }

    /// <summary>A separate first scene keeps game initialization behind model installation.</summary>
    public sealed class AudioModelStartup : MonoBehaviour
    {
        sealed class InstallProgress
        {
            public readonly string Message;
            public readonly float Value;
            public InstallProgress(string message, float value) { Message = message; Value = value; }
        }
        volatile InstallProgress _progress = new InstallProgress("문서 폴더의 공용 모델 확인 중", 0);
        Task<bool> _task;
        CancellationTokenSource _cancel;
        AsyncOperation _loading;
        string _gameScene;
        bool _quitRequested;
        float _lastPaint;
        TextMeshProUGUI _message, _percent, _actionText;
        UnityEngine.UI.Button _action;
        RectTransform _fill;
        TMP_FontAsset _font;
        Font _sourceFont;
        bool _ownsFont;

        void Start()
        {
            CreateView();
            Application.wantsToQuit += WantsToQuit;
            try
            {
                var asset = Resources.Load<TextAsset>(AudioAnswerLaunchConfiguration.ResourcePath);
                if (asset == null) throw new InvalidOperationException("게임 시작 설정이 없습니다. 초기 세팅 후 다시 빌드하세요.");
                var config = JsonUtility.FromJson<AudioAnswerLaunchConfiguration>(asset.text);
                _gameScene = config?.gameScene;
                if (string.IsNullOrEmpty(_gameScene) || _gameScene == SceneManager.GetActiveScene().path
                    || !Application.CanStreamedLevelBeLoaded(_gameScene))
                    throw new InvalidOperationException("게임 시작 씬을 찾을 수 없습니다. 초기 세팅 후 다시 빌드하세요.");
                var check = System.Diagnostics.Stopwatch.StartNew();
                bool ready = SharedSpeechModel.HasVerificationReceipt(SharedSpeechModel.FilePath,
                    SharedSpeechModel.Bytes, SharedSpeechModel.Sha256);
                if (ready)
                {
                    Debug.Log("[오디오 정답 처리] 기존 모델 설치 확인 " + check.Elapsed.TotalMilliseconds.ToString("F1")
                        + " ms · 전체 해시 검사 생략");
                    LoadGame();
                }
                else BeginInstallation();
            }
            catch (Exception e) { Fail(e.Message); }
        }

        void BeginInstallation()
        {
            if (_task != null || _loading != null) return;
            if (string.IsNullOrEmpty(_gameScene) || !Application.CanStreamedLevelBeLoaded(_gameScene))
            { Fail("게임 시작 설정을 확인한 뒤 다시 빌드하세요."); return; }
            try
            {
                string destination = SharedSpeechModel.FilePath;
                _cancel?.Dispose();
                _cancel = new CancellationTokenSource();
                var token = _cancel.Token;
                _progress = new InstallProgress("문서 폴더의 공용 모델 확인 중", 0);
                _action.interactable = true;
                _actionText.text = "취소";
                Paint(_progress);
                _task = Task.Run(() => SharedModelInstaller.EnsureAsync(destination, SharedSpeechModel.Bytes,
                    SharedSpeechModel.Sha256, null, SharedSpeechModel.DownloadUrls,
                    (message, value) => _progress = new InstallProgress(message, value), token));
            }
            catch (Exception e) { Fail(e.Message); }
        }

        void Update()
        {
            if (_loading != null)
            {
                Paint(new InstallProgress("게임 불러오는 중", Mathf.Clamp01(_loading.progress / .9f)));
                return;
            }
            if (_task == null) return;
            if (!_task.IsCompleted)
            {
                if (!_cancel.IsCancellationRequested && Time.unscaledTime - _lastPaint >= .1f)
                { Paint(_progress); _lastPaint = Time.unscaledTime; }
                return;
            }
            var completed = _task;
            _task = null;
            try
            {
                bool installed = completed.GetAwaiter().GetResult();
                _cancel.Dispose(); _cancel = null;
                if (_quitRequested) { Application.Quit(); return; }
                Debug.Log("[오디오 정답 처리] " + (installed ? "공용 모델 설치 완료" : "기존 공용 모델 사용"));
                LoadGame();
            }
            catch (OperationCanceledException)
            {
                _cancel?.Dispose(); _cancel = null;
                if (_quitRequested) Application.Quit();
                else Fail("설치를 취소했습니다. 다시 시도하면 설치를 시작합니다.");
            }
            catch (Exception e)
            {
                _cancel?.Dispose(); _cancel = null;
                if (_quitRequested) Application.Quit();
                else Fail("모델 준비에 실패했습니다. 네트워크·저장 공간·다른 앱의 설치 상태를 확인하세요.\n" + e.Message);
            }
        }

        void LoadGame()
        {
            SharedSpeechModel.RequireInstalled();
            _action.interactable = false;
            _actionText.text = "준비 완료";
            Paint(new InstallProgress("게임 불러오는 중", 0));
            _loading = SceneManager.LoadSceneAsync(_gameScene, LoadSceneMode.Single);
            if (_loading == null) throw new InvalidOperationException("게임 씬을 불러오지 못했습니다.");
        }

        void ActionClicked()
        {
            if (_task == null) { BeginInstallation(); return; }
            _cancel.Cancel();
            _action.interactable = false;
            _message.text = "설치 취소 중 · 잠시 기다려 주세요";
        }

        bool WantsToQuit()
        {
            if (_task == null || _task.IsCompleted) return true;
            _quitRequested = true;
            _cancel.Cancel();
            _action.interactable = false;
            _message.text = "설치를 정리하고 종료하는 중";
            return false;
        }

        void Fail(string message)
        {
            Debug.LogWarning("[오디오 정답 처리] " + message);
            _message.text = message;
            _percent.text = "";
            _actionText.text = "다시 시도";
            _action.interactable = true;
        }

        void Paint(InstallProgress progress)
        {
            float value = Mathf.Clamp01(progress.Value);
            _message.text = progress.Message;
            _percent.text = Mathf.RoundToInt(value * 100) + "%";
            _fill.anchorMax = new Vector2(value, 1);
        }

        void OnDestroy()
        {
            Application.wantsToQuit -= WantsToQuit;
            _cancel?.Cancel();
            var cancellation = _cancel;
            if (_task != null && !_task.IsCompleted)
                _task.ContinueWith(t => { var observed = t.Exception; cancellation?.Dispose(); }, TaskScheduler.Default);
            else _cancel?.Dispose();
            if (_ownsFont && _font != null)
            {
                foreach (var texture in _font.atlasTextures) if (texture != null) Destroy(texture);
                if (_font.material != null) Destroy(_font.material);
                Destroy(_font);
            }
            if (_sourceFont != null) Destroy(_sourceFont);
        }

        void CreateView()
        {
            // A font file path provides font data to TextCore in Windows Players.
            string koreanFont = Font.GetPathsToOSFonts().FirstOrDefault(path =>
                string.Equals(Path.GetFileName(path), "malgun.ttf", StringComparison.OrdinalIgnoreCase));
            _sourceFont = koreanFont != null ? new Font(koreanFont)
                : Font.CreateDynamicFontFromOSFont(new[] { "Malgun Gothic", "Arial" }, 28);
            _font = TMP_FontAsset.CreateFontAsset(_sourceFont);
            _ownsFont = _font != null;
            if (_font == null) _font = TMP_Settings.defaultFontAsset;
            var canvasObject = new GameObject("AudioModelSetupCanvas", typeof(RectTransform), typeof(Canvas),
                typeof(UnityEngine.UI.CanvasScaler), typeof(UnityEngine.UI.GraphicRaycaster));
            canvasObject.transform.SetParent(transform, false);
            canvasObject.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasObject.GetComponent<UnityEngine.UI.CanvasScaler>();
            scaler.uiScaleMode = UnityEngine.UI.CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080); scaler.matchWidthOrHeight = .5f;
            var backdrop = Rect("Backdrop", canvasObject.transform, Vector2.zero, Vector2.zero);
            backdrop.anchorMin = Vector2.zero; backdrop.anchorMax = Vector2.one;
            backdrop.gameObject.AddComponent<UnityEngine.UI.Image>().color = new Color(.055f, .075f, .12f);
            var panel = Rect("Setup", backdrop, new Vector2(820, 380), Vector2.zero);
            panel.gameObject.AddComponent<UnityEngine.UI.Image>().color = new Color(.11f, .15f, .22f);
            Label("Title", panel, "오디오 초기 세팅", 34, new Vector2(740, 54), new Vector2(0, 138));
            _message = Label("Status", panel, "공용 모델 확인 중", 23, new Vector2(740, 106), new Vector2(0, 52));
            _message.overflowMode = TextOverflowModes.Ellipsis;
            var track = Rect("ProgressTrack", panel, new Vector2(700, 18), new Vector2(0, -30));
            track.gameObject.AddComponent<UnityEngine.UI.Image>().color = new Color(.2f, .25f, .32f);
            _fill = Rect("Progress", track, Vector2.zero, Vector2.zero);
            _fill.anchorMin = Vector2.zero; _fill.anchorMax = new Vector2(0, 1);
            _fill.gameObject.AddComponent<UnityEngine.UI.Image>().color = new Color(.2f, .68f, .9f);
            _percent = Label("Percentage", panel, "0%", 24, new Vector2(700, 38), new Vector2(0, -64));
            _action = Button(panel, "취소", new Vector2(-110, -135), ActionClicked, out _actionText);
            Button(panel, "종료", new Vector2(110, -135), () => Application.Quit(), out _);
            new GameObject("SetupEventSystem", typeof(EventSystem), typeof(StandaloneInputModule)).transform.SetParent(transform, false);
        }

        static RectTransform Rect(string name, Transform parent, Vector2 size, Vector2 position)
        {
            var value = new GameObject(name, typeof(RectTransform)).GetComponent<RectTransform>();
            value.SetParent(parent, false); value.anchorMin = value.anchorMax = value.pivot = new Vector2(.5f, .5f);
            value.sizeDelta = size; value.anchoredPosition = position; return value;
        }

        TextMeshProUGUI Label(string name, Transform parent, string text, int size, Vector2 dimensions, Vector2 position)
        {
            var label = Rect(name, parent, dimensions, position).gameObject.AddComponent<TextMeshProUGUI>();
            label.font = _font; label.fontSize = size; label.text = text; label.color = Color.white;
            label.alignment = TextAlignmentOptions.Center; label.raycastTarget = false; return label;
        }

        UnityEngine.UI.Button Button(Transform parent, string text, Vector2 position, UnityEngine.Events.UnityAction action,
            out TextMeshProUGUI label)
        {
            var rect = Rect(text, parent, new Vector2(190, 54), position);
            var background = rect.gameObject.AddComponent<UnityEngine.UI.Image>(); background.color = new Color(.16f, .38f, .57f);
            var button = rect.gameObject.AddComponent<UnityEngine.UI.Button>(); button.targetGraphic = background;
            button.onClick.AddListener(action); label = Label("Text", rect, text, 24, new Vector2(182, 50), Vector2.zero);
            return button;
        }
    }
}
