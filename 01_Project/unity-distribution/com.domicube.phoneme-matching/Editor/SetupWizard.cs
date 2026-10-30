using System;
using System.Threading;
using System.Threading.Tasks;
using DomiCube.PhonemeMatching.Unity;
using UnityEditor;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Editor
{
    [InitializeOnLoad]
    public static class SetupWizard
    {
        internal const string ResumeKey = "DomiCube.AudioAnswer.ResumeSetup";
        static Task<bool> _task;
        static CancellationTokenSource _cancel;
        static volatile string _status;
        static volatile float _progress;
        static bool _reloadLocked;
        static bool _showingProgress;
        static double _lastReport;

        static SetupWizard()
        {
            EditorApplication.update += Tick;
            EditorApplication.quitting += Cancel;
            AssemblyReloadEvents.beforeAssemblyReload += ClearProgress;
            EditorApplication.playModeStateChanged += state =>
            {
                if (state == PlayModeStateChange.ExitingEditMode) Cancel();
            };
            if (SessionState.GetBool(ResumeKey, false)) EditorApplication.delayCall += () =>
            {
                SessionState.SetBool(ResumeKey, false);
                Run();
            };
        }

        [MenuItem("Tools/오디오 정답 처리 기능/초기 세팅", false, 0)]
        public static void Run()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                Debug.LogWarning("[오디오 정답 처리] Play 종료 후 초기 세팅을 실행하세요.");
                return;
            }
            if (_task != null)
            {
                ShowProgress();
                Debug.Log("[오디오 정답 처리] 초기 세팅이 진행 중입니다.");
                return;
            }
            if (!InferenceEngineInstaller.HasEngine)
            {
                SessionState.SetBool(ResumeKey, true);
                InferenceEngineInstaller.AddPackage();
                return;
            }
            try
            {
                AudioAnswerContent.Load();
                AudioAnswerBuildSetup.Configure();
                string destination = SharedSpeechModel.FilePath;
                _cancel = new CancellationTokenSource();
                var token = _cancel.Token;
                _status = "문서 폴더의 공용 모델 확인 중";
                _progress = 0;
                ShowProgress();
                EditorApplication.LockReloadAssemblies();
                _reloadLocked = true;
                _task = Task.Run(() => ModelFileInstaller.EnsureAsync(destination, SharedSpeechModel.Bytes,
                    SharedSpeechModel.Sha256, null, SharedSpeechModel.DownloadUrls,
                    (status, value) => { _status = status; _progress = value; }, token));
            }
            catch (Exception e)
            {
                Finish();
                Debug.LogError("[오디오 정답 처리] 초기 세팅 실패: " + e.Message);
            }
        }

        static void Cancel()
        {
            _cancel?.Cancel();
            ClearProgress();
        }

        static void ShowProgress()
        {
            if (_cancel == null || _cancel.IsCancellationRequested) return;
            _showingProgress = true;
            float value = Mathf.Clamp01(_progress);
            if (EditorUtility.DisplayCancelableProgressBar("오디오 초기 세팅",
                    Mathf.RoundToInt(value * 100) + "% · " + _status, value))
                Cancel();
        }

        internal static void ShowEngineProgress()
        {
            if (!SessionState.GetBool(ResumeKey, false)) return;
            if (_showingProgress && EditorApplication.timeSinceStartup - _lastReport < .1) return;
            _lastReport = EditorApplication.timeSinceStartup;
            _showingProgress = true;
            // UPM does not expose download percentages. Do not invent a percentage.
            EditorUtility.DisplayProgressBar("오디오 초기 세팅", "추론 엔진 설치 중 · 잠시 기다려 주세요", 0);
        }

        internal static void ClearProgress()
        {
            if (!_showingProgress) return;
            _showingProgress = false;
            EditorUtility.ClearProgressBar();
        }

        static void Tick()
        {
            if (_task == null) return;
            if (!_task.IsCompleted)
            {
                if (EditorApplication.timeSinceStartup - _lastReport > .1)
                {
                    ShowProgress();
                    _lastReport = EditorApplication.timeSinceStartup;
                }
                return;
            }
            try
            {
                bool installed = _task.GetAwaiter().GetResult();
                Finish();
                Debug.Log("[오디오 정답 처리] 초기 세팅 완료 · " + (installed ? "공용 모델 설치 완료" : "기존 공용 모델 사용")
                    + "\n" + SharedSpeechModel.FilePath);
            }
            catch (OperationCanceledException)
            {
                Finish();
                Debug.Log("[오디오 정답 처리] 초기 세팅 취소. 메뉴에서 다시 실행할 수 있습니다.");
            }
            catch (Exception e)
            {
                Finish();
                Debug.LogError("[오디오 정답 처리] 초기 세팅 실패: " + e.Message
                    + "\n네트워크와 다른 프로젝트의 설치 상태를 확인한 뒤 다시 실행하세요.");
            }
        }

        static void Finish()
        {
            ClearProgress();
            _task = null;
            _cancel?.Dispose();
            _cancel = null;
            if (_reloadLocked)
            {
                _reloadLocked = false;
                EditorApplication.UnlockReloadAssemblies();
            }
        }
    }

    // Close any setup window serialized by previews 1–3 after the package reloads.
    // Retained only for compatibility; initial setup no longer opens an EditorWindow.
    public sealed class AudioAnswerSetupWindow : EditorWindow
    {
        public static void Open() => SetupWizard.Run();
        void OnEnable() => EditorApplication.delayCall += () => { if (this != null) Close(); };
    }
}
