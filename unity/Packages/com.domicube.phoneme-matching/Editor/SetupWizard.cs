using UnityEditor;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Editor
{
    /// <summary>
    /// One button that does whatever is still missing.
    ///
    /// The three install steps have to be separate menu items because
    /// installing the inference engine recompiles the project, which throws
    /// away whatever was running. So the wizard writes down where it got to
    /// and picks up again after the reload.
    ///
    /// Skips what is already there, which makes it the right button in both
    /// situations: a fresh project runs all three, and a teammate who cloned
    /// a project that already uses this only gets the model download.
    /// </summary>
    [InitializeOnLoad]
    public static class SetupWizard
    {
        private const string StepKey = "DomiCube.PhonemeMatching.SetupStep";

        private const int Idle = 0;
        private const int WaitingForEngine = 1;

        static SetupWizard()
        {
            if (SessionState.GetInt(StepKey, Idle) != WaitingForEngine)
            {
                return;
            }

            // Let the reload finish before touching the package manager.
            EditorApplication.delayCall += ResumeAfterEngine;
        }

        [MenuItem("Tools/Phoneme Matching/초기 세팅", false, 0)]
        public static void Run()
        {
            if (!InferenceEngineInstaller.HasEngine)
            {
                if (!Confirm(engineNeeded: true))
                {
                    return;
                }

                Debug.Log(
                    "[PhonemeMatching] 1/3 추론 엔진 설치 중 "
                    + $"({InferenceEngineInstaller.EngineName}). 컴파일이 "
                    + "끝나면 이어서 진행합니다.");

                SessionState.SetInt(StepKey, WaitingForEngine);
                InferenceEngineInstaller.AddPackage();
                return;
            }

            if (DataInstaller.IsInstalled && ModelDownloader.IsInstalled)
            {
                EditorUtility.DisplayDialog(
                    "Phoneme Matching",
                    "이미 다 설치돼 있습니다.\n\n"
                    + "Tools > Phoneme Matching > 발음 테스트 (마이크) 로 "
                    + "확인해보세요.",
                    "확인");
                return;
            }

            if (!Confirm(engineNeeded: false))
            {
                return;
            }

            RunRemaining();
        }

        private static void ResumeAfterEngine()
        {
            SessionState.SetInt(StepKey, Idle);

            if (!InferenceEngineInstaller.HasEngine)
            {
                // Either the download failed or the user cancelled it; the
                // installer has already logged why.
                Debug.LogError(
                    "[PhonemeMatching] 추론 엔진 설치가 끝나지 않아 초기 "
                    + "세팅을 멈췄습니다. Tools > Phoneme Matching > "
                    + "초기 세팅 으로 다시 시작하세요.");
                return;
            }

            Debug.Log("[PhonemeMatching] 1/3 추론 엔진 설치 완료.");
            RunRemaining();
        }

        private static void RunRemaining()
        {
            if (!DataInstaller.IsInstalled)
            {
                Debug.Log("[PhonemeMatching] 2/3 데이터 파일 설치 중…");
                if (!DataInstaller.Install(askBeforeOverwriting: false))
                {
                    Debug.LogError("[PhonemeMatching] 2/3 실패. 위 오류를 보세요.");
                    return;
                }
            }

            if (!ModelDownloader.IsInstalled)
            {
                Debug.Log("[PhonemeMatching] 3/3 음향 모델 내려받는 중…");
                if (!ModelDownloader.Download(askFirst: false))
                {
                    Debug.LogError("[PhonemeMatching] 3/3 실패. 위 오류를 보세요.");
                    return;
                }
            }

            Debug.Log(
                "[PhonemeMatching] 초기 세팅 완료. "
                + "Tools > Phoneme Matching > 발음 테스트 (마이크) 로 "
                + "확인해보세요.");
        }

        private static bool Confirm(bool engineNeeded)
        {
            string steps = "";
            if (engineNeeded)
            {
                steps += $"· 추론 엔진 설치 ({InferenceEngineInstaller.EngineName})\n";
            }

            if (!DataInstaller.IsInstalled)
            {
                steps += "· 데이터 파일 설치 (JSON 3개)\n";
            }

            if (!ModelDownloader.IsInstalled)
            {
                steps += "· 음향 모델 내려받기 (약 1.18GB, 몇 분 걸립니다)\n";
            }

            return EditorUtility.DisplayDialog(
                "Phoneme Matching",
                "없는 것만 차례대로 설치합니다.\n\n" + steps
                + "\n이미 있는 것은 건너뜁니다.",
                "시작", "취소");
        }
    }
}
