using UnityEditor;
using UnityEditor.PackageManager;
using UnityEditor.PackageManager.Requests;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Editor
{
    /// <summary>
    /// Installs the inference engine this editor can actually run.
    ///
    /// The package cannot declare it: UPM dependencies are a single list
    /// with no way to say "Sentis 1.2 on 2022, Inference Engine 2.x on
    /// Unity 6", and naming either one breaks the other editor. So the
    /// package ships without an engine and this puts it one click away
    /// rather than making everyone look up a version string.
    ///
    /// Deliberately not automatic. Adding packages to someone's project
    /// without asking is rude, and a project may be pinned to a
    /// particular version for reasons this script cannot see.
    /// </summary>
    [InitializeOnLoad]
    public static class InferenceEngineInstaller
    {
#if UNITY_6000_0_OR_NEWER
        private const string PackageId = "com.unity.ai.inference@2.6.1";
        private const string DisplayName = "Sentis (com.unity.ai.inference) 2.6.1";
#else
        private const string PackageId = "com.unity.sentis@1.2.0-exp.2";
        private const string DisplayName = "Sentis 1.2.0-exp.2";
#endif

        private const string NoticeKey =
            "DomiCube.PhonemeMatching.EngineNoticeShown";

        private static AddRequest _request;

        static InferenceEngineInstaller()
        {
            // Once per project, not once per recompile.
            if (HasEngine || SessionState.GetBool(NoticeKey, false))
            {
                return;
            }

            SessionState.SetBool(NoticeKey, true);
            Debug.LogWarning(
                "[PhonemeMatching] 추론 엔진이 없어 음향 모델을 실행할 수 "
                + "없습니다. 매칭 계층은 그대로 동작합니다.\n"
                + $"설치: Tools > Phoneme Matching > 추론 엔진 설치 "
                + $"({DisplayName})");
        }

        private static bool HasEngine
        {
            get
            {
#if DOMICUBE_INFERENCE_ENGINE || DOMICUBE_SENTIS_1
                return true;
#else
                return false;
#endif
            }
        }

        [MenuItem("Tools/Phoneme Matching/추론 엔진 설치", false, 30)]
        public static void Install()
        {
            if (HasEngine)
            {
                EditorUtility.DisplayDialog(
                    "Phoneme Matching",
                    "추론 엔진이 이미 설치돼 있습니다.",
                    "확인");
                return;
            }

            if (!EditorUtility.DisplayDialog(
                    "Phoneme Matching",
                    $"이 에디터에 맞는 추론 엔진을 설치합니다.\n\n{DisplayName}\n\n"
                    + "Package Manager 가 다운로드하고 프로젝트를 다시 "
                    + "컴파일합니다.",
                    "설치", "취소"))
            {
                return;
            }

            _request = Client.Add(PackageId);
            EditorApplication.update += Poll;
        }

        [MenuItem("Tools/Phoneme Matching/추론 엔진 설치", true)]
        private static bool InstallEnabled()
        {
            return _request == null;
        }

        private static void Poll()
        {
            if (_request == null || !_request.IsCompleted)
            {
                return;
            }

            EditorApplication.update -= Poll;

            if (_request.Status == StatusCode.Success)
            {
                Debug.Log(
                    $"[PhonemeMatching] {_request.Result.displayName} "
                    + $"{_request.Result.version} 설치 완료. 컴파일이 끝나면 "
                    + "Tools > Phoneme Matching > 발음 테스트 (마이크) 로 "
                    + "확인하세요.");
            }
            else
            {
                Debug.LogError(
                    $"[PhonemeMatching] {PackageId} 설치 실패: "
                    + $"{_request.Error?.message}");
            }

            _request = null;
        }
    }
}
