using DomiCube.PhonemeMatching.Unity;
using UnityEditor;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Editor
{
    [InitializeOnLoad]
    public static class MusicPresentationSetup
    {
        private const string WaitingKey = "DomiCube.PhonemeMatching.MusicSetupWaiting";

        static MusicPresentationSetup()
        {
            if (SessionState.GetBool(WaitingKey, false))
                EditorApplication.delayCall += Resume;
        }

        [MenuItem("Tools/Phoneme Matching/악기놀이 발표 설정", false, 1)]
        public static void Run()
        {
            bool engine = !InferenceEngineInstaller.HasEngine;
            bool model = !ModelDownloader.ChildIsInstalled;
            if ((engine || model) && !EditorUtility.DisplayDialog(
                "악기놀이 발표 설정",
                "12개 정답의 발표 설정과 9단계는 패키지에 포함되어 있습니다.\n\n"
                + (engine ? $"· 추론 엔진 설치: {InferenceEngineInstaller.EngineName}\n" : "")
                + (model ? "· 아동 음향 모델 받기: 약 1.18GB (최초 한 번)\n" : "")
                + "\n준비가 끝나면 아동 모델과 발표 설정을 연결한 테스트 창을 엽니다.",
                "시작", "취소")) return;

            if (engine)
            {
                SessionState.SetBool(WaitingKey, true);
                InferenceEngineInstaller.AddPackage();
                return;
            }
            Finish();
        }

        private static void Resume()
        {
            SessionState.SetBool(WaitingKey, false);
            if (!InferenceEngineInstaller.HasEngine)
            {
                Debug.LogError("[MusicPresentation] 추론 엔진 설치가 완료되지 않았습니다. 악기놀이 발표 설정을 다시 실행하세요.");
                return;
            }
            Finish();
        }

        private static void Finish()
        {
            // Read package resources directly; never overwrite project JSONs.
            var profile = MusicPresentation.Load();
            if (!ModelDownloader.EnsureChild()) return;
            AssetDatabase.Refresh();
            PronunciationTestWindow.OpenMusicDemo();
            Debug.Log($"[MusicPresentation] {profile.ProfileId} · 12개 정답 / 9단계 적용. "
                + "게임에서는 GameObject > Phoneme Matching > 악기놀이 발표를 추가하세요.");
        }

        [MenuItem("GameObject/Phoneme Matching/악기놀이 발표", false, 10)]
        public static void CreateLesson()
        {
            var go = new GameObject("악기놀이 발표");
            Undo.RegisterCreatedObjectUndo(go, "악기놀이 발표 추가");
            Undo.AddComponent<MusicPresentationLesson>(go);
            Selection.activeGameObject = go;
        }
    }
}
