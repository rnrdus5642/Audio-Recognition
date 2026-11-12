using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Editor
{
    public static class OrthographicModelInstaller
    {
        public const string ResourcePath = "Models/hubert_ko_child_jamo";
        public const string Destination = "Assets/Resources/Models/hubert_ko_child_jamo.sentis";

        [MenuItem("Tools/Phoneme Matching/원문 자모 아동 모델 파일에서 가져오기…", false, 34)]
        public static void Import()
        {
#if !UNITY_6000_0_OR_NEWER
            EditorUtility.DisplayDialog("원문 자모 모델", "이번 .sentis 모델은 Unity 6 + Inference Engine 2.6.1용입니다.", "확인");
#else
            if (!InferenceEngineInstaller.HasEngine)
            {
                EditorUtility.DisplayDialog("추론 엔진 설치 필요",
                    "Tools > Phoneme Matching > 추론 엔진 설치를 먼저 실행하고 컴파일 완료 후 다시 가져오세요.", "확인");
                return;
            }
            string source = EditorUtility.OpenFilePanel("hubert_ko_child_jamo.sentis", "", "sentis");
            if (string.IsNullOrEmpty(source)) return;
            if (string.Equals(Path.GetFullPath(source), Path.GetFullPath(Destination), StringComparison.OrdinalIgnoreCase)) return;
            if (File.Exists(Destination) && !EditorUtility.DisplayDialog("원문 자모 모델 교체",
                    "같은 경로의 모델을 교체합니다. 게임 코드와 모델 에셋 참조는 유지됩니다.", "교체", "취소")) return;
            string temporary = Destination + "." + Guid.NewGuid().ToString("N") + ".copying";
            try
            {
                if (new FileInfo(source).Length < 1024) throw new InvalidDataException("모델 파일이 너무 작습니다.");
                if (!DataInstaller.InstallMissing()) return;
                Directory.CreateDirectory(Path.GetDirectoryName(Destination));
                EditorUtility.DisplayProgressBar("원문 자모 모델", "모델 파일을 복사하고 있습니다.", .3f);
                File.Copy(source, temporary, overwrite: false);
                if (File.Exists(Destination)) File.Replace(temporary, Destination, null);
                else File.Move(temporary, Destination);
                // The existing .meta is deliberately retained during later weight swaps.
                AssetDatabase.ImportAsset(Destination, ImportAssetOptions.ForceSynchronousImport);
                Debug.Log("[PhonemeMatching] 원문 자모 모델 설치 완료. 발음 테스트 (마이크)에서 "
                          + "hubert_ko_child_jamo 모델을 선택하세요. 새 모델의 판정 임계값은 실제 발화로 확인해야 합니다.");
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog("모델 설치 실패", exception.Message, "확인");
            }
            finally
            {
                EditorUtility.ClearProgressBar();
                if (File.Exists(temporary)) File.Delete(temporary);
            }
#endif
        }
    }
}
