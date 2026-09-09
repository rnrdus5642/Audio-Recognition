#if UNITY_5_3_OR_NEWER
using System;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Unity
{
    /// <summary>Optional display evidence. Missing evidence withholds guidance, never an answer.</summary>
    public static class ModelFeedbackResources
    {
        private const string ChildPath = "DomiCube/PhonemeMatching/Feedback/reference_errors_child";

        public static ModelErrorProfile Load(string matrixId)
        {
            if (matrixId != "ko_child_v2") return null;
            var asset = Resources.Load<TextAsset>(ChildPath);
            if (asset == null) return null;
            try
            {
                var profile = ModelErrorProfile.FromJson(asset.text);
                return profile.MatrixId == matrixId ? profile : null;
            }
            catch (Exception e)
            {
                Debug.LogWarning("[PhonemeMatching] 모델 오류표를 읽지 못해 발음 안내를 보류합니다: " + e.Message);
                return null;
            }
        }
    }
}
#endif
