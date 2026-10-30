#if UNITY_5_3_OR_NEWER
using System;
using System.Linq;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Unity
{
    /// <summary>
    /// Load the policy directly from the installed package. Old or customized
    /// StreamingAssets files do not replace the presentation profile.
    /// </summary>
    public static class MusicPresentation
    {
        public static MusicPresentationProfile Load() => MusicPresentationProfile.FromJson(
            Read(MusicPresentationProfile.MatrixFile),
            Read(MusicPresentationProfile.TargetsFile),
            Read(MusicPresentationProfile.VocabularyFile), ModelFeedbackResources.Load("ko_child_v2"));

        public static void ApplyTo(PronunciationListener listener, int stage = 1)
        {
            if (listener == null) throw new ArgumentNullException(nameof(listener));
            var profile = Load();
            var selected = profile.GetStage(stage); // Validate before stopping an existing question.
            listener.StopListening();
            listener.LoadData(Read(MusicPresentationProfile.MatrixFile), Read(MusicPresentationProfile.TargetsFile));
            listener.MatrixFileName = MusicPresentationProfile.MatrixFile;
            listener.TargetsFileName = MusicPresentationProfile.TargetsFile;
            listener.WindowSeconds = MusicPresentationProfile.WindowSeconds;
            listener.HopSeconds = MusicPresentationProfile.HopSeconds;
            listener.Consecutive = MusicPresentationProfile.Consecutive;
            listener.TargetWords = selected.Answers.ToArray();
        }

        private static string Read(string fileName)
        {
            string path = MusicPresentationProfile.ResourceFolder
                + System.IO.Path.GetFileNameWithoutExtension(fileName);
            var asset = Resources.Load<TextAsset>(path);
            if (asset == null)
                throw new InvalidOperationException(
                    $"패키지의 악기놀이 데이터가 없습니다: {path}. 패키지를 업데이트하세요.");
            return asset.text;
        }
    }
}
#endif
