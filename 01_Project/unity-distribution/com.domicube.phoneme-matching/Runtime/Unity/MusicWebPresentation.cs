#if UNITY_5_3_OR_NEWER
using System;
using System.Linq;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Unity
{
    /// <summary>Music's 82 stage expressions with the shared web v4 decision policy.</summary>
    public static class MusicWebPresentation
    {
        public const string ResourceFolder = "DomiCube/PhonemeMatching/MusicWebV4/";
        public const string ModelResourcePath = "Models/hubert_ko_child_jamo_sentis1";
        public const string TargetsFile = "targets_music_web_v4.json";
        public const string VocabularyFile = "hubert_ko_child_jamo_vocab.json";

        public static MusicPresentationProfile Load() => MusicPresentationProfile.FromJson(
            Read("ko_child_v2.json"), ProjectTargets(), Read(VocabularyFile));

        public static void ApplyTo(PronunciationListener listener, int stage = 1)
        {
            if (listener == null) throw new ArgumentNullException(nameof(listener));
            var selected = Load().GetStage(stage);
            listener.ConfigureOrthographicWordPolicy();
            listener.LoadData(Read("ko_child_v2.json"), ProjectTargets());
            listener.TargetWords = selected.Answers.ToArray();
        }

        private static string ProjectTargets() => Resources.Load<TextAsset>(AudioAnswerContent.ConfigurationResource) != null
            ? AudioAnswerContent.PrepareTargets(AudioAnswerContent.ReadTargets()) : Read(TargetsFile);

        private static string Read(string fileName)
        {
            var path = ResourceFolder + System.IO.Path.GetFileNameWithoutExtension(fileName);
            var asset = Resources.Load<TextAsset>(path);
            if (asset == null) throw new InvalidOperationException("Missing Music web v4 data: " + path);
            return asset.text;
        }
    }
}
#endif
