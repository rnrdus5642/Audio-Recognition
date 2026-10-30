using System;
using System.Linq;
using DomiCube.PhonemeMatching.Korean;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Unity
{
    public static class AudioAnswerContent
    {
        public const string ConfigurationResource = "CRC/AudioAnswerProject";
        public static string TargetsResourcePath => (string)Configuration()["targetsResourcePath"];
        public static string DisplayName => (string)Configuration()["displayName"];
        static string _source, _prepared;
        static MusicPresentationProfile _profile;

        static JObject Configuration()
        {
            var asset = Resources.Load<TextAsset>(ConfigurationResource);
            if (asset == null) throw new InvalidOperationException("프로젝트의 음성 정답 설정이 없습니다: " + ConfigurationResource);
            var config = JObject.Parse(asset.text);
            if (string.IsNullOrWhiteSpace((string)config["targetsResourcePath"]))
                throw new InvalidOperationException("정답 목록 경로가 없습니다.");
            return config;
        }
        public static string ReadTargets()
        {
            var asset = Resources.Load<TextAsset>(TargetsResourcePath);
            if (asset == null) throw new InvalidOperationException("정답 목록이 없습니다: " + TargetsResourcePath);
            return asset.text;
        }
        public static MusicPresentationProfile Load()
        {
            string source = ReadTargets();
            if (_profile != null && source == _source) return _profile;
            _prepared = PrepareTargets(source);
            _profile = Validate(source);
            _source = source;
            return _profile;
        }
        public static string PrepareTargets(string source)
        {
            var doc = JObject.Parse(source);
            foreach (var answer in (JArray)doc["answers"])
                answer["phonemes"] = new JArray(JamoIpa.ToPhonemes(Rules.Apply((string)answer["text"])));
            return doc.ToString(Formatting.None);
        }
        public static MusicPresentationProfile Validate(string source) => MusicPresentationProfile.FromJson(
            ReadShared("ko_child_v2"), PrepareTargets(source), ReadShared("hubert_ko_child_jamo_vocab"));
        public static void ApplyTo(PronunciationListener listener, int stage)
        {
            if (listener == null) throw new ArgumentNullException(nameof(listener));
            var selected = Load().GetStage(stage);
            listener.ConfigureOrthographicWordPolicy();
            listener.LoadData(ReadShared("ko_child_v2"), _prepared);
            listener.TargetWords = selected.Answers.ToArray();
        }
        static string ReadShared(string name)
        {
            var asset = Resources.Load<TextAsset>(MusicWebPresentation.ResourceFolder + name);
            if (asset == null) throw new InvalidOperationException("공통 음성 데이터가 없습니다: " + name);
            return asset.text;
        }
    }
}
