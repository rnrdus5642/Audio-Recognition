using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace DomiCube.PhonemeMatching
{
    /// <summary>The packaged twelve-word policy and its nine lesson stages.</summary>
    public sealed class MusicPresentationProfile
    {
        public const float WindowSeconds = 2.5f;
        public const float HopSeconds = 0.5f;
        public const int Consecutive = 2;
        public const string ModelResourcePath = "Models/wav2vec2_ko_child";
        public const string ResourceFolder = "DomiCube/PhonemeMatching/MusicDemo/";
        public const string MatrixFile = "ko_child_v2.json";
        public const string TargetsFile = "targets_music_demo.json";
        public const string VocabularyFile = "wav2vec2_ko_vocab.json";

        private readonly Dictionary<int, MusicPresentationStage> _stages;
        public string ProfileId { get; }
        public bool UsesOrthographicWordPolicy { get; }
        public ConfusionMatrix Matrix { get; }
        public TargetCatalog Catalog { get; }
        public CtcVocabulary Vocabulary { get; }
        public ModelErrorProfile FeedbackErrors { get; }
        public IReadOnlyList<MusicPresentationStage> Stages { get; }

        private MusicPresentationProfile(string matrixJson, string targetsJson, string vocabularyJson, ModelErrorProfile modelErrors)
        {
            var document = JObject.Parse(targetsJson);
            UsesOrthographicWordPolicy = (string)document["word_policy"] == OrthographicWordPolicy.Version;
            Matrix = PhonemeData.LoadMatrix(matrixJson, UsesOrthographicWordPolicy ? "ko_child_v2" : null);
            Catalog = PhonemeData.LoadTargets(targetsJson);
            Vocabulary = PhonemeData.LoadCtcVocabulary(vocabularyJson);
            FeedbackErrors = modelErrors;
            ProfileId = (string)document["profile_id"];
            if (string.IsNullOrWhiteSpace(ProfileId) || Matrix.MatrixId != "ko_child_v2"
                || Catalog.ConfusionMatrixId != Matrix.MatrixId || (UsesOrthographicWordPolicy ? Catalog.Answers.Count < 1 : Catalog.Answers.Count != 12))
                throw new ArgumentException("악기놀이 발표용 아동 데이터 조합이 아닙니다.");

            var items = document["presentation_stages"] as JArray;
            if (items == null || items.Count != 9)
                throw new ArgumentException("악기놀이 단계는 1~9까지 필요합니다.");
            _stages = new Dictionary<int, MusicPresentationStage>();
            foreach (var item in items)
            {
                if (item["id"]?.Type != JTokenType.Integer)
                    throw new ArgumentException("악기놀이 단계 번호가 올바르지 않습니다.");
                int id = (int)item["id"];
                string title = (string)item["title"];
                var words = item["answers"] as JArray;
                if (id < 1 || id > 9 || _stages.ContainsKey(id) || string.IsNullOrWhiteSpace(title)
                    || words == null || words.Count == 0
                    || words.Any(w => w.Type != JTokenType.String || Catalog.Find((string)w) == null))
                    throw new ArgumentException($"악기놀이 {id}단계의 정답 목록이 올바르지 않습니다.");
                var answers = words.Select(w => (string)w).ToArray();
                if (answers.Distinct().Count() != answers.Length)
                    throw new ArgumentException($"악기놀이 {id}단계의 정답이 중복됩니다.");
                _stages.Add(id, new MusicPresentationStage(id, title, answers));
            }
            Stages = Array.AsReadOnly(_stages.Values.OrderBy(s => s.Id).ToArray());
        }

        public static MusicPresentationProfile FromJson(
            string matrixJson, string targetsJson, string vocabularyJson, ModelErrorProfile modelErrors = null) =>
            new MusicPresentationProfile(matrixJson, targetsJson, vocabularyJson, modelErrors);

        public MusicPresentationStage GetStage(int stage)
        {
            if (!_stages.TryGetValue(stage, out var result))
                throw new ArgumentOutOfRangeException(nameof(stage), "단계는 1~9입니다.");
            return result;
        }

        /// <summary>For applications that already own their audio capture.</summary>
        public PronunciationSession CreateSession(IPhonemeRecognizer recognizer, AudioWindowBuffer buffer = null) =>
            UsesOrthographicWordPolicy ? PronunciationSession.ForOrthographicWord(Matrix, recognizer, .85, Consecutive, buffer)
                : new PronunciationSession(Matrix, Catalog, recognizer, Consecutive, buffer, FeedbackErrors);
    }

    public sealed class MusicPresentationStage
    {
        public int Id { get; }
        public string Title { get; }
        public IReadOnlyList<string> Answers { get; }

        internal MusicPresentationStage(int id, string title, string[] answers)
        {
            Id = id;
            Title = title;
            Answers = Array.AsReadOnly(answers);
        }
    }
}
