using System;
using System.IO;

namespace DomiCube.PhonemeMatching.Editor
{
    internal enum TestDataChoice
    {
        Unspecified,
        Base,
        Child,
        OrthographicChild
    }

    /// <summary>
    /// Model/data pairing for the editor test window only. Standard download
    /// names resolve automatically; renamed/custom models need an explicit
    /// choice, not a silent fallback to the base model's thresholds.
    /// </summary>
    internal sealed class PronunciationTestSettings
    {
        internal static readonly PronunciationTestSettings Base =
            new PronunciationTestSettings("기본 모델", "ko_child_v1.json", "targets.json");
        internal static readonly PronunciationTestSettings Child =
            new PronunciationTestSettings("아동 파인튜닝 모델", "ko_child_v2.json", "targets_child.json");
        internal static readonly PronunciationTestSettings OrthographicChild =
            new PronunciationTestSettings("원문 자모 아동 모델 (미리보기)", "ko_child_v2.json", "targets_child.json",
                "hubert_ko_child_jamo_vocab.json");
        internal const string VocabularyFile = "wav2vec2_ko_vocab.json";

        internal string Label { get; }
        internal string MatrixFile { get; }
        internal string TargetsFile { get; }
        internal string SelectedVocabularyFile { get; }

        private PronunciationTestSettings(string label, string matrixFile, string targetsFile,
            string vocabularyFile = VocabularyFile)
        {
            Label = label;
            MatrixFile = matrixFile;
            TargetsFile = targetsFile;
            SelectedVocabularyFile = vocabularyFile;
        }

        internal static PronunciationTestSettings FromModelPath(string modelPath)
        {
            string file = Path.GetFileName((modelPath ?? "").Replace('\\', '/'));
            if (string.Equals(file, "hubert_ko_child_jamo.sentis", StringComparison.OrdinalIgnoreCase)
                || string.Equals(file, "hubert_ko_child_jamo.onnx", StringComparison.OrdinalIgnoreCase))
                return OrthographicChild;
            if (string.Equals(file, Path.GetFileName(ModelDownloader.ChildResourcePath) + ".onnx",
                    StringComparison.OrdinalIgnoreCase))
                return Child;
            if (string.Equals(file, Path.GetFileName(ModelDownloader.ResourcePath) + ".onnx",
                    StringComparison.OrdinalIgnoreCase))
                return Base;
            return null;
        }

        internal static PronunciationTestSettings Resolve(string modelPath, TestDataChoice choice)
        {
            // A saved manual choice for a renamed model must not override a
            // known filename when it is restored to a standard model name.
            var known = FromModelPath(modelPath);
            if (known != null) return known;
            switch (choice)
            {
                case TestDataChoice.Base: return Base;
                case TestDataChoice.Child: return Child;
                case TestDataChoice.OrthographicChild: return OrthographicChild;
                default: return null;
            }
        }

        internal void Load(string directory, out ConfusionMatrix matrix,
            out TargetCatalog catalog, out CtcVocabulary vocabulary)
        {
            // Validate every selected file before model allocation/warmup.
            // Missing child data must never fall back to the base pair, nor
            // trigger an installer that could replace customized JSON files.
            var required = this == OrthographicChild
                ? new[] { MatrixFile, SelectedVocabularyFile }
                : new[] { MatrixFile, TargetsFile, SelectedVocabularyFile };
            foreach (string file in required)
            {
                string path = Path.Combine(directory, file);
                if (!File.Exists(path))
                    throw new FileNotFoundException(
                        $"{Label} 채점 파일이 없습니다: {file}\n"
                        + "Tools > Phoneme Matching > 초기 세팅에서 누락된 파일을 설치하세요.", path);
            }
            matrix = PhonemeData.LoadMatrix(File.ReadAllText(Path.Combine(directory, MatrixFile)));
            string targetsPath = Path.Combine(directory, TargetsFile);
            catalog = this == OrthographicChild
                ? new TargetCatalog() : PhonemeData.LoadTargets(File.ReadAllText(targetsPath));
            vocabulary = PhonemeData.LoadCtcVocabulary(File.ReadAllText(Path.Combine(directory, SelectedVocabularyFile)));
        }
    }
}
