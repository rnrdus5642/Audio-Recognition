using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Editor
{
    /// <summary>
    /// Puts the three JSON files the runtime loads into StreamingAssets.
    ///
    /// They ship in the package's Samples~ folder, which Unity's own
    /// "Import Sample" button copies into Assets/Samples/... - one
    /// directory further from where the runtime actually reads them. This
    /// copies them where they belong instead.
    ///
    /// The acoustic model is not handled here: it is 1.18 GB, cannot live
    /// in a UPM package, and dropping it into a git-backed project without
    /// asking would be a rude thing to do to someone's repository. This
    /// only reports whether one is present.
    /// </summary>
    public static class DataInstaller
    {
        private const string StreamingAssets = "Assets/StreamingAssets";

        private static readonly string[] Files =
        {
            "ko_child_v1.json",
            "targets.json",
            "wav2vec2_ko_vocab.json",
        };

        /// <summary>True when all three files are already in place.</summary>
        internal static bool IsInstalled
        {
            get
            {
                foreach (string file in Files)
                {
                    if (!File.Exists(Path.Combine(StreamingAssets, file)))
                    {
                        return false;
                    }
                }

                return true;
            }
        }

        [MenuItem("Tools/Phoneme Matching/데이터 파일 설치", false, 21)]
        public static void Install()
        {
            Install(askBeforeOverwriting: true);
        }

        /// <param name="askBeforeOverwriting">
        /// False from the setup wizard, which has already asked once for
        /// the whole sequence.
        /// </param>
        internal static bool Install(bool askBeforeOverwriting)
        {
            string source = SampleFolder();
            if (source == null)
            {
                Debug.LogError(
                    "[PhonemeMatching] 패키지 안에서 Samples~/KoreanData 를 "
                    + "찾지 못했습니다. 패키지가 온전히 설치됐는지 "
                    + "확인하세요.");
                return false;
            }

            var missing = new List<string>();
            var existing = new List<string>();
            foreach (string file in Files)
            {
                if (!File.Exists(Path.Combine(source, file)))
                {
                    missing.Add(file);
                }
                else if (File.Exists(Path.Combine(StreamingAssets, file)))
                {
                    existing.Add(file);
                }
            }

            if (missing.Count > 0)
            {
                Debug.LogError(
                    $"[PhonemeMatching] {source} 에 없는 파일: "
                    + string.Join(", ", missing));
                return false;
            }

            if (askBeforeOverwriting && existing.Count > 0
                && !EditorUtility.DisplayDialog(
                    "Phoneme Matching",
                    $"{StreamingAssets} 에 이미 있는 파일 {existing.Count}개를 "
                    + "덮어씁니다.\n\n" + string.Join("\n", existing),
                    "덮어쓰기", "취소"))
            {
                return false;
            }

            Directory.CreateDirectory(StreamingAssets);
            foreach (string file in Files)
            {
                File.Copy(
                    Path.Combine(source, file),
                    Path.Combine(StreamingAssets, file),
                    overwrite: true);
            }

            AssetDatabase.Refresh();

            // Read them back the way the runtime will, so a truncated copy
            // is caught here rather than in front of a child.
            var catalog = PhonemeData.LoadTargets(
                File.ReadAllText(Path.Combine(StreamingAssets, "targets.json")));
            var vocab = PhonemeData.LoadCtcVocabulary(
                File.ReadAllText(
                    Path.Combine(StreamingAssets, "wav2vec2_ko_vocab.json")));

            Debug.Log(
                $"[PhonemeMatching] {StreamingAssets} 에 {Files.Length}개 복사 "
                + $"완료 - 단어 {catalog.Answers.Count}개, 어휘 {vocab.Size}개.\n"
                + ModelNotice());
            return true;
        }

        /// <summary>
        /// Samples~ lives inside the installed package - the git cache,
        /// the local folder, or the embedded copy - so ask the package
        /// manager where that is rather than guessing.
        /// </summary>
        private static string SampleFolder()
        {
            // Fully qualified: UnityEditor also has a legacy PackageInfo,
            // and an unqualified name would be ambiguous.
            var info = UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                typeof(DataInstaller).Assembly);
            if (info == null)
            {
                return null;
            }

            string path = Path.Combine(
                info.resolvedPath, "Samples~", "KoreanData");
            return Directory.Exists(path) ? path : null;
        }

        private static string ModelNotice()
        {
            string[] models = Directory.GetFiles(
                Application.dataPath, "*.onnx", SearchOption.AllDirectories);
            if (models.Length > 0)
            {
                return $"음향 모델: {models.Length}개 발견.";
            }

            return "음향 모델(.onnx)이 없습니다. "
                + "Tools > Phoneme Matching > 음향 모델 내려받기 로 "
                + "받으세요. 없어도 매칭 계층은 동작하지만, 소리를 "
                + "판정하려면 필요합니다.";
        }
    }
}
