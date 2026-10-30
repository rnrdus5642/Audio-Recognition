using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace DomiCube.PhonemeMatching.Editor
{
    /// <summary>
    /// Puts the JSON files the runtime loads into StreamingAssets.
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

        // Adult and child ship side by side rather than one replacing
        // the other. The two acoustic models mishear differently, so the
        // thresholds and the streaming profile fitted for one are wrong
        // for the other - the child settings applied to the adult model
        // confirm 6.47% of sessions the child never spoke in, against a
        // 1% budget. PronunciationListener names which pair it wants.
        private static readonly string[] Files =
        {
            "ko_child_v1.json",
            "targets.json",
            "ko_child_v2.json",
            "targets_child.json",
            "targets_music_demo.json",
            "wav2vec2_ko_vocab.json",
        };

        /// <summary>True when every file is already in place.</summary>
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
        public static void Install()
        {
            Install(overwriteExisting: true);
        }

        /// <summary>Initial setup only fills gaps; never replaces user data.</summary>
        internal static bool InstallMissing()
        {
            return Install(overwriteExisting: false);
        }

        private static bool Install(bool overwriteExisting)
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

            int copied;
            try
            {
                if (!InstallFiles(source, StreamingAssets, overwriteExisting,
                        existing => EditorUtility.DisplayDialog(
                            "Phoneme Matching",
                            $"{StreamingAssets} 의 기존 파일 {existing.Count}개를 "
                            + "패키지 기본 데이터로 교체합니다.\n"
                            + "직접 추가한 단어와 수정한 판정 설정도 덮어씁니다.\n\n"
                            + string.Join("\n", existing),
                            "기본 데이터로 교체", "취소"), out copied))
                {
                    return false;
                }
            }
            catch (IOException e)
            {
                Debug.LogError($"[PhonemeMatching] 데이터 설치 실패: {e.Message}");
                return false;
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
                $"[PhonemeMatching] {StreamingAssets}: {copied}개 설치, "
                + $"기존 {Files.Length - copied}개 유지 - "
                + $"단어 {catalog.Answers.Count}개, 어휘 {vocab.Size}개.\n"
                + ModelNotice());
            return true;
        }

        /// <summary>
        /// File policy shared by initial setup and explicit replacement.
        /// Replacement requires approval of the existing filenames before
        /// any writes. Kept independent of dialogs and AssetDatabase so
        /// tests can exercise it in disposable directories.
        /// </summary>
        internal static bool InstallFiles(
            string source, string destination, bool overwriteExisting,
            Func<IReadOnlyList<string>, bool> confirmOverwrite, out int copied)
        {
            copied = 0;
            var existing = new List<string>();
            foreach (string file in Files)
            {
                // Validate the entire source before changing anything.
                string sourceFile = Path.Combine(source, file);
                if (!File.Exists(sourceFile))
                {
                    throw new FileNotFoundException(
                        $"패키지 기본 데이터가 없습니다: {file}", sourceFile);
                }

                if (File.Exists(Path.Combine(destination, file)))
                {
                    existing.Add(file);
                }
            }

            if (overwriteExisting && existing.Count > 0
                && (confirmOverwrite == null
                    || !confirmOverwrite(existing.AsReadOnly())))
            {
                return false;
            }

            Directory.CreateDirectory(destination);
            foreach (string file in Files)
            {
                string target = Path.Combine(destination, file);
                if (!overwriteExisting && File.Exists(target))
                {
                    continue;
                }

                // A file appearing after the prompt was not approved for
                // replacement. File.Copy must fail rather than clobber it.
                File.Copy(Path.Combine(source, file), target,
                    overwrite: overwriteExisting && existing.Contains(file));
                copied++;
            }

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
