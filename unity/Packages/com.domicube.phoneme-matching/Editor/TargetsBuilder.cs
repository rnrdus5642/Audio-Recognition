using System;
using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace DomiCube.PhonemeMatching.Editor
{
    /// <summary>
    /// Rebuilds targets.json from words.csv without leaving Unity.
    ///
    /// The conversion itself cannot move into the editor: Korean
    /// phonological rules need g2pkk, which needs a morphological
    /// analyser (mecab) and its dictionary. Skipping the rules was
    /// measured and it costs accuracy - positives 69.4% -> 66.7%,
    /// negative rejection 91.7% -> 88.2% - because the targets stop
    /// describing how the words are actually pronounced (학교 is said
    /// [학꾜]). So the rules stay, Python stays, and this just removes
    /// the command line.
    ///
    /// Needs the repository checked out and its dependencies installed:
    ///
    ///     pip install -r requirements.txt
    ///
    /// which is g2pkk, eunjeon, jamo - no PyTorch, no model.
    /// </summary>
    public static class TargetsBuilder
    {
        private const string RepoKey = "DomiCube.PhonemeMatching.RepoPath";
        private const string StreamingTargets = "Assets/StreamingAssets/targets.json";

        private const string RepoUrl =
            "https://github.com/rnrdus5642/Audio-Recognition";

        /// <summary>
        /// Shown when neither menu can find a checkout. Most people never
        /// need one - the built targets.json is committed - so the dialog
        /// says that first, then how to get one.
        /// </summary>
        private static bool AskForRepository(string why)
        {
            int choice = EditorUtility.DisplayDialogComplex(
                "Phoneme Matching",
                why + "\n\n"
                + "정답 단어를 바꿀 때만 필요합니다. 한 사람이 바꿔서 "
                + "targets.json 을 커밋하면 나머지는 받기만 하면 됩니다.\n\n"
                + "저장소가 없으면 여기서 받을 수 있습니다 (약 3MB).",
                "저장소 받기", "취소", "이미 있음 - 폴더 고르기");

            if (choice == 2)
            {
                SetRepository();
                return FindRepository() != null;
            }

            if (choice != 0)
            {
                return false;
            }

            return Clone();
        }

        /// <summary>
        /// Clone the repository so nobody has to leave the editor for it.
        ///
        /// git is safe to assume: Unity cannot install a package from a git
        /// URL without it, which is how this package got here.
        /// </summary>
        private static bool Clone()
        {
            string git = OnPath("git");
            if (git == null)
            {
                Debug.LogError(
                    "[PhonemeMatching] git 을 찾지 못했습니다. "
                    + "https://git-scm.com 에서 설치한 뒤 Unity 를 다시 "
                    + $"시작하거나, 직접 클론하세요:\n    git clone {RepoUrl}.git");
                return false;
            }

            string parent = EditorUtility.OpenFolderPanel(
                "저장소를 받을 위치 (안에 Audio-Recognition 폴더가 생깁니다)",
                "", "");
            if (string.IsNullOrEmpty(parent))
            {
                return false;
            }

            string target = Path.Combine(parent, "Audio-Recognition");
            if (Directory.Exists(target))
            {
                if (IsRepository(target))
                {
                    EditorPrefs.SetString(RepoKey, target);
                    Debug.Log($"[PhonemeMatching] 이미 있어서 그대로 씁니다: {target}");
                    return true;
                }

                Debug.LogError(
                    $"[PhonemeMatching] {target} 이 이미 있는데 저장소가 "
                    + "아닙니다. 다른 위치를 고르거나 그 폴더를 치우세요.");
                return false;
            }

            bool ok;
            try
            {
                EditorUtility.DisplayProgressBar(
                    "Phoneme Matching", "저장소를 받는 중…", 0.5f);
                ok = Run(
                    git, $"clone {RepoUrl}.git \"{target}\"", parent, out string log);
                if (!ok)
                {
                    Debug.LogError($"[PhonemeMatching] 클론 실패:\n{log}");
                }
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }

            if (!ok || !IsRepository(target))
            {
                return false;
            }

            EditorPrefs.SetString(RepoKey, target);
            Debug.Log(
                $"[PhonemeMatching] 저장소를 받았습니다: {target}\n"
                + "정답 데이터를 다시 만들려면 파이썬 의존성도 한 번 "
                + "설치해야 합니다 (torch 는 필요 없습니다):\n"
                + $"    cd \"{target}\"\n"
                + "    python -m venv .venv\n"
                + "    .venv\\Scripts\\pip install -r requirements.txt");
            return true;
        }

        [MenuItem("Tools/Phoneme Matching/정답 데이터 다시 만들기", false, 61)]
        public static void Rebuild()
        {
            string repo = FindRepository();
            if (repo == null)
            {
                if (!AskForRepository(
                        "정답 데이터를 다시 만들려면 Audio-Recognition "
                        + "저장소와 파이썬이 필요합니다."))
                {
                    return;
                }

                repo = FindRepository();
            }

            string python = FindPython(repo);
            if (python == null)
            {
                Debug.LogError(
                    $"[PhonemeMatching] 파이썬을 찾지 못했습니다. {repo} 에서\n"
                    + "  python -m venv .venv\n"
                    + "  .\\.venv\\Scripts\\Activate.ps1\n"
                    + "  pip install -r requirements.txt\n"
                    + "를 먼저 실행하세요.");
                return;
            }

            if (!Run(python, "-m python.build.build_targets", repo, out string log))
            {
                Debug.LogError($"[PhonemeMatching] 빌드 실패:\n{log}");
                return;
            }

            Debug.Log($"[PhonemeMatching] {log.Trim()}");

            string built = Path.Combine(repo, "shared", "targets.json");
            if (!File.Exists(built))
            {
                Debug.LogError(
                    $"[PhonemeMatching] 빌드는 됐는데 {built} 이 없습니다.");
                return;
            }

            Directory.CreateDirectory(
                Path.GetDirectoryName(StreamingTargets) ?? "Assets/StreamingAssets");
            File.Copy(built, StreamingTargets, overwrite: true);
            AssetDatabase.Refresh();

            // Read it back the way the runtime will, so a malformed file
            // is caught here rather than in front of a child.
            var catalog = PhonemeData.LoadTargets(File.ReadAllText(StreamingTargets));
            Debug.Log(
                $"[PhonemeMatching] {StreamingTargets} 갱신 완료 - "
                + $"{catalog.Answers.Count}개 단어");
        }

        [MenuItem("Tools/Phoneme Matching/단어 목록 열기 (words.csv)", false, 60)]
        public static void OpenWords()
        {
            string repo = FindRepository();
            if (repo == null)
            {
                if (!AskForRepository(
                        "words.csv 는 Audio-Recognition 저장소에 있습니다."))
                {
                    return;
                }

                repo = FindRepository();
            }

            EditorUtility.OpenWithDefaultApp(
                Path.Combine(repo, "shared", "words.csv"));
        }

        [MenuItem("Tools/Phoneme Matching/저장소 경로 지정…", false, 80)]
        public static void SetRepository()
        {
            string picked = EditorUtility.OpenFolderPanel(
                "Audio-Recognition 저장소 폴더", EditorPrefs.GetString(RepoKey, ""), "");
            if (string.IsNullOrEmpty(picked))
            {
                return;
            }

            if (!IsRepository(picked))
            {
                Debug.LogError(
                    $"[PhonemeMatching] {picked} 에 shared/words.csv 와 "
                    + "python/build/build_targets.py 가 없습니다.");
                return;
            }

            EditorPrefs.SetString(RepoKey, picked);
            Debug.Log($"[PhonemeMatching] 저장소: {picked}");
        }

        /// <summary>
        /// Saved path first, then upwards from the package - which finds
        /// it automatically when the package is embedded in the repo.
        /// </summary>
        private static string FindRepository()
        {
            string saved = EditorPrefs.GetString(RepoKey, "");
            if (!string.IsNullOrEmpty(saved) && IsRepository(saved))
            {
                return saved;
            }

            var dir = new DirectoryInfo(
                Path.GetFullPath("Packages/com.domicube.phoneme-matching"));
            while (dir != null)
            {
                if (IsRepository(dir.FullName))
                {
                    return dir.FullName;
                }

                dir = dir.Parent;
            }

            return null;
        }

        private static bool IsRepository(string path)
        {
            return File.Exists(Path.Combine(path, "shared", "words.csv"))
                && File.Exists(Path.Combine(
                    path, "python", "build", "build_targets.py"));
        }

        /// <summary>
        /// The repository's own virtual environment if there is one -
        /// the dependencies are pinned there, and a system Python may
        /// have none of them.
        /// </summary>
        private static string FindPython(string repo)
        {
            string[] candidates =
            {
                Path.Combine(repo, ".venv", "Scripts", "python.exe"),
                Path.Combine(repo, ".venv", "bin", "python"),
            };

            foreach (var candidate in candidates)
            {
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            return OnPath("python") ?? OnPath("python3");
        }

        private static string OnPath(string exe)
        {
            string paths = Environment.GetEnvironmentVariable("PATH") ?? "";
            foreach (var dir in paths.Split(Path.PathSeparator))
            {
                if (string.IsNullOrWhiteSpace(dir))
                {
                    continue;
                }

                foreach (var name in new[] { exe, exe + ".exe" })
                {
                    string full = Path.Combine(dir.Trim(), name);
                    if (File.Exists(full))
                    {
                        return full;
                    }
                }
            }

            return null;
        }

        private static bool Run(
            string exe, string arguments, string workingDirectory, out string log)
        {
            var info = new ProcessStartInfo(exe, arguments)
            {
                WorkingDirectory = workingDirectory,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = System.Text.Encoding.UTF8,
                StandardErrorEncoding = System.Text.Encoding.UTF8,
            };

            try
            {
                using (var process = Process.Start(info))
                {
                    string stdout = process.StandardOutput.ReadToEnd();
                    string stderr = process.StandardError.ReadToEnd();
                    if (!process.WaitForExit(120000))
                    {
                        process.Kill();
                        log = "2분 안에 끝나지 않아 중단했습니다.";
                        return false;
                    }

                    log = string.IsNullOrWhiteSpace(stderr)
                        ? stdout
                        : stdout + "\n" + stderr;
                    return process.ExitCode == 0;
                }
            }
            catch (Exception e)
            {
                log = e.Message;
                return false;
            }
        }
    }
}
