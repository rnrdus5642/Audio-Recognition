using System;
using System.IO;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;
using UnityEngine.Networking;

namespace DomiCube.PhonemeMatching.Editor
{
    /// <summary>
    /// Fetches the acoustic model so a teammate does not have to build it.
    ///
    /// The model is 1.18 GB - too big for a UPM package and too big for
    /// a git repository - so it is hosted separately and downloaded once
    /// per project. Without this, using the package means cloning the
    /// build repository, installing PyTorch, and running an export, which
    /// is more than anyone should need to do to add a feature.
    ///
    /// The download goes to a temporary file and is verified before it is
    /// moved into Assets, because a truncated model imports fine and then
    /// fails at inference with nothing pointing back to here.
    /// </summary>
    [InitializeOnLoad]
    public static class ModelDownloader
    {
        private const string UrlKey = "DomiCube.PhonemeMatching.ModelUrl";

        /// <summary>
        /// Under Resources on purpose. The model is gitignored, so every
        /// machine imports its own copy and gets its own asset GUID - an
        /// inspector reference saved by one person resolves to nothing on
        /// everyone else's. Committing the .meta does not help either:
        /// Unity deletes a .meta whose asset is absent. Loading by path
        /// sidesteps GUIDs entirely.
        /// </summary>
        private const string Destination =
            "Assets/Resources/Models/wav2vec2_ko.onnx";

        /// <summary>What Resources.Load takes - no folder, no extension.</summary>
        public const string ResourcePath = "Models/wav2vec2_ko";

        private const string NoticeKey =
            "DomiCube.PhonemeMatching.ModelNoticeShown";

        /// <summary>
        /// The model is gitignored, so a teammate who clones a project that
        /// already uses this package gets everything except the one file
        /// that makes it work. Without a notice they meet it as a runtime
        /// error in the middle of a lesson instead.
        /// </summary>
        static ModelDownloader()
        {
            if (SessionState.GetBool(NoticeKey, false))
            {
                return;
            }

            SessionState.SetBool(NoticeKey, true);

            // Only nag projects that are actually set up to use this: the
            // data files are the signal, since they are committed.
            bool configured = File.Exists(
                Path.Combine(Application.streamingAssetsPath, "targets.json"));
            if (!configured || File.Exists(Destination))
            {
                return;
            }

            Debug.LogWarning(
                "[PhonemeMatching] 음향 모델이 없습니다. 저장소에 커밋되지 "
                + "않는 파일이라 각자 한 번 받아야 합니다.\n"
                + "Tools > Phoneme Matching > 음향 모델 내려받기 (약 1.18GB)");
        }

        /// <summary>
        /// Where the model is published. Override per project with
        /// <see cref="SetUrl"/> if the team hosts it somewhere else.
        /// </summary>
        private const string DefaultUrl =
            "https://github.com/rnrdus5642/Audio-Recognition/releases/"
            + "download/model-v1/wav2vec2_ko.onnx";

        /// <summary>Bytes the published file has. 0 disables the check.</summary>
        private const long ExpectedBytes = 1269455457;

        /// <summary>SHA-256 of the published file. Empty disables the check.</summary>
        private const string ExpectedSha256 =
            "bd81ae2fb270870ebbbc4b76df04ccab4eda6e355cfd65460c1dda7f1b0a7ce9";

        internal static bool IsInstalled => File.Exists(Destination);

        [MenuItem("Tools/Phoneme Matching/음향 모델 내려받기", false, 22)]
        public static void Download()
        {
            if (File.Exists(Destination) && !EditorUtility.DisplayDialog(
                    "Phoneme Matching",
                    $"{Destination} 이 이미 있습니다. 다시 받을까요?",
                    "다시 받기", "취소"))
            {
                return;
            }

            if (!EditorUtility.DisplayDialog(
                    "Phoneme Matching",
                    $"음향 모델을 내려받습니다 (약 1.18GB).\n\n{Url}\n\n"
                    + "회선에 따라 몇 분 걸리고, 받은 뒤 Unity 가 임포트하는 "
                    + "데 몇 분 더 걸립니다.",
                    "내려받기", "취소"))
            {
                return;
            }

            Download(askFirst: false);
        }

        /// <param name="askFirst">
        /// False from the setup wizard, which has already asked once for
        /// the whole sequence.
        /// </param>
        internal static bool Download(bool askFirst)
        {
            string temp = Path.Combine(
                Path.GetTempPath(), "wav2vec2_ko.onnx.download");
            try
            {
                return Fetch(Url, temp) && Install(temp);
            }
            finally
            {
                EditorUtility.ClearProgressBar();
                if (File.Exists(temp))
                {
                    File.Delete(temp);
                }
            }
        }

        /// <summary>
        /// For a teammate who already has the file - a network share, a
        /// USB stick, another project - or for testing this path without
        /// a download.
        /// </summary>
        [MenuItem("Tools/Phoneme Matching/음향 모델 파일에서 가져오기…", false, 23)]
        public static void ImportFromFile()
        {
            string picked = EditorUtility.OpenFilePanel(
                "wav2vec2_ko.onnx", "", "onnx");
            if (string.IsNullOrEmpty(picked))
            {
                return;
            }

            try
            {
                Install(picked);
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }
        }

        private static bool Install(string source)
        {
            if (!Verify(source))
            {
                return false;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(Destination));
            File.Copy(source, Destination, overwrite: true);
            EditorUtility.ClearProgressBar();

            AssetDatabase.Refresh();
            Debug.Log(
                $"[PhonemeMatching] {Destination} 준비 완료. Unity 임포트에 "
                + "몇 분 걸립니다.\n"
                + "코드에서는 경로로 부르세요 - 인스펙터에 끌어다 놓으면 "
                + "다른 팀원 PC 에서 참조가 비어 있게 됩니다.\n"
                + $"    Resources.Load<ModelAsset>(\"{ResourcePath}\")");

            OfferToIgnore();
            return true;
        }

        /// <summary>
        /// A 1.18 GB file in a git-backed project is a push failure waiting
        /// to happen - GitHub refuses anything over 100 MB, and by then it
        /// is in the history. Asking beats hoping everyone read the README,
        /// and asking beats editing someone's VCS config unannounced.
        /// </summary>
        private static void OfferToIgnore()
        {
            string root = Directory.GetParent(Application.dataPath)?.FullName;
            if (root == null || !Directory.Exists(Path.Combine(root, ".git")))
            {
                return;
            }

            string ignorePath = Path.Combine(root, ".gitignore");
            string rule = "Assets/Resources/Models/*.onnx";
            if (File.Exists(ignorePath)
                && File.ReadAllText(ignorePath).Contains(rule))
            {
                return;
            }

            if (!EditorUtility.DisplayDialog(
                    "Phoneme Matching",
                    "이 프로젝트는 git 저장소입니다.\n\n"
                    + "받은 모델은 1.18GB 라 커밋하면 push 가 거부됩니다 "
                    + "(GitHub 제한 100MB). .gitignore 에 아래 한 줄을 "
                    + $"추가할까요?\n\n    {rule}",
                    "추가", "나중에"))
            {
                return;
            }

            File.AppendAllText(
                ignorePath,
                Environment.NewLine
                + "# 음향 모델 (1.18GB). Tools > Phoneme Matching > "
                + "음향 모델 내려받기 로 각자 받습니다." + Environment.NewLine
                + rule + Environment.NewLine
                + rule + ".meta" + Environment.NewLine);

            Debug.Log($"[PhonemeMatching] {ignorePath} 에 규칙을 추가했습니다.");
        }

        private static string Url
        {
            get
            {
                string saved = EditorPrefs.GetString(UrlKey, "");
                return string.IsNullOrEmpty(saved) ? DefaultUrl : saved;
            }
        }

        private static bool Fetch(string url, string temp)
        {
            using (var request = UnityWebRequest.Get(url))
            {
                request.downloadHandler = new DownloadHandlerFile(temp);
                var operation = request.SendWebRequest();

                while (!operation.isDone)
                {
                    double mb = request.downloadedBytes / 1048576.0;
                    // The MB counter stays: a bar that only says "wait"
                    // reads as frozen on a download this long.
                    if (EditorUtility.DisplayCancelableProgressBar(
                            "음향 모델을 내려받는 중입니다",
                            $"핸드폰 하지 말고 다른 일 하세요 · {mb:F0} MB",
                            request.downloadProgress))
                    {
                        request.Abort();
                        Debug.Log("[PhonemeMatching] 내려받기를 취소했습니다.");
                        return false;
                    }

                    System.Threading.Thread.Sleep(100);
                }

#if UNITY_2020_2_OR_NEWER
                bool failed = request.result != UnityWebRequest.Result.Success;
#else
                bool failed = request.isNetworkError || request.isHttpError;
#endif
                if (failed)
                {
                    Debug.LogError(
                        $"[PhonemeMatching] 내려받기 실패: {request.error}\n"
                        + $"주소: {url}\n"
                        + "주소가 맞는지, 저장소 Release 에 파일이 올라가 "
                        + "있는지 확인하세요.");
                    return false;
                }
            }

            return true;
        }

        private static bool Verify(string path)
        {
            var info = new FileInfo(path);
            if (ExpectedBytes > 0 && info.Length != ExpectedBytes)
            {
                Debug.LogError(
                    $"[PhonemeMatching] 받은 파일 크기가 다릅니다: "
                    + $"{info.Length} != {ExpectedBytes}. 내려받기가 중간에 "
                    + "끊겼거나 다른 파일입니다.");
                return false;
            }

            if (string.IsNullOrEmpty(ExpectedSha256))
            {
                return true;
            }

            EditorUtility.DisplayProgressBar("음향 모델 내려받기", "검증 중…", 1f);
            using (var stream = File.OpenRead(path))
            using (var sha = SHA256.Create())
            {
                string actual = BitConverter
                    .ToString(sha.ComputeHash(stream))
                    .Replace("-", "")
                    .ToLowerInvariant();
                if (actual != ExpectedSha256)
                {
                    Debug.LogError(
                        $"[PhonemeMatching] 해시가 다릅니다.\n"
                        + $"받은 값: {actual}\n기대 값: {ExpectedSha256}");
                    return false;
                }
            }

            return true;
        }
    }
}
