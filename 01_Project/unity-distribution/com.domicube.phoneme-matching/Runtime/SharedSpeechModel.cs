using System;
using System.IO;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

namespace DomiCube.PhonemeMatching
{
    /// <summary>One verified model per Windows user, independent of Unity projects.</summary>
    public static class SharedSpeechModel
    {
        public const string FileName = "hubert_ko_child_jamo_sentis1.sentis";
        public const long Bytes = 3851161113L;
        public const string Sha256 = "f676db58234bd48f5e4530211e623248afbb22a972c54b89a39288eb07ef29ff";
        public const string ReleaseUrl = "https://github.com/rnrdus5642/Audio-Recognition/releases/download/model-v3-child-jamo-sentis1/";
        public static string DirectoryPath
        {
            get
            {
                string documents = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
                if (string.IsNullOrWhiteSpace(documents))
                    throw new IOException("Windows 문서 폴더를 찾을 수 없습니다.");
                return Path.Combine(documents, "오디오정답처리", "Models");
            }
        }
        public static string FilePath => Path.Combine(DirectoryPath, FileName);
        public static string[] DownloadUrls => new[] { ReleaseUrl + FileName + ".part1", ReleaseUrl + FileName + ".part2" };
        public const string SetupInstructions = "Unity에서는 Tools > 오디오 정답 처리 기능 > 초기 세팅을 실행하세요. 빌드 앱에서는 시작 화면에서 다시 시도하세요.";
        public static bool HasExpectedSize => File.Exists(FilePath) && new FileInfo(FilePath).Length == Bytes;

        public static string RequireInstalled()
        {
            if (!HasVerificationReceipt(FilePath, Bytes, Sha256))
                throw new FileNotFoundException("공용 음성 모델 설치 확인이 필요합니다. " + SetupInstructions + "\n" + FilePath, FilePath);
            return FilePath;
        }

        // Full SHA-256 is checked by setup in a background task. Runtime only reads
        // this small receipt and file metadata, never hashes 3.85 GB on the game thread.
        public static bool HasVerificationReceipt(string path, long bytes, string sha256)
        {
            try
            {
                var model = new FileInfo(path);
                var receipt = new FileInfo(path + ".verified");
                if (!model.Exists || model.Length != bytes || !receipt.Exists || receipt.Length > 1024) return false;
                return File.ReadAllText(receipt.FullName, Encoding.UTF8) == Receipt(bytes, sha256, model.LastWriteTimeUtc.Ticks);
            }
            catch (IOException) { return false; }
            catch (UnauthorizedAccessException) { return false; }
        }

        /// <summary>Called by the installer only after full hash verification, under its installation lock.</summary>
        public static void WriteVerificationReceipt(string path, long bytes, string sha256, long verifiedWriteTicks)
        {
            var model = new FileInfo(path);
            if (!model.Exists || model.Length != bytes || model.LastWriteTimeUtc.Ticks != verifiedWriteTicks)
                throw new IOException("검증 중 모델 파일이 변경됐습니다. 초기 세팅을 다시 실행하세요.");
            string receipt = path + ".verified";
            string contents = Receipt(bytes, sha256, verifiedWriteTicks);
            if (File.Exists(receipt) && File.ReadAllText(receipt, Encoding.UTF8) == contents) return;
            string temp = receipt + "." + Guid.NewGuid().ToString("N") + ".tmp";
            try
            {
                File.WriteAllText(temp, contents, new UTF8Encoding(false));
                for (int attempt = 0; ; attempt++)
                {
                    try
                    {
                        if (File.Exists(receipt)) File.Replace(temp, receipt, null, true);
                        else File.Move(temp, receipt);
                        break;
                    }
                    // Windows indexers can briefly open a newly created receipt
                    // without delete sharing. Keep the old receipt intact and retry.
                    catch (IOException) when (attempt < 4) { Thread.Sleep(50 * (attempt + 1)); }
                }
            }
            finally { if (File.Exists(temp)) File.Delete(temp); }
        }

        static string Receipt(long bytes, string sha256, long ticks) =>
            "audio-answer-model-v1\n" + bytes.ToString(CultureInfo.InvariantCulture) + "\n" +
            sha256.ToLowerInvariant() + "\n" + ticks.ToString(CultureInfo.InvariantCulture) + "\n";

        public static bool VerifyFile(string path, long bytes, string sha256,
            CancellationToken cancel = default, Action<float> progress = null)
        {
            if (!File.Exists(path) || new FileInfo(path).Length != bytes) return false;
            using (var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1048576))
            using (var hash = SHA256.Create())
            {
                var buffer = new byte[1048576];
                long read = 0;
                int count;
                while ((count = input.Read(buffer, 0, buffer.Length)) != 0)
                {
                    cancel.ThrowIfCancellationRequested();
                    hash.TransformBlock(buffer, 0, count, null, 0);
                    read += count;
                    progress?.Invoke((float)(read / (double)bytes));
                }
                hash.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
                return string.Equals(BitConverter.ToString(hash.Hash).Replace("-", ""), sha256, StringComparison.OrdinalIgnoreCase);
            }
        }
    }
}
