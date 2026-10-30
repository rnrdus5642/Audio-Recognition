using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace DomiCube.PhonemeMatching
{
    /// <summary>Stream, verify, then atomically replace. Never expose an incomplete model.</summary>
    public static class SharedModelInstaller
    {
        public static async Task<bool> EnsureAsync(string destination, long bytes, string sha256,
            string source, string[] urls, Action<string, float> progress, CancellationToken cancel)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            using (var lease = new FileStream(destination + ".install.lock", FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None))
            {
                progress?.Invoke("공용 모델 확인 중", 0);
                long existingTicks = File.Exists(destination) ? File.GetLastWriteTimeUtc(destination).Ticks : 0;
                if (SharedSpeechModel.VerifyFile(destination, bytes, sha256, cancel,
                        value => progress?.Invoke("공용 모델 무결성 확인 중", value)))
                {
                    cancel.ThrowIfCancellationRequested();
                    SharedSpeechModel.WriteVerificationReceipt(destination, bytes, sha256, existingTicks);
                    return false;
                }
                cancel.ThrowIfCancellationRequested();
                string temporary = destination + "." + Guid.NewGuid().ToString("N") + ".partial";
                try
                {
                    using (var output = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1048576, true))
                    {
                        long written = 0;
                        if (!string.IsNullOrEmpty(source))
                        {
                            if (!File.Exists(source) || new FileInfo(source).Length != bytes)
                                throw new InvalidDataException("선택한 모델의 크기가 올바르지 않습니다.");
                            using (var input = new FileStream(source, FileMode.Open, FileAccess.Read, FileShare.Read, 1048576, true))
                                written = await Transfer(input, output, written, bytes, "기존 모델을 문서 폴더에 설치 중", progress, cancel).ConfigureAwait(false);
                        }
                        else
                        {
                            if (urls == null || urls.Length == 0) throw new InvalidOperationException("모델 다운로드 주소가 없습니다.");
                            using (var http = new HttpClient { Timeout = TimeSpan.FromMinutes(120) })
                            {
                                http.DefaultRequestHeaders.UserAgent.ParseAdd("AudioAnswer-Setup/0.8.8");
                                foreach (var url in urls)
                                {
                                    using (var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancel).ConfigureAwait(false))
                                    {
                                        response.EnsureSuccessStatusCode();
                                        // Mono may ignore cancellation on an in-flight socket read.
                                        // Disposing the response closes that read as well as the HTTP request.
                                        using (cancel.Register(() => response.Dispose()))
                                        using (var input = await response.Content.ReadAsStreamAsync().ConfigureAwait(false))
                                            written = await Transfer(input, output, written, bytes, "모델 다운로드 중", progress, cancel).ConfigureAwait(false);
                                    }
                                }
                            }
                        }
                        if (written != bytes) throw new InvalidDataException("모델 다운로드가 불완전합니다. 초기 세팅을 다시 실행하세요.");
                    }
                    long verifiedTicks = File.GetLastWriteTimeUtc(temporary).Ticks;
                    if (!SharedSpeechModel.VerifyFile(temporary, bytes, sha256, cancel,
                            value => progress?.Invoke("설치 파일 검증 중", value)))
                        throw new InvalidDataException("모델 SHA-256이 일치하지 않습니다. 기존 파일은 변경하지 않았습니다.");
                    cancel.ThrowIfCancellationRequested();
                    if (File.Exists(destination)) File.Replace(temporary, destination, null);
                    else File.Move(temporary, destination);
                    SharedSpeechModel.WriteVerificationReceipt(destination, bytes, sha256, verifiedTicks);
                    return true;
                }
                catch (Exception) when (cancel.IsCancellationRequested)
                {
                    throw new OperationCanceledException(cancel);
                }
                finally { if (File.Exists(temporary)) File.Delete(temporary); }
            }
        }

        static async Task<long> Transfer(Stream input, Stream output, long written, long total,
            string status, Action<string, float> progress, CancellationToken cancel)
        {
            var buffer = new byte[1048576];
            int count;
            while ((count = await input.ReadAsync(buffer, 0, buffer.Length, cancel).ConfigureAwait(false)) > 0)
            {
                written += count;
                if (written > total) throw new InvalidDataException("배포 모델의 크기가 예상 크기를 초과했습니다.");
                await output.WriteAsync(buffer, 0, count, cancel).ConfigureAwait(false);
                progress?.Invoke(status, (float)(written / (double)total));
            }
            return written;
        }
    }
}
