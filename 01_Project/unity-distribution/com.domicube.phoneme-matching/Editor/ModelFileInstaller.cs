using System;
using System.Threading;
using System.Threading.Tasks;

namespace DomiCube.PhonemeMatching.Editor
{
    // Keep the existing Editor API while sharing the verified installer with Player builds.
    public static class ModelFileInstaller
    {
        public static Task<bool> EnsureAsync(string destination, long bytes, string sha256,
            string source, string[] urls, Action<string, float> progress, CancellationToken cancel) =>
            SharedModelInstaller.EnsureAsync(destination, bytes, sha256, source, urls, progress, cancel);
    }
}
