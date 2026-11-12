using System;
using System.IO;
using System.Diagnostics;
using System.Threading;
using Newtonsoft.Json.Linq;
using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using UnityEditor;
using UnityEngine;

// Copy into an isolated Unity 2022 project's Assets/Editor, never a game project.
public static class FinalPackageValidation
{
    static string Argument(string key)
    {
        var args = Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < args.Length; i++) if (args[i] == key) return args[i + 1];
        throw new ArgumentException("Missing argument: " + key);
    }

    public static void Run()
    {
        string jobPath = Argument("-finalPackageJob");
        var job = JObject.Parse(File.ReadAllText(jobPath));
        string reportPath = (string)job["report"];
        var results = new JArray();
        var package = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(OrthographicWordPolicy).Assembly);
        var report = new JObject { ["success"] = false, ["unity"] = Application.unityVersion,
            ["package_version"] = package.version, ["package_source"] = package.source.ToString(),
            ["policy"] = OrthographicWordPolicy.Version, ["cases"] = results };
        try
        {
            if (package.version != "0.9.0-preview.1" || package.source.ToString() != "LocalTarball")
                throw new InvalidOperationException("Test the distributed tarball, not a source checkout.");
            string modelPath = (string)job["model"];
            if (!SharedSpeechModel.VerifyFile(modelPath, SharedSpeechModel.Bytes, SharedSpeechModel.Sha256))
                throw new InvalidDataException("Final model identity mismatch");
            var vocab = PhonemeData.LoadCtcVocabulary(File.ReadAllText(Path.Combine(package.resolvedPath,
                "Runtime/Resources/DomiCube/PhonemeMatching/MusicWebV4/hubert_ko_child_jamo_vocab.json")));
            var matrix = PhonemeData.LoadMatrix(File.ReadAllText(Path.Combine(package.resolvedPath,
                "Runtime/Resources/DomiCube/PhonemeMatching/MusicWebV4/ko_child_v2.json")));
            using (var recognizer = new SentisPhonemeRecognizer(modelPath, vocab))
            {
                recognizer.Warmup();
                foreach (var item in (JArray)job["cases"])
                {
                    byte[] bytes = File.ReadAllBytes((string)item["input_file"]);
                    var audio = new float[bytes.Length / 4];
                    Buffer.BlockCopy(bytes, 0, audio, 0, bytes.Length);
                    recognizer.RecognizeWithText(audio, out string text, out var phonemes);
                    var row = new JObject { ["id"] = item["id"], ["text"] = text,
                        ["expected"] = item["text"], ["text_equal"] = text == (string)item["text"] };
                    results.Add(row);
                    if (item["targets"] != null)
                    {
                        var session = PronunciationSession.ForOrthographicWord(matrix, recognizer);
                        session.Begin(item["targets"].ToObject<string[]>());
                        var first = session.Push(audio, .5);
                        var second = session.Push(audio, 1);
                        row["confirmed"] = second.Confirmed;
                        row["target"] = second.WordDecision.MatchedTarget;
                        row["comparison"] = second.WordDecision.ComparisonText;
                        row["normalization"] = second.WordDecision.Normalization;
                        if (first.Confirmed || !second.Confirmed || second.WordDecision.MatchedTarget != (string)item["expected_target"])
                            throw new InvalidOperationException("Full recognition/postprocessing failed: " + item["id"]);
                    }
                    else if (!(bool)row["text_equal"])
                        throw new InvalidOperationException("CTC decode differs from reference: " + item["id"]);
                }
            }
            report["success"] = true;
        }
        catch (Exception e) { report["error"] = e.ToString(); UnityEngine.Debug.LogException(e); }
        File.WriteAllText(reportPath, report.ToString());
        if (!(bool)report["success"]) EditorApplication.Exit(1);
    }

    public static void InstallFromRelease()
    {
        string reportPath = Argument("-installReport");
        var report = new JObject { ["success"] = false, ["path"] = SharedSpeechModel.FilePath };
        try
        {
            double last = -1;
            bool installed = SharedModelInstaller.EnsureAsync(SharedSpeechModel.FilePath, SharedSpeechModel.Bytes,
                SharedSpeechModel.Sha256, null, SharedSpeechModel.DownloadUrls,
                (phase, progress) => {
                    double bucket = Math.Floor(progress * 10);
                    if (bucket != last) { last = bucket; UnityEngine.Debug.Log(phase + " " + (int)(progress * 100) + "%"); }
                }, CancellationToken.None).GetAwaiter().GetResult();
            var timer = Stopwatch.StartNew();
            string path = SharedSpeechModel.RequireInstalled();
            timer.Stop();
            report["downloaded"] = installed;
            report["receipt_reused"] = path == SharedSpeechModel.FilePath;
            report["repeat_check_ms"] = timer.Elapsed.TotalMilliseconds;
            report["sha256"] = SharedSpeechModel.Sha256;
            report["bytes"] = new FileInfo(path).Length;
            report["success"] = true;
        }
        catch (Exception e) { report["error"] = e.ToString(); UnityEngine.Debug.LogException(e); }
        File.WriteAllText(reportPath, report.ToString());
        if (!(bool)report["success"]) EditorApplication.Exit(1);
    }
}
