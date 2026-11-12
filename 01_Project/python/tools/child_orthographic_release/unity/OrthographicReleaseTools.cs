using System;
using System.IO;
using System.Reflection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

public static class OrthographicReleaseTools
{
    public static string Argument(string name)
    {
        var values = Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < values.Length; i++)
            if (values[i] == name) return values[i + 1];
        throw new ArgumentException("Missing argument: " + name);
    }

    public static void Convert()
    {
        string source = Argument("-onnxSource"), target = Argument("-sentisTarget");
        Directory.CreateDirectory(Path.GetDirectoryName(target));
        var started = DateTime.UtcNow;
        try
        {
            var assembly = Assembly.Load("Unity.InferenceEngine.Editor");
            var type = assembly.GetType("Unity.InferenceEngine.Editor.Onnx.ONNXModelConverter", true);
            var converter = Activator.CreateInstance(type, new object[] { source });
            var model = (Model)type.GetMethod("Convert", BindingFlags.Instance | BindingFlags.Public).Invoke(converter, null);
            ModelWriter.Save(target, model);
            File.WriteAllText(target + ".import.json", JsonConvert.SerializeObject(new {
                success = true, unity = Application.unityVersion, source, target,
                bytes = new FileInfo(target).Length, seconds = (DateTime.UtcNow - started).TotalSeconds
            }, Formatting.Indented));
            Debug.Log("Orthographic model converted: " + target);
        }
        catch (Exception exception)
        {
            File.WriteAllText(target + ".import.json", JsonConvert.SerializeObject(new {
                success = false, error = exception.ToString()
            }, Formatting.Indented));
            throw;
        }
    }

    public static void Validate()
    {
        string directory = Argument("-caseDirectory");
        string source = Argument("-sentisTarget");
        string reportPath = Path.Combine(directory, "unity-validation-installed.json");
        Unity.Collections.NativeLeakDetection.Mode = Unity.Collections.NativeLeakDetectionMode.EnabledWithStackTrace;
        var results = new JArray();
        var report = new JObject { ["success"] = false, ["unity"] = Application.unityVersion,
            ["backend"] = "CPU", ["cases"] = results, ["model"] = source };
        var package = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(CtcVocabulary).Assembly);
        report["package_version"] = package.version;
        report["package_source"] = package.source.ToString();
        report["package_path"] = package.resolvedPath;
        try
        {
            const string assetPath = "Assets/Resources/Models/hubert_ko_child_jamo.sentis";
            Directory.CreateDirectory(Path.GetDirectoryName(assetPath));
            if (!File.Exists(assetPath)) File.Copy(source, assetPath);
            AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
            var asset = AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
            if (asset == null) throw new InvalidOperationException("The .sentis importer did not create a ModelAsset");
            report["asset_import_passed"] = true;
            report["asset_guid"] = AssetDatabase.AssetPathToGUID(assetPath);
            File.WriteAllText(reportPath, report.ToString());

            var vocabulary = PhonemeData.LoadCtcVocabulary(File.ReadAllText(
                Path.Combine(directory, "hubert_ko_child_jamo_vocab.json")));
            if (vocabulary.Size != 69 || vocabulary.Normalize || !vocabulary.ComposeHangulJamo)
                throw new InvalidOperationException("Incorrect vocabulary contract");
            var cases = JArray.Parse(File.ReadAllText(Path.Combine(directory, "cases.json")));
            using (var recognizer = SentisPhonemeRecognizer.FromFile(source, vocabulary, BackendType.CPU))
            {
                foreach (var item in cases)
                {
                    byte[] bytes = File.ReadAllBytes((string)item["input_file"]);
                    var samples = new float[bytes.Length / sizeof(float)];
                    Buffer.BlockCopy(bytes, 0, samples, 0, bytes.Length);
                    if (samples.Length != 40000) throw new InvalidDataException("Unexpected window size");
                    var started = DateTime.UtcNow;
                    recognizer.RecognizeWithText(samples, out var text, out var phonemes);
                    bool equal = text == (string)item["text"];
                    results.Add(new JObject { ["id"] = item["id"], ["expected"] = item["text"],
                        ["text"] = text, ["equal"] = equal, ["phoneme_count"] = phonemes.Count,
                        ["seconds"] = (DateTime.UtcNow - started).TotalSeconds });
                    File.WriteAllText(reportPath, report.ToString());
                    Debug.Log("Orthographic parity " + item["id"] + ": " + equal + " / " + text);
                    if (!equal) throw new InvalidOperationException("CTC output differs from Python: " + item["id"]);
                }
            }
            report["success"] = true;
            report["completed_utc"] = DateTime.UtcNow.ToString("o");
            File.WriteAllText(reportPath, report.ToString());
        }
        catch (Exception exception)
        {
            report["error"] = exception.ToString();
            File.WriteAllText(reportPath, report.ToString());
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }
}
