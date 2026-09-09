using System;
using System.IO;
using NUnit.Framework;
using UnityEditor.PackageManager;

namespace DomiCube.PhonemeMatching.Editor.Tests
{
    public sealed class PronunciationTestSettingsTests
    {
        private string _source;
        private string _scratch;

        [SetUp]
        public void SetUp()
        {
            var package = PackageInfo.FindForAssembly(typeof(PronunciationTestSettings).Assembly);
            Assert.That(package, Is.Not.Null);
            _source = Path.Combine(package.resolvedPath, "Samples~/KoreanData");
            _scratch = Path.Combine(Path.GetTempPath(),
                "DomiCube-TestSettings-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_scratch);
        }

        [TearDown]
        public void TearDown()
        {
            if (_scratch == null || !Directory.Exists(_scratch)) return;
            string temp = Path.GetFullPath(Path.GetTempPath())
                .TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
            Assert.That(Path.GetFullPath(_scratch).StartsWith(temp, StringComparison.OrdinalIgnoreCase), Is.True);
            Assert.That(Path.GetFileName(_scratch).StartsWith("DomiCube-TestSettings-"), Is.True);
            Assert.That((File.GetAttributes(_scratch) & FileAttributes.ReparsePoint) == 0, Is.True);
            Directory.Delete(_scratch, recursive: true);
        }

        [TestCase("Assets/Resources/Models/wav2vec2_ko.onnx", false)]
        [TestCase("Assets/Resources/Models/wav2vec2_ko_child.onnx", true)]
        [TestCase("Assets/Moved/wav2vec2_ko_child.onnx", true)]
        [TestCase("Assets/Moved/WAV2VEC2_KO_CHILD.ONNX", true)]
        [TestCase(@"Assets\Moved\wav2vec2_ko_child.onnx", true)]
        public void StandardNamesResolveAutomatically(string path, bool child)
        {
            Assert.That(PronunciationTestSettings.Resolve(path, TestDataChoice.Unspecified),
                Is.SameAs(child ? PronunciationTestSettings.Child : PronunciationTestSettings.Base));
        }

        [TestCase(null)]
        [TestCase("")]
        [TestCase("Assets/child-models/custom.onnx")]
        [TestCase("Assets/wav2vec2_ko_child_copy.onnx")]
        public void UnknownNameDoesNotSilentlyUseBaseSettings(string path)
        {
            Assert.That(PronunciationTestSettings.FromModelPath(path), Is.Null);
            Assert.That(PronunciationTestSettings.Resolve(path, TestDataChoice.Unspecified), Is.Null);
        }

        [TestCase(false)]
        [TestCase(true)]
        public void RenamedModelCanSelectItsDataExplicitly(bool child)
        {
            Assert.That(PronunciationTestSettings.Resolve("Assets/renamed.onnx",
                child ? TestDataChoice.Child : TestDataChoice.Base),
                Is.SameAs(child ? PronunciationTestSettings.Child : PronunciationTestSettings.Base));
        }

        [TestCase("Assets/wav2vec2_ko_child.onnx", true)]
        [TestCase("Assets/wav2vec2_ko.onnx", false)]
        public void StandardNameOverridesStaleManualChoice(string path, bool child)
        {
            Assert.That(PronunciationTestSettings.Resolve(path,
                child ? TestDataChoice.Base : TestDataChoice.Child),
                Is.SameAs(child ? PronunciationTestSettings.Child : PronunciationTestSettings.Base));
        }

        [TestCase(false, 0.65, 0.05, 6.0)]
        [TestCase(true, 0.875, 0.005, 8.0)]
        public void LoadsMatchingThresholdAndStreamingProfile(
            bool child, double threshold, double skipCost, double context)
        {
            var settings = child ? PronunciationTestSettings.Child : PronunciationTestSettings.Base;
            // Only this model's three files are required, not the other pair.
            Copy(settings.MatrixFile, settings.TargetsFile, PronunciationTestSettings.VocabularyFile);
            settings.Load(_scratch, out var matrix, out var catalog, out var vocabulary);
            var matcher = Matcher.ForStreaming(matrix);
            Assert.That(catalog.Find("사과").Threshold, Is.EqualTo(threshold));
            Assert.That(matcher.SkipCost, Is.EqualTo(skipCost));
            Assert.That(matcher.ContextMult, Is.EqualTo(context));
            Assert.That(vocabulary, Is.Not.Null);
        }

        [TestCase("ko_child_v2.json")]
        [TestCase("targets_child.json")]
        [TestCase("wav2vec2_ko_vocab.json")]
        public void MissingChildFileDoesNotFallBackToBaseOrInstallAnything(string missing)
        {
            foreach (string file in new[] { "ko_child_v1.json", "targets.json",
                "ko_child_v2.json", "targets_child.json", "wav2vec2_ko_vocab.json" })
                if (file != missing) Copy(file);
            var error = Assert.Throws<FileNotFoundException>(() =>
                PronunciationTestSettings.Child.Load(_scratch, out _, out _, out _));
            Assert.That(error.FileName, Is.EqualTo(Path.Combine(_scratch, missing)));
            Assert.That(error.Message, Does.Contain(missing));
            Assert.That(Directory.GetFiles(_scratch).Length, Is.EqualTo(4));
            Assert.That(File.Exists(Path.Combine(_scratch, missing)), Is.False);
        }

        [Test]
        public void CustomizedChildTargetsAreLoadedWithoutReplacement()
        {
            Copy("ko_child_v2.json", "wav2vec2_ko_vocab.json");
            const string custom = "{\"answers\":[{\"id\":\"audit_custom\",\"text\":\"내단어\","
                + "\"phonemes\":[\"n\",\"ɛ\"],\"threshold\":0.91}]}";
            string path = Path.Combine(_scratch, "targets_child.json");
            File.WriteAllText(path, custom);
            PronunciationTestSettings.Child.Load(_scratch, out _, out var catalog, out _);
            Assert.That(catalog.Find("audit_custom").Threshold, Is.EqualTo(0.91));
            Assert.That(catalog.Answers.Count, Is.EqualTo(1));
            Assert.That(File.ReadAllText(path), Is.EqualTo(custom));
        }

        private void Copy(params string[] files)
        {
            foreach (string file in files)
                File.Copy(Path.Combine(_source, file), Path.Combine(_scratch, file));
        }

        [Test]
        public void ChildModelCanOptIntoMusicDemoWithoutChangingOrdinaryChildSettings()
        {
            Assert.That(PronunciationTestSettings.Resolve("Assets/wav2vec2_ko_child.onnx", TestDataChoice.MusicDemo),
                Is.SameAs(PronunciationTestSettings.MusicDemo));
            Assert.That(PronunciationTestSettings.Resolve("Assets/wav2vec2_ko_child.onnx", TestDataChoice.Unspecified),
                Is.SameAs(PronunciationTestSettings.Child));
            Assert.That(PronunciationTestSettings.Resolve("Assets/wav2vec2_ko.onnx", TestDataChoice.MusicDemo), Is.Null);
            PronunciationTestSettings.MusicDemo.Load(_scratch, out _, out var catalog, out _);
            Assert.That(catalog.Answers.Count, Is.EqualTo(12));
            Assert.That(catalog.Find("북"), Is.Null);
            Assert.That(catalog.Find("잡아").ConfirmationFrames, Is.EqualTo(2));
            Assert.That(catalog.Find("잡아").ShortUtterancePolicy.Threshold, Is.EqualTo(.85));
            Assert.That(catalog.Find("쳐").ContextPrefixes.Count, Is.EqualTo(4));
        }

        [Test]
        public void MusicDemoUsesPackageResourcesAndLeavesOldProjectDataUntouched()
        {
            var path = Path.Combine(_scratch, "targets_music_demo.json");
            const string old = "old project customization";
            File.WriteAllText(path, old);
            PronunciationTestSettings.MusicDemo.Load(_scratch, out var matrix, out var catalog, out var vocab);
            Assert.That(matrix.MatrixId, Is.EqualTo("ko_child_v2"));
            Assert.That(catalog.Find("흔들어").ConfirmationFrames, Is.EqualTo(1));
            Assert.That(catalog.Find("들어").ContrastMargin, Is.EqualTo(0.08));
            Assert.That(vocab, Is.Not.Null);
            Assert.That(File.ReadAllText(path), Is.EqualTo(old));
            Assert.That(Directory.GetFiles(_scratch).Length, Is.EqualTo(1));
            PronunciationTestSettings.MusicDemo.Load(Path.Combine(_scratch, "not-installed"), out _, out _, out _);
            Assert.That(Directory.Exists(Path.Combine(_scratch, "not-installed")), Is.False);
        }
    }
}
