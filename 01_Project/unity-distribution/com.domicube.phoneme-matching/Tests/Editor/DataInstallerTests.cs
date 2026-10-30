using System;
using System.IO;
using System.Linq;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Editor.Tests
{
    public sealed class DataInstallerTests
    {
        private static readonly string[] FileNames =
        {
            "ko_child_v1.json", "targets.json", "ko_child_v2.json",
            "targets_child.json", "targets_music_demo.json", "wav2vec2_ko_vocab.json"
        };
        private string _root;
        private string _source;
        private string _destination;

        [SetUp]
        public void SetUp()
        {
            _root = Path.Combine(Path.GetTempPath(),
                "DomiCube-DataInstaller-Tests-" + Guid.NewGuid().ToString("N"));
            _source = Path.Combine(_root, "package");
            _destination = Path.Combine(_root, "StreamingAssets");
            Directory.CreateDirectory(_source);
            Directory.CreateDirectory(_destination);
            foreach (string file in FileNames)
                File.WriteAllText(Path.Combine(_source, file), "package default: " + file);
        }

        [TearDown]
        public void TearDown()
        {
            if (_root == null || !Directory.Exists(_root)) return;
            string temp = Path.GetFullPath(Path.GetTempPath())
                .TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
            Assert.That(Path.GetFullPath(_root).StartsWith(temp, StringComparison.OrdinalIgnoreCase), Is.True);
            Assert.That(Path.GetFileName(_root).StartsWith("DomiCube-DataInstaller-Tests-"), Is.True);
            Assert.That((File.GetAttributes(_root) & FileAttributes.ReparsePoint) == 0, Is.True);
            Directory.Delete(_root, recursive: true);
        }

        [Test]
        public void FreshSetupInstallsAllSixFilesWithoutAnOverwritePrompt()
        {
            bool ok = DataInstaller.InstallFiles(_source, _destination, false,
                _ => throw new Exception("Missing-only setup must not prompt"), out int copied);
            Assert.That(ok, Is.True);
            Assert.That(copied, Is.EqualTo(6));
            AssertDefaults();
        }

        [Test]
        public void MissingChildFilesDoNotReplaceCustomAdultData()
        {
            string[] adultFiles = { "ko_child_v1.json", "targets.json", "wav2vec2_ko_vocab.json" };
            var timestamps = adultFiles.ToDictionary(file => file, file =>
            {
                string path = Path.Combine(_destination, file);
                File.WriteAllText(path, "custom word / threshold / data: " + file);
                File.SetLastWriteTimeUtc(path, new DateTime(2020, 1, 2, 0, 0, 0, DateTimeKind.Utc));
                return File.GetLastWriteTimeUtc(path);
            });
            Assert.That(DataInstaller.InstallFiles(_source, _destination, false,
                _ => throw new Exception("Existing data must be preserved"), out int copied), Is.True);
            Assert.That(copied, Is.EqualTo(3));
            foreach (string file in adultFiles)
            {
                string path = Path.Combine(_destination, file);
                Assert.That(File.ReadAllText(path), Is.EqualTo("custom word / threshold / data: " + file));
                Assert.That(File.GetLastWriteTimeUtc(path), Is.EqualTo(timestamps[file]));
            }
            foreach (string file in new[] { "ko_child_v2.json", "targets_child.json", "targets_music_demo.json" })
                Assert.That(File.ReadAllBytes(Path.Combine(_destination, file)),
                    Is.EqualTo(File.ReadAllBytes(Path.Combine(_source, file))));
        }

        [Test]
        public void RepeatedMissingOnlySetupLeavesEveryExistingFileUntouched()
        {
            foreach (string file in FileNames)
                File.WriteAllText(Path.Combine(_destination, file), "custom: " + file);
            Assert.That(DataInstaller.InstallFiles(_source, _destination, false,
                null, out int copied), Is.True);
            Assert.That(copied, Is.EqualTo(0));
            foreach (string file in FileNames)
                Assert.That(File.ReadAllText(Path.Combine(_destination, file)), Is.EqualTo("custom: " + file));
        }

        [Test]
        public void ExplicitReplacementRequiresApprovalAndThenCopiesDefaults()
        {
            foreach (string file in FileNames)
                File.WriteAllText(Path.Combine(_destination, file), "custom: " + file);
            int prompts = 0;
            bool ok = DataInstaller.InstallFiles(_source, _destination, true, existing =>
            {
                prompts++;
                Assert.That(existing, Is.EquivalentTo(FileNames));
                Assert.That(File.ReadAllText(Path.Combine(_destination, "targets.json")),
                    Is.EqualTo("custom: targets.json"));
                return true;
            }, out int copied);
            Assert.That(ok, Is.True);
            Assert.That(prompts, Is.EqualTo(1));
            Assert.That(copied, Is.EqualTo(6));
            AssertDefaults();
        }

        [Test]
        public void CancellingReplacementMakesNoChangesIncludingMissingFiles()
        {
            File.WriteAllText(Path.Combine(_destination, "targets.json"), "custom targets");
            Assert.That(DataInstaller.InstallFiles(_source, _destination, true,
                _ => false, out int copied), Is.False);
            Assert.That(copied, Is.EqualTo(0));
            Assert.That(Directory.GetFiles(_destination).Length, Is.EqualTo(1));
            Assert.That(File.ReadAllText(Path.Combine(_destination, "targets.json")), Is.EqualTo("custom targets"));
        }

        [Test]
        public void ReplacementWithoutAnApprovalCallbackDoesNotOverwrite()
        {
            File.WriteAllText(Path.Combine(_destination, "targets.json"), "custom targets");
            Assert.That(DataInstaller.InstallFiles(_source, _destination, true,
                null, out int copied), Is.False);
            Assert.That(copied, Is.EqualTo(0));
            Assert.That(File.ReadAllText(Path.Combine(_destination, "targets.json")), Is.EqualTo("custom targets"));
        }

        [Test]
        public void MissingPackageSourceFailsBeforeWritingAnything()
        {
            File.WriteAllText(Path.Combine(_destination, "targets.json"), "custom targets");
            File.Delete(Path.Combine(_source, "wav2vec2_ko_vocab.json"));
            Assert.Throws<FileNotFoundException>(() =>
                DataInstaller.InstallFiles(_source, _destination, false, null, out _));
            Assert.That(Directory.GetFiles(_destination).Length, Is.EqualTo(1));
            Assert.That(File.ReadAllText(Path.Combine(_destination, "targets.json")), Is.EqualTo("custom targets"));
        }

        [Test]
        public void FilesAppearingAfterApprovalAreNotSilentlyOverwritten()
        {
            File.WriteAllText(Path.Combine(_destination, "targets.json"), "custom targets");
            string newlyCreated = Path.Combine(_destination, "targets_child.json");
            Assert.Throws<IOException>(() => DataInstaller.InstallFiles(_source, _destination, true, existing =>
            {
                Assert.That(existing, Is.EquivalentTo(new[] { "targets.json" }));
                File.WriteAllText(newlyCreated, "created after approval");
                return true;
            }, out _));
            Assert.That(File.ReadAllText(newlyCreated), Is.EqualTo("created after approval"));
        }

        private void AssertDefaults()
        {
            foreach (string file in FileNames)
                Assert.That(File.ReadAllBytes(Path.Combine(_destination, file)),
                    Is.EqualTo(File.ReadAllBytes(Path.Combine(_source, file))));
        }
    }
}
