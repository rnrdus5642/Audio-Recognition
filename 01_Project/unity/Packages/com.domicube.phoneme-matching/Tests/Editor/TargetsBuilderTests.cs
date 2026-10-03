using System;
using System.IO;
using System.Reflection;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Editor.Tests
{
    public sealed class TargetsBuilderTests
    {
        private string _root;

        [SetUp]
        public void SetUp()
        {
            _root = Path.Combine(Path.GetTempPath(),
                "DomiCube-TargetsBuilder-Tests-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_root);
        }

        [TearDown]
        public void TearDown()
        {
            string temp = Path.GetFullPath(Path.GetTempPath())
                .TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
            Assert.That(Path.GetFullPath(_root).StartsWith(temp, StringComparison.OrdinalIgnoreCase), Is.True);
            Assert.That(Path.GetFileName(_root).StartsWith("DomiCube-TargetsBuilder-Tests-"), Is.True);
            Assert.That((File.GetAttributes(_root) & FileAttributes.ReparsePoint) == 0, Is.True);
            Directory.Delete(_root, recursive: true);
        }

        private static string Resolve(string path)
        {
            var method = typeof(TargetsBuilder).GetMethod("ResolveProjectDirectory",
                BindingFlags.NonPublic | BindingFlags.Static);
            Assert.That(method, Is.Not.Null);
            return (string)method.Invoke(null, new object[] { path });
        }

        private static void CreateProject(string directory)
        {
            Directory.CreateDirectory(Path.Combine(directory, "shared"));
            Directory.CreateDirectory(Path.Combine(directory, "python", "build"));
            File.WriteAllText(Path.Combine(directory, "shared", "words.csv"), "word\n");
            File.WriteAllText(Path.Combine(directory, "python", "build", "build_targets.py"), "# test\n");
        }

        [Test]
        public void WorkspaceSelectionReturnsNestedSourceDirectory()
        {
            string project = Path.Combine(_root, "01_Project");
            CreateProject(project);
            Assert.That(Resolve(_root), Is.EqualTo(project));
        }

        [Test]
        public void DirectProjectSelectionStillWorks()
        {
            string project = Path.Combine(_root, "01_Project");
            CreateProject(project);
            Assert.That(Resolve(project), Is.EqualTo(project));
        }

        [Test]
        public void LegacyReleaseLayoutStillWorks()
        {
            CreateProject(_root);
            Assert.That(Resolve(_root), Is.EqualTo(_root));
        }

        [Test]
        public void ExplicitSourceSelectionWinsWhenBothLayoutsExist()
        {
            CreateProject(_root);
            CreateProject(Path.Combine(_root, "01_Project"));
            Assert.That(Resolve(_root), Is.EqualTo(_root));
        }

        [TestCase(null)]
        [TestCase("")]
        public void EmptyPreferenceDoesNotSelectTheWorkingDirectory(string path)
        {
            Assert.That(Resolve(path), Is.Null);
        }

        [Test]
        public void EmptyOrMissingFolderIsRejected()
        {
            Assert.That(Resolve(_root), Is.Null);
            Assert.That(Resolve(Path.Combine(_root, "missing")), Is.Null);
        }

        [TestCase("shared/words.csv")]
        [TestCase("python/build/build_targets.py")]
        public void PartialCheckoutDoesNotCountAsAProject(string existingFile)
        {
            string path = Path.Combine(_root, "01_Project", existingFile);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, "incomplete checkout");
            Assert.That(Resolve(_root), Is.Null);
        }
    }
}
