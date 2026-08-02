using System;
using System.Collections.Generic;
using DomiCube.PhonemeMatching.Korean;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    /// <summary>
    /// Checks this port against reference values produced by the Python
    /// implementation (python/tools/export_parity_vectors.py).
    ///
    /// The thresholds, skip_cost and streaming profile were all tuned in
    /// Python against recorded audio. If C# computes even slightly
    /// different scores those measurements stop describing what ships,
    /// and nothing would visibly break - both sides still return a
    /// plausible number. These tests make the drift loud.
    ///
    /// Regenerate the vectors after changing the matcher or the matrix:
    ///     python -m python.tools.export_parity_vectors
    /// </summary>
    [TestFixture]
    public sealed class ParityTests
    {
        private const double Tolerance = 1e-9;

        private static JObject _vectors;
        private static ConfusionMatrix _matrix;
        private static Matcher _batch;
        private static Matcher _streaming;
        private static Dictionary<string, Answer> _targetsById;

        [OneTimeSetUp]
        public void LoadFixtures()
        {
            _vectors = JObject.Parse(ReadRepoFile(
                "unity/Packages/com.domicube.phoneme-matching/Tests/Runtime/"
                + "parity_vectors.json"));
            _matrix = PhonemeData.LoadMatrix(ReadRepoFile(
                "shared/confusion_matrices/ko_child_v1.json"));
            _batch = new Matcher(_matrix);
            _streaming = Matcher.ForStreaming(_matrix);

            _targetsById = new Dictionary<string, Answer>();
            foreach (var t in (JArray)_vectors["targets"])
            {
                var answer = new Answer
                {
                    Id = (string)t["id"],
                    Text = (string)t["text"],
                    Threshold = (double)t["threshold"]
                };
                foreach (var p in (JArray)t["phonemes"])
                {
                    answer.Phonemes.Add((string)p);
                }

                _targetsById[answer.Id] = answer;
            }
        }

        /// <summary>
        /// Resolve a repo-relative path by walking up from the running
        /// assembly, so the same tests work under `dotnet test` and the
        /// Unity Test Runner regardless of working directory.
        /// </summary>
        private static string ReadRepoFile(string relative)
        {
            return RepoFiles.Read(relative);
        }

        private static List<string> Strings(JToken token)
        {
            var list = new List<string>();
            foreach (var t in (JArray)token)
            {
                list.Add((string)t);
            }

            return list;
        }

        [Test]
        public void VectorFileIsPopulated()
        {
            // Every parity check below iterates a JArray, so an empty or
            // truncated vector file would let them all pass silently.
            Assert.That(((JArray)_vectors["hangul_to_ipa"]).Count,
                Is.GreaterThanOrEqualTo(20));
            Assert.That(((JArray)_vectors["matrix_costs"]["substitution"]).Count,
                Is.GreaterThanOrEqualTo(10));
            Assert.That(((JArray)_vectors["score_against"]["batch"]).Count,
                Is.GreaterThanOrEqualTo(50));
            Assert.That(((JArray)_vectors["score_against"]["streaming"]).Count,
                Is.GreaterThanOrEqualTo(50));
            Assert.That(((JArray)_vectors["best_match"]).Count,
                Is.GreaterThanOrEqualTo(20));
            Assert.That(((JArray)_vectors["streaming_sessions"]).Count,
                Is.GreaterThanOrEqualTo(4));
            Assert.That(_targetsById.Count, Is.GreaterThanOrEqualTo(10));

            // The vectors must describe the matrix the tests load.
            Assert.That((string)_vectors["matrix_version"],
                Is.EqualTo(_matrix.Version),
                "parity vectors are stale - re-run "
                + "python -m python.tools.export_parity_vectors");
        }

        // ------------------------------------------------------------------
        // 1. Hangul -> IPA
        // ------------------------------------------------------------------

        [Test]
        public void HangulToIpaMatchesPython()
        {
            var failures = new List<string>();
            foreach (var c in (JArray)_vectors["hangul_to_ipa"])
            {
                var text = (string)c["text"];
                var expected = Strings(c["phonemes"]);
                var actual = JamoIpa.ToPhonemes(text);
                if (!ListEquals(expected, actual))
                {
                    failures.Add(
                        $"{text}: expected [{string.Join(" ", expected)}] "
                        + $"got [{string.Join(" ", actual)}]");
                }
            }

            Assert.That(failures, Is.Empty, string.Join("\n", failures));
        }

        [Test]
        public void NonHangulIsSkipped()
        {
            Assert.That(JamoIpa.ToPhonemes("abc 123 !?"), Is.Empty);
            Assert.That(JamoIpa.ToPhonemes(null), Is.Empty);
            Assert.That(JamoIpa.ToPhonemes(string.Empty), Is.Empty);
        }

        // ------------------------------------------------------------------
        // 2. Confusion matrix costs
        // ------------------------------------------------------------------

        [Test]
        public void MatrixCostsMatchPython()
        {
            var costs = _vectors["matrix_costs"];

            foreach (var c in (JArray)costs["substitution"])
            {
                Assert.That(
                    _matrix.SubCost((string)c["a"], (string)c["b"]),
                    Is.EqualTo((double)c["cost"]).Within(Tolerance),
                    $"sub {(string)c["a"]}|{(string)c["b"]}");
            }

            foreach (var c in (JArray)costs["deletion"])
            {
                Assert.That(
                    _matrix.DelCost((string)c["phoneme"]),
                    Is.EqualTo((double)c["cost"]).Within(Tolerance),
                    $"del {(string)c["phoneme"]}");
            }

            foreach (var c in (JArray)costs["insertion"])
            {
                Assert.That(
                    _matrix.InsCost((string)c["phoneme"]),
                    Is.EqualTo((double)c["cost"]).Within(Tolerance),
                    $"ins {(string)c["phoneme"]}");
            }
        }

        [Test]
        public void SubstitutionIsCommutative()
        {
            Assert.That(_matrix.SubCost("k", "k͈"),
                Is.EqualTo(_matrix.SubCost("k͈", "k")).Within(Tolerance));
        }

        [Test]
        public void ProfilesMatchPython()
        {
            var p = _vectors["profiles"];

            Assert.That(_batch.SkipCost,
                Is.EqualTo((double)p["batch"]["skip_cost"]).Within(Tolerance));
            Assert.That(_batch.Coverage,
                Is.EqualTo((double)p["batch"]["coverage"]).Within(Tolerance));
            Assert.That(_batch.ContextMult, Is.Null);

            Assert.That(_streaming.SkipCost,
                Is.EqualTo((double)p["streaming"]["skip_cost"]).Within(Tolerance));
            Assert.That(_streaming.Coverage,
                Is.EqualTo((double)p["streaming"]["coverage"]).Within(Tolerance));
            Assert.That(_streaming.ContextMult,
                Is.EqualTo((double)p["streaming"]["context_mult"]).Within(Tolerance));
        }

        // ------------------------------------------------------------------
        // 3. score_against
        // ------------------------------------------------------------------

        [Test]
        public void BatchScoresMatchPython()
        {
            AssertScoreVectors("batch", _batch);
        }

        [Test]
        public void StreamingScoresMatchPython()
        {
            AssertScoreVectors("streaming", _streaming);
        }

        private static void AssertScoreVectors(string profile, Matcher matcher)
        {
            var failures = new List<string>();
            int i = 0;
            foreach (var c in (JArray)_vectors["score_against"][profile])
            {
                var user = Strings(c["user"]);
                var target = Strings(c["target"]);
                double d = matcher.ScoreAgainst(
                    user, target, out double score, out int ws, out int we);

                var label = $"[{i}] user=[{string.Join(" ", user)}] "
                    + $"target=[{string.Join(" ", target)}]";

                if (Math.Abs(d - (double)c["distance"]) > 1e-6)
                {
                    failures.Add(
                        $"{label} distance {d} != {(double)c["distance"]}");
                }

                if (Math.Abs(score - (double)c["score"]) > 1e-6)
                {
                    failures.Add(
                        $"{label} score {score} != {(double)c["score"]}");
                }

                if (ws != (int)c["window_start"] || we != (int)c["window_end"])
                {
                    failures.Add(
                        $"{label} window [{ws},{we}] != "
                        + $"[{(int)c["window_start"]},{(int)c["window_end"]}]");
                }

                matcher.ContextSlice(user, target, out int start, out int dropped);
                if (dropped != (int)c["context_dropped"])
                {
                    failures.Add(
                        $"{label} dropped {dropped} != "
                        + $"{(int)c["context_dropped"]}");
                }

                i++;
            }

            Assert.That(failures, Is.Empty,
                $"{profile}: {failures.Count} mismatch\n"
                + string.Join("\n", failures.GetRange(
                    0, Math.Min(10, failures.Count))));
        }

        // ------------------------------------------------------------------
        // 4. best_match
        // ------------------------------------------------------------------

        [Test]
        public void BestMatchMatchesPython()
        {
            var failures = new List<string>();
            int i = 0;
            foreach (var c in (JArray)_vectors["best_match"])
            {
                var matcher = (string)c["profile"] == "streaming"
                    ? _streaming
                    : _batch;
                var user = Strings(c["user"]);
                var candidates = Candidates(c["candidate_ids"]);
                var r = matcher.BestMatch(user, candidates);

                var label = $"[{i}] {(string)c["profile"]} "
                    + $"user=[{string.Join(" ", user)}]";

                if (r.TargetId != (string)c["target_id"])
                {
                    failures.Add(
                        $"{label} id {r.TargetId} != {(string)c["target_id"]}");
                }

                if (Math.Abs(r.Score - (double)c["score"]) > 1e-6)
                {
                    failures.Add(
                        $"{label} score {r.Score} != {(double)c["score"]}");
                }

                if (r.Passed != (bool)c["passed"])
                {
                    failures.Add(
                        $"{label} passed {r.Passed} != {(bool)c["passed"]}");
                }

                i++;
            }

            Assert.That(failures, Is.Empty,
                string.Join("\n", failures.GetRange(
                    0, Math.Min(10, failures.Count))));
        }

        private static List<Answer> Candidates(JToken ids)
        {
            var list = new List<Answer>();
            foreach (var id in (JArray)ids)
            {
                list.Add(_targetsById[(string)id]);
            }

            return list;
        }

        // ------------------------------------------------------------------
        // 5. StreamingMatcher
        // ------------------------------------------------------------------

        [Test]
        public void StreamingSessionsMatchPython()
        {
            var failures = new List<string>();
            foreach (var c in (JArray)_vectors["streaming_sessions"])
            {
                var caseId = (string)c["case_id"];
                int consecutive = (int)c["consecutive"];
                var sm = new StreamingMatcher(
                    Matcher.ForStreaming(_matrix),
                    Candidates(c["candidate_ids"]),
                    consecutive);

                var expectedStreaks = new List<int>();
                foreach (var s in (JArray)c["streaks"])
                {
                    expectedStreaks.Add((int)s);
                }

                int? firedAt = null;
                string firedId = null;
                int frame = 0;
                foreach (var f in (JArray)c["frames"])
                {
                    var hit = sm.Push(Strings(f));
                    if (sm.Streak != expectedStreaks[frame])
                    {
                        failures.Add(
                            $"{caseId}/c{consecutive} frame {frame}: streak "
                            + $"{sm.Streak} != {expectedStreaks[frame]}");
                    }

                    if (hit != null && firedAt == null)
                    {
                        firedAt = frame;
                        firedId = hit.Result.TargetId;
                    }

                    frame++;
                }

                var expectedFire = c["fired_at_frame"].Type == JTokenType.Null
                    ? (int?)null
                    : (int)c["fired_at_frame"];
                if (firedAt != expectedFire)
                {
                    failures.Add(
                        $"{caseId}/c{consecutive}: fired at {firedAt} != "
                        + $"{expectedFire}");
                }

                var expectedId = (string)c["fired_target_id"];
                if (firedId != expectedId)
                {
                    failures.Add(
                        $"{caseId}/c{consecutive}: fired id {firedId} != "
                        + $"{expectedId}");
                }
            }

            Assert.That(failures, Is.Empty,
                string.Join("\n", failures.GetRange(
                    0, Math.Min(10, failures.Count))));
        }

        // ------------------------------------------------------------------
        // Behavioural guards that do not depend on the vectors
        // ------------------------------------------------------------------

        [Test]
        public void StreakRequiresTheSameAnswer()
        {
            var apple = _targetsById["apple"];
            var mom = _targetsById["mom"];
            var sm = new StreamingMatcher(
                Matcher.ForStreaming(_matrix),
                new List<Answer> { apple, mom }, 3);

            Assert.That(sm.Push(apple.Phonemes), Is.Null);
            Assert.That(sm.Push(mom.Phonemes), Is.Null);
            Assert.That(sm.Push(apple.Phonemes), Is.Null);
            Assert.That(sm.Streak, Is.EqualTo(1));
        }

        [Test]
        public void ResetClearsStreak()
        {
            var apple = _targetsById["apple"];
            var sm = new StreamingMatcher(
                Matcher.ForStreaming(_matrix),
                new List<Answer> { apple }, 2);

            sm.Push(apple.Phonemes);
            sm.Reset();
            Assert.That(sm.Streak, Is.EqualTo(0));
            Assert.That(sm.Push(apple.Phonemes), Is.Null);
        }

        [Test]
        public void ContextLimitBoundsTheScoredWindow()
        {
            var target = _targetsById["apple"].Phonemes;
            var filler = new List<string>
                { "n", "i", "l", "o", "m", "u", "t", "e" };

            _streaming.ScoreAgainst(
                Repeat(filler, 5, target), target, out double a, out _, out _);
            _streaming.ScoreAgainst(
                Repeat(filler, 20, target), target, out double b, out _, out _);
            Assert.That(a, Is.EqualTo(b).Within(Tolerance),
                "speech older than the context limit must stop mattering");

            // Batch has no limit, so the same input keeps degrading. Kept
            // short enough that neither score has bottomed out at 0.
            _batch.ScoreAgainst(
                Repeat(filler, 1, target), target, out double c, out _, out _);
            _batch.ScoreAgainst(
                Repeat(filler, 3, target), target, out double d, out _, out _);
            Assert.That(d, Is.GreaterThan(0.0));
            Assert.That(d, Is.LessThan(c),
                "batch has no context limit, so it keeps degrading");
        }

        [Test]
        public void RejectsInvalidConfiguration()
        {
            Assert.Throws<ArgumentOutOfRangeException>(
                () => new Matcher(_matrix, coverage: 0.0));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => new StreamingMatcher(
                    _batch, new List<Answer>(), 0));
        }

        /// <summary>filler repeated `times`, then the target appended.</summary>
        private static List<string> Repeat(
            List<string> filler, int times, List<string> tail)
        {
            var result = new List<string>();
            for (int i = 0; i < times; i++)
            {
                result.AddRange(filler);
            }

            result.AddRange(tail);
            return result;
        }

        private static bool ListEquals(List<string> a, List<string> b)
        {
            if (a.Count != b.Count)
            {
                return false;
            }

            for (int i = 0; i < a.Count; i++)
            {
                if (!string.Equals(a[i], b[i], StringComparison.Ordinal))
                {
                    return false;
                }
            }

            return true;
        }
    }
}
