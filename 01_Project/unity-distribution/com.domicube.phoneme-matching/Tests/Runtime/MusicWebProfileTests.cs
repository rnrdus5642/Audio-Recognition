#if UNITY_EDITOR && UNITY_INCLUDE_TESTS
using System;
using System.Linq;
using DomiCube.PhonemeMatching.Unity;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    public sealed class MusicWebProfileTests
    {
        [Test]
        public void ProfileKeepsAll82ExpressionsInTheirRequestedStages()
        {
            var p = MusicWebPresentation.Load();
            Assert.That(p.UsesOrthographicWordPolicy, Is.True);
            Assert.That(p.Stages.Select(s => s.Answers.Count), Is.EqualTo(new[] {10,9,9,9,9,9,8,10,9}));
            Assert.That(p.Catalog.Answers.Count, Is.EqualTo(47));
            Assert.That(p.GetStage(7).Answers, Does.Not.Contain("놓아야죠"));
            Assert.That(p.GetStage(3).Answers, Does.Contain("놓아야죠"));
            Assert.That(p.GetStage(1).Answers, Does.Contain("들까요"));
            Assert.That(p.Vocabulary.ComposeHangulJamo, Is.True);
            Assert.That(p.Vocabulary.Normalize, Is.False);
        }

        [Test]
        public void All82ExactWordsConfirmAfterAnUnrelatedPrefix()
        {
            var p = MusicWebPresentation.Load();
            foreach (var stage in p.Stages)
            foreach (var word in stage.Answers)
            {
                var policy = new OrthographicWordPolicy(p.Matrix);
                policy.Begin(stage.Answers);
                var a = policy.Push("오늘은 여기서 " + word, 2.5, 1);
                var b = policy.Push("오늘은 여기서 " + word, 3, 1);
                Assert.That(a.Confirmed, Is.False, word);
                Assert.That(b.Confirmed, Is.True, "stage " + stage.Id + " / " + word + " / " + b.DecisionReason);
                Assert.That(b.MatchedTarget, Is.EqualTo(word));
            }
        }

        [Test]
        public void ListenerUsesWebPolicyAndResetsWhenStageChanges()
        {
            var go = new UnityEngine.GameObject("MusicWebProfileTest");
            try
            {
                var listener = go.AddComponent<PronunciationListener>();
                MusicWebPresentation.ApplyTo(listener, 1);
                Assert.That(listener.UseOrthographicWordPolicy, Is.True);
                Assert.That(listener.TargetWords.Length, Is.EqualTo(10));
                Assert.That(listener.WindowSeconds, Is.EqualTo(2.5f));
                Assert.That(listener.HopSeconds, Is.EqualTo(.5f));
                Assert.That(listener.WordThreshold, Is.EqualTo(.85f));
                MusicWebPresentation.ApplyTo(listener, 7);
                Assert.That(listener.TargetWords.Length, Is.EqualTo(8));
                Assert.That(listener.IsListening, Is.False);
            }
            finally { UnityEngine.Object.DestroyImmediate(go); }
        }
    }
}
#endif
