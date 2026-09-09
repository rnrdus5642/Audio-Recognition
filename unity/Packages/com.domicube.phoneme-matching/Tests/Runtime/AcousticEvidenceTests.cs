using System;
using System.Collections.Generic;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    public sealed class AcousticEvidenceTests
    {
        [TestCase("")]
        [TestCase("가")]
        [TestCase("가가")]
        [TestCase("가나")]
        [TestCase("나가나")]
        public void CtcForwardSumAgreesWithEveryPossibleAlignment(string wanted)
        {
            var probabilities = new[] { .5,.2,.3, .1,.4,.5, .3,.4,.3, .2,.3,.5 };
            var logits = Array.ConvertAll(probabilities, p => (float)Math.Log(p));
            var vocab = new CtcVocabulary { Tokens = new[] { "가", "나", "_" }, BlankId = 2 };
            var scorer = new CtcSequenceScorer(logits, 4, 3, vocab);
            double sum = 0;
            for (int code = 0; code < 81; code++)
            {
                int remaining = code;
                var path = new List<int>();
                double probability = 1;
                for (int f = 0; f < 4; f++)
                {
                    int token = remaining % 3;
                    remaining /= 3;
                    path.Add(token);
                    probability *= probabilities[f * 3 + token];
                }
                if (CtcDecoder.Decode(path, vocab) == wanted) sum += probability;
            }
            Assert.That(scorer.Score(wanted), Is.EqualTo(Math.Log(sum)).Within(1e-6));
            Assert.That(scorer.Score("미등록"), Is.EqualTo(double.NegativeInfinity));
        }

        private static TargetCatalog Catalog() => PhonemeData.LoadTargets(RepoFiles.Read(
            "unity/Assets/StreamingAssets/targets_music_demo.json"));
        private static ConfusionMatrix Matrix() => PhonemeData.LoadMatrix(RepoFiles.Read(
            "unity/Assets/StreamingAssets/ko_child_v2.json"));
        private static AcousticEvidence Support(double score = -4) => new AcousticEvidence
        {
            TargetText = "잡아", ContrastText = "작아", LogProbability = score,
            GreedyLogProbability = -1, ContrastLogProbability = -1
        };

        [Test]
        public void SameRawTranscriptNeedsCurrentAudioEvidenceAndTwoFrames()
        {
            var a = Catalog().Find("잡아");
            var m = Matcher.ForStreaming(Matrix());
            var ipa = new[] { "tɕ", "a", "k", "a" };
            var evidence = new Dictionary<string, AcousticEvidence> { ["잡아"] = Support() };
            Assert.That(m.BestMatch(ipa, new[] { a }).Passed, Is.False);
            var stream = new StreamingMatcher(m, new[] { a });
            Assert.That(stream.Push(ipa, evidence), Is.Null);
            Assert.That(stream.Push(ipa), Is.Null);
            Assert.That(stream.Push(ipa, evidence), Is.Null);
            var hit = stream.Push(ipa, evidence);
            Assert.That(hit, Is.Not.Null);
            Assert.That(hit.Streak, Is.EqualTo(2));
            Assert.That(hit.Result.UserPhonemes, Is.EqualTo(ipa));
            Assert.That(hit.Feedback.KoreanFeedback.Items, Is.Empty);
            Assert.That(hit.Feedback.KoreanFeedback.Summary, Is.EqualTo("발음에 문제 없어요."));
            Assert.That(hit.Feedback.RawKoreanFeedback.Items, Is.Not.Empty);
            Assert.That(hit.Feedback.FeedbackReview.Withheld[0].Reason, Is.EqualTo("acoustic_keyword_recovery"));
            evidence["잡아"] = Support(-14);
            Assert.That(m.BestMatch(ipa, new[] { a }, evidence).Passed, Is.False);
            a.ContrastMargin = 0;
            a.ContrastiveAlternatives.Add(new ContrastiveAlternative
                { Text = "작아", ActionGroup = "small", Phonemes = new List<string>(ipa) });
            evidence["잡아"] = Support();
            Assert.That(m.BestMatch(ipa, new[] { a }, evidence).CompetitorRejected, Is.True);
        }

        [Test]
        public void SessionConnectsEvidenceWithoutChangingRawTextAndDiagnostics()
        {
            var session = new PronunciationSession(Matrix(), Catalog(), new EvidenceRecognizer());
            session.Begin("잡아");
            var first = session.Push(new float[40000]);
            Assert.That(first.Confirmed, Is.False);
            var second = session.Push(new float[40000]);
            Assert.That(second.Confirmed, Is.True);
            Assert.That(second.Text, Is.EqualTo("작아"));
            Assert.That(second.Best.AcousticEvidence, Is.Not.Null);
            var json = RecognitionDiagnostics.ToJson(new[] { "잡아" }, new[] { second });
            Assert.That(json, Does.Contain("acoustic_log_probability"));
            Assert.That(json, Does.Contain("작아"));
        }

        private sealed class EvidenceRecognizer : IPhonemeRecognizer, IAcousticEvidenceRecognizer
        {
            public string Language => "ko";
            public int SampleRate => 16000;
            public List<string> Recognize(float[] audio) => new List<string> { "tɕ", "a", "k", "a" };
            public void RecognizeWithText(float[] audio, out string text, out List<string> phonemes)
            { text = "작아"; phonemes = Recognize(audio); }
            public AcousticEvidence GetAcousticEvidence(string target, string contrast) => Support();
        }

        [TestCase(double.NaN)]
        [TestCase(double.PositiveInfinity)]
        [TestCase(double.NegativeInfinity)]
        [TestCase(1)]
        public void InvalidEvidenceDoesNotGrantAcceptance(double value)
        {
            var a = Catalog().Find("잡아");
            Assert.That(Support(value).Permits(a, 4, .8), Is.False);
        }

        [Test]
        public void InvalidPolicyFailsAtLoad()
        {
            const string json = "{\"answers\":[{\"id\":\"a\",\"text\":\"a\",\"phonemes\":[\"a\"],\"acoustic_keyword_policy\":{}}]}";
            Assert.That(() => PhonemeData.LoadTargets(json), Throws.ArgumentException.With.Message.Contains("acoustic_keyword_policy"));
        }
    }
}
