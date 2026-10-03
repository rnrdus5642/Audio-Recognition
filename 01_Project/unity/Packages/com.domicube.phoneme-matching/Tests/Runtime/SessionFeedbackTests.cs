using System.Collections.Generic;
using NUnit.Framework;

namespace DomiCube.PhonemeMatching.Tests
{
    public sealed class SessionFeedbackTests
    {
        private sealed class Recognizer : IPhonemeRecognizer
        {
            public string Language => "ko";
            public int Calls;
            public List<string> Recognize(float[] audio)
                => new List<string> { "t", "a", "k", "w", "a" };
            public void RecognizeWithText(float[] audio, out string text, out List<string> phonemes)
            {
                Calls++;
                text = "다과";
                phonemes = Recognize(audio);
            }
        }

        [Test]
        public void SessionReturnsAndRetainsConfirmingFeedbackThenClearsIt()
        {
            var matrix = PhonemeData.LoadMatrix(RepoFiles.Read("shared/confusion_matrices/ko_child_v2.json"));
            var catalog = new TargetCatalog
            {
                Answers = new List<Answer>
                {
                    new Answer { Id = "apple", Text = "사과", Threshold = 0.65,
                        Phonemes = new List<string> { "s", "a", "k", "w", "a" } }
                }
            };
            var recognizer = new Recognizer();
            var session = new PronunciationSession(matrix, catalog, recognizer);
            Assert.That(session.Consecutive, Is.EqualTo(2));
            session.Begin("사과");
            var first = session.Push(new float[40000]);
            Assert.That(first.Confirmed, Is.False);
            Assert.That(first.Feedback, Is.Null);
            Assert.That(session.LastConfirmation, Is.Null);

            var second = session.Push(new float[40000]);
            Assert.That(second.Confirmed, Is.True);
            Assert.That(second.Feedback, Is.SameAs(session.LastConfirmation));
            Assert.That(second.Feedback.Frames, Is.EqualTo(2));
            Assert.That(second.Feedback.SubstitutionCount, Is.EqualTo(1));
            Assert.That(second.Feedback.KoreanFeedback.Items[0].HeardSyllable, Is.EqualTo("다"));
            Assert.That(session.IsActive, Is.False);
            Assert.That(recognizer.Calls, Is.EqualTo(2));
            session.End();
            Assert.That(session.LastConfirmation, Is.SameAs(second.Feedback));
            session.Begin("사과");
            Assert.That(session.LastConfirmation, Is.Null);
            Assert.That(second.Feedback.Frames, Is.EqualTo(2));
        }
    }
}
