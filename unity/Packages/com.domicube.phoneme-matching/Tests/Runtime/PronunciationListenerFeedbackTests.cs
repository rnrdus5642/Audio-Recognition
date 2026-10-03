#if UNITY_EDITOR && UNITY_INCLUDE_TESTS
using System.Collections.Generic;
using System.Reflection;
using DomiCube.PhonemeMatching.Unity;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace DomiCube.PhonemeMatching.Tests
{
    public sealed class PronunciationListenerFeedbackTests
    {
        [Test]
        public void DetailedEventAndSnapshotAreAvailableBeforeLegacyCallback()
        {
            var gameObject = new GameObject("Feedback test (no microphone)");
            try
            {
                var listener = gameObject.AddComponent<PronunciationListener>();
                var result = new MatchResult
                {
                    TargetId = "apple", TargetText = "사과", Score = 0.9,
                    Distance = 0.5, Threshold = 0.65, Passed = true,
                    UserPhonemes = new List<string> { "tʰ", "a", "k", "w", "a" },
                    TargetPhonemes = new List<string> { "s", "a", "k", "w", "a" },
                    WindowEnd = 5,
                    Alignment = new List<AlignStep>
                    {
                        new AlignStep("tʰ", "s", AlignOp.Substitute),
                        new AlignStep("a", "a", AlignOp.Match), new AlignStep("k", "k", AlignOp.Match),
                        new AlignStep("w", "w", AlignOp.Match), new AlignStep("a", "a", AlignOp.Match)
                    }
                };
                var hit = new StreamingHit
                {
                    Result = result, Frames = 2, Streak = 2,
                    Feedback = PronunciationFeedback.FromMatch(result, 2, 2)
                };
                var order = new List<string>();
                listener.OnConfirmedDetailed.AddListener(feedback =>
                {
                    Assert.That(listener.IsListening, Is.False);
                    Assert.That(listener.LastConfirmation, Is.SameAs(feedback));
                    Assert.That(feedback.SubstitutionCount, Is.EqualTo(1));
                    Assert.That(feedback.KoreanFeedback.Items[0].Message,
                        Is.EqualTo("‘사과’의 ‘사’가 ‘타’에 가까운 소리로 인식됐어요."));
                    order.Add("detailed");
                });
                listener.OnConfirmed.AddListener((word, score) =>
                {
                    Assert.That(word, Is.EqualTo("사과"));
                    Assert.That(score, Is.EqualTo(0.9f));
                    Assert.That(listener.LastConfirmation, Is.Not.Null);
                    order.Add("legacy");
                });
                typeof(PronunciationListener).GetMethod("Confirm",
                    BindingFlags.Instance | BindingFlags.NonPublic).Invoke(listener, new object[] { hit.Feedback });
                Assert.That(order, Is.EqualTo(new[] { "detailed", "legacy" }));
                listener.StopListening();
                Assert.That(listener.LastConfirmation, Is.SameAs(hit.Feedback));

                // A new attempt clears old feedback, even if it cannot start.
                LogAssert.Expect(LogType.Error,
                    "[PronunciationListener] recognizer가 없습니다. SetRecognizer()를 먼저 호출하세요.");
                listener.Listen();
                Assert.That(listener.LastConfirmation, Is.Null);
            }
            finally
            {
                Object.DestroyImmediate(gameObject);
            }
        }
    }
}
#endif
