"""The Python/UI payload and C# payload share explicit contract vectors."""

import json
from pathlib import Path

import pytest

from python.runtime.matching import (
    ConfusionMatrix, Matcher, MatchResult, PronunciationFeedback, StreamingMatcher,
)
from python.tools._web_feedback import pronunciation_feedback_html


ROOT = Path(__file__).resolve().parents[2]
VECTORS = json.loads((ROOT / (
    "unity/Packages/com.domicube.phoneme-matching/Tests/Runtime/feedback_vectors.json"
)).read_text(encoding="utf-8"))
APPLE = {"id": "apple", "text": "사과", "phonemes": ["s", "a", "k", "w", "a"], "threshold": 0.65}


@pytest.mark.parametrize("case", VECTORS, ids=[c["name"] for c in VECTORS])
def test_shared_feedback_contract(case):
    feedback = PronunciationFeedback.from_match(
        MatchResult(**case["result"]), case["frames"], case["streak"]
    )
    assert feedback.to_dict() == case["expected"]


def streaming(consecutive=2):
    matrix = ConfusionMatrix.from_json(ROOT / "shared/confusion_matrices/ko_child_v1.json")
    candidate = {**APPLE, "phonemes": APPLE["phonemes"].copy()}
    return StreamingMatcher(Matcher.for_streaming(matrix), [candidate], consecutive)


def test_confirmation_uses_last_frame_and_copies_mutable_lists():
    sm = streaming()
    assert sm.push(APPLE["phonemes"].copy()) is None
    user = ["t", "a", "k", "w", "a"]
    hit = sm.push(user)
    assert hit is not None
    feedback = hit.feedback
    assert (feedback.frames, feedback.streak) == (2, 2)
    assert feedback.substitution_count == 1
    assert feedback.score == hit.result.score < 1.0
    assert feedback.threshold == 0.65
    user[0] = "x"
    hit.result.target_phonemes[0] = "x"
    hit.result.alignment.clear()
    detached = feedback.to_dict()
    detached["alignment"].clear()
    sm.reset()
    assert feedback.user_phonemes[0] == "t"
    assert feedback.target_phonemes[0] == "s"
    assert len(feedback.alignment) == 5
    assert feedback.korean_feedback.items[0].heard_syllable == "다"


def test_context_limit_keeps_full_user_indices_and_excludes_prefix():
    sm = streaming(consecutive=1)
    hit = sm.push(["z"] * 40 + APPLE["phonemes"])
    assert hit is not None
    assert hit.feedback.window_start == 40
    assert hit.feedback.alignment[0].user_index == 40
    assert hit.feedback.insertion_count == 0


def test_rejected_frame_does_not_produce_confirmation():
    assert streaming(consecutive=1).push([]) is None


def test_html_shows_all_difference_types_and_excludes_outside_phonemes():
    case = VECTORS[0]
    feedback = PronunciationFeedback.from_match(MatchResult(**case["result"]), 4, 2)
    html = pronunciation_feedback_html(feedback)
    for label in ("교체 1개", "누락 1개", "추가 1개", "일치 3개", "구간 밖 2개"):
        assert label in html
    assert "4번째 확정 프레임" in html
    assert "실제 발음 오류를 단정" in html


def test_html_does_not_claim_no_differences_when_alignment_is_missing():
    case = VECTORS[2]
    feedback = PronunciationFeedback.from_match(MatchResult(**case["result"]), 4, 2)
    html = pronunciation_feedback_html(feedback)
    assert "정렬 정보가 없습니다" in html
    assert "다른 음소가 발견되지 않았습니다" not in html


def test_html_escapes_target_text_and_phonemes():
    case = VECTORS[0]
    result = MatchResult(**{**case["result"], "target_text": "<script>alert(1)</script>"})
    result.alignment = [("<svg/onload=alert(1)>", "<img>", "sub")]
    html = pronunciation_feedback_html(PronunciationFeedback.from_match(result, 4, 2))
    assert "<script>" not in html
    assert "<svg" not in html
    assert "<img>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img&gt;" in html
