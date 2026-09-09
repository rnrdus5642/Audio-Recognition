"""Model uncertainty changes guidance only, including the reported 놔 -> 와 case."""
import json
from pathlib import Path

import pytest

from python.runtime.matching import ConfusionMatrix, Matcher, MatchResult, ModelErrors, PronunciationFeedback, StreamingMatcher

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "unity/Packages/com.domicube.phoneme-matching/Tests/Runtime"
VECTORS = json.loads((DATA / "model_feedback_vectors.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", VECTORS, ids=lambda c: c["name"])
def test_shared_model_guidance_vectors(case):
    errors = ModelErrors.from_json(ROOT / "shared/reference/reference_errors.json") if case["profile"] else None
    alignment = case["alignment"]
    result = MatchResult(
        target_text=case["word"], target_id="fixture", score=.9, threshold=.7, passed=True, distance=.2,
        target_phonemes=[s[1] for s in alignment if s[2] != "ins"],
        user_phonemes=[s[0] for s in alignment if s[2] != "del"], alignment=alignment)
    feedback = PronunciationFeedback.from_match(result, 2, 2, matrix_id=case["matrix"], model_errors=errors)
    assert feedback.feedback_review.status == case["status"]
    assert [(w.alignment_index, w.reason) for w in feedback.feedback_review.withheld] == [tuple(w) for w in case["withheld"]]
    assert [h.syllable_index for h in feedback.korean_feedback.items] == case["hint_syllables"]
    assert feedback.passed and feedback.score == .9 and feedback.threshold == .7
    assert len(feedback.alignment) == len(alignment)
    if case["status"] == "withheld":
        assert feedback.korean_feedback.summary == "발음에 문제 없어요."
        assert feedback.raw_korean_feedback.items
    elif case["status"] == "partial":
        assert feedback.korean_feedback.summary == "아래 부분만 조금 더 또렷하게 말해 보세요."
    before = feedback.to_dict()
    feedback.apply_model_errors(case["matrix"], errors)
    assert feedback.to_dict() == before


def test_nwa_still_confirms_and_does_not_blame_missing_n():
    matrix = ConfusionMatrix.from_json(ROOT / "shared/confusion_matrices/ko_child_v2.json")
    catalog = json.loads((ROOT / "shared/targets_music_demo.json").read_text(encoding="utf-8"))
    word = next(a for a in catalog["answers"] if a["text"] == "놔")
    errors = ModelErrors.from_json(ROOT / "shared/reference/reference_errors.json")
    baseline = StreamingMatcher(Matcher.for_streaming(matrix), [word])
    reviewed = StreamingMatcher(Matcher.for_streaming(matrix), [word], feedback_errors=errors)
    for _ in range(2):
        old = baseline.push(["w", "a"])
        new = reviewed.push(["w", "a"])
        assert bool(old) == bool(new)
    assert new is not None
    assert (new.result.score, new.frames, new.streak) == (old.result.score, old.frames, old.streak)
    assert new.feedback.deletion_count == 1
    assert new.feedback.korean_feedback.items == []
    assert new.feedback.feedback_review.status == "withheld"


def test_packaged_evidence_matches_source():
    packaged = ROOT / "unity/Packages/com.domicube.phoneme-matching/Runtime/Resources/DomiCube/PhonemeMatching/Feedback/reference_errors_child.json"
    assert packaged.read_bytes() == (ROOT / "shared/reference/reference_errors.json").read_bytes()
