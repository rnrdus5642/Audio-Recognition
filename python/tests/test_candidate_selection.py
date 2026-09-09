"""Accept-any questions rank passing answers before failing ones.

Use the actual child-model package data as well as threshold boundary cases:
the original regression was a passing 강아지 hidden by a higher-scoring 빵
that did not clear its own (stricter) threshold.
"""

import json
from pathlib import Path

import pytest

from python.runtime.matching import ConfusionMatrix, Matcher, StreamingMatcher
from python.runtime.matching.confirmation import PronunciationFeedback


ROOT = Path(__file__).resolve().parents[2]
DATA = (ROOT / "unity/Packages/com.domicube.phoneme-matching"
        / "Samples~/KoreanData")
UTTERANCE = ["p", "a", "ŋ", "a", "tɕ", "i"]


@pytest.fixture
def matrix():
    return ConfusionMatrix.from_json(DATA / "ko_child_v2.json")


@pytest.fixture
def matcher(matrix):
    return Matcher.for_streaming(matrix)


@pytest.fixture
def answers():
    """The shipped catalog with the calibrated escape switched off.

    These cases are about the streak and the candidate switch, which the
    strict two-frame rule owns. `test_streaming.py` covers the escape
    itself; leaving it on here would confirm an exact frame before the
    streak logic under test ever ran.
    """
    catalog = {a["id"]: dict(a) for a in json.loads(
        (DATA / "targets_child.json").read_text(encoding="utf-8")
    )["answers"]}
    for answer in catalog.values():
        answer.pop("allow_exact_single_frame", None)
    return catalog


@pytest.fixture
def calibrated_answers():
    """The shipped catalog exactly as it ships, escape included."""
    return {a["id"]: a for a in json.loads(
        (DATA / "targets_child.json").read_text(encoding="utf-8")
    )["answers"]}


@pytest.mark.parametrize("reverse", [False, True])
def test_passing_answer_is_not_hidden_by_higher_failing_score(
    matcher, answers, reverse,
):
    puppy, bread = answers["puppy"], answers["bread"]
    passing = matcher.best_match(UTTERANCE, [puppy])
    failing = matcher.best_match(UTTERANCE, [bread])
    assert passing.passed and not failing.passed
    assert passing.score == pytest.approx(0.8991666666666667)
    assert failing.score == pytest.approx(0.9283333333333333)
    assert failing.score > passing.score

    candidates = [bread, puppy] if reverse else [puppy, bread]
    # Includes threshold, target IPA, distance, alignment and window offsets.
    assert matcher.best_match(UTTERANCE, candidates) == passing


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("puppy_threshold,bread_threshold,expected,passed", [
    (0.8, 0.975, "puppy", True),   # higher score fails
    (0.8, 0.9, "bread", True),     # both pass: highest raw score wins
    (0.95, 0.9, "bread", True),    # only the higher score passes
    (0.95, 0.975, "bread", False), # neither passes: keep the display leader
])
def test_selection_policy(matcher, answers, reverse, puppy_threshold,
                          bread_threshold, expected, passed):
    puppy = dict(answers["puppy"], threshold=puppy_threshold)
    bread = dict(answers["bread"], threshold=bread_threshold)
    candidates = [bread, puppy] if reverse else [puppy, bread]
    result = matcher.best_match(UTTERANCE, candidates)
    chosen = next(c for c in candidates if c["id"] == expected)
    assert result == matcher.best_match(UTTERANCE, [chosen])
    assert result.passed is passed


@pytest.mark.parametrize("first_threshold,second_threshold,expected,passed", [
    (0.8, 0.8, "first", True),
    (0.975, 0.975, "first", False),
    (0.975, 0.8, "second", True),
    (0.8, 0.975, "first", True),
])
def test_ties_keep_input_order_only_within_same_pass_status(
    matcher, answers, first_threshold, second_threshold, expected, passed,
):
    candidates = [
        dict(answers["bread"], id="first", threshold=first_threshold),
        dict(answers["bread"], id="second", threshold=second_threshold),
    ]
    result = matcher.best_match(UTTERANCE, candidates)
    assert result.target_id == expected
    assert result.passed is passed


@pytest.mark.parametrize("profile", ["batch", "streaming", "exact"])
def test_threshold_equality_passes_in_every_profile(matrix, answers, profile):
    matcher = (Matcher.for_streaming(matrix) if profile == "streaming"
               else Matcher(matrix, mode="exact" if profile == "exact"
                            else "substring"))
    candidates = [dict(answers["puppy"]), dict(answers["bread"])]
    candidates.sort(key=lambda c: matcher.score_against(
        UTTERANCE, c["phonemes"])[1])
    lower, higher = candidates
    lower["threshold"] = matcher.score_against(UTTERANCE, lower["phonemes"])[1]
    higher["threshold"] = 1.0
    assert lower["threshold"] < matcher.score_against(
        UTTERANCE, higher["phonemes"])[1] < higher["threshold"]
    result = matcher.best_match(UTTERANCE, [higher, lower])
    assert result == matcher.best_match(UTTERANCE, [lower])
    assert result.passed


def test_only_supplied_answers_are_accepted(matcher, answers):
    # The catalog containing 강아지 must not make a bread-only question pass.
    result = matcher.best_match(UTTERANCE, [answers["bread"]])
    assert result.target_id == "bread"
    assert not result.passed
    assert not matcher.best_match(UTTERANCE, []).passed


@pytest.mark.parametrize("interruption,expected_id,frame_count", [
    (None, "puppy", 2),
    ("silence", "puppy", 4),
    ("bread", "bread", 3),
])
def test_two_frames_require_the_same_passing_answer_and_its_feedback(
    matcher, answers, interruption, expected_id, frame_count,
):
    sm = StreamingMatcher(matcher, [answers["bread"], answers["puppy"]])
    assert sm.consecutive == 2
    assert sm.push(UTTERANCE) is None
    assert sm.streak == 1
    final = UTTERANCE
    if interruption == "silence":
        assert sm.push([]) is None
        assert sm.streak == 0
        assert sm.push(UTTERANCE) is None
        assert sm.streak == 1
    elif interruption == "bread":
        final = answers["bread"]["phonemes"]
        assert sm.push(final) is None
        assert sm.streak == 1

    hit = sm.push(final)
    assert hit is not None
    assert hit.result.target_id == expected_id
    assert hit.frames == frame_count
    assert hit.streak == 2
    expected = matcher.best_match(final, [answers[expected_id]])
    assert hit.feedback.to_dict() == PronunciationFeedback.from_match(
        expected, frame_count, 2, matrix_id=matcher.matrix.matrix_id).to_dict()
    sm.reset()
    assert sm.streak == 0
    assert sm.push(final) is None
