"""One exact frame may confirm, but only where calibration said so.

The escape exists because a short word often gets exactly one frame in
which the recogniser reproduces the target and never a second: by the
next hop the utterance is still inside the window but sits elsewhere in
it, and what the recogniser makes of it changes with that position. It is
deliberately narrow - exactly 1.0, exactly `consecutive == 2`, and only
for a word the calibration switched on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from python.runtime.matching import ConfusionMatrix, Matcher, StreamingMatcher

DATA = Path(__file__).resolve().parents[2] / "shared"
BREAD = ["p͈", "a", "ŋ"]


@pytest.fixture
def matrix():
    return ConfusionMatrix.from_json(
        DATA / "confusion_matrices" / "ko_child_v2.json")


@pytest.fixture
def catalog():
    return {a["text"]: a for a in json.loads(
        (DATA / "targets_child.json").read_text(encoding="utf-8"))["answers"]}


def flagged(catalog, word="빵"):
    answer = dict(catalog[word])
    assert answer["allow_exact_single_frame"] is True
    return answer


def unflagged(catalog, word="빵"):
    answer = dict(catalog[word])
    answer.pop("allow_exact_single_frame", None)
    return answer


def test_the_shipped_catalog_calibrated_28_of_35(catalog):
    on = [w for w, a in catalog.items() if a.get("allow_exact_single_frame")]
    # The six action words added for the music lesson were derived at the
    # ordinary pass mark only, so they confirm on two frames like every
    # word did before the escape was calibrated for anything.
    assert len(catalog) == 35 and len(on) == 28
    for word in ("들기", "치기", "놓아", "놓기", "흔들기", "두드리기"):
        assert word not in on
    assert "와요" not in on, "와요 exceeds the budget and must stay strict"
    assert catalog["책"]["threshold"] == 0.826


def test_one_exact_frame_confirms_with_a_genuine_streak_of_one(matrix, catalog):
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [flagged(catalog)])
    hit = sm.push(BREAD)
    assert hit is not None
    assert hit.result.score == 1.0
    # Not a fabricated second frame.
    assert hit.streak == 1 and sm.streak == 1
    assert hit.frames == 1
    assert hit.feedback.streak == 1 and hit.feedback.frames == 1
    assert json.loads(json.dumps(hit.feedback.to_dict()))["streak"] == 1


def test_without_the_flag_one_exact_frame_is_not_enough(matrix, catalog):
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [unflagged(catalog)])
    assert sm.push(BREAD) is None
    assert sm.streak == 1
    assert sm.push(BREAD) is not None


def test_a_near_perfect_frame_does_not_qualify(matrix, catalog):
    matcher = Matcher.for_streaming(matrix)
    answer = flagged(catalog)
    nearly = BREAD + ["i"]
    score = matcher.score_against(nearly, answer["phonemes"])[1]
    assert answer["threshold"] <= score < 1.0, score
    sm = StreamingMatcher(matcher, [answer])
    assert sm.push(nearly) is None, "only exactly 1.0 may confirm alone"
    assert sm.streak == 1
    assert sm.push(nearly) is not None, "the ordinary rule still applies"


def test_empty_input_confirms_nothing(matrix, catalog):
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [flagged(catalog)])
    assert sm.push([]) is None
    assert sm.streak == 0


@pytest.mark.parametrize("consecutive", [1, 3])
def test_only_the_calibrated_streak_length_gets_the_escape(
    matrix, catalog, consecutive
):
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [flagged(catalog)],
                          consecutive=consecutive)
    hit = sm.push(BREAD)
    if consecutive == 1:
        assert hit is not None and hit.streak == 1
    else:
        assert hit is None, "consecutive 3 must not inherit a one-frame rule"
        assert sm.push(BREAD) is None
        assert sm.push(BREAD) is not None


def test_a_hand_tuned_profile_turns_the_escape_off(matrix, catalog):
    off_profile = Matcher(matrix, skip_cost=0.2, coverage=0.5, context_mult=8.0)
    sm = StreamingMatcher(off_profile, [flagged(catalog)])
    assert sm.push(BREAD) is None, "uncalibrated profile must stay strict"
    assert sm.push(BREAD) is not None


def test_reset_and_a_candidate_switch_still_behave(matrix, catalog):
    answers = [unflagged(catalog, "빵"), unflagged(catalog, "강아지")]
    sm = StreamingMatcher(Matcher.for_streaming(matrix), answers)
    assert sm.push(BREAD) is None and sm.streak == 1
    sm.reset()
    assert sm.streak == 0
    assert sm.push(BREAD) is None
    assert sm.push(BREAD) is not None


def test_a_shared_id_cannot_borrow_another_answers_approval(matrix, catalog):
    flagged_bread = flagged(catalog)
    impostor = dict(catalog["강아지"])
    impostor["id"] = flagged_bread["id"]
    impostor.pop("allow_exact_single_frame", None)
    sm = StreamingMatcher(Matcher.for_streaming(matrix),
                          [flagged_bread, impostor])
    assert sm.push(BREAD) is None, "a duplicate id must void the shortcut"
    assert sm.push(BREAD) is not None


def test_approval_is_re_read_and_cannot_go_stale(matrix, catalog):
    answer = flagged(catalog)
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [answer])
    # The caller edits the candidate after construction.
    answer["allow_exact_single_frame"] = False
    assert sm.push(BREAD) is None, "approval was cached from construction"
    answer["allow_exact_single_frame"] = True
    sm.reset()
    assert sm.push(BREAD) is not None


def test_a_stub_matcher_without_a_matrix_still_works(matrix, catalog):
    class Stub:
        def __init__(self, inner):
            self._inner = inner

        def best_match(self, phonemes, candidates):
            return self._inner.best_match(phonemes, candidates)

    sm = StreamingMatcher(Stub(Matcher.for_streaming(matrix)),
                          [unflagged(catalog)])
    assert sm.push(BREAD) is None
    assert sm.push(BREAD) is not None


def test_a_missing_snapshot_grants_nothing(matrix, catalog):
    answer = flagged(catalog)
    answer.pop("exact_single_frame_calibration")
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [answer])
    assert sm.push(BREAD) is None
    assert sm.push(BREAD) is not None


@pytest.mark.parametrize("field,value", [
    ("threshold", 0.5),
    ("text", "다른말"),
    ("phonemes", ["k", "a"]),
    ("consecutive", 3),
    ("matrix_id", "ko_child_v1"),
])
def test_a_snapshot_that_no_longer_describes_the_answer_grants_nothing(
    matrix, catalog, field, value
):
    answer = flagged(catalog)
    answer["exact_single_frame_calibration"] = dict(
        answer["exact_single_frame_calibration"])
    answer["exact_single_frame_calibration"][field] = value
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [answer])
    assert sm.push(BREAD) is None, f"{field} mismatch still granted the shortcut"


@pytest.mark.parametrize("field,value", [
    ("threshold", 0.5),
    ("phonemes", ["p͈", "a"]),
    ("text", "다른말"),
])
def test_editing_the_answer_after_calibration_grants_nothing(
    matrix, catalog, field, value
):
    answer = flagged(catalog)
    answer[field] = value
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [answer])
    assert sm.push(answer["phonemes"] if field == "phonemes" else BREAD) is None


def test_another_matrix_with_its_own_valid_profile_grants_nothing(catalog):
    other = ConfusionMatrix.from_json(
        DATA / "confusion_matrices" / "ko_child_v1.json")
    sm = StreamingMatcher(Matcher.for_streaming(other), [flagged(catalog)])
    assert sm.push(BREAD) is None, "approval must not travel to another matrix"


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_a_flag_that_is_not_a_real_boolean_grants_nothing(matrix, catalog, value):
    answer = flagged(catalog)
    answer["allow_exact_single_frame"] = value
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [answer])
    assert sm.push(BREAD) is None


def test_a_real_candidate_switch_resets_the_streak(matrix, catalog):
    bread, puppy = unflagged(catalog, "빵"), unflagged(catalog, "강아지")
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [bread, puppy])
    assert sm.push(bread["phonemes"]) is None
    assert sm.streak == 1
    # A different winning candidate: the streak starts again, not continues.
    assert sm.push(puppy["phonemes"]) is None
    assert sm.streak == 1
    assert sm.push(puppy["phonemes"]) is not None


def test_the_session_ends_once_on_a_single_frame_confirmation(matrix, catalog):
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [flagged(catalog)])
    first = sm.push(BREAD)
    assert first is not None and first.frames == 1
    # Pushing again keeps returning hits (documented), but the frame count
    # advances honestly rather than re-reporting the first.
    second = sm.push(BREAD)
    assert second is not None and second.frames == 2


def test_the_shipped_matrix_names_itself(tmp_path):
    """The file says which matrix it is, so no loader has to guess.

    C# reads the JSON and nothing else, so an identity derived from the
    file name exists only in Python and cannot be compared on the device.
    """
    source = DATA / "confusion_matrices" / "ko_child_v2.json"
    assert json.loads(source.read_text(encoding="utf-8"))["matrix_id"] \
        == "ko_child_v2"
    renamed = tmp_path / "copied.json"
    renamed.write_bytes(source.read_bytes())
    assert ConfusionMatrix.from_json(renamed).matrix_id == "ko_child_v2"


@pytest.mark.parametrize("actual", [None, "", "   ", "unknown", " unknown "])
def test_a_matrix_that_names_nothing_gets_no_shortcut(matrix, catalog, actual):
    matrix.matrix_id = actual
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [flagged(catalog)])
    assert sm.push(BREAD) is None, "an unnamed matrix must not be assumed"
    assert sm.push(BREAD) is not None


@pytest.mark.parametrize("name", ["", "unknown", " unknown ", "   "])
def test_two_anonymous_matrices_are_not_the_same_matrix(matrix, catalog, name):
    """Both sides saying "unknown" is not agreement about anything."""
    answer = flagged(catalog)
    snapshot = dict(answer["exact_single_frame_calibration"])
    snapshot["matrix_id"] = name
    answer["exact_single_frame_calibration"] = snapshot
    matrix.matrix_id = name
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [answer])
    assert sm.push(BREAD) is None


@pytest.mark.parametrize("field", ["matrix_id", "text", "phonemes",
                                   "threshold", "consecutive",
                                   "streaming_profile"])
def test_a_snapshot_missing_a_field_grants_nothing(matrix, catalog, field):
    answer = flagged(catalog)
    snapshot = dict(answer["exact_single_frame_calibration"])
    del snapshot[field]
    answer["exact_single_frame_calibration"] = snapshot
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [answer])
    assert sm.push(BREAD) is None, f"a snapshot with no {field} granted it"


@pytest.mark.parametrize("field,value", [
    ("matrix_id", ""),
    ("matrix_id", 2),
    ("text", None),
    ("phonemes", []),
    ("phonemes", "p a ng"),
    ("phonemes", ["p", 3]),
    ("threshold", "0.975"),
    ("threshold", True),
    ("threshold", float("nan")),
    ("consecutive", True),
    ("consecutive", "2"),
    ("streaming_profile", None),
])
def test_a_malformed_snapshot_grants_nothing(matrix, catalog, field, value):
    answer = flagged(catalog)
    snapshot = dict(answer["exact_single_frame_calibration"])
    snapshot[field] = value
    answer["exact_single_frame_calibration"] = snapshot
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [answer])
    assert sm.push(BREAD) is None, f"{field}={value!r} granted the shortcut"


@pytest.mark.parametrize("profile", [
    [],
    ["skip_cost"],
    "",
    "0.005",
    {"skip_cost": 0.005, "coverage": 0.8},
    {"skip_cost": "0.005", "coverage": 0.8, "context_mult": 8.0},
    {"skip_cost": True, "coverage": 0.8, "context_mult": 8.0},
    {"skip_cost": float("inf"), "coverage": 0.8, "context_mult": 8.0},
])
def test_a_malformed_profile_disables_instead_of_raising(matrix, catalog,
                                                         profile):
    """A non-empty list or string used to raise out of the match path."""
    answer = flagged(catalog)
    snapshot = dict(answer["exact_single_frame_calibration"])
    snapshot["streaming_profile"] = profile
    answer["exact_single_frame_calibration"] = snapshot
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [answer])
    assert sm.push(BREAD) is None
    assert sm.push(BREAD) is not None, "the ordinary rule still applies"


@pytest.mark.parametrize("field,value", [
    ("threshold", 0.5),
    ("phonemes", ["p͈", "a"]),
    ("text", "다른말"),
])
def test_editing_the_answer_after_construction_grants_nothing(
    matrix, catalog, field, value
):
    """The existing tests edit before construction; this one edits after."""
    answer = flagged(catalog)
    sm = StreamingMatcher(Matcher.for_streaming(matrix), [answer])
    answer[field] = value
    frame = value if field == "phonemes" else BREAD
    assert sm.push(frame) is None, "approval was cached at construction"
    assert sm.push(frame) is not None, "the ordinary rule still applies"


def test_retuning_the_matcher_after_construction_grants_nothing(matrix,
                                                                catalog):
    matcher = Matcher.for_streaming(matrix)
    sm = StreamingMatcher(matcher, [flagged(catalog)])
    matcher.skip_cost = 0.2
    assert sm.push(BREAD) is None, "the profile was cached at construction"
    assert sm.push(BREAD) is not None


def test_swapping_the_matrix_identity_after_construction_grants_nothing(
    matrix, catalog
):
    matcher = Matcher.for_streaming(matrix)
    sm = StreamingMatcher(matcher, [flagged(catalog)])
    matcher.matrix.matrix_id = "ko_child_v1"
    assert sm.push(BREAD) is None
