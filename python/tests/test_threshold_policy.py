"""Calibration of the exact-single-frame escape, and what it refuses.

Synthetic public score sequences, so this runs in CI without the private
reference corpus. `.venv/demo-priority-0906-3c91` holds the audit against
the real reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from python.build.build_targets import (
    WordRow, build_answer_entries, load_policy, load_thresholds)
from python.tools.child_tuning.derive_thresholds import (
    CANDIDATES, exact_single_frame_policy, fires, lower_bound_status)

SHARED = Path(__file__).resolve().parents[2] / "shared"
COPIES = (
    SHARED / "targets_child.json",
    SHARED.parent / "unity/Assets/StreamingAssets/targets_child.json",
    SHARED.parent / ("unity/Packages/com.domicube.phoneme-matching/"
                     "Samples~/KoreanData/targets_child.json"),
)


def test_the_candidate_grid_is_a_clean_0_025_ladder():
    assert CANDIDATES[0] == 0.5 and CANDIDATES[-1] == 1.0
    assert len(CANDIDATES) == 21
    for value in CANDIDATES:
        assert round(value, 4) == value, "must survive 4-decimal rounding"


def test_a_word_earns_the_escape_by_tightening_not_by_relaxing():
    # seconds=10 makes the rate equal the firing count, so budget=32 is
    # "at most 32 firing utterances", the same shape production uses.
    scored = [[0.83, 0.83] for _ in range(31)] + [[1.0, 0.0]]
    threshold, allowed, fired, rate = exact_single_frame_policy(
        scored, 0.825, seconds=10.0, budget=32.0, consecutive=2)
    assert allowed is True
    assert threshold == 0.825, "no tightening was needed"
    assert fired == 32 and rate == 32.0


def test_over_budget_by_one_is_paid_for_by_one_grid_step():
    # 33 firing sequences at 0.825; a single 0.001 step drops exactly the
    # one sitting between the two grid points.
    scored = [[0.83, 0.83] for _ in range(32)] + [[0.8255, 0.8255]]
    assert sum(fires(s, 0.825, 2) for s in scored) == 33
    threshold, allowed, fired, rate = exact_single_frame_policy(
        scored, 0.825, seconds=10.0, budget=32.0, consecutive=2)
    assert allowed is True
    assert threshold == 0.826
    assert fired == 32


def test_exact_frames_alone_over_budget_keeps_the_strict_rule():
    # Every sequence reaches 1.0, so no normal-path tightening can help.
    scored = [[1.0, 0.0] for _ in range(40)]
    threshold, allowed, fired, rate = exact_single_frame_policy(
        scored, 0.825, seconds=10.0, budget=32.0, consecutive=2)
    assert allowed is False
    assert threshold == 0.825, "the ordinary pass mark is left alone"
    assert fired == 40


def test_lower_bound_status_separates_a_fallback_from_a_real_choice():
    class Fixed:
        def __init__(self, score):
            self.score = score

        def score_against(self, frame, phonemes):
            return (0.0, self.score, 0, 0, [])

    reference = [{"text": "다른 말", "frames": ["a", "b"]}]
    infeasible = lower_bound_status(Fixed(1.0), ["k", "a"], "가", reference,
                                    10.0, 0.01, 2)
    feasible = lower_bound_status(Fixed(0.99), ["k", "a"], "가", reference,
                                  10.0, 0.01, 2)
    assert infeasible["threshold"] == feasible["threshold"] == 1.0
    assert infeasible["feasible"] is False
    assert feasible["feasible"] is True


def written(tmp_path, thresholds, policy, **overrides):
    payload = {
        "matrix_id": "ko_child_v2", "consecutive": 2, "budget": 0.01,
        "streaming_profile": {"skip_cost": 0.005, "coverage": 0.8,
                              "context_mult": 8.0},
        "reference": {"utterances": 10, "seconds": 1000.0,
                      "seconds_exact": 1000.0},
        "thresholds": thresholds, "policy": policy,
    }
    payload.update(overrides)
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def record(**overrides):
    # 0 firings over the reference is a rate of 0: the builder now checks
    # the two against each other, so the fixture has to be consistent.
    base = {"threshold": 0.8, "feasible": True, "rate_per_10s": 0.0,
            "false_accepts": 0, "allow_exact_single_frame": True,
            "text": "가", "phonemes": ["k", "a"]}
    base.update(overrides)
    return base


def test_the_builder_refuses_an_infeasible_policy(tmp_path):
    path = written(tmp_path, {"가": 1.0},
                   {"가": record(threshold=1.0, feasible=False)})
    with pytest.raises(ValueError, match="budget"):
        load_policy(path)


def test_a_missing_feasible_field_is_not_treated_as_feasible(tmp_path):
    bad = record()
    del bad["feasible"]
    path = written(tmp_path, {"가": 0.8}, {"가": bad})
    with pytest.raises(ValueError, match="budget"):
        load_policy(path)


def test_the_builder_refuses_a_policy_that_contradicts_its_threshold(tmp_path):
    path = written(tmp_path, {"가": 0.8}, {"가": record(threshold=0.9)})
    with pytest.raises(ValueError, match="disagrees"):
        load_policy(path)


def test_the_builder_refuses_a_rate_outside_the_budget(tmp_path):
    path = written(tmp_path, {"가": 0.8},
                   {"가": record(rate_per_10s=0.9, false_accepts=90)})
    with pytest.raises(ValueError, match="budget"):
        load_policy(path)


def test_a_policy_not_tied_to_its_calibration_is_refused(tmp_path):
    path = written(tmp_path, {"가": 0.8}, {"가": record()})
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["matrix_id"]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix_id"):
        load_policy(path)


def test_a_policy_from_another_matrix_is_refused(tmp_path):
    path = written(tmp_path, {"가": 0.8}, {"가": record()})
    with pytest.raises(ValueError, match="recalibrate"):
        load_policy(path, matrix_id="ko_child_v1")


def test_a_rate_that_does_not_match_its_firing_count_is_refused(tmp_path):
    path = written(tmp_path, {"가": 0.8},
                   {"가": record(false_accepts=5, rate_per_10s=0.004)})
    with pytest.raises(ValueError, match="does not match"):
        load_policy(path)


def test_a_policy_calibrated_for_another_streak_length_is_refused(tmp_path):
    path = written(tmp_path, {"가": 0.8}, {"가": record()}, consecutive=3)
    with pytest.raises(ValueError, match="consecutive 2"):
        load_policy(path)


def test_a_policy_that_switches_nothing_on_may_use_another_streak(tmp_path):
    """Only the escape was calibrated at 2.

    Deriving ordinary thresholds at consecutive 3 is a normal thing to do,
    and the file it writes must stay buildable.
    """
    path = written(tmp_path, {"가": 0.8},
                   {"가": record(allow_exact_single_frame=False)},
                   consecutive=3)
    calibration = load_policy(path)
    assert calibration["consecutive"] == 3
    assert calibration["policy"]["가"]["allow_exact_single_frame"] is False


def test_a_rate_inside_the_budget_only_after_rounding_is_refused(tmp_path):
    """0.0100001 written as 0.010000 is still over a budget of 0.01."""
    path = written(tmp_path, {"가": 0.8},
                   {"가": record(false_accepts=1, rate_per_10s=0.01)},
                   reference={"utterances": 10, "seconds": 999.99,
                              "seconds_exact": 999.99})
    with pytest.raises(ValueError, match="over the budget"):
        load_policy(path)


def test_the_same_count_just_inside_the_budget_is_accepted(tmp_path):
    """The other side of that boundary: 0.00999... is under 0.01."""
    path = written(tmp_path, {"가": 0.8},
                   {"가": record(false_accepts=1, rate_per_10s=0.01)},
                   reference={"utterances": 10, "seconds": 1000.01,
                              "seconds_exact": 1000.01})
    assert load_policy(path)["policy"]["가"]["false_accepts"] == 1


@pytest.mark.parametrize("budget", [True, "0.01", None, float("nan"), -0.01])
def test_a_budget_that_is_not_a_number_is_refused(tmp_path, budget):
    path = written(tmp_path, {"가": 0.8}, {"가": record()}, budget=budget)
    with pytest.raises(ValueError, match="budget"):
        load_policy(path)


@pytest.mark.parametrize("fired", [True, 1.0, "0", -1])
def test_a_firing_count_that_is_not_a_count_is_refused(tmp_path, fired):
    path = written(tmp_path, {"가": 0.8}, {"가": record(false_accepts=fired)})
    with pytest.raises(ValueError, match="count|does not match|budget"):
        load_policy(path)


@pytest.mark.parametrize("matrix_id", ["", "   ", 2, None])
def test_a_policy_with_no_usable_matrix_id_is_refused(tmp_path, matrix_id):
    path = written(tmp_path, {"가": 0.8}, {"가": record()}, matrix_id=matrix_id)
    with pytest.raises(ValueError, match="matrix"):
        load_policy(path)


def test_a_threshold_that_does_not_survive_serialisation_is_refused(tmp_path):
    """0.82594 would ship as 0.8259 - a looser pass mark nobody measured."""
    path = written(tmp_path, {"가": 0.82594}, {"가": record(threshold=0.82594)})
    with pytest.raises(ValueError, match="4-decimal"):
        build_answer_entries([WordRow("w_ga", "가", "ko")],
                             thresholds=load_thresholds(path),
                             calibration=load_policy(path))


def test_a_threshold_on_the_production_grid_is_written_through(tmp_path):
    path = written(tmp_path, {"가": 0.826}, {"가": record(threshold=0.826)})
    entries = build_answer_entries([WordRow("w_ga", "가", "ko")],
                                   thresholds=load_thresholds(path),
                                   calibration=load_policy(path))
    assert entries[0]["threshold"] == 0.826
    assert entries[0]["allow_exact_single_frame"] is True
    snapshot = entries[0]["exact_single_frame_calibration"]
    assert snapshot["threshold"] == 0.826
    assert snapshot["matrix_id"] == "ko_child_v2"


def test_a_non_boolean_flag_is_refused(tmp_path):
    path = written(tmp_path, {"가": 0.8},
                   {"가": record(allow_exact_single_frame="yes")})
    with pytest.raises(ValueError, match="boolean"):
        load_policy(path)


def test_a_threshold_file_without_a_policy_block_grants_nothing(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"thresholds": {"가": 0.8}}), encoding="utf-8")
    assert load_policy(path) == {"policy": {}}
    assert load_thresholds(path) == {"가": 0.8}


def test_the_shipped_policy_matches_the_shipped_catalog():
    stored = json.loads(
        (SHARED / "thresholds_child.json").read_text(encoding="utf-8"))
    catalog = json.loads(COPIES[0].read_text(encoding="utf-8"))["answers"]
    by_word = {a["text"]: a for a in catalog}
    assert stored["consecutive"] == 2
    for word, record in stored["policy"].items():
        answer = by_word[word]
        assert record["feasible"] is True
        assert answer["threshold"] == round(record["threshold"], 4)
        assert bool(answer.get("allow_exact_single_frame")) == \
            record["allow_exact_single_frame"]
        if record["allow_exact_single_frame"]:
            snapshot = answer["exact_single_frame_calibration"]
            assert snapshot["matrix_id"] == stored["matrix_id"]
            assert snapshot["consecutive"] == stored["consecutive"]
            assert snapshot["streaming_profile"] == stored["streaming_profile"]
            assert snapshot["text"] == word
            assert snapshot["phonemes"] == answer["phonemes"]
            assert snapshot["threshold"] == answer["threshold"]
    assert by_word["책"]["threshold"] == 0.826
    assert not by_word["와요"].get("allow_exact_single_frame")


def test_all_three_shipped_copies_agree():
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in COPIES]
    first = payloads[0]["answers"]
    for other in payloads[1:]:
        assert other["answers"] == first
