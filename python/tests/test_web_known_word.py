"""The live web tab must not re-measure a word the catalog already ships.

A catalog word carries a configured threshold and, sometimes, a
calibrated one-exact-frame escape. Measuring it again here would answer a
slightly different question than the device asks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from python.tools.test_real_audio import AudioTester
from python.tools.web_test import _confirmed_status_html, _streak_summary_md

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ROOT / "shared/targets_child.json"
CHILD_MATRIX = ROOT / "shared/confusion_matrices/ko_child_v2.json"
ADULT_MATRIX = ROOT / "shared/confusion_matrices/ko_child_v1.json"


@pytest.fixture
def child_tester():
    return AudioTester(targets_path=TARGETS, matrix_path=CHILD_MATRIX)


def test_a_catalog_word_inherits_the_shipped_threshold_and_policy(child_tester):
    _, candidate = child_tester._make_custom_target("책", "ko")
    assert candidate["threshold"] == 0.826
    assert candidate["threshold_source"] == "configured"
    assert candidate["allow_exact_single_frame"] is True
    snapshot = candidate["exact_single_frame_calibration"]
    assert snapshot["matrix_id"] == "ko_child_v2"
    assert snapshot["text"] == "책" and snapshot["threshold"] == 0.826
    # Only the runtime id is rewritten; the policy travels intact.
    assert candidate["id"] == "custom_target"


def test_an_excluded_word_inherits_no_escape(child_tester):
    _, candidate = child_tester._make_custom_target("와요", "ko")
    assert candidate["threshold_source"] == "configured"
    assert "allow_exact_single_frame" not in candidate
    assert "exact_single_frame_calibration" not in candidate


def test_a_word_outside_the_catalog_stays_strict(child_tester):
    _, candidate = child_tester._make_custom_target("포도", "ko")
    assert candidate["threshold_source"] in ("measured", "default")
    assert "allow_exact_single_frame" not in candidate
    assert "exact_single_frame_calibration" not in candidate


def test_a_catalog_built_for_another_matrix_is_not_inherited():
    # Child data with the adult matrix is not the combination anything
    # was calibrated for.
    tester = AudioTester(targets_path=TARGETS, matrix_path=ADULT_MATRIX)
    _, candidate = tester._make_custom_target("책", "ko")
    assert candidate["threshold_source"] != "configured"
    assert "allow_exact_single_frame" not in candidate


def test_the_ui_does_not_claim_two_passes_for_one_frame():
    class Hit:
        streak = 1
        frames = 3

        class result:
            target_text = "책"
            score = 1.0

    html = _confirmed_status_html(Hit, 1.5)
    assert "정확일치 1프레임" in html
    assert "연속 2회 통과로 확정" not in html

    points = [{"score": 1.0, "threshold": 0.826, "word": "책"}]
    summary = _streak_summary_md(points, 2, confirmed_on_exact=True)
    assert "정확일치 1프레임으로 확정" in summary
    assert "확정에 2회 필요" not in summary


def test_the_ordinary_confirmation_still_reads_as_two_passes():
    class Hit:
        streak = 2
        frames = 2

        class result:
            target_text = "사과"
            score = 0.9

    html = _confirmed_status_html(Hit, 1.0)
    assert "연속 2회 통과로 확정" in html
