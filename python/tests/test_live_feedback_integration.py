"""Preserve the desktop error-table policy when adding readable feedback."""

from pathlib import Path

import pytest

from python.runtime.matching import ConfusionMatrix, MatchResult, PronunciationFeedback
from python.tools._web_syllable_feedback import confirmed_feedback_html
from python.tools.test_real_audio import AudioTester

ROOT = Path(__file__).resolve().parents[2]


def feedback():
    result = MatchResult(
        target_id="dad", target_text="아빠", score=0.94, distance=0.2,
        threshold=0.65, passed=True,
        user_phonemes=["a", "p", "a"], target_phonemes=["a", "p͈", "a"],
        window_start=0, window_end=3,
        alignment=[("a", "a", "match"), ("p", "p͈", "sub"), ("a", "a", "match")],
    )
    return PronunciationFeedback.from_match(result, 2, 2)


def candidate():
    return {"syllables": [
        {"text": "아", "spoken": "아", "span": [0, 1]},
        {"text": "빠", "spoken": "빠", "span": [1, 3]},
    ]}


def test_known_model_error_is_withheld_but_raw_contract_is_preserved():
    tester = AudioTester(targets_path=None, matrix_path=ROOT / "shared/confusion_matrices/ko_child_v2.json")
    result = feedback()
    before = result.to_dict()
    html = confirmed_feedback_html(tester, result, candidate(), "ko")
    primary = html.split("<details>", 1)[0]
    assert "판단 보류" in primary
    assert "발음 지적을 보류" in primary
    assert "빠’가 ‘바" not in primary
    assert "빠 → 바" in html  # original comparison remains in closed details
    assert result.to_dict() == before


def test_child_error_table_is_not_used_for_adult_profile():
    tester = AudioTester(targets_path=None, matrix_path=ROOT / "shared/confusion_matrices/ko_child_v1.json")
    html = confirmed_feedback_html(tester, feedback(), candidate(), "ko")
    primary = html.split("<details>", 1)[0]
    assert "모델 오류표가 없습니다" in primary
    assert "빠’가 ‘바" in primary


def test_report_text_is_escaped():
    tester = AudioTester(targets_path=None)
    malicious = candidate()
    malicious["syllables"][1]["text"] = "<script>alert(1)</script>"
    html = confirmed_feedback_html(tester, feedback(), malicious, "ko")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_live_ui_keeps_latest_default_of_two_frames():
    pytest.importorskip("gradio")
    from python.tools.web_test import build_app

    app = build_app(AudioTester(targets_path=None))
    try:
        slider = next(c for c in app.config["components"] if c["props"].get("label") == "확정에 필요한 연속 횟수")
        assert slider["props"]["value"] == 2
    finally:
        app.close()


def test_child_streaming_profile_was_not_replaced_by_legacy_defaults():
    from python.runtime.matching import Matcher

    matrix = ConfusionMatrix.from_json(ROOT / "shared/confusion_matrices/ko_child_v2.json")
    matcher = Matcher.for_streaming(matrix)
    assert matcher.skip_cost == 0.005
    assert matcher.context_mult == 8
    assert matcher.coverage == 0.8
