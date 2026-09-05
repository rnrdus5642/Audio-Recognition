"""Live-only web UI and confirmation tests, without model downloads or a mic."""

from pathlib import Path

import pytest

pytest.importorskip("gradio")
import numpy as np

from python.tools.test_real_audio import AudioTester
from python.tools.web_test import SAMPLE_RATE, build_app


MATRIX_PATH = (
    Path(__file__).resolve().parents[2]
    / "shared/confusion_matrices/ko_child_v1.json"
)
APPLE_IPA = ["s", "a", "k", "w", "a"]


class FakeRecognizer:
    def __init__(self):
        self.windows = []
        self.warmups = 0
        self.phonemes = APPLE_IPA.copy()

    def recognize(self, audio):
        self.warmups += 1
        return []

    def recognize_with_text(self, audio):
        self.windows.append(audio.copy())
        return "사과", self.phonemes.copy()


@pytest.fixture
def live_app(monkeypatch):
    tester = AudioTester(targets_path=None, matrix_path=MATRIX_PATH)
    recognizer = FakeRecognizer()
    monkeypatch.setattr(tester, "_recognizer_for", lambda language: recognizer)
    monkeypatch.setattr(
        tester,
        "_make_custom_target",
        lambda text, language: ("__custom__", {
            "id": "custom_target", "text": text,
            "phonemes": APPLE_IPA, "threshold": 0.65,
        }),
    )
    app = build_app(tester)
    callbacks = {fn.fn.__name__: fn.fn for fn in app.fns.values() if fn.fn}
    yield app, callbacks, recognizer
    app.close()


def test_only_live_microphone_ui_is_exposed(live_app):
    app, callbacks, recognizer = live_app
    components = app.config["components"]
    assert not any(c["type"] in {"tabs", "tabitem", "file"} for c in components)
    audio = [c for c in components if c["type"] == "audio"]
    assert len(audio) == 1
    assert audio[0]["props"]["sources"] == ["microphone"]
    assert audio[0]["props"]["streaming"] is True
    assert set(callbacks) == {"live_warmup", "live_reset", "live_step"}
    assert not recognizer.windows
    assert recognizer.warmups == 0


def test_two_passing_frames_confirm_and_stop_recording(live_app):
    _, callbacks, recognizer = live_app
    step = callbacks["live_step"]
    chunk = (SAMPLE_RATE, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
    first = step(chunk, {}, "사과", "ko", 2.5, 2)
    assert len(first) == 10
    assert first[8] is None
    assert first[0]["sm"].streak == 1
    assert not first[0].get("done")
    recognizer.phonemes = ["t", "a", "k", "w", "a"]
    second = step(chunk, first[0], "사과", "ko", 2.5, 2)
    assert second[0]["done"] is True
    assert second[0]["sm"].streak == 2
    assert "2번째 프레임" in second[1]
    assert second[-1]["recording"] is False
    assert "교체 1개" in second[7]
    assert "사 → 다" in second[7]
    hint = second[8]["korean_feedback"]["items"][0]
    assert hint["message"] == "‘사과’의 ‘사’가 ‘다’에 가까운 소리로 인식됐어요."
    assert hint["syllable_index"] == 0
    assert second[8]["schema_version"] == 2
    assert second[8]["frames"] == 2
    assert second[8]["substitution_count"] == 1
    assert second[8]["user_phonemes"] == recognizer.phonemes
    assert second[8]["threshold"] == 0.65
    assert [len(window) for window in recognizer.windows] == [8000, 16000]

    # Late audio after confirmation must not cause another inference.
    third = step(chunk, second[0], "사과", "ko", 2.5, 2)
    assert third[1] == second[1]
    assert third[7:9] == second[7:9]
    assert len(recognizer.windows) == 2


def test_live_buffer_keeps_only_recent_window(live_app):
    _, callbacks, recognizer = live_app
    state = {}
    for _ in range(3):
        result = callbacks["live_step"](
            (SAMPLE_RATE, np.zeros(8000, dtype=np.float32)),
            state, "사과", "ko", 1.0, 6,
        )
        state = result[0]
    assert [len(window) for window in recognizer.windows] == [8000, 16000, 16000]
    assert state["elapsed"] == 1.5


def test_reset_clears_session_and_displays(live_app):
    _, callbacks, _ = live_app
    reset = callbacks["live_reset"]()
    assert len(reset) == 9
    assert reset[0] == {}
    assert reset[6] == []
    assert reset[8] is None
    assert "정답이 확정되면" in reset[7]
    assert "초기화됨" in reset[1]


def test_empty_candidates_do_not_start_inference(live_app):
    _, callbacks, recognizer = live_app
    result = callbacks["live_step"](
        (SAMPLE_RATE, np.zeros(8000, dtype=np.float32)),
        {}, "", "ko", 2.5, 2,
    )
    assert len(result) == 10
    assert result[8] is None
    assert "정답 단어를 입력하세요" in result[1]
    assert not recognizer.windows


def test_warmup_uses_live_candidates_without_creating_a_session(live_app):
    _, callbacks, recognizer = live_app
    status, thresholds = callbacks["live_warmup"]("사과", "ko")
    assert "준비 완료" in status
    assert "사과" in thresholds
    assert recognizer.warmups == 1
    assert not recognizer.windows


def test_live_tester_does_not_require_a_built_target_catalog():
    tester = AudioTester(targets_path=None)
    assert tester.targets == {"answers": []}
    assert tester.available_target_languages() == []
