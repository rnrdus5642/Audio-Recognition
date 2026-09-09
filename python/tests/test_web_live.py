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
HOP = 8000  # samples that make one frame, whatever the chunk size is
CHUNK_SPLITS = {
    "16000x1": [16000],
    "8000x2": [8000],
    "4000x4": [4000],
    "irregular": [1, 7999, 3, 5000, 4001],
    "longer than the window": [64000],
}


class FakeRecognizer:
    def __init__(self):
        self.windows = []
        self.warmups = 0
        self.phonemes = APPLE_IPA.copy()
        self.calls = 0
        # Which recognition attempt blows up, and whether the recogniser
        # can be obtained at all. Both are ways a live session dies
        # halfway through a recording.
        self.fail_on = None
        self.acquire_error = None

    def recognize(self, audio):
        self.warmups += 1
        return []

    def recognize_with_text(self, audio):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("모델이 응답하지 않습니다")
        self.windows.append(audio.copy())
        return "사과", self.phonemes.copy()


@pytest.fixture
def live_app(monkeypatch):
    tester = AudioTester(targets_path=None, matrix_path=MATRIX_PATH)
    recognizer = FakeRecognizer()

    def provider(language):
        if recognizer.acquire_error:
            raise RuntimeError(recognizer.acquire_error)
        return recognizer

    monkeypatch.setattr(tester, "_recognizer_for", provider)
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
    assert second[8]["korean_feedback"]["items"] == []
    assert second[8]["feedback_review"]["status"] == "withheld"
    assert "사 → 다" not in second[7].split("<details", 1)[0]
    hint = second[8]["raw_korean_feedback"]["items"][0]
    assert hint["message"] == "‘사과’의 ‘사’가 ‘다’에 가까운 소리로 인식됐어요."
    assert hint["syllable_index"] == 0
    assert second[8]["schema_version"] == 3
    assert second[8]["frames"] == 2
    assert second[8]["substitution_count"] == 1
    assert second[8]["user_phonemes"] == recognizer.phonemes
    assert second[8]["threshold"] == 0.65
    assert [len(window) for window in recognizer.windows] == [40000, 40000]
    assert second[0]["elapsed"] == 1.0  # startup silence is not elapsed audio

    # Late audio after confirmation must not cause another inference.
    third = step(chunk, second[0], "사과", "ko", 2.5, 2)
    assert third[1] == second[1]
    assert third[7:9] == second[7:9]
    assert len(recognizer.windows) == 2


def test_live_buffer_keeps_only_recent_window(live_app):
    _, callbacks, recognizer = live_app
    state = {}
    for i in range(3):
        result = callbacks["live_step"](
            (SAMPLE_RATE, np.full(8000, (i + 1) / 10, dtype=np.float32)),
            state, "사과", "ko", 1.0, 6,
        )
        state = result[0]
    assert [len(window) for window in recognizer.windows] == [16000] * 3
    np.testing.assert_array_equal(recognizer.windows[0][:8000], 0)
    np.testing.assert_array_equal(recognizer.windows[0][8000:], np.float32(0.1))
    np.testing.assert_array_equal(recognizer.windows[-1][:8000], np.float32(0.2))
    np.testing.assert_array_equal(recognizer.windows[-1][8000:], np.float32(0.3))
    assert state["elapsed"] == 1.5


@pytest.mark.parametrize("window_s", [1.0, 2.5, 4.0])
def test_live_windows_are_cut_on_sample_boundaries(live_app, window_s):
    """One window per 8000 new samples, left-padded like Unity.

    A chunk that finishes three hops scores three windows, each holding
    the audio as it stood at its own boundary - not the newest one three
    times over.
    """
    _, callbacks, recognizer = live_app
    recognizer.phonemes = []  # keep listening past a full window
    keep = int(window_s * SAMPLE_RATE)
    state, received, expected = {}, np.zeros(0, dtype=np.float32), []
    # Tiny, irregular and larger-than-window inputs, not just whole hops.
    for size in [1, 4000, 8000, 28000, 64001, 8000]:
        chunk = np.linspace(-0.75, 0.75, size, dtype=np.float32)
        original = chunk.copy()
        before = len(received)
        result = callbacks["live_step"](
            (SAMPLE_RATE, chunk), state, "사과", "ko", window_s, 2,
        )
        state = result[0]
        received = np.concatenate([received, chunk])
        for end in range((before // HOP + 1) * HOP, len(received) + 1, HOP):
            tail = received[max(0, end - keep):end]
            expected.append(np.pad(tail, (keep - len(tail), 0)))

        assert len(recognizer.windows) == len(expected)
        for got, want in zip(recognizer.windows, expected):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(chunk, original)

        window = state["framer"].snapshot()
        np.testing.assert_array_equal(
            window,
            np.pad(received[-keep:], (max(0, keep - len(received)), 0)))
        assert window.dtype == np.float32
        assert state["elapsed"] == pytest.approx(
            len(received) // HOP * HOP / SAMPLE_RATE)
        assert result[8] is None
        assert state["sm"].streak == 0


@pytest.mark.parametrize("window_s", [1.0, 4.0])
def test_changing_window_restarts_audio_and_confirmation(live_app, window_s):
    _, callbacks, recognizer = live_app
    step = callbacks["live_step"]
    first = step(
        (SAMPLE_RATE, np.full(8000, 0.25, dtype=np.float32)),
        {}, "사과", "ko", 2.5, 2,
    )
    assert first[0]["sm"].streak == 1
    second = step(
        (SAMPLE_RATE, np.full(8000, 0.5, dtype=np.float32)),
        first[0], "사과", "ko", window_s, 2,
    )
    assert second[0]["sm"].streak == 1  # do not combine different windows
    assert second[0]["elapsed"] == 0.5
    assert len(second[0]["rows"]) == len(second[0]["points"]) == 1
    assert second[8] is None
    assert len(recognizer.windows[-1]) == int(window_s * SAMPLE_RATE)
    np.testing.assert_array_equal(recognizer.windows[-1][:-8000], 0)
    np.testing.assert_array_equal(recognizer.windows[-1][-8000:], 0.5)


def test_reset_starts_with_silence_not_the_previous_audio(live_app):
    _, callbacks, recognizer = live_app
    step = callbacks["live_step"]
    first = step(
        (SAMPLE_RATE, np.full(8000, 0.25, dtype=np.float32)),
        {}, "사과", "ko", 2.5, 2,
    )
    assert first[0]["sm"].streak == 1
    reset = callbacks["live_reset"]()
    result = step(
        (SAMPLE_RATE, np.full(8000, 0.5, dtype=np.float32)),
        reset[0], "사과", "ko", 2.5, 2,
    )
    np.testing.assert_array_equal(recognizer.windows[-1][:-8000], 0)
    np.testing.assert_array_equal(recognizer.windows[-1][-8000:], 0.5)
    assert result[0]["elapsed"] == 0.5
    assert result[0]["sm"].streak == 1
    assert result[8] is None


@pytest.mark.parametrize("empty", [None, (SAMPLE_RATE, np.zeros(0, dtype=np.float32))])
def test_empty_audio_does_not_score_startup_silence(live_app, empty):
    _, callbacks, recognizer = live_app
    result = callbacks["live_step"](empty, {}, "사과", "ko", 2.5, 2)
    assert not recognizer.windows
    assert result[8] is None
    assert result[0].get("elapsed", 0) == 0
    assert not result[0].get("done")


@pytest.mark.parametrize("name", list(CHUNK_SPLITS))
def test_confirmation_does_not_move_with_the_chunk_size(live_app, name):
    """The same PCM confirms on the same sample, however it arrives.

    This is the bug the framing replaces: one second of audio delivered
    as 16000x1 never confirmed, as 8000x2 confirmed, and as 4000x4
    confirmed halfway through it - because a frame was a callback.
    """
    _, callbacks, recognizer = live_app
    step = callbacks["live_step"]
    audio = np.linspace(-0.5, 0.5, 64000, dtype=np.float32)
    sizes = CHUNK_SPLITS[name]
    state, offset, i = {}, 0, 0
    while offset < len(audio) and not state.get("done"):
        size = min(sizes[i % len(sizes)], len(audio) - offset)
        state = step((SAMPLE_RATE, audio[offset:offset + size]),
                     state, "사과", "ko", 2.5, 2)[0]
        offset += size
        i += 1

    assert state["done"] is True
    assert state["elapsed"] == 1.0, "confirmed somewhere other than 1.0 s"
    assert len(recognizer.windows) == 2, "audio after the answer was scored"
    assert state["confirmation"]["frames"] == 2
    assert state["confirmation"]["streak"] == 2
    assert [row[0] for row in state["rows"]] == [1.0, 0.5]
    np.testing.assert_array_equal(
        recognizer.windows[1][-16000:], audio[:16000])


def test_a_partial_hop_waits_for_the_rest_of_it(live_app):
    _, callbacks, recognizer = live_app
    step = callbacks["live_step"]

    first = step((SAMPLE_RATE, np.full(HOP - 1, 0.25, dtype=np.float32)),
                 {}, "사과", "ko", 2.5, 2)
    assert not recognizer.windows
    assert first[0]["elapsed"] == 0.0
    assert first[0]["rows"] == []
    assert first[0]["sm"].streak == 0

    second = step((SAMPLE_RATE, np.full(1, 0.25, dtype=np.float32)),
                  first[0], "사과", "ko", 2.5, 2)
    assert len(recognizer.windows) == 1
    assert second[0]["elapsed"] == 0.5
    assert second[0]["sm"].streak == 1


def test_repeated_polls_without_a_whole_hop_never_score(live_app):
    _, callbacks, recognizer = live_app
    step = callbacks["live_step"]
    state = {}
    for _ in range(6):
        state = step((SAMPLE_RATE, np.full(1000, 0.1, dtype=np.float32)),
                     state, "사과", "ko", 2.5, 2)[0]

    assert not recognizer.windows
    assert state["framer"].total_samples == 6000
    assert state["elapsed"] == 0.0

    state = step((SAMPLE_RATE, np.full(2000, 0.1, dtype=np.float32)),
                 state, "사과", "ko", 2.5, 2)[0]
    assert len(recognizer.windows) == 1
    assert state["elapsed"] == 0.5


def test_reset_drops_a_half_finished_hop(live_app):
    _, callbacks, recognizer = live_app
    step = callbacks["live_step"]
    half = np.full(4000, 0.25, dtype=np.float32)

    first = step((SAMPLE_RATE, half), {}, "사과", "ko", 2.5, 2)
    assert not recognizer.windows

    state = callbacks["live_reset"]()[0]
    state = step((SAMPLE_RATE, np.full(4000, 0.5, dtype=np.float32)),
                 state, "사과", "ko", 2.5, 2)[0]
    assert not recognizer.windows, "the abandoned half hop completed a frame"

    state = step((SAMPLE_RATE, np.full(4000, 0.5, dtype=np.float32)),
                 state, "사과", "ko", 2.5, 2)[0]
    assert len(recognizer.windows) == 1
    np.testing.assert_array_equal(recognizer.windows[0][-8000:], 0.5)
    assert state["elapsed"] == 0.5


@pytest.mark.parametrize("sizes", [[8000, 8000], [16000]])
def test_a_failed_frame_ends_the_recording_instead_of_bridging_the_streak(
    live_app, sizes
):
    """A hop that never scored must not join the hops either side of it.

    Gradio hands the same state object back on the next chunk, so a
    streak of 1 left behind by a failed frame would be waiting there for
    the next passing frame to confirm on - two passes with a hole in the
    middle, which is not what "연속 2회" means.
    """
    _, callbacks, recognizer = live_app
    step = callbacks["live_step"]
    recognizer.fail_on = 2
    state, result = {}, None
    for size in sizes:
        result = step((SAMPLE_RATE, np.zeros(size, dtype=np.float32)),
                      state, "사과", "ko", 2.5, 2)
        state = result[0]

    assert recognizer.calls == 2
    assert state["done"] is True, "the recording carried on after a failure"
    assert state["sm"].streak == 0, "the streak survived the failed frame"
    assert state["framer"].total_samples == 0
    assert state["rows"] == [] and state["points"] == []
    assert result[8] is None
    assert state["confirmation"] is None
    assert result[-1]["recording"] is False
    assert "채점 실패" in result[1]
    assert "모델이 응답하지 않습니다" in result[1]

    # Whatever arrives after that must not reach the model, and certainly
    # must not confirm.
    late = step((SAMPLE_RATE, np.zeros(16000, dtype=np.float32)),
                state, "사과", "ko", 2.5, 2)
    assert recognizer.calls == 2
    assert late[8] is None
    assert late[0]["sm"].streak == 0


def test_a_recogniser_that_cannot_be_loaded_ends_the_recording(live_app):
    _, callbacks, recognizer = live_app
    recognizer.acquire_error = "<script>alert(1)</script> 모델 없음"

    result = callbacks["live_step"](
        (SAMPLE_RATE, np.zeros(8000, dtype=np.float32)),
        {}, "사과", "ko", 2.5, 2,
    )
    state = result[0]

    assert state["done"] is True
    assert not recognizer.windows
    # feed() spent the hop before the recogniser was asked for, so the
    # framer has to be rewound too.
    assert state["framer"].total_samples == 0
    assert state["sm"].streak == 0
    assert result[-1]["recording"] is False
    assert "<script>" not in result[1]
    assert "&lt;script&gt;" in result[1]


def test_a_new_recording_after_a_failure_needs_two_passes_again(live_app):
    _, callbacks, recognizer = live_app
    step = callbacks["live_step"]
    chunk = (SAMPLE_RATE, np.zeros(8000, dtype=np.float32))
    recognizer.fail_on = 2
    state = step(chunk, {}, "사과", "ko", 2.5, 2)[0]
    state = step(chunk, state, "사과", "ko", 2.5, 2)[0]
    assert state["done"] is True

    recognizer.fail_on = None
    state = callbacks["live_reset"]()[0]

    first = step(chunk, state, "사과", "ko", 2.5, 2)
    assert first[0]["sm"].streak == 1
    assert first[8] is None, "one pass confirmed after a failed recording"

    second = step(chunk, first[0], "사과", "ko", 2.5, 2)
    assert second[0]["done"] is True
    assert second[8]["frames"] == 2
    assert second[8]["streak"] == 2


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


@pytest.mark.parametrize("words", ["빵, 강아지", "강아지, 빵"])
def test_live_multi_answer_reports_the_passing_word_and_its_feedback(monkeypatch, words):
    # Exercise real live target construction and child thresholds, replacing
    # only the acoustic recognizer with the reproducible audit phonemes.
    tester = AudioTester(
        targets_path=None,
        matrix_path=MATRIX_PATH.with_name("ko_child_v2.json"),
    )
    recognizer = FakeRecognizer()
    recognizer.phonemes = ["p", "a", "ŋ", "a", "tɕ", "i"]
    monkeypatch.setattr(tester, "_recognizer_for", lambda language: recognizer)
    app = build_app(tester)
    try:
        step = next(fn.fn for fn in app.fns.values()
                    if fn.fn and fn.fn.__name__ == "live_step")
        chunk = (SAMPLE_RATE, np.zeros(8000, dtype=np.float32))
        first = step(chunk, {}, words, "ko", 2.5, 2)
        assert first[0]["sm"].streak == 1
        assert first[6][0][2:] == ["강아지", 0.899, 0.8]
        assert first[8] is None

        second = step(chunk, first[0], words, "ko", 2.5, 2)
        assert second[0]["done"] is True
        assert second[-1]["recording"] is False
        payload = second[8]
        assert payload["target_text"] == "강아지"
        assert payload["threshold"] == 0.8
        assert payload["score"] == pytest.approx(0.8991666666666667)
        assert payload["frames"] == payload["streak"] == 2
        assert payload["target_phonemes"] == ["k", "a", "ŋ", "a", "tɕ", "i"]
        assert "강아지" in second[1] and "강아지" in second[7]
        assert len(recognizer.windows) == 2

        # A bread-only question must keep its failing score for display,
        # without accepting the puppy from the wider catalog.
        bread = step(chunk, {}, "빵", "ko", 2.5, 2)
        assert bread[0]["sm"].streak == 0
        assert bread[6][0][2:] == ["빵", 0.928, 0.975]
        assert bread[8] is None
        assert not bread[0].get("done")
    finally:
        app.close()
