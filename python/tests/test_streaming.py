"""Streaming (continuous listening) regression tests.

Fixtures hold real wav2vec2 output: each case was synthesised with
edge-tts, cut with `rolling_windows(2.5s window, 0.4s hop)` and
re-recognised frame by frame. That keeps these tests model-free while
still exercising the phoneme sequences the recogniser actually produces.

What continuous listening must get right, beyond batch scoring:

  * detect the answer even when the user rambles first
  * NOT fire before the answer was spoken (the child is still thinking)
  * NOT fire at all when the answer is never spoken, at ANY frame
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python.runtime.matching.confusion_matrix import ConfusionMatrix
from python.runtime.matching.matcher import Matcher
from python.runtime.matching.streaming import StreamingMatcher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = (
    PROJECT_ROOT / "shared" / "confusion_matrices" / "ko_child_v1.json"
)
FRAMES_PATH = (
    PROJECT_ROOT / "python" / "tests" / "fixtures" / "streaming_frames.json"
)

# Frames are 0.4 s apart and the onset of each answer was measured by
# synthesising the preamble separately, so allow one hop of slack when
# checking "did it fire before the answer was spoken".
ONSET_TOLERANCE_S = 0.5

# 'mom_ok' ends the moment the answer finishes, so there is no trailing
# audio to confirm the streak over - covered by its own test below.
UNCONFIRMABLE = {"mom_ok"}


@pytest.fixture(scope="module")
def matrix() -> ConfusionMatrix:
    return ConfusionMatrix.from_json(MATRIX_PATH)


@pytest.fixture(scope="module")
def frames() -> dict:
    return json.loads(FRAMES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def answers_by_text() -> dict[str, list[dict]]:
    """Text -> the candidate list for a question asking that word.

    One entry: a question asks for one word and scores that word. Words
    used to compete inside a segment, which measured identically and
    forced callers to know what else was on screen.
    """
    targets = json.loads(
        (PROJECT_ROOT / "shared" / "targets.json").read_text(encoding="utf-8")
    )
    return {a["text"]: [a] for a in targets["answers"]}


def run_case(case: dict, matrix: ConfusionMatrix, candidates: list[dict],
             consecutive: int = 3):
    """Replay one case. Returns (fire_time, answer_text) or (None, None)."""
    sm = StreamingMatcher(
        Matcher.for_streaming(matrix), candidates, consecutive=consecutive
    )
    for fr in case["frames"]:
        hit = sm.push(fr["ipa"])
        if hit:
            return fr["t"], hit.result.target_text
    return None, None


def cases_where(frames: dict, contains: bool) -> list[dict]:
    return [c for c in frames["cases"] if c["contains_answer"] is contains]


class TestStreamingDetection:
    def test_answer_after_rambling_is_detected(
        self, frames, matrix, answers_by_text
    ) -> None:
        """The user may say anything before the answer."""
        missed = []
        for case in cases_where(frames, True):
            if case["id"] in UNCONFIRMABLE:
                continue
            t, picked = run_case(
                case, matrix, answers_by_text[case["target_text"]]
            )
            if t is None or picked != case["target_text"]:
                missed.append(f"{case['id']}: got {picked!r} at {t}")
        assert not missed, "; ".join(missed)

    def test_never_fires_before_the_answer_is_spoken(
        self, frames, matrix, answers_by_text
    ) -> None:
        """Firing early would skip to the next question mid-thought.

        Regression: with free skipping, "음 이건 과일..." matched 사과 on
        the shared [k, w, a] long before 사과 was said.
        """
        early = []
        for case in cases_where(frames, True):
            if case["id"] in UNCONFIRMABLE:
                continue
            t, _ = run_case(
                case, matrix, answers_by_text[case["target_text"]]
            )
            onset = case["answer_onset_s"] or 0.0
            if t is not None and t < onset - ONSET_TOLERANCE_S:
                early.append(f"{case['id']}: fired {t}s, answer at {onset}s")
        assert not early, "; ".join(early)


class TestStreamingFalseAccept:
    def test_utterance_without_the_answer_never_fires(
        self, frames, matrix, answers_by_text
    ) -> None:
        """Must hold at EVERY frame, not just the last one."""
        leaks = []
        for case in cases_where(frames, False):
            t, picked = run_case(
                case, matrix, answers_by_text[case["target_text"]]
            )
            if t is not None:
                leaks.append(f"{case['id']}: {picked!r} at {t}s")
        assert not leaks, "; ".join(leaks)

    def test_single_frame_scoring_no_longer_leaks_here(
        self, frames, matrix, answers_by_text
    ) -> None:
        """The streak is no longer what stops these fixtures leaking.

        It was, when a question scored a whole segment: a chance
        alignment could crown some other word, and requiring the same
        winner several frames running threw that out. Scoring only the
        asked word removed that path, and on these fixtures
        `consecutive=1` now false-accepts nothing while `consecutive=3`
        costs one of five detections.

        Five recordings is not enough to drop the streak - a frame-level
        error rate still compounds over a long session - but the evidence
        that justified 3 is gone, so revisit it with real child
        recordings rather than assuming.
        """
        leaks = [
            case["id"]
            for case in cases_where(frames, False)
            if run_case(
                case, matrix, answers_by_text[case["target_text"]],
                consecutive=1,
            )[0] is not None
        ]
        assert not leaks, (
            f"single-frame scoring false-accepted {leaks}; the streak is "
            "load-bearing again and this test should say so"
        )

    def test_needs_trailing_audio_to_confirm(
        self, frames, matrix, answers_by_text
    ) -> None:
        """Confirmation costs `consecutive` frames of latency.

        'mom_ok' ends immediately after the answer, so the streak never
        completes. The recorder must keep listening for at least
        consecutive x hop seconds after speech stops.
        """
        case = next(c for c in frames["cases"] if c["id"] == "mom_ok")
        cands = answers_by_text[case["target_text"]]
        assert run_case(case, matrix, cands, consecutive=3)[0] is None
        # The answer *is* there - a shorter streak finds it.
        t, picked = run_case(case, matrix, cands, consecutive=1)
        assert t is not None and picked == case["target_text"]


class TestStreamingMatcherMechanics:
    @pytest.fixture
    def candidates(self) -> list[dict]:
        return [
            {"id": "apple", "text": "사과",
             "phonemes": ["s", "a", "k", "w", "a"], "threshold": 0.65},
            {"id": "mom", "text": "엄마",
             "phonemes": ["ʌ", "m", "m", "a"], "threshold": 0.70},
        ]

    def test_streak_must_be_the_same_answer(
        self, matrix, candidates
    ) -> None:
        """Alternating winners must not accumulate into a hit."""
        sm = StreamingMatcher(
            Matcher.for_streaming(matrix), candidates, consecutive=3
        )
        apple = ["s", "a", "k", "w", "a"]
        mom = ["ʌ", "m", "m", "a"]
        assert sm.push(apple) is None
        assert sm.push(mom) is None
        assert sm.push(apple) is None
        assert sm.streak == 1

    def test_hit_after_consecutive_frames(self, matrix, candidates) -> None:
        sm = StreamingMatcher(
            Matcher.for_streaming(matrix), candidates, consecutive=3
        )
        apple = ["s", "a", "k", "w", "a"]
        assert sm.push(apple) is None
        assert sm.push(apple) is None
        hit = sm.push(apple)
        assert hit is not None
        assert hit.result.target_id == "apple"
        assert hit.frames == 3

    def test_failing_frame_breaks_the_streak(
        self, matrix, candidates
    ) -> None:
        sm = StreamingMatcher(
            Matcher.for_streaming(matrix), candidates, consecutive=2
        )
        apple = ["s", "a", "k", "w", "a"]
        sm.push(apple)
        assert sm.push(["n", "n", "n"]) is None
        assert sm.streak == 0
        assert sm.push(apple) is None  # streak restarted

    def test_reset_clears_streak(self, matrix, candidates) -> None:
        sm = StreamingMatcher(
            Matcher.for_streaming(matrix), candidates, consecutive=2
        )
        apple = ["s", "a", "k", "w", "a"]
        sm.push(apple)
        sm.reset()
        assert sm.streak == 0
        assert sm.push(apple) is None

    def test_rejects_bad_consecutive(self, matrix, candidates) -> None:
        with pytest.raises(ValueError):
            StreamingMatcher(
                Matcher.for_streaming(matrix), candidates, consecutive=0
            )


class TestStreamingProfile:
    def test_profile_is_stricter_than_batch(self, matrix) -> None:
        batch = Matcher(matrix)
        stream = Matcher.for_streaming(matrix)
        assert stream.coverage > batch.coverage
        assert stream.skip_cost < batch.skip_cost
        assert batch.context_mult is None
        assert stream.context_mult is not None

    def test_context_limit_bounds_the_scored_window(self, matrix) -> None:
        """Speech older than the context limit must stop mattering.

        The score may drop for filler still inside the window, but once
        the limit is reached more speech changes nothing - that is what
        stops a long session from drifting in either direction.
        """
        stream = Matcher.for_streaming(matrix)
        target = ["s", "a", "k", "w", "a"]
        recent = ["s", "a", "k", "w", "a"]
        filler = ["n", "i", "l", "o", "m", "u", "t", "e"]

        _, short, _, _, _ = stream.score_against(filler * 5 + recent, target)
        _, long_, _, _, _ = stream.score_against(filler * 20 + recent, target)
        assert short == pytest.approx(long_)

        # Batch scoring has no limit, so the same input keeps degrading.
        # (Kept short enough that neither score has bottomed out at 0.)
        batch = Matcher(matrix)
        _, b_short, _, _, _ = batch.score_against(filler + recent, target)
        _, b_long, _, _, _ = batch.score_against(filler * 3 + recent, target)
        assert 0.0 < b_long < b_short
