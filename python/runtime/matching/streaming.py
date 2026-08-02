"""Continuous-listening wrapper around `Matcher`.

The VR flow keeps the microphone open and re-scores as audio arrives,
stopping as soon as an answer is recognised. That is a different problem
from scoring one finished recording:

  * The user may say anything before the answer ("음.. 이건 과일 같은데..
    빨간걸 보니 사과!"), so the preamble must not count against them.
  * Scoring runs many times per session. With a 0.4 s hop over 10 s and
    four candidate answers that is ~100 scoring events; a 1% per-event
    false-accept rate compounds to ~63% per session. Bounding the
    context helps but does not remove this - the fix is to require the
    *same* answer to win several consecutive frames before committing.
    A chance alignment drifts to a different candidate on the next
    frame; a real utterance stays in the rolling window for its whole
    duration and wins every frame.

This module is deliberately free of audio and model dependencies: it
consumes phoneme lists, so it ports to C# as mechanically as `Matcher`
does. The caller owns the microphone, the rolling audio window and the
recogniser (see `python.runtime.audio.rolling_windows`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .matcher import Matcher, MatchResult


@dataclass
class StreamingHit:
    """A confirmed answer."""

    result: MatchResult
    frames: int        # how many frames were consumed before confirming
    streak: int        # consecutive frames the winner held


class StreamingMatcher:
    """Feed one frame of recognised phonemes at a time; get a hit when an
    answer has won `consecutive` frames in a row.

    Each frame's phonemes should come from re-recognising the most recent
    audio window (NOT from recognising isolated chunks and concatenating
    - wav2vec2 is a context model and mangles short fragments).

    Usage::

        sm = StreamingMatcher(Matcher.for_streaming(matrix), answers)
        for window in rolling_windows(...):
            hit = sm.push(recognizer.recognize(window))
            if hit:
                ...   # stop recording, move on
                break
    """

    def __init__(
        self,
        matcher: Matcher,
        candidates: list[dict],
        consecutive: int = 3,
    ) -> None:
        if consecutive < 1:
            raise ValueError(
                f"consecutive must be >= 1, got {consecutive!r}"
            )
        self.matcher = matcher
        self.candidates = candidates
        self.consecutive = consecutive
        self.reset()

    def reset(self) -> None:
        """Clear the streak. Call between questions."""
        self._streak_id: str | None = None
        self._streak = 0
        self._frames = 0

    @property
    def streak(self) -> int:
        """Consecutive frames the current leader has held (0 if none)."""
        return self._streak

    def push(self, phonemes: list[str]) -> StreamingHit | None:
        """Score one frame. Returns a hit once the streak is long enough.

        After a hit the streak is left intact, so pushing more frames
        keeps returning hits; call `reset()` when moving on.
        """
        self._frames += 1
        result = self.matcher.best_match(phonemes, self.candidates)

        if result.passed and result.target_id is not None:
            if result.target_id == self._streak_id:
                self._streak += 1
            else:
                self._streak_id = result.target_id
                self._streak = 1
        else:
            self._streak_id = None
            self._streak = 0

        if self._streak >= self.consecutive:
            return StreamingHit(
                result=result,
                frames=self._frames,
                streak=self._streak,
            )
        return None


__all__ = ["StreamingMatcher", "StreamingHit"]
