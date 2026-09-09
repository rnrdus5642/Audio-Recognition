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

import math
from dataclasses import InitVar, dataclass, field

from .matcher import Matcher, MatchResult, validate_word_policy
from .feedback import PronunciationFeedback

# An id that identifies nothing. A matrix file that never said which
# matrix it is loads under this name, and two of them are not the same
# matrix just because they are both anonymous.
UNNAMED_MATRIX = ("", "unknown")


def _names_a_matrix(matrix_id: object) -> bool:
    """Whether an id picks out one particular matrix."""
    return (isinstance(matrix_id, str)
            and matrix_id.strip() not in UNNAMED_MATRIX)


def _number(value: object) -> bool:
    """A real, finite number. `True` is not one, and neither is "0.8"."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _validated_profile(snapshot: object) -> dict | None:
    """The snapshot's streaming profile, if the snapshot is well formed.

    Calibration writes one fixed shape and the runtime compares itself
    against every field of it. A snapshot that is missing a field, or
    carries a string where a number belongs, describes nothing that can be
    checked - so it grants nothing, exactly as the C# parser returns null
    for the same file, rather than raising out of the matching path.
    """
    if not isinstance(snapshot, dict):
        return None
    if not _names_a_matrix(snapshot.get("matrix_id")):
        return None
    if not isinstance(snapshot.get("text"), str):
        return None
    phonemes = snapshot.get("phonemes")
    if not isinstance(phonemes, list) or not phonemes:
        return None
    if not all(isinstance(p, str) for p in phonemes):
        return None
    if not _number(snapshot.get("threshold")):
        return None
    consecutive = snapshot.get("consecutive")
    if not isinstance(consecutive, int) or isinstance(consecutive, bool):
        return None
    profile = snapshot.get("streaming_profile")
    if not isinstance(profile, dict):
        return None
    if not all(_number(profile.get(key))
               for key in ("skip_cost", "coverage", "context_mult")):
        return None
    return profile


@dataclass
class StreamingHit:
    """A confirmed answer."""

    result: MatchResult
    frames: int        # how many frames were consumed before confirming
    streak: int        # consecutive frames the winner held
    required_frames: int = 2
    feedback: PronunciationFeedback = field(init=False)
    matrix_id: InitVar[str | None] = None
    model_errors: InitVar[object] = None

    def __post_init__(self, matrix_id, model_errors) -> None:
        # Snapshot the confirming frame, not an average of the streak.
        self.feedback = PronunciationFeedback.from_match(
            self.result, self.frames, self.streak, matrix_id=matrix_id, model_errors=model_errors
        )


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

    `consecutive` defaults to 2. Swept over rolling sessions built from
    all 36 golden clips (0.5 s hop), asking each clip's own word and
    every other word in turn:

        streak   detected      false accepts   confirmed after
          1      18/36 (50%)   19/612 (3.1%)   1.5 s
          2      18/36 (50%)   16/612 (2.6%)   2.1 s
          3      15/36 (42%)   10/612 (1.6%)   2.5 s
          4      15/36 (42%)    9/612 (1.5%)   3.0 s

    2 dominates 1 outright - same detections, fewer false accepts. Going
    to 3 buys 1 percentage point of false accepts and costs 8 points of
    detection, which is the wrong trade for a child who did say the word
    (see REPORT_PHASE2: false reject is the worse failure here).

    Adult TTS, though. Child speech scores less steadily frame to frame,
    which is exactly the condition a streak defends against, so re-run
    the sweep on real recordings before trusting 2 in the field.

    One exception, and only where calibration has licensed it. A word can
    carry `allow_exact_single_frame`, and then a single frame that scores
    exactly 1.0 confirms on its own. 1.0 means the phonemes that were
    SCORED - the trailing slice left by the context limit - were the
    target's sequence and nothing else; it does not say the whole
    transcript held nothing but the answer, and it is not "close" or
    0.9999. Short words often get exactly one such frame and never a
    second: the utterance is still inside the 2.5 s window on the next
    hop, but it sits somewhere else in it, and what the recogniser makes
    of it changes with that position. Measured on held-out child
    recordings this recovered 12 of 241 isolated words for 6 extra false
    confirmations over 2.2 hours of continuous speech.

    The escape is off unless the word was calibrated for it, `consecutive`
    is exactly 2, and the matcher still matches the snapshot calibration
    wrote into that word - same matrix, same streaming profile, same
    threshold and pronunciation. A caller who tunes skip_cost or coverage
    by hand, or who swapped the matrix, is outside what was calibrated, so
    the escape turns itself off rather than trust a number nobody
    measured.
    """

    def __init__(
        self,
        matcher: Matcher,
        candidates: list[dict],
        consecutive: int = 2,
        feedback_errors=None,
    ) -> None:
        if consecutive < 1:
            raise ValueError(
                f"consecutive must be >= 1, got {consecutive!r}"
            )
        self.matcher = matcher
        self.candidates = candidates
        self.consecutive = consecutive
        self.feedback_errors = feedback_errors
        for candidate in candidates:
            validate_word_policy(candidate)
        self.reset()

    def required_frames(self, target_id: str | None = None) -> int:
        """Word override, or the caller's default for legacy/unknown IDs."""
        winners = [c for c in self.candidates if c.get("id") == target_id]
        if target_id is None or len(winners) != 1:
            return self.consecutive
        validate_word_policy(winners[0])
        return winners[0].get("confirmation_frames") or self.consecutive

    def _exact_single_frame(self, result: MatchResult) -> bool:
        """A lone frame that reproduced the target exactly.

        Checked against the snapshot calibration wrote into the winning
        answer, not against whatever the current matrix happens to say.
        The approval belongs to one word, scored one way, under one
        matrix; if any of that has moved - a different matrix with its own
        valid profile, an edited threshold or pronunciation - the licence
        does not transfer. Everything is re-read on the frame that would
        confirm, so nothing can go stale after construction.
        """
        if not result.passed or result.score != 1.0 or self.consecutive != 2:
            return False
        if result.target_id is None:
            return False

        # An id shared by two candidates cannot identify which one won, so
        # neither may borrow the other's approval.
        winners = [c for c in self.candidates
                   if c.get("id") == result.target_id]
        if len(winners) != 1:
            return False
        answer = winners[0]
        # Explicit word policy replaces the older exact-only exception.
        if answer.get("confirmation_frames") is not None:
            return False
        # Strictly True: "false" and 1 are not approvals.
        if answer.get("allow_exact_single_frame") is not True:
            return False

        snapshot = answer.get("exact_single_frame_calibration")
        profile = _validated_profile(snapshot)
        if profile is None:
            return False
        if snapshot["consecutive"] != 2:
            return False
        if (snapshot["text"] != answer.get("text")
                or snapshot["phonemes"] != answer.get("phonemes")
                or snapshot["threshold"] != answer.get("threshold")):
            return False

        # Nothing above touches the matcher, so a caller with a stub and
        # no calibrated words is untouched.
        if getattr(self.matcher, "mode", "substring") != "substring":
            return False
        matrix = getattr(self.matcher, "matrix", None)
        actual = getattr(matrix, "matrix_id", None)
        # Both sides must name a matrix, and the same one: "unknown" is
        # not an identity, and matching it against itself would hand every
        # anonymous matrix an approval measured under one named matrix.
        if not _names_a_matrix(actual) or actual != snapshot["matrix_id"]:
            return False
        return (
            self.matcher.skip_cost == profile["skip_cost"]
            and self.matcher.coverage == profile["coverage"]
            and (self.matcher.context_mult or 0.0)
            == (profile["context_mult"] or 0.0)
        )

    def reset(self) -> None:
        """Clear the streak. Call between questions."""
        self._streak_id: str | None = None
        self._streak = 0
        self._frames = 0

    @property
    def streak(self) -> int:
        """Consecutive frames the current leader has held (0 if none)."""
        return self._streak

    def push(self, phonemes: list[str], acoustic_evidence=None) -> StreamingHit | None:
        """Score one frame. Returns a hit once the streak is long enough.

        After a hit the streak is left intact, so pushing more frames
        keeps returning hits; call `reset()` when moving on.
        """
        self._frames += 1
        result = (self.matcher.best_match(phonemes, self.candidates, acoustic_evidence)
                  if acoustic_evidence else self.matcher.best_match(phonemes, self.candidates))

        if result.passed and result.target_id is not None:
            if result.target_id == self._streak_id:
                self._streak += 1
            else:
                self._streak_id = result.target_id
                self._streak = 1
        else:
            self._streak_id = None
            self._streak = 0

        if (self._streak >= self.required_frames(result.target_id)
                or self._exact_single_frame(result)):
            return StreamingHit(
                result=result,
                frames=self._frames,
                # The real streak, not a fabricated second frame: a
                # display that says "2 of 2" for one frame would be a lie.
                streak=self._streak,
                required_frames=self.required_frames(result.target_id),
                matrix_id=getattr(getattr(self.matcher, "matrix", None), "matrix_id", None),
                model_errors=self.feedback_errors,
            )
        return None


__all__ = ["StreamingMatcher", "StreamingHit"]
