"""Display-neutral phoneme feedback from an existing scoring alignment.

The snake_case JSON contract matches Unity's PronunciationFeedback. Indices
refer to IPA tokens, not Hangul characters or audio timestamps. No acoustic
inference or second alignment is performed here.
"""

from dataclasses import asdict, dataclass, field

from .matcher import MatchResult
from .korean_feedback import KoreanFeedback, build_korean_feedback


@dataclass
class PhonemeComparison:
    operation: str
    target_phoneme: str
    user_phoneme: str
    target_index: int  # zero-based; -1 for an insertion
    user_index: int    # zero-based in full user_phonemes; -1 for a deletion


@dataclass
class PronunciationFeedback:
    schema_version: int
    target_id: str | None
    target_text: str | None
    score: float
    distance: float
    threshold: float
    passed: bool
    frames: int
    streak: int
    target_phonemes: list[str]
    user_phonemes: list[str]
    window_start: int
    window_end: int
    alignment_available: bool
    alignment: list[PhonemeComparison]
    match_count: int
    substitution_count: int
    deletion_count: int
    insertion_count: int
    korean_feedback: KoreanFeedback = field(default_factory=KoreanFeedback)

    @classmethod
    def from_match(
        cls, result: MatchResult, frames: int, streak: int
    ) -> "PronunciationFeedback":
        steps = []
        counts = {"match": 0, "sub": 0, "del": 0, "ins": 0}
        user_index, target_index = result.window_start, 0
        for user, target, operation in result.alignment or []:
            steps.append(PhonemeComparison(
                operation=operation,
                target_phoneme=target,
                user_phoneme=user,
                target_index=-1 if operation == "ins" else target_index,
                user_index=-1 if operation == "del" else user_index,
            ))
            counts[operation] += 1
            if operation != "ins":
                target_index += 1
            if operation != "del":
                user_index += 1
        feedback = cls(
            schema_version=2,
            target_id=result.target_id,
            target_text=result.target_text,
            score=result.score,
            distance=result.distance,
            threshold=result.threshold,
            passed=result.passed,
            frames=frames,
            streak=streak,
            target_phonemes=list(result.target_phonemes or []),
            user_phonemes=list(result.user_phonemes),
            window_start=result.window_start,
            window_end=result.window_end,
            # Exact-mode matching currently has no alignment trace. Do
            # not misrepresent its absence as perfect pronunciation.
            alignment_available=bool(steps),
            alignment=steps,
            match_count=counts["match"],
            substitution_count=counts["sub"],
            deletion_count=counts["del"],
            insertion_count=counts["ins"],
        )
        feedback.korean_feedback = build_korean_feedback(feedback)
        return feedback

    def to_dict(self) -> dict:
        """A detached, JSON-serializable copy for UI and future clients."""
        return asdict(self)
