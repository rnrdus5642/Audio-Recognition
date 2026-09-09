"""Conservative display review of ASR differences; never changes scoring.

An absent pair in a sparse error table is unknown, not a measured zero.
Mirrored by Runtime/ModelFeedback.cs and shared contract vectors.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from .korean_feedback import KoreanFeedback


@dataclass
class WithheldDifference:
    alignment_index: int
    reason: str
    observations: int = 0
    total: int = 0
    model_error_rate: float | None = None


@dataclass
class FeedbackReview:
    matrix_id: str = ""
    profile_matrix_id: str = ""
    reference_utterances: int = 0
    status: str = "unavailable"
    withheld: list[WithheldDifference] = field(default_factory=list)


def withholding_reason(operation, statistics, profile_matches, *, mute_above=.02, min_observations=2):
    if not profile_matches:
        return "missing_or_mismatched_profile"
    if operation == "ins":
        return "unmeasured_insertion"
    if statistics is None:
        return "unmeasured_difference"
    count, total = statistics
    if (type(count) is not int or type(total) is not int
            or count < 0 or total <= 0 or count > total or count < min_observations):
        return "insufficient_observations"
    return "known_model_confusion" if count / total >= mute_above else ""


def review_feedback(feedback, matrix_id=None, errors=None):
    """Return UI-safe hints and an audit of withheld raw differences."""
    raw = feedback.raw_korean_feedback
    matches = bool(matrix_id and matrix_id != "unknown" and errors
                   and errors.matrix_id == matrix_id)
    review = FeedbackReview(
        matrix_id=matrix_id or "", profile_matrix_id=errors.matrix_id if errors else "",
        reference_utterances=errors.utterances if matches else 0)
    for index, step in enumerate(feedback.alignment):
        if step.operation == "match":
            continue
        stats = errors.statistics(step.operation, step.target_phoneme, step.user_phoneme) if matches else None
        reason = ("acoustic_keyword_recovery" if getattr(feedback, "_acoustic_recovery", False)
                  else withholding_reason(step.operation, stats, matches))
        if reason:
            count, total = stats or (0, 0)
            review.withheld.append(WithheldDifference(
                index, reason, count, total, count / total if total > 0 else None))

    muted = {item.alignment_index for item in review.withheld}
    # A reconstructed syllable can include several differences. Withhold
    # the entire hint if even one of its ingredients cannot be attributed.
    items = [deepcopy(hint) for hint in raw.items if not muted.intersection(hint.alignment_indices)]
    if not feedback.alignment_available:
        status, summary = "unavailable", raw.summary
    elif review.withheld:
        status = "partial" if items else "withheld"
        # Presentation wording does not change the internal uncertainty record.
        summary = ("아래 부분만 조금 더 또렷하게 말해 보세요."
                   if items else "발음에 문제 없어요.")
    elif items:
        status, summary = "review", raw.summary
    else:
        status, summary = "matched", raw.summary
    review.status = status
    return KoreanFeedback(raw.syllable_mapping_available, summary, items), review
