"""Opt-in CTC evidence for a short answer, separate from its raw transcript.

Log likelihoods measure model support, not pronunciation quality or accuracy.
No evidence (including recognizers without logits) preserves ordinary matching.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AcousticEvidence:
    target_text: str
    contrast_text: str
    log_probability: float
    greedy_log_probability: float
    contrast_log_probability: float

    def permits(self, answer: dict, phoneme_count: int, score: float) -> bool:
        p = answer.get("acoustic_keyword_policy")
        if not p or self.target_text != answer["text"] or self.contrast_text != p["contrast_text"]:
            return False
        values = (self.log_probability, self.greedy_log_probability, self.contrast_log_probability)
        return (all(math.isfinite(v) and v <= 1e-5 for v in values)
                and max(1, len(answer["phonemes"]) - 1) <= phoneme_count
                <= len(answer["phonemes"]) + p["max_extra_phonemes"]
                and score >= p["min_score"]
                and self.log_probability >= p["min_log_probability"]
                and self.greedy_log_probability - self.log_probability <= p["max_log_gap"]
                and self.contrast_log_probability - self.log_probability <= p["max_contrast_log_gap"])


def validate_acoustic_policy(answer: dict) -> None:
    p = answer.get("acoustic_keyword_policy")
    if p is None:
        return
    error = "acoustic_keyword_policy requires finite limits, a contrast_text and max_extra_phonemes in [0, 6]"
    if not isinstance(p, dict):
        raise ValueError(error)
    for key in ("min_score", "min_log_probability", "max_log_gap", "max_contrast_log_gap"):
        v = p.get(key)
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
            raise ValueError(error)
    if (not 0 < p["min_score"] <= 1 or not -100 <= p["min_log_probability"] < 0
            or not 0 <= p["max_log_gap"] <= 100 or not 0 <= p["max_contrast_log_gap"] <= 100
            or not isinstance(p.get("contrast_text"), str) or not p["contrast_text"].strip()
            or p["contrast_text"] == answer.get("text")
            or type(p.get("max_extra_phonemes")) is not int or not 0 <= p["max_extra_phonemes"] <= 6):
        raise ValueError(error)


def collect_acoustic_evidence(recognizer, candidates):
    """Called immediately after recognition; compatible with legacy recognizers."""
    method = getattr(recognizer, "acoustic_evidence", None)
    return {a["text"]: method(a["text"], a["acoustic_keyword_policy"]["contrast_text"])
            for a in candidates if a.get("acoustic_keyword_policy")} if method else None
