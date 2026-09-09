"""Show model-aware syllable verdicts and readable, unmuted sound hints.

The shared contract reviews display hints while retaining raw alignment.
"""

from html import escape
from copy import deepcopy
from pathlib import Path

from python.runtime.matching.feedback import ModelErrors, explain
from python.tools._web_feedback import pronunciation_feedback_html

ERRORS_PATH = Path(__file__).resolve().parents[2] / "shared/reference/reference_errors.json"
_MODEL_ERRORS = {}
_VERDICT = {
    "clean": ("#22c55e", "문제 없음"),
    "near": ("#eab308", "거의 맞음"),
    "blurred": ("#f97316", "흐림"),
    "different": ("#ef4444", "다르게 인식"),
    "no_comment": ("#22c55e", "문제 없음"),
}


def model_errors_for(matrix_id):
    if matrix_id not in _MODEL_ERRORS:
        try:
            table = ModelErrors.from_json(ERRORS_PATH) if ERRORS_PATH.exists() else None
        except (OSError, ValueError, TypeError):
            table = None
        _MODEL_ERRORS[matrix_id] = (
            table if table is not None and table.matrix_id == matrix_id else None
        )
    return _MODEL_ERRORS[matrix_id]


def confirmed_feedback_html(tester, feedback, candidate, language):
    matrix = tester._matrix_for(language)
    errors = model_errors_for(matrix.matrix_id)
    feedback = deepcopy(feedback)
    feedback.apply_model_errors(matrix.matrix_id, errors)
    guidance_html = pronunciation_feedback_html(feedback)
    syllables = candidate.get("syllables") or []
    if not syllables or not feedback.alignment_available:
        return guidance_html

    alignment = [
        (step.user_phoneme, step.target_phoneme, step.operation)
        for step in feedback.alignment
    ]
    try:
        reports = explain(alignment, syllables, matrix, errors)
    except ValueError:
        return guidance_html

    cards = []
    for report in reports:
        colour, label = _VERDICT[report.status]
        spoken = (
            f"<small>소리: {escape(report.spoken)}</small>"
            if report.resyllabified else ""
        )
        cards.append(
            f"<div style='display:inline-block;padding:10px;margin:4px;"
            f"border:2px solid {colour};border-radius:8px;'>"
            f"<div style='font-size:28px;'>{escape(report.text)}</div>"
            f"{spoken}<div>{label}</div></div>"
        )

    return (
        "<section><h3>정답 확정 · 음절별 참고 안내</h3>"
        f"{''.join(cards)}{guidance_html}</section>"
    )
