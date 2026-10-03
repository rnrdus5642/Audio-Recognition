"""Show model-aware syllable verdicts and readable, unmuted sound hints.

The JSON/Unity contract still describes the raw recognised alignment.
The error-table overlay is a separate Python display policy, not new scoring.
"""

from html import escape
from pathlib import Path

from python.runtime.matching.feedback import ModelErrors, explain
from python.tools._web_feedback import pronunciation_feedback_html

ERRORS_PATH = Path(__file__).resolve().parents[2] / "shared/reference/reference_errors.json"
_MODEL_ERRORS = {}
_VERDICT = {
    "clean": ("#22c55e", "음소 일치"),
    "near": ("#eab308", "거의 맞음"),
    "blurred": ("#f97316", "흐림"),
    "different": ("#ef4444", "다르게 인식"),
    "no_comment": ("#9ca3af", "판단 보류"),
}


def _model_errors(matrix_id):
    if matrix_id not in _MODEL_ERRORS:
        table = ModelErrors.from_json(ERRORS_PATH) if ERRORS_PATH.exists() else None
        _MODEL_ERRORS[matrix_id] = (
            table if table is not None and table.matrix_id == matrix_id else None
        )
    return _MODEL_ERRORS[matrix_id]


def confirmed_feedback_html(tester, feedback, candidate, language):
    raw_html = pronunciation_feedback_html(feedback)
    syllables = candidate.get("syllables") or []
    if not syllables or not feedback.alignment_available:
        return raw_html

    matrix = tester._matrix_for(language)
    errors = _model_errors(matrix.matrix_id)
    alignment = [
        (step.user_phoneme, step.target_phoneme, step.operation)
        for step in feedback.alignment
    ]
    try:
        reports = explain(alignment, syllables, matrix, errors)
    except ValueError:
        return raw_html

    differences = [
        index for index, step in enumerate(feedback.alignment)
        if step.operation != "match"
    ]
    slips = [slip for report in reports for slip in report.slips]
    muted_indices = {
        index for index, slip in zip(differences, slips) if slip.muted
    }
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

    # A hint may group several changes. If any is muted, withhold the
    # whole approximation rather than reconstructing a misleading syllable.
    hints = [
        f"<p>{escape(hint.message)}</p>"
        for hint in feedback.korean_feedback.items
        if not muted_indices.intersection(hint.alignment_indices)
    ]
    if muted_indices:
        hints.append(
            "<p>인식기가 자주 혼동하거나 근거가 부족한 차이는 "
            "발음 지적을 보류했습니다.</p>"
        )
    if errors is None:
        note = (
            f"⚠ {escape(matrix.matrix_id)}에 맞는 모델 오류표가 없습니다. "
            "인식 오류가 섞일 수 있으며 실제 발음의 진단이 아닙니다."
        )
    else:
        note = (
            f"{errors.utterances}발화의 모델 오류표를 참고한 안내입니다. "
            "음소 일치나 판단 보류가 실제 발음의 정확성을 보장하지는 않습니다."
        )
    return (
        "<section><h3>정답 확정 · 음절별 참고 안내</h3>"
        f"<p>{note}</p>{''.join(cards)}{''.join(hints)}"
        "<details><summary>인식 결과의 전체 한글·IPA 비교 "
        "(판단 보류 항목 포함)</summary>"
        "<p>아래 원본 비교와 공통 JSON에는 보류한 차이도 남아 있습니다. "
        "Unity의 기본 피드백도 이 원본 비교를 제공합니다.</p>"
        f"{raw_html}</details></section>"
    )
