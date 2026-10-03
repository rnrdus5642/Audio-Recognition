"""Render the shared feedback contract; no pronunciation scoring lives here."""

from html import escape

from python.runtime.matching import PronunciationFeedback


def pronunciation_feedback_html(feedback: PronunciationFeedback | None) -> str:
    if feedback is None:
        return (
            "<p style='color:#6b7280;'>정답이 확정되면 쉬운 한글 발음 설명이 "
            "여기에 표시됩니다.</p>"
        )

    title = (
        "<section style='padding:16px;border:1px solid #d1d5db;"
        "border-radius:8px;'>"
        "<h3 style='margin:0 0 8px;'>확정된 발음 비교 · 쉬운 설명</h3>"
        f"<p><b>{escape(feedback.target_text or '')}</b> · "
        f"{feedback.frames}번째 확정 프레임 기준 "
        f"(연속 {feedback.streak}회)</p>"
    )
    caution = (
        "<p style='font-size:13px;color:#6b7280;'>"
        "인식된 소리와 정답을 비교한 안내입니다. 한글 예시는 소리를 쉽게 풀어 쓴 것으로, "
        "인식 문장의 표기와는 다를 수 있습니다. "
        "실제 발음 오류를 단정하는 진단은 아닙니다.</p>"
    )
    guide = feedback.korean_feedback
    readable = f"<p>{escape(guide.summary)}</p>"
    kinds = {"sub": "다른 소리", "del": "빠진 소리", "ins": "더해진 소리", "mixed": "여러 차이"}
    for hint in guide.items:
        label = kinds[hint.kind]
        if hint.syllable_index >= 0:
            label = f"{hint.syllable_index + 1}번째 한글 · {label}"
        example = escape(hint.target_syllable)
        if hint.heard_syllable:
            example += " → " + escape(hint.heard_syllable)
        readable += (
            "<div style='margin:10px 0;padding:12px 14px;border-left:4px solid #d97706;"
            "background:#fffbeb;color:#78350f;border-radius:6px;'>"
            f"<div style='font-size:13px;'>{label}</div>"
            + (f"<div style='font-size:22px;font-weight:700;'>{example}</div>" if example else "")
            + f"<p style='margin:6px 0 0;'>{escape(hint.message)}</p></div>"
        )
    if not feedback.alignment_available:
        return title + readable + "<p>이 결과에는 음소 정렬 정보가 없습니다.</p>" + caution + "</section>"

    labels = {
        "match": ("일치", "#dcfce7", "#166534"),
        "sub": ("교체", "#ffedd5", "#9a3412"),
        "del": ("누락", "#fee2e2", "#991b1b"),
        "ins": ("추가", "#dbeafe", "#1e40af"),
    }
    counts = (
        f"<p>일치 {feedback.match_count}개 · "
        f"<b>교체 {feedback.substitution_count}개 · "
        f"누락 {feedback.deletion_count}개 · "
        f"추가 {feedback.insertion_count}개</b></p>"
    )
    differences = (
        feedback.substitution_count + feedback.deletion_count + feedback.insertion_count
    )
    if differences == 0:
        counts += "<p>비교 구간에서 다른 음소가 발견되지 않았습니다.</p>"

    rows = []
    for step in feedback.alignment:
        label, background, color = labels[step.operation]
        target_position = "—" if step.target_index < 0 else str(step.target_index + 1)
        user_position = "—" if step.user_index < 0 else str(step.user_index + 1)
        cells = [
            target_position, escape(step.target_phoneme) or "—",
            user_position, escape(step.user_phoneme) or "—",
        ]
        rows.append(
            "<tr>" + "".join(
                f"<td style='padding:6px 10px;text-align:center;'>{cell}</td>"
                for cell in cells
            ) + f"<td style='padding:6px 10px;text-align:center;"
            f"background:{background};color:{color};font-weight:600;'>{label}</td></tr>"
        )
    table = (
        "<div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr>"
        + "".join(f"<th style='padding:6px;'>{h}</th>" for h in (
            "정답 음소 번호", "정답 소리", "인식 음소 번호", "인식된 소리", "비교"
        ))
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    outside = feedback.window_start + len(feedback.user_phonemes) - feedback.window_end
    note = (
        "<p style='font-size:13px;color:#6b7280;'>"
        "번호는 각 IPA 배열에서 1부터 표시합니다. 한글 글자 위치나 발음 시각은 아닙니다."
    )
    if outside:
        note += f" 비교 구간 밖 {outside}개 음소는 위 교체·누락·추가 개수에서 제외합니다."
    note += " 점수는 차이 개수의 단순 비율이 아닌 가중 유사도입니다.</p>"
    details = (
        "<details style='margin-top:14px;'><summary style='cursor:pointer;'>"
        "IPA 상세 비교 (개발·디버깅용)</summary>"
        + counts + table + note + "</details>"
    )
    return title + readable + details + caution + "</section>"
