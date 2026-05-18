"""Minimal target-data builder for the Gradio web UI.

언어 선택 + 단어 여러 줄 입력 → IPA 추출 → JSON 다운로드.
세그먼트/정답 ID, 임계값, 검증 같은 빌드 파이프라인용 메타데이터는
모두 제거. 단어 → 음소만 뽑는 단순한 유틸.
"""

from __future__ import annotations

import json
import tempfile
from typing import Optional

from python.build.g2p import get_g2p


_G2P_CACHE: dict[str, object] = {}


def _g2p_for(language: str):
    if language not in _G2P_CACHE:
        _G2P_CACHE[language] = get_g2p(language)
    return _G2P_CACHE[language]


def _parse_words(raw_text: str) -> list[str]:
    """한 줄에 한 단어. 빈 줄과 '#' 주석은 무시."""
    words: list[str] = []
    seen: set[str] = set()
    for line in (raw_text or "").splitlines():
        w = line.strip()
        if not w or w.startswith("#"):
            continue
        if w in seen:
            continue
        seen.add(w)
        words.append(w)
    return words


def extract(language: str, raw_text: str) -> tuple[list[dict], list[str]]:
    """단어 목록을 IPA로 변환. (entries, errors) 반환."""
    words = _parse_words(raw_text)
    if not words:
        return [], ["입력된 단어가 없습니다."]

    try:
        g2p = _g2p_for(language)
    except KeyError as e:
        return [], [f"언어 미지원: {e}"]

    entries: list[dict] = []
    errors: list[str] = []
    for w in words:
        try:
            phonemes = g2p.to_ipa(w)
        except Exception as e:
            errors.append(f"'{w}': 변환 실패 ({e})")
            continue
        if not phonemes:
            errors.append(f"'{w}': 음소 생성 실패")
            continue
        entries.append(
            {"text": w, "language": language, "phonemes": phonemes}
        )
    return entries, errors


def render_table(entries: list[dict]) -> list[list]:
    return [
        [i, e["text"], " ".join(e["phonemes"]), len(e["phonemes"])]
        for i, e in enumerate(entries, start=1)
    ]


def export_json(entries: list[dict]) -> Optional[str]:
    if not entries:
        return None
    tmp = tempfile.NamedTemporaryFile(
        prefix="words_ipa_",
        suffix=".json",
        delete=False,
        mode="w",
        encoding="utf-8",
    )
    json.dump(entries, tmp, ensure_ascii=False, indent=2)
    tmp.close()
    return tmp.name


def status_ok(msg: str) -> str:
    return (
        "<div style='padding:8px;background:#dcfce7;color:#166534;"
        f"border-radius:6px;'>{msg}</div>"
    )


def status_warn(msg: str) -> str:
    return (
        "<div style='padding:8px;background:#fef3c7;color:#92400e;"
        f"border-radius:6px;'>{msg}</div>"
    )


def status_error(msg: str) -> str:
    return (
        "<div style='padding:8px;background:#fee2e2;color:#991b1b;"
        f"border-radius:6px;'>{msg}</div>"
    )
