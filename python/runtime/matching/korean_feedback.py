"""Readable Korean hints from the *scored* IPA alignment, without rescoring.

Only localize to written syllables when their canonical IPA exactly matches
the target. Phonological changes, unknown sounds and ambiguous insertions
must not turn into invented spellings or falsely highlighted letters.
Mirrored by Runtime/Korean/KoreanFeedbackBuilder.cs; tested with shared vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from python.build.g2p.ko.jamo_ipa import (
    CODA_IPA, FINALS, INITIALS, MEDIALS, NUCLEUS_IPA, ONSET_IPA,
    decompose_syllable,
)

if TYPE_CHECKING:
    from .feedback import PhonemeComparison, PronunciationFeedback


@dataclass
class PronunciationHint:
    kind: str  # sub / del / ins / mixed; one hint may cover several differences
    syllable_index: int  # Hangul-only ordinal, zero-based; -1 when not localized
    target_syllable: str
    heard_syllable: str  # approximate sound, NOT an ASR transcript; "" if unavailable
    message: str
    alignment_indices: list[int]


@dataclass
class KoreanFeedback:
    syllable_mapping_available: bool = False
    summary: str = ""
    items: list[PronunciationHint] = field(default_factory=list)


def _compose(initial: str, medial: str, final: str) -> str:
    return chr(0xAC00 + (INITIALS.index(initial) * 21 + MEDIALS.index(medial)) * 28
               + FINALS.index(final))


_INITIAL_NAMES = (
    "기역", "쌍기역", "니은", "디귿", "쌍디귿", "리을", "미음", "비읍", "쌍비읍",
    "시옷", "쌍시옷", "이응", "지읒", "쌍지읒", "치읓", "키읔", "티읕", "피읖", "히읗",
)
_SOUND_LABELS = {
    ONSET_IPA[jamo][0]: name + " 소리"
    for jamo, name in zip(INITIALS, _INITIAL_NAMES) if ONSET_IPA[jamo]
}
_SOUND_LABELS.update({
    tokens[0]: f"‘{_compose('ㅇ', jamo, '')}’의 모음 소리"
    for jamo, tokens in NUCLEUS_IPA.items() if len(tokens) == 1
})
_SOUND_LABELS.update({
    "j": "‘야’에서 앞에 붙는 소리", "w": "‘와’에서 앞에 붙는 소리",
    "ɰ": "‘의’에서 앞에 붙는 소리", "k̚": "기역 계열 받침 소리",
    "t̚": "디귿 계열 받침 소리", "p̚": "비읍 계열 받침 소리",
    "l": "리을 소리", "ŋ": "받침 이응 소리",
})
# Representative *sounds*, not inferred spellings. Preserve the target coda
# spelling whenever its IPA is unchanged (e.g. 옷 must not become 옫).
_SIMPLE_CODAS = {jamo: CODA_IPA[jamo] for jamo in "ㄱㄴㄷㄹㅁㅂㅇ"}


def _sound(phoneme: str) -> str:
    return _SOUND_LABELS.get(phoneme, "한글로 옮기기 어려운 소리")


def _describe(step: PhonemeComparison) -> str:
    if step.operation == "sub":
        return f"{_sound(step.target_phoneme)}가 {_sound(step.user_phoneme)}로 인식됐어요."
    if step.operation == "del":
        return f"{_sound(step.target_phoneme)}가 인식 결과에서 빠졌어요."
    return f"정답에 없는 {_sound(step.user_phoneme)}가 추가로 인식됐어요."


def _reverse(tokens: list[str], table: dict[str, list[str]], original: str) -> str | None:
    if table.get(original) == tokens:
        return original
    candidates = [jamo for jamo, ipa in table.items() if ipa == tokens]
    return candidates[0] if len(candidates) == 1 else None


def _heard_syllable(jamo: tuple[str, str, str], parts: list[list[str]]) -> str:
    initial = _reverse(parts[0], ONSET_IPA, jamo[0])
    medial = _reverse(parts[1], NUCLEUS_IPA, jamo[1])
    if not parts[2]:
        final = ""
    elif CODA_IPA.get(jamo[2]) == parts[2]:
        final = jamo[2]
    else:
        final = _reverse(parts[2], _SIMPLE_CODAS, jamo[2])
    if initial is None or medial is None or final is None:
        return ""
    return _compose(initial, medial, final)


def build_korean_feedback(feedback: PronunciationFeedback) -> KoreanFeedback:
    guide = KoreanFeedback()
    if not feedback.alignment_available:
        guide.summary = "발음 비교 정보가 없어 자세히 설명할 수 없어요."
        return guide

    syllables = []
    owners: list[tuple[int, int]] = []  # target IPA index -> syllable, onset/vowel/coda
    canonical = []
    for char in feedback.target_text or "":
        jamo = decompose_syllable(char)
        if jamo is None:
            continue
        parts = [ONSET_IPA[jamo[0]], NUCLEUS_IPA[jamo[1]], CODA_IPA.get(jamo[2], [])]
        for part_index, tokens in enumerate(parts):
            canonical.extend(tokens)
            owners.extend([(len(syllables), part_index)] * len(tokens))
        syllables.append((char, jamo))

    target_steps = [step for step in feedback.alignment if step.operation != "ins"]
    guide.syllable_mapping_available = bool(syllables) and (
        canonical == feedback.target_phonemes
        and [step.target_phoneme for step in target_steps] == canonical
        and [step.target_index for step in target_steps] == list(range(len(canonical)))
    )
    differences = [i for i, step in enumerate(feedback.alignment) if step.operation != "match"]
    if not differences:
        guide.summary = "비교한 구간에서 정답과 다른 소리가 발견되지 않았어요."
        return guide
    guide.summary = "정답과 다르게 인식된 부분이에요."
    if not guide.syllable_mapping_available:
        guide.summary += " 글자 위치를 정확히 연결하기 어려워 소리만 안내해요."
        guide.items = [PronunciationHint(
            feedback.alignment[i].operation, -1, "", "", _describe(feedback.alignment[i]), [i]
        ) for i in differences]
        return guide

    # An insertion has no target position. Do not force it into a syllable;
    # its neighbors cannot be safely reconstructed as complete syllables.
    blocked = set()
    for i, step in enumerate(feedback.alignment):
        if step.operation != "ins":
            continue
        for candidates in (reversed(feedback.alignment[:i]), feedback.alignment[i + 1:]):
            neighbor = next((s for s in candidates if s.target_index >= 0), None)
            if neighbor is not None:
                blocked.add(owners[neighbor.target_index][0])

    heard_parts = [[[], [], []] for _ in syllables]
    groups: dict[int, list[int]] = {}
    for i, step in enumerate(feedback.alignment):
        if step.operation == "ins":
            guide.items.append(PronunciationHint("ins", -1, "", "", _describe(step), [i]))
            continue
        owner, part = owners[step.target_index]
        if step.operation != "del":
            heard_parts[owner][part].append(step.user_phoneme)
        if step.operation != "match":
            groups.setdefault(owner, []).append(i)

    for owner, indices in groups.items():
        char, jamo = syllables[owner]
        operations = {feedback.alignment[i].operation for i in indices}
        kind = next(iter(operations)) if len(operations) == 1 else "mixed"
        subject = f"‘{feedback.target_text}’의 ‘{char}’"
        heard = "" if owner in blocked else _heard_syllable(jamo, heard_parts[owner])
        if heard and heard != char:
            particle = "이" if jamo[2] else "가"
            message = f"{subject}{particle} ‘{heard}’에 가까운 소리로 인식됐어요."
        elif not any(heard_parts[owner]) and owner not in blocked:
            message = f"{subject} 소리가 인식 결과에서 빠졌어요."
        else:
            heard = ""
            message = subject + ": " + " ".join(_describe(feedback.alignment[i]) for i in indices)
        guide.items.append(PronunciationHint(kind, owner, char, heard, message, indices))
    guide.items.sort(key=lambda item: item.alignment_indices[0])
    return guide
