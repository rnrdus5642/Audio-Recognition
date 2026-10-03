"""Korean Jamo -> IPA mapping.

This mapping is the SINGLE SOURCE OF TRUTH for Korean phoneme representation
in this project. Both the build pipeline (targets) and the runtime recognizer
(Korean ASR output -> IPA) MUST use this same mapping for the build and
runtime IPA sequences to be comparable.

Conventions:
  * Diphthongs are split into glide + vowel ("ㅑ" -> ["j", "a"]).
  * Aspirated consonants use the IPA aspiration diacritic ("kʰ").
  * Tense (fortis) consonants use the IPA "stiff voice" diacritic ("k͈").
  * Coda stops use the "unreleased" diacritic ("k̚").
  * Empty list means the jamo contributes no phoneme (silent ㅇ in onset).

Input is expected to be the *surface form* Hangul produced by g2pkk
(phonological rules already applied).
"""

from __future__ import annotations

# Standard Hangul syllable block offsets (Unified Hangul Block, U+AC00..U+D7A3)
HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3

# Cardinality per position in the Hangul syllable structure
N_INITIAL = 19
N_MEDIAL = 21
N_FINAL = 28  # includes "no final" (index 0)

# Compatibility-Jamo characters in canonical Hangul order
INITIALS = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
MEDIALS = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
# Index 0 = no final consonant
FINALS = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# ---------------------------------------------------------------------------
# IPA mapping tables
# ---------------------------------------------------------------------------

# Onset (initial) -> IPA phoneme list.
# Per Korean phonology: ㅇ in onset position is silent.
ONSET_IPA: dict[str, list[str]] = {
    "ㄱ": ["k"],
    "ㄲ": ["k͈"],
    "ㅋ": ["kʰ"],
    "ㄷ": ["t"],
    "ㄸ": ["t͈"],
    "ㅌ": ["tʰ"],
    "ㅂ": ["p"],
    "ㅃ": ["p͈"],
    "ㅍ": ["pʰ"],
    "ㅈ": ["tɕ"],
    "ㅉ": ["tɕ͈"],
    "ㅊ": ["tɕʰ"],
    "ㅅ": ["s"],
    "ㅆ": ["s͈"],
    "ㅎ": ["h"],
    "ㄴ": ["n"],
    "ㅁ": ["m"],
    "ㄹ": ["ɾ"],  # tap/flap; intervocalic and onset
    "ㅇ": [],     # silent
}

# Medial (vowel/diphthong) -> IPA list.
# Diphthongs deliberately split into glide + vowel so matching can be
# tolerant to glide deletion (a common child speech variation).
NUCLEUS_IPA: dict[str, list[str]] = {
    "ㅏ": ["a"],
    "ㅓ": ["ʌ"],
    "ㅗ": ["o"],
    "ㅜ": ["u"],
    "ㅡ": ["ɯ"],
    "ㅣ": ["i"],
    "ㅐ": ["ɛ"],
    "ㅔ": ["e"],
    "ㅑ": ["j", "a"],
    "ㅕ": ["j", "ʌ"],
    "ㅛ": ["j", "o"],
    "ㅠ": ["j", "u"],
    "ㅒ": ["j", "ɛ"],
    "ㅖ": ["j", "e"],
    "ㅘ": ["w", "a"],
    "ㅙ": ["w", "ɛ"],
    "ㅝ": ["w", "ʌ"],
    "ㅞ": ["w", "e"],
    "ㅚ": ["w", "e"],   # modern monophthongization, often realized as [we]
    "ㅟ": ["w", "i"],
    "ㅢ": ["ɰ", "i"],
}

# Coda (final) -> IPA list. Korean coda stops are unreleased.
# Consonant clusters in coda are reduced to a single representative per
# Korean 자음군 단순화 (cluster simplification). g2pkk should normally
# resolve clusters already, but we include them for safety.
CODA_IPA: dict[str, list[str]] = {
    # Single consonants (already neutralized by g2pkk's surface form)
    "ㄱ": ["k̚"],
    "ㄲ": ["k̚"],
    "ㅋ": ["k̚"],
    "ㄴ": ["n"],
    "ㄷ": ["t̚"],
    "ㅅ": ["t̚"],
    "ㅆ": ["t̚"],
    "ㅈ": ["t̚"],
    "ㅊ": ["t̚"],
    "ㅌ": ["t̚"],
    "ㅎ": ["t̚"],
    "ㄹ": ["l"],
    "ㅁ": ["m"],
    "ㅂ": ["p̚"],
    "ㅍ": ["p̚"],
    "ㅇ": ["ŋ"],
    # Consonant clusters (fallback; reduced form)
    "ㄳ": ["k̚"],
    "ㄵ": ["n"],
    "ㄶ": ["n"],
    "ㄺ": ["k̚"],
    "ㄻ": ["m"],
    "ㄼ": ["l"],
    "ㄽ": ["l"],
    "ㄾ": ["l"],
    "ㄿ": ["p̚"],
    "ㅀ": ["l"],
    "ㅄ": ["p̚"],
}


def decompose_syllable(syllable: str) -> tuple[str, str, str] | None:
    """Decompose a single Hangul syllable into (initial, medial, final).

    Uses the standard Unicode Hangul algorithm. Final is "" when absent.
    Returns None for non-Hangul characters.
    """
    if len(syllable) != 1:
        raise ValueError(f"decompose_syllable expects a single char, got {syllable!r}")

    code = ord(syllable)
    if not HANGUL_BASE <= code <= HANGUL_END:
        return None

    offset = code - HANGUL_BASE
    initial_idx = offset // (N_MEDIAL * N_FINAL)
    medial_idx = (offset // N_FINAL) % N_MEDIAL
    final_idx = offset % N_FINAL

    return (
        INITIALS[initial_idx],
        MEDIALS[medial_idx],
        FINALS[final_idx],  # "" if no final
    )


def hangul_to_ipa_phonemes(text: str) -> list[str]:
    """Convert (surface-form) Hangul text to an IPA phoneme list.

    Non-Hangul characters (spaces, punctuation, digits) are skipped silently.
    The input is expected to be the output of a Korean phonological rule
    applier such as g2pkk; this function does NOT apply phonological rules
    itself.
    """
    result: list[str] = []
    for ch in text:
        decomp = decompose_syllable(ch)
        if decomp is None:
            continue
        initial, medial, final = decomp
        result.extend(ONSET_IPA.get(initial, []))
        result.extend(NUCLEUS_IPA.get(medial, []))
        if final:
            result.extend(CODA_IPA.get(final, []))
    return result


__all__ = [
    "ONSET_IPA",
    "NUCLEUS_IPA",
    "CODA_IPA",
    "decompose_syllable",
    "hangul_to_ipa_phonemes",
]
