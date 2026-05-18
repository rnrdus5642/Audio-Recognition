"""Unit tests for the Hangul -> IPA mapping table.

These tests do NOT require g2pkk; they only verify the deterministic
syllable decomposition and the IPA lookup. They run quickly and can be
used as the first sanity check after any mapping change.
"""

from __future__ import annotations

import pytest

from python.build.g2p.ko.jamo_ipa import (
    decompose_syllable,
    hangul_to_ipa_phonemes,
)


class TestDecomposeSyllable:
    def test_simple_syllable_with_final(self) -> None:
        assert decompose_syllable("강") == ("ㄱ", "ㅏ", "ㅇ")

    def test_simple_syllable_no_final(self) -> None:
        assert decompose_syllable("가") == ("ㄱ", "ㅏ", "")

    def test_with_diphthong(self) -> None:
        assert decompose_syllable("과") == ("ㄱ", "ㅘ", "")

    def test_with_complex_final(self) -> None:
        assert decompose_syllable("닭") == ("ㄷ", "ㅏ", "ㄺ")

    def test_silent_onset(self) -> None:
        assert decompose_syllable("아") == ("ㅇ", "ㅏ", "")

    def test_non_hangul_returns_none(self) -> None:
        assert decompose_syllable("a") is None
        assert decompose_syllable("?") is None
        assert decompose_syllable("1") is None

    def test_rejects_multi_char(self) -> None:
        with pytest.raises(ValueError):
            decompose_syllable("강아")


class TestHangulToIpaPhonemes:
    """These tests assume the INPUT is already the surface form
    (i.e., g2pkk has already applied phonological rules).
    """

    def test_simple_word_no_rules(self) -> None:
        # 사과 has no phonological rules to apply
        assert hangul_to_ipa_phonemes("사과") == ["s", "a", "k", "w", "a"]

    def test_word_with_final(self) -> None:
        # 강 -> [k, a, ŋ]
        assert hangul_to_ipa_phonemes("강") == ["k", "a", "ŋ"]

    def test_silent_onset_omitted(self) -> None:
        # 아 -> just [a]; the silent ㅇ contributes nothing
        assert hangul_to_ipa_phonemes("아") == ["a"]

    def test_diphthong_splits(self) -> None:
        # 야 -> [j, a]
        assert hangul_to_ipa_phonemes("야") == ["j", "a"]
        # 의 -> [ɰ, i]
        assert hangul_to_ipa_phonemes("의") == ["ɰ", "i"]

    def test_aspirated_and_tense(self) -> None:
        # 칸 -> [kʰ, a, n], 깐 -> [k͈, a, n]
        assert hangul_to_ipa_phonemes("칸") == ["kʰ", "a", "n"]
        assert hangul_to_ipa_phonemes("깐") == ["k͈", "a", "n"]

    def test_unreleased_coda_stops(self) -> None:
        # 학 -> [h, a, k̚]
        assert hangul_to_ipa_phonemes("학") == ["h", "a", "k̚"]
        # 압 -> [a, p̚]
        assert hangul_to_ipa_phonemes("압") == ["a", "p̚"]

    def test_skips_non_hangul(self) -> None:
        # Spaces, punctuation should be silently dropped
        assert hangul_to_ipa_phonemes("강 아") == ["k", "a", "ŋ", "a"]
        assert hangul_to_ipa_phonemes("사과!") == ["s", "a", "k", "w", "a"]

    def test_empty_input(self) -> None:
        assert hangul_to_ipa_phonemes("") == []
        assert hangul_to_ipa_phonemes("   ") == []

    def test_compound_word(self) -> None:
        # 엄마 -> [ʌ, m, m, a]
        # 엄 = (ㅇ silent, ㅓ, ㅁ) -> [ʌ, m]
        # 마 = (ㅁ, ㅏ, "")       -> [m, a]
        assert hangul_to_ipa_phonemes("엄마") == ["ʌ", "m", "m", "a"]
