"""End-to-end tests for the Korean G2P pipeline.

Requires g2pkk + eunjeon to be installed. The tests verify that the
combination of g2pkk's phonological rules and our jamo->IPA mapping
produces the expected IPA sequence for words that exercise each major
Korean phonological rule.
"""

from __future__ import annotations

import pytest

from python.build.g2p import available_languages, get_g2p
from python.build.g2p.ko import KoreanG2P


@pytest.fixture(scope="module")
def g2p() -> KoreanG2P:
    return KoreanG2P()


class TestRegistry:
    def test_ko_is_available(self) -> None:
        assert "ko" in available_languages()

    def test_get_g2p_korean(self) -> None:
        g = get_g2p("ko")
        assert isinstance(g, KoreanG2P)
        assert g.language == "ko"

    def test_get_g2p_unknown_language(self) -> None:
        with pytest.raises(KeyError):
            get_g2p("xx")


class TestPhonologicalRules:
    """Each test verifies a specific Korean phonological rule.
    We check the surface-form (g2pkk output) separately from the IPA
    so that a failure tells us WHICH stage broke.
    """

    def test_no_rule_words(self, g2p: KoreanG2P) -> None:
        assert g2p.apply_rules("사과") == "사과"
        assert g2p.to_ipa("사과") == ["s", "a", "k", "w", "a"]

        assert g2p.apply_rules("나비") == "나비"
        assert g2p.to_ipa("나비") == ["n", "a", "p", "i"]

    def test_nasal_assimilation_b_to_m(self, g2p: KoreanG2P) -> None:
        # 앞문 -> 암문 (ㅂ→ㅁ before nasal)
        assert g2p.apply_rules("앞문") == "암문"
        assert g2p.to_ipa("앞문") == ["a", "m", "m", "u", "n"]

    def test_nasal_assimilation_g_to_ng(self, g2p: KoreanG2P) -> None:
        # 국물 -> 궁물 (ㄱ→ㅇ before nasal)
        assert g2p.apply_rules("국물") == "궁물"
        assert g2p.to_ipa("국물") == ["k", "u", "ŋ", "m", "u", "l"]

    def test_liquid_assimilation(self, g2p: KoreanG2P) -> None:
        # 신라 -> 실라 (ㄴ→ㄹ after/before ㄹ)
        assert g2p.apply_rules("신라") == "실라"
        assert g2p.to_ipa("신라") == ["s", "i", "l", "ɾ", "a"]

    def test_palatalization(self, g2p: KoreanG2P) -> None:
        # 같이 -> 가치 (ㅌ+ㅣ→ㅊ)
        assert g2p.apply_rules("같이") == "가치"
        assert g2p.to_ipa("같이") == ["k", "a", "tɕʰ", "i"]
        # 굳이 -> 구지 (ㄷ+ㅣ→ㅈ)
        assert g2p.apply_rules("굳이") == "구지"
        assert g2p.to_ipa("굳이") == ["k", "u", "tɕ", "i"]

    def test_h_deletion(self, g2p: KoreanG2P) -> None:
        # 좋아 -> 조아 (intervocalic ㅎ drops)
        assert g2p.apply_rules("좋아") == "조아"
        assert g2p.to_ipa("좋아") == ["tɕ", "o", "a"]

    def test_linking(self, g2p: KoreanG2P) -> None:
        # 옷이 -> 오시 (final ㅅ moves to next onset)
        assert g2p.apply_rules("옷이") == "오시"
        assert g2p.to_ipa("옷이") == ["o", "s", "i"]

    def test_tensification(self, g2p: KoreanG2P) -> None:
        # 학교 -> 학꾜 (after obstruent coda)
        assert g2p.apply_rules("학교") == "학꾜"
        assert g2p.to_ipa("학교") == ["h", "a", "k̚", "k͈", "j", "o"]


class TestEdgeCases:
    def test_empty(self, g2p: KoreanG2P) -> None:
        assert g2p.to_ipa("") == []
        assert g2p.to_ipa("   ") == []

    def test_whitespace_trimmed(self, g2p: KoreanG2P) -> None:
        assert g2p.to_ipa("  사과  ") == ["s", "a", "k", "w", "a"]


class TestChildVocabulary:
    """Words from the sample words.csv. Catches regressions in the
    real authoring data and serves as a quick sanity check that the
    target IPA sequences look reasonable.
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("엄마", ["ʌ", "m", "m", "a"]),
            ("아빠", ["a", "p͈", "a"]),
            ("강아지", ["k", "a", "ŋ", "a", "tɕ", "i"]),
            ("나비", ["n", "a", "p", "i"]),
            ("사과", ["s", "a", "k", "w", "a"]),
            ("우유", ["u", "j", "u"]),
            ("빵", ["p͈", "a", "ŋ"]),
            ("책", ["tɕʰ", "ɛ", "k̚"]),
        ],
    )
    def test_vocabulary(
        self, g2p: KoreanG2P, text: str, expected: list[str]
    ) -> None:
        assert g2p.to_ipa(text) == expected
