"""Korean G2P implementation.

Pipeline (executed by `to_ipa`):
  1. `.rules` applies Korean phonological rules:
       자음동화, 비음화, 경음화, 구개음화, ㅎ 탈락, 연음, 자음군 단순화 ...
     Input is orthographic Hangul; output is surface-form Hangul.
  2. Each surface-form Hangul syllable is decomposed into
     (initial, medial, final) compatibility-jamo using the standard
     Unicode Hangul algorithm.
  3. Each jamo is mapped to one or more IPA phonemes using the project's
     canonical mapping table (`.jamo_ipa`).

Both steps run at runtime too, on the recogniser's output, and both have
C# counterparts the device runs (`Korean.Rules`, `Korean.JamoIpa`).
Answers and user speech therefore reach IPA by the same route, which is
what the matching assumes. They did not always: the rules used to be
build-time only, so a perfectly spoken 쳐 arrived as 쳐 and was compared
against 처.
"""

from __future__ import annotations

from ..base import BaseG2P
from . import rules
from .jamo_ipa import hangul_to_ipa_phonemes


class KoreanG2P(BaseG2P):
    """Korean text -> IPA phoneme list.

    Lazy-initialises the g2pkk model on first use so that simply importing
    the class does not download NLTK data or build the mecab wrapper.
    """

    def __init__(self) -> None:
        self._g2p = None  # lazy init

    @property
    def language(self) -> str:
        return "ko"

    def _ensure_g2p(self) -> None:
        if self._g2p is None:
            # Import inside the method so the package import is cheap even
            # when g2pkk is not strictly required (e.g., during pure unit
            # tests of the IPA mapping table).
            from g2pkk import G2p

            self._g2p = G2p()

    def apply_rules(self, text: str) -> str:
        """Return the surface-form Hangul (phonological rules applied).

        Uses the mecab-free implementation, which is the one C# can run.
        Answers and user speech both pass through here, so anything the
        device cannot reproduce would show up as a mismatch between two
        spellings of the same sound rather than as a missing rule.

        `g2pkk` itself is still available through `apply_rules_g2pkk` for
        comparing the two; it needs a 112 MB dictionary and, measured on
        the curriculum, changes nothing.
        """
        return rules.apply_rules(text.strip())

    def apply_rules_g2pkk(self, text: str) -> str:
        """The full g2pkk path, morphological analyser included."""
        self._ensure_g2p()
        return self._g2p(text.strip())

    def to_ipa(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        surface = self.apply_rules(text)
        return hangul_to_ipa_phonemes(surface)


__all__ = ["KoreanG2P"]
