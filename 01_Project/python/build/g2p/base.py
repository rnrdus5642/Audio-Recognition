"""Base interface for Grapheme-to-Phoneme (G2P) implementations.

Each language provides its own subclass. The output is a list of IPA phonemes
that the matching engine compares against the user's recognized phoneme sequence.

Build-time only: this module is not used at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseG2P(ABC):
    """Abstract base for language-specific G2P implementations."""

    @property
    @abstractmethod
    def language(self) -> str:
        """ISO language code (e.g., 'ko', 'en')."""
        raise NotImplementedError

    @abstractmethod
    def to_ipa(self, text: str) -> list[str]:
        """Convert orthographic text to an IPA phoneme sequence.

        Args:
            text: Orthographic input in the implementation's language
                (e.g., "사과" for Korean).

        Returns:
            List of IPA phoneme strings, one phoneme per element.
            Diphthongs are split into glide + vowel (e.g., "야" -> ["j", "a"]).
            Aspirated/tense consonants are kept as single units with combining
            diacritics (e.g., "ㅋ" -> "kʰ", "ㄲ" -> "k͈").
        """
        raise NotImplementedError

    def to_ipa_syllables(
        self, text: str
    ) -> list[tuple[str, list[str]]] | None:
        """`to_ipa`, grouped into the units it is pronounced in.

        Returns (spoken unit, its phonemes) pairs. Concatenating the
        phonemes must reproduce `to_ipa(text)` exactly - the grouping
        only says which phonemes came from where, so feedback can point
        at a syllable instead of at an IPA symbol.

        The unit is the *spoken* one, which need not be the written one:
        Korean 먹어요 is pronounced 머거요, and the ㄱ written at the end
        of 먹 is spoken at the start of 거.

        Returns None when the language cannot draw that boundary; callers
        must treat the grouping as optional.
        """
        return None
