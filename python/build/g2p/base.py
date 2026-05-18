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
