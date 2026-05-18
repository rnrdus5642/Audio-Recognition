"""Grapheme-to-Phoneme registry.

Adding a new language is a 3-step change:

    1. Create a folder `python/build/g2p/<lang>/` containing your
       implementation that subclasses `BaseG2P`.
    2. Import the class here.
    3. Add one entry to `G2P_REGISTRY`.

Nothing else in the codebase needs to change — the build pipeline,
matching engine, web UI and tests all consume languages through this
registry.
"""

from __future__ import annotations

from .base import BaseG2P
from .ko import KoreanG2P

G2P_REGISTRY: dict[str, type[BaseG2P]] = {
    "ko": KoreanG2P,
    # 영어 추가 예시:
    # "en": EnglishG2P,
    # 일본어 추가 예시:
    # "ja": JapaneseG2P,
}


def get_g2p(language: str) -> BaseG2P:
    """Instantiate the G2P implementation registered for `language`.

    Raises KeyError when the language has no registered implementation.
    """
    if language not in G2P_REGISTRY:
        raise KeyError(
            f"No G2P registered for language {language!r}. "
            f"Available: {available_languages()}"
        )
    return G2P_REGISTRY[language]()


def available_languages() -> list[str]:
    """List ISO codes for which a G2P is registered, sorted."""
    return sorted(G2P_REGISTRY)


__all__ = [
    "BaseG2P",
    "KoreanG2P",
    "G2P_REGISTRY",
    "get_g2p",
    "available_languages",
]
