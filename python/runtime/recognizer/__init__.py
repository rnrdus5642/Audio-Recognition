"""Phoneme recognizer registry.

Adding a new language is a 3-step change:

    1. Create a folder `python/runtime/recognizer/<lang>/` containing your
       implementation that subclasses `BaseRecognizer`.
    2. Import the class here.
    3. Add one entry to `RECOGNIZER_REGISTRY`.

Each recognizer is lazy-instantiated (the heavy HuggingFace model only
loads on first `recognize` call), so importing this registry is cheap
even when many languages are registered.
"""

from __future__ import annotations

from .base import BaseRecognizer
from .ko import DEFAULT_MODEL as KOREAN_DEFAULT_MODEL
from .ko import KoreanASRRecognizer

RECOGNIZER_REGISTRY: dict[str, type[BaseRecognizer]] = {
    "ko": KoreanASRRecognizer,
    # 영어 추가 예시:
    # "en": EnglishASRRecognizer,
    # 일본어 추가 예시:
    # "ja": JapaneseASRRecognizer,
}


def get_recognizer(language: str, **kwargs) -> BaseRecognizer:
    """Instantiate the recognizer registered for `language`.

    Extra kwargs are forwarded to the recognizer's constructor (e.g.,
    a non-default model name, a torch device).
    """
    if language not in RECOGNIZER_REGISTRY:
        raise KeyError(
            f"No recognizer registered for language {language!r}. "
            f"Available: {available_languages()}"
        )
    return RECOGNIZER_REGISTRY[language](**kwargs)


def available_languages() -> list[str]:
    """List ISO codes for which a recognizer is registered, sorted."""
    return sorted(RECOGNIZER_REGISTRY)


__all__ = [
    "BaseRecognizer",
    "KoreanASRRecognizer",
    "KOREAN_DEFAULT_MODEL",
    "RECOGNIZER_REGISTRY",
    "get_recognizer",
    "available_languages",
]
