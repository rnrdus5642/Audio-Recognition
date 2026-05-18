"""Base interface for runtime audio -> IPA recognizers.

Each language implementation wraps:
  * an acoustic model (audio -> orthography or audio -> phonemes directly)
  * (if needed) a language-specific G2P that converts orthographic output
    to the same IPA representation used by the build pipeline.

The runtime matching engine consumes only the IPA phoneme list returned
by `recognize`; it does not care which model produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseRecognizer(ABC):
    """Abstract base for language-specific phoneme recognizers."""

    @property
    @abstractmethod
    def language(self) -> str:
        """ISO language code (e.g., 'ko')."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Human-readable identifier, used in evaluation logs."""
        return self.__class__.__name__

    @abstractmethod
    def recognize(self, audio_16k_mono: np.ndarray) -> list[str]:
        """Return an IPA phoneme list for the given audio.

        Args:
            audio_16k_mono: 1-D float32 numpy array, mono, sampled at 16kHz,
                normalized to roughly [-1.0, 1.0].

        Returns:
            List of IPA phoneme strings. Empty list if no speech detected.
        """
        raise NotImplementedError
