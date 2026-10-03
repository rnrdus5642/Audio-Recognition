"""Korean recognizer: wav2vec2 ASR (Hangul output) -> g2pkk -> IPA.

This path leverages a Korean-specialised ASR model for acoustic accuracy,
then re-uses the SAME g2pkk + jamo->IPA pipeline as the build script so
that target and user IPA sequences are guaranteed to be drawn from the
same phoneme inventory.

We deliberately use raw CTC argmax (no LM, no beam search) to avoid the
STT-style hallucination where the model commits to a different, more
common Korean word when the user mispronounces.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from ..base import BaseRecognizer

# Default model: well-known, fine-tuned for Korean ASR on KsponSpeech / Zeroth
DEFAULT_MODEL = "kresnik/wav2vec2-large-xlsr-korean"

# Allowed character set in CTC output: Korean Hangul + space.
# Anything else (e.g., punctuation tokens emitted by some processors)
# is stripped before passing to g2pkk so the rule applier does not
# choke on stray symbols.
_HANGUL_RE = re.compile(r"[가-힣\s]")


class KoreanASRRecognizer(BaseRecognizer):
    """Wav2Vec2 (Korean) + g2pkk pipeline.

    Lazy-loaded; instantiation is cheap, the heavy HF download happens
    on first call to `recognize` (or `_load`).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._processor = None  # type: ignore[assignment]
        self._model = None      # type: ignore[assignment]
        self._g2p = None        # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def language(self) -> str:
        return "ko"

    @property
    def name(self) -> str:
        return f"KoreanASRRecognizer({self.model_name})"

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        from python.build.g2p.ko import KoreanG2P

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()
        self._g2p = KoreanG2P()
        # Warm up g2pkk so the first real call doesn't pay for it
        self._g2p.to_ipa("사과")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transcribe_hangul(self, audio_16k: np.ndarray) -> str:
        """Run only the ASR step. Useful for debugging the acoustic stage
        in isolation from the G2P step.
        """
        self._load()
        import torch

        if audio_16k.ndim != 1:
            raise ValueError(
                f"Expected 1-D mono audio, got shape {audio_16k.shape}"
            )

        inputs = self._processor(
            audio_16k,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self._device)

        with torch.no_grad():
            logits = self._model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        # Wav2Vec2Processor.batch_decode strips CTC blanks and collapses
        # repeats internally.
        text = self._processor.batch_decode(pred_ids)[0]
        return self._sanitize(text)

    def recognize(self, audio_16k_mono: np.ndarray) -> list[str]:
        return self.recognize_with_text(audio_16k_mono)[1]

    def recognize_with_text(
        self, audio_16k_mono: np.ndarray
    ) -> tuple[str, list[str]]:
        """Hangul + IPA from one acoustic pass (see `BaseRecognizer`)."""
        hangul = self.transcribe_hangul(audio_16k_mono)
        if not hangul.strip():
            return hangul, []
        return hangul, self._g2p.to_ipa(hangul)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(text: str) -> str:
        """Keep only Hangul characters and spaces; drop punctuation, latin
        letters, digits and other artefacts that occasionally appear in
        CTC output.
        """
        return "".join(_HANGUL_RE.findall(text)).strip()
