"""Audio I/O for the runtime pipeline.

Single source of truth for "what the recognizer expects as input":
16 kHz mono float32 in the [-1, 1] range. All callers (web UI, CLI,
batch evaluation) must go through `load_audio_16k_mono`.
"""

from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16000


def load_audio_16k_mono(path: str | Path) -> np.ndarray:
    """Load an audio file, resample to 16 kHz mono, return float32 ndarray.

    Librosa handles every format we care about (WAV, MP3, M4A, FLAC, ...)
    and applies the high-quality `soxr_hq` resampler by default, which is
    what wav2vec2-family feature extractors expect.
    """
    import librosa

    audio, _ = librosa.load(
        str(path),
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )
    return audio.astype(np.float32)
