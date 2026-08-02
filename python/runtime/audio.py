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


def rolling_windows(
    audio: np.ndarray,
    window_s: float = 2.5,
    hop_s: float = 0.4,
    sample_rate: int = TARGET_SAMPLE_RATE,
):
    """Yield `(end_time_s, window)` over the trailing `window_s` of audio.

    This is the input shape `StreamingMatcher` expects: each window is
    re-recognised in full rather than recognising isolated chunks and
    concatenating the phonemes. Measured on a 4.4 s utterance, 0.5 s
    chunks recognised independently lost a quarter of the phonemes and
    garbled whole words, while re-recognising the trailing window kept
    them intact - wav2vec2 needs the surrounding context.

    `window_s` also caps how much silence reaches the model: padding one
    clip out to 13.6 s made the recogniser emit 2 phonemes instead of 5.
    """
    hop = max(1, int(hop_s * sample_rate))
    window = max(1, int(window_s * sample_rate))
    for end in range(hop, len(audio) + hop, hop):
        end = min(end, len(audio))
        yield end / sample_rate, audio[max(0, end - window):end]
