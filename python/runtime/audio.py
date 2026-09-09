"""Audio I/O for the runtime pipeline.

Single source of truth for "what the recognizer expects as input":
16 kHz mono float32 in the [-1, 1] range. All callers (web UI, CLI,
batch evaluation) must go through `load_audio_16k_mono`.
"""

from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16000


class HopWindow:
    """One window and the sample it ended on."""

    __slots__ = ("end_sample", "samples")

    def __init__(self, end_sample: int, samples: np.ndarray) -> None:
        self.end_sample = end_sample
        self.samples = samples

    def end_seconds(self, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
        """Where this window ends in the recording, not how long the
        caller has been running."""
        return self.end_sample / sample_rate


class HopFramer:
    """Cut arriving audio into windows on exact hop boundaries.

    `rolling_windows` does this for a finished recording; this does it
    for one that is still arriving, a chunk at a time, and the two agree
    on where the boundaries fall.

    A frame is `hop` new samples - never "one callback". The live web UI
    used to re-recognise once per browser chunk, so the same second of
    speech was judged twice when chunks arrived at 8000 samples and four
    times at 4000: the buffering decided how many chances a child got and
    when the answer confirmed. Counting samples instead makes split and
    merged input produce the same windows, in the same order, ending on
    the same sample.

    Windows are left-padded with silence until enough audio has arrived,
    which is what Unity's `AudioWindowBuffer.Snapshot()` hands the model:
    a statically exported graph accepts exactly one input length.

    Feed it 16 kHz mono float32. The guarantee is about identical
    normalised PCM - resampling anything else is a separate stage with
    its own behaviour, unchanged by this.
    """

    def __init__(self, window_samples: int, hop_samples: int) -> None:
        if window_samples < 1:
            raise ValueError(f"window_samples must be >= 1, got {window_samples}")
        if hop_samples < 1:
            raise ValueError(f"hop_samples must be >= 1, got {hop_samples}")
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self.reset()

    def reset(self) -> None:
        """Forget the audio, the queue and the boundary phase.

        A leftover part-hop would shift every boundary of the next
        question, so it goes too.
        """
        self._window = np.zeros(self.window_samples, dtype=np.float32)
        self.total_samples = 0
        self._next_boundary = self.hop_samples

    def snapshot(self) -> np.ndarray:
        """The window as it stands, including a part-finished hop."""
        return self._window.copy()

    def feed(self, chunk) -> list[HopWindow]:
        """Take some samples; return the windows they completed.

        Empty input completes nothing. One large chunk completes several
        windows, each cut where its own boundary falls rather than all of
        them showing the newest audio - otherwise merging chunks would
        change what was scored.
        """
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        frames: list[HopWindow] = []
        offset = 0
        while offset < len(audio):
            room = min(len(audio) - offset,
                       self._next_boundary - self.total_samples)
            segment = audio[offset:offset + room]
            self._window = np.concatenate(
                [self._window, segment])[-self.window_samples:]
            offset += room
            self.total_samples += room
            if self.total_samples >= self._next_boundary:
                frames.append(
                    HopWindow(self.total_samples, self._window.copy()))
                self._next_boundary += self.hop_samples
        return frames


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
