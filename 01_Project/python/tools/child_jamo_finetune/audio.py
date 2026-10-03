"""Read one selected recording straight out of its archive.

The archives stay exactly as the vendor shipped them: opened read-only,
seeked to the byte range the catalogue recorded, and closed. Handles are
cached per archive because training touches all eight repeatedly.
"""
from pathlib import Path
import io
import sys
import threading

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW

SAMPLE_RATE = 16000
TRIM_PAD = 0.25          # keep a little air around the annotated speech span
CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)

_handles = {}
_lock = threading.Lock()


def frames(samples):
    """Encoder output frames for a waveform of `samples` samples."""
    n = samples
    for k, s in zip(CONV_KERNEL, CONV_STRIDE):
        n = (n - k) // s + 1
        if n <= 0:
            return 0
    return n


def _handle(archive):
    with _lock:
        fh = _handles.get(archive)
        if fh is None:
            matches = [p for p in RAW.rglob(archive)]
            if len(matches) != 1:
                raise FileNotFoundError(f'{archive}: {len(matches)} matches')
            fh = _handles[archive] = matches[0].open('rb')
        return fh


def payload(archive, offset, size):
    """The exact WAV bytes of one member."""
    fh = _handle(archive)
    with _lock:
        fh.seek(offset)
        blob = fh.read(size)
    if len(blob) != size:
        raise IOError(f'{archive}@{offset}: read {len(blob)} of {size}')
    return blob


def decode(blob):
    """Mono float32 at 16 kHz, or a raised error describing why not."""
    x, rate = sf.read(io.BytesIO(blob), dtype='float32', always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(f'sample rate {rate}')
    if x.ndim == 2:
        x = x.mean(axis=1)
    if not np.isfinite(x).all():
        raise ValueError('non-finite samples')
    return np.ascontiguousarray(x, dtype=np.float32)


def span(start_field, end_field, total_seconds):
    """The crop policy, in one place, as (start, end) seconds.

    Selection, the audit and training all call this, so a recording can
    never be measured with one crop and trained with another. Malformed
    or missing `SpeechStart`/`SpeechEnd` is a recorded policy rather than
    an exception: the whole file is used and the caller is told, instead
    of one bad row taking down a pool or quietly changing the crop.
    """
    try:
        start, end = float(start_field), float(end_field)
    except (TypeError, ValueError):
        return 0.0, total_seconds, 'whole_file_unparsable_span'
    if not (0.0 <= start < end):
        return 0.0, total_seconds, 'whole_file_implausible_span'
    lo = max(0.0, start - TRIM_PAD)
    hi = min(total_seconds, end + TRIM_PAD)
    if hi <= lo:
        return 0.0, total_seconds, 'whole_file_empty_after_padding'
    return lo, hi, 'annotated_span'


def crop(x, start_field, end_field):
    """Apply `span` to a decoded waveform."""
    lo, hi, policy = span(start_field, end_field, len(x) / SAMPLE_RATE)
    a = max(0, round(lo * SAMPLE_RATE))
    b = min(len(x), round(hi * SAMPLE_RATE))
    return np.ascontiguousarray(x[a:b], dtype=np.float32), policy


def close():
    with _lock:
        for fh in _handles.values():
            fh.close()
        _handles.clear()
