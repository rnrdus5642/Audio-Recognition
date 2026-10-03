"""Vary existing literal-zero margins as well as added silence and gain.

Only exactly-zero outer samples can be removed. Retain >=80 ms of available
zero margin on each side, remove multiples of 320 samples to preserve the
20-ms convolution phase, and keep every nonzero sample in its original order.
Inference/evaluation requests remain the primary experiment's transformations.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

import augmentation
import context_augmentation as core

VERSION = 'child-jamo-existing-zero-context-v1'
MAX_SIDE_SAMPLES = core.MAX_SIDE_SAMPLES
EVAL = core.EVAL
GUARD = 1280
_minimum_frames = {}


@dataclass(frozen=True)
class TrainingMargin:
    added_left: int
    minimum_frames: int
    padding_band: str


def configure(rows):
    global _minimum_frames
    _minimum_frames = {r['wav']: augmentation.minimum_ctc_frames(r['label_ids']) for r in rows}


def bounds(wave):
    x = np.asarray(wave)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty mono waveform')
    nonzero = np.flatnonzero(x)
    if not len(nonzero):
        return 0, len(x)
    leading = int(nonzero[0])
    trailing = len(x) - int(nonzero[-1]) - 1
    remove_left = max(0, (leading - GUARD) // 320 * 320)
    remove_right = max(0, (trailing - GUARD) // 320 * 320)
    return remove_left, len(x) - remove_right


def parameters(wav, seed, epoch, policy='combined'):
    left, right, db, mode = core.parameters(wav, seed, epoch, policy)
    if mode == 'original':
        return left, right, db, mode
    if wav not in _minimum_frames:
        raise RuntimeError('Configure the audited targets before training')
    raw = hashlib.sha256(f'{VERSION}:{seed}:{epoch}:{wav}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(raw[:8], 'big'))
    band = 'none'
    if mode in ('padding', 'both'):
        choice = int(rng.integers(0, 3))
        band = ('none', 'short', 'wide')[choice]
        if choice == 0:
            left = right = 0
        elif choice == 1:
            left, right = int(rng.integers(0, 7)) * 320, int(rng.integers(0, 7)) * 320
    return TrainingMargin(left, _minimum_frames[wav], band), right, db, 'core_' + mode


def apply(wave, left=0, right=0, db=0.0):
    if not isinstance(left, TrainingMargin):
        # The complete primary evaluation is unchanged, including raw inputs.
        return core.apply(wave, left, right, db)
    request = left
    x = np.ascontiguousarray(wave, dtype=np.float32)
    lo, hi = bounds(x)
    fallback = augmentation.frames(hi - lo + request.added_left + right) < request.minimum_frames
    if fallback:
        lo, hi = 0, len(x)
    assert not x[:lo].any() and not x[hi:].any()
    assert lo % 320 == 0 and (len(x) - hi) % 320 == 0
    changed, meta = core.apply(x[lo:hi], request.added_left, right, db)
    assert augmentation.frames(len(changed)) >= request.minimum_frames
    meta.update(original_samples=len(x), removed_left_zero_samples=lo,
                removed_right_zero_samples=len(x) - hi, nonzero_samples_removed=0,
                padding_band=request.padding_band, ctc_length_fallback=bool(fallback),
                training_context_policy=VERSION)
    return changed, meta
