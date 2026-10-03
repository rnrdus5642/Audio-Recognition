"""Reproducible, content-preserving zero padding and scalar gain changes."""
import hashlib
import math

import numpy as np

VERSION = 'child-jamo-context-v1'
RATE = 16000
MAX_SIDE_SAMPLES = 19200
# Evaluation offsets differ from the sampled training grid.
EVAL = {
    'original': (0, 0, 0.0),
    'left_370ms': (5920, 0, 0.0),
    'right_830ms': (0, 13280, 0.0),
    'both_310_670ms': (4960, 10720, 0.0),
    'gain_minus12': (0, 0, -12.0),
    'gain_plus6': (0, 0, 6.0),
    'both_gain_minus9': (4960, 10720, -9.0),
}


def parameters(wav, seed, epoch, policy='combined'):
    """Half original; remaining half balanced among padding, gain, both."""
    raw = hashlib.sha256(f'{VERSION}:{seed}:{epoch}:{wav}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(raw[:8], 'big'))
    choice = int(rng.integers(0, 6))
    if policy == 'control' or choice < 3:
        return 0, 0, 0.0, 'original'
    if policy not in ('combined', 'padding', 'gain'):
        raise ValueError(policy)
    use_pad = policy == 'padding' or (policy == 'combined' and choice in (3, 5))
    use_gain = policy == 'gain' or (policy == 'combined' and choice in (4, 5))
    # A 20-ms grid preserves the original convolution phase in training.
    left = int(rng.integers(0, 61)) * 320 if use_pad else 0
    right = int(rng.integers(0, 61)) * 320 if use_pad else 0
    db = float(rng.uniform(-15.0, 6.0)) if use_gain else 0.0
    name = 'both' if use_pad and use_gain else 'padding' if use_pad else 'gain'
    return left, right, db, name


def apply(wave, left=0, right=0, db=0.0):
    x = np.ascontiguousarray(wave, dtype=np.float32)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty mono waveform')
    if int(left) != left or int(right) != right or left < 0 or right < 0 or not math.isfinite(db):
        raise ValueError('Invalid padding or gain')
    peak = float(np.max(np.abs(x)))
    requested_gain = 10.0 ** (db / 20.0)
    gain = requested_gain
    # Never clip or compress; cap positive amplification with one global scalar.
    if db > 0 and peak:
        gain = min(requested_gain, max(1.0, .98 / peak))
    scaled = x.copy() if gain == 1.0 else x * np.float32(gain)
    y = np.pad(scaled, (int(left), int(right)), mode='constant')
    assert np.isfinite(y).all()
    return np.ascontiguousarray(y), {
        'left_samples': int(left), 'right_samples': int(right),
        'requested_db': float(db), 'applied_gain': gain,
        'applied_db': 20 * math.log10(gain), 'gain_capped': gain < requested_gain,
        'original_samples': len(x), 'samples': len(y),
        'speech_samples_removed': 0,
    }
