"""Deterministic training-only acoustic variations with unchanged CTC targets.

No inference audio is transformed. The original manifest and its hashes
remain the source of truth; callers verify the original before augmentation.
Half the examples remain unchanged. A per-recording key makes results
independent of prefetch timing and the position from which training resumes.
"""
from fractions import Fraction
import hashlib

import numpy as np
from scipy.signal import resample_poly

VERSION = 'child-jamo-acoustic-v1'
MODES = ('identity',) * 5 + ('rate_down_2', 'rate_up_2', 'pitch_down_2',
                           'pitch_up_2', 'tempo_0.65')
MAX_LENGTH_FACTOR = 1 / .65


def mode_for(wav, seed, epoch):
    key = f'{VERSION}:{seed}:{epoch}:{wav}'.encode('utf-8')
    return MODES[int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % len(MODES)]


def frames(samples):
    """Output length of the encoder's audited seven convolutional layers."""
    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        samples = (samples - kernel) // stride + 1
    return max(0, samples)


def minimum_ctc_frames(targets):
    return len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))


def apply(wave, mode, *, target_ids=()):
    original = np.ascontiguousarray(wave, dtype=np.float32)
    if original.ndim != 1 or not len(original) or not np.isfinite(original).all():
        raise ValueError('augmentation requires a nonempty finite mono waveform')
    if mode not in MODES:
        raise ValueError(f'unknown training augmentation: {mode}')
    if mode == 'identity':
        changed = original.copy()
    elif mode.startswith('rate_'):
        semitones = -2 if mode == 'rate_down_2' else 2
        ratio = Fraction(2 ** (-semitones / 12)).limit_denominator(1024)
        changed = resample_poly(original, ratio.numerator, ratio.denominator)
    else:
        import librosa
        if mode == 'tempo_0.65':
            changed = librosa.effects.time_stretch(original, rate=.65, n_fft=1024, hop_length=256)
        else:
            steps = -2 if mode == 'pitch_down_2' else 2
            changed = librosa.effects.pitch_shift(original, sr=16000, n_steps=steps,
                                                  n_fft=1024, hop_length=256, scale=False)
    changed = np.ascontiguousarray(changed, dtype=np.float32)
    if not np.isfinite(changed).all():
        raise ValueError('non-finite augmented waveform')
    meta = {'requested': mode, 'applied': mode, 'attenuation': 1.0,
            'original_samples': len(original)}
    if frames(len(changed)) < minimum_ctc_frames(target_ids):
        # Keeping the original avoids silent CTC-infinite examples when
        # speeding up an already short utterance. No target is shortened.
        changed = original.copy()
        meta.update(applied='identity', skipped='insufficient_ctc_frames')
    if meta['applied'] != 'identity':
        peak = float(np.max(np.abs(changed)))
        attenuation = min(1.0, .98 / peak) if peak else 1.0
        changed *= np.float32(attenuation)
        meta['attenuation'] = attenuation
    meta['samples'] = len(changed)
    return changed, meta
