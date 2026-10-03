"""Acoustic review signals, never automatic speech or transcript verdicts."""
from pathlib import Path

import numpy as np

from common import digest

VERSION = 'child-jamo-quality-2026-09-27'
LIMITATION = ('Energy is not a speech detector. Quiet speech and background '
              'events can trigger these flags. Review candidates must not be '
              'automatically deleted, relabelled, amplified or recropped.')


def contract_sha256():
    return digest(Path(__file__))


def _dbfs(value):
    return float(20 * np.log10(max(float(value), 1e-12)))


def _stats(wave):
    if not len(wave):
        return dict(rms_dbfs=-240.0, peak_frame_dbfs=-240.0, energy=0.0)
    power = np.square(wave.astype(np.float64))
    energy = float(power.sum())
    width = (len(power) + 319) // 320
    frames = np.pad(power, (0, width * 320 - len(power))).reshape(-1, 320)
    return dict(rms_dbfs=_dbfs(np.sqrt(energy / len(power))),
                peak_frame_dbfs=_dbfs(np.sqrt(frames.mean(axis=1).max())),
                energy=energy)


def measure(full, cropped, reference_jamo):
    """Inspect the exact 16 kHz input and what the crop left outside it."""
    source, clip = _stats(full), _stats(cropped)
    share = clip['energy'] / source['energy'] if source['energy'] else 0.0
    gap = source['peak_frame_dbfs'] - clip['peak_frame_dbfs']
    flags = []
    if clip['rms_dbfs'] < -50:
        flags.append('low_crop_rms_below_minus50_dbfs')
    if source['rms_dbfs'] < -50:
        flags.append('low_source_rms_below_minus50_dbfs')
    if (len(full) - len(cropped) >= 1600 and share < .05 and
            clip['rms_dbfs'] < -35 and source['peak_frame_dbfs'] > -25 and gap > 15):
        flags.append('crop_excludes_most_strong_audio')
    if len(cropped) < 24000 and reference_jamo >= 20:
        flags.append('short_input_with_long_reference')
    return dict(full=source, crop=clip, crop_energy_share=share,
                peak_frame_gap_db=gap, flags=flags)
