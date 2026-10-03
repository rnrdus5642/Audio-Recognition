"""Signal, CTC and resume invariants for the training-only augmentation."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augmentation as a


@pytest.fixture
def tone():
    return (.3 * np.sin(2 * np.pi * 400 * np.arange(16000) / 16000)).astype(np.float32)


def dominant(wave):
    return np.argmax(np.abs(np.fft.rfft(wave * np.hanning(len(wave))))) * 16000 / len(wave)


def test_identity_is_bit_exact_and_does_not_alias(tone):
    result, meta = a.apply(tone, 'identity')
    assert np.array_equal(result, tone) and not np.shares_memory(result, tone)
    assert meta['attenuation'] == 1 and meta['applied'] == 'identity'


@pytest.mark.parametrize('mode,frequency_ratio,length_ratio', [
    ('rate_down_2', 2 ** (-2 / 12), 2 ** (2 / 12)),
    ('rate_up_2', 2 ** (2 / 12), 2 ** (-2 / 12)),
    ('pitch_down_2', 2 ** (-2 / 12), 1),
    ('pitch_up_2', 2 ** (2 / 12), 1),
    ('tempo_0.65', 1, 1 / .65),
])
def test_frequency_and_length_interventions(tone, mode, frequency_ratio, length_ratio):
    original = tone.copy()
    result, meta = a.apply(tone, mode)
    assert abs(len(result) - len(tone) * length_ratio) <= 2
    assert abs(dominant(result) - 400 * frequency_ratio) < 4
    assert np.array_equal(tone, original)
    assert np.isfinite(result).all() and np.abs(result).max() <= .980001


def test_per_record_determinism_ignores_other_calls(tone):
    expected = a.mode_for('record.wav', 917, 2)
    for i in range(50):
        a.mode_for(f'other{i}.wav', 4, 0)
    assert a.mode_for('record.wav', 917, 2) == expected
    assert {a.mode_for('record.wav', 917, e) for e in range(100)} == set(a.MODES)
    first, _ = a.apply(tone, expected)
    second, _ = a.apply(tone, expected)
    assert np.array_equal(first, second)


def test_too_short_for_ctc_keeps_original_and_targets():
    wave = np.sin(np.arange(3600) * .02).astype(np.float32) * .1
    targets = [1] * 5  # 9 frames including the required intervening blanks.
    before = targets.copy()
    assert a.frames(len(wave)) >= a.minimum_ctc_frames(targets)
    # Select a boundary where speed-up makes the target infeasible.
    assert a.frames(len(wave)) == 11
    result, meta = a.apply(wave, 'rate_up_2', target_ids=targets)
    # This boundary retains nine frames and is still valid.
    assert a.frames(len(result)) >= 9
    tighter = [1, 1, 1, 1, 2, 2]  # 10 frames.
    result, meta = a.apply(wave, 'rate_up_2', target_ids=tighter)
    assert meta['applied'] == 'identity' and meta['skipped'] == 'insufficient_ctc_frames'
    assert np.array_equal(result, wave) and targets == before


@pytest.mark.parametrize('wave,mode', [(np.array([]), 'identity'),
                                    (np.array([np.nan]), 'identity'),
                                    (np.zeros((2, 1600)), 'identity'),
                                    (np.zeros(1600), 'typo')])
def test_invalid_input_refused(wave, mode):
    with pytest.raises(ValueError):
        a.apply(wave, mode)
