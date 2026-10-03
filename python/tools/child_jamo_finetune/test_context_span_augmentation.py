import numpy as np

import context_augmentation as original
import context_span_augmentation as span


def test_removes_only_outer_zeros_retains_guard_phase_and_all_signal():
    speech = np.array([.01, 0., -.4, 0., .2] * 1000, dtype=np.float32)
    wave = np.pad(speech, (4801, 9603))
    lo, hi = span.bounds(wave)
    assert lo % 320 == 0 and (len(wave) - hi) % 320 == 0
    assert 1280 <= 4801 - lo < 1600
    assert 1280 <= hi - (4801 + len(speech)) < 1600
    y, meta = span.apply(wave, span.TrainingMargin(640, 5, 'short'), 320, -6.)
    offset = 640 + 4801 - lo
    np.testing.assert_array_equal(y[offset:offset + len(speech)], speech * np.float32(meta['applied_gain']))
    assert meta['nonzero_samples_removed'] == 0
    assert not meta['ctc_length_fallback']


def test_too_short_ctc_span_keeps_original_instead_of_dropping_labels():
    wave = np.zeros(40000, dtype=np.float32)
    wave[20000:20010] = .1
    y, meta = span.apply(wave, span.TrainingMargin(0, 50, 'none'))
    assert meta['ctc_length_fallback']
    np.testing.assert_array_equal(y, wave)


def test_primary_evaluation_remains_byte_identical():
    wave = np.pad(np.linspace(-.7, .7, 5000, dtype=np.float32), (5000, 3000))
    for parameters in original.EVAL.values():
        expected, _ = original.apply(wave, *parameters)
        actual, _ = span.apply(wave, *parameters)
        np.testing.assert_array_equal(actual, expected)


def test_no_nonzero_boundary_samples_are_removed():
    wave = np.array([.1, 0., .2, 0., -.1], dtype=np.float32)
    assert span.bounds(wave) == (0, len(wave))
