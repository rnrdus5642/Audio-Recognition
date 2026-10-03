import numpy as np
import pytest

import context_augmentation as aug
import robustness


def test_padding_retains_every_original_sample_and_has_true_length_mask():
    wave = np.linspace(-.5, .5, 1000, dtype=np.float32)
    changed, meta = aug.apply(wave, 37, 91)
    assert np.array_equal(changed[37:1037], wave)
    assert not changed[:37].any() and not changed[1037:].any()
    batch = robustness.pack([changed, wave], [{'label_ids': [2]}, {'label_ids': [2]}], device='cpu')
    assert batch['attention_mask'][0].sum() == len(changed)
    assert batch['attention_mask'][1].sum() == len(wave)
    assert meta['speech_samples_removed'] == 0


def test_gain_does_not_clip_or_distort_relative_samples():
    wave = np.array([-.9, -.3, 0., .6, .9], dtype=np.float32)
    changed, meta = aug.apply(wave, db=6)
    assert meta['gain_capped']
    assert np.max(np.abs(changed)) <= .980001
    np.testing.assert_allclose(changed, wave * np.float32(meta['applied_gain']), rtol=0, atol=0)
    assert np.array_equal(wave, np.array([-.9, -.3, 0., .6, .9], dtype=np.float32))


def test_deterministic_per_recording_transform_and_full_policy_coverage():
    specs = [aug.parameters(str(i), 92801, 0) for i in range(1000)]
    assert specs == [aug.parameters(str(i), 92801, 0) for i in range(1000)]
    assert specs != [aug.parameters(str(i), 92801, 1) for i in range(1000)]
    assert {s[3] for s in specs} == {'original', 'padding', 'gain', 'both'}
    assert all(0 <= s[0] <= aug.MAX_SIDE_SAMPLES and 0 <= s[1] <= aug.MAX_SIDE_SAMPLES for s in specs)
    assert all(aug.parameters(str(i), 92801, 0, 'control') == (0, 0, 0., 'original') for i in range(1000))


@pytest.mark.parametrize('wave,kwargs', [([], {}), ([float('nan')], {}), ([1.], {'left': -1}), ([1.], {'db': float('inf')})])
def test_invalid_input_fails_without_silent_audio_repair(wave, kwargs):
    with pytest.raises(ValueError):
        aug.apply(wave, **kwargs)
