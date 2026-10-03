"""Protect paired exposure, source identity and true lengths in the experiment."""
from pathlib import Path
import hashlib
import io
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robustness as r
import common


def test_selection_is_order_independent_and_excludes_review_examples():
    rows = [dict(wav=f'{i}.wav', surface='두 단어' if i % 3 == 0 else f'말{i % 7}') for i in range(100)]
    excluded = {'3.wav', '6.wav', '15.wav'}
    a, balance = r.select_training(rows, 40, 927, excluded)
    b, _ = r.select_training(list(reversed(rows)), 40, 927, excluded)
    assert a == b and balance == 20
    assert len({row['wav'] for row in a}) == 40
    assert not excluded & {row['wav'] for row in a}
    assert all(len(row['surface'].split()) == 1 for row in a[:balance])


def test_packing_preserves_targets_and_masks_real_augmented_lengths():
    rows = [{'label_ids': [1, 2, 2]}, {'label_ids': [3]}]
    waves = [np.ones(5000, dtype=np.float32), np.full(2400, .5, dtype=np.float32)]
    batch = r.pack(waves, rows, 'cpu')
    assert batch['attention_mask'].sum(-1).tolist() == [5000, 2400]
    assert batch['targets'].tolist() == [1, 2, 2, 3]
    assert batch['target_lengths'].tolist() == [3, 1]
    assert batch['input_values'][1, 2400:].count_nonzero() == 0


def test_source_validation_precedes_training_transform(monkeypatch):
    buffer = io.BytesIO()
    sf.write(buffer, np.sin(np.arange(16000) * .02).astype(np.float32) * .1,
             16000, format='WAV', subtype='PCM_16')
    blob = buffer.getvalue()
    wave, policy = r.audio.crop(r.audio.decode(blob), '0', '1')
    row = dict(archive='test.tar', offset=0, size=len(blob), wav='test.wav',
               source_sha256=hashlib.sha256(blob).hexdigest(),
               pcm_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
               speech_start='0', speech_end='1', samples=len(wave), span_policy=policy)
    monkeypatch.setattr(r.audio, 'payload', lambda *args: blob)
    assert np.array_equal(r.verify_read(row), wave)
    with pytest.raises(RuntimeError, match='Audited crop changed'):
        r.verify_read(dict(row, pcm_sha256='incorrect'))
    with pytest.raises(RuntimeError, match='Source changed'):
        r.verify_read(dict(row, source_sha256='incorrect'))


def test_status_replace_retries_reader_lock_without_partial_json(tmp_path, monkeypatch):
    path = tmp_path / 'progress.json'
    common.dump(path, {'update': 32})
    replace = common.os.replace
    calls = []
    def briefly_locked(source, target):
        calls.append(target)
        assert common.load(path) == {'update': 32}
        if len(calls) < 3:
            raise PermissionError('temporary Windows reader handle')
        return replace(source, target)
    monkeypatch.setattr(common.os, 'replace', briefly_locked)
    monkeypatch.setattr(common.time, 'sleep', lambda _: None)
    common.dump(path, {'update': 64})
    assert len(calls) == 3 and common.load(path) == {'update': 64}


def test_persistent_status_permission_error_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / 'progress.json'
    common.dump(path, {'update': 32})
    calls = []
    def denied(*_args):
        calls.append(1)
        raise PermissionError('persistent')
    monkeypatch.setattr(common.os, 'replace', denied)
    monkeypatch.setattr(common.time, 'sleep', lambda _: None)
    with pytest.raises(PermissionError, match='persistent'):
        common.dump(path, {'update': 64})
    assert len(calls) == 40 and common.load(path) == {'update': 32}
