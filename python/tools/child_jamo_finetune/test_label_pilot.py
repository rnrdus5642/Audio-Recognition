"""Guard human supervision changes, exact playback identity and pronunciation scoring."""
from contextlib import closing
from pathlib import Path
import hashlib
import sqlite3
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_pilot as pilot


def test_review_audio_rejects_same_duration_different_sound(tmp_path):
    path = tmp_path / 'heard.wav'
    sf.write(path, .1 * np.sin(np.arange(8000) * .02), 16000, subtype='PCM_16')
    wave, _ = sf.read(path, dtype='float32')
    row = {'id': 'review:1', 'path': str(path), 'file_sha256': pilot.digest(path),
           'samples': len(wave), 'pcm_sha256': hashlib.sha256(wave.tobytes()).hexdigest()}
    assert np.array_equal(pilot.reviewed_wave(row), wave)
    with pytest.raises(RuntimeError, match='PCM changed'):
        pilot.reviewed_wave(dict(row, pcm_sha256='wrong'))
    sf.write(path, .1 * np.sin(np.arange(8000) * .04), 16000, subtype='PCM_16')
    with pytest.raises(RuntimeError, match='audio file changed'):
        pilot.reviewed_wave(row)


def test_new_ambiguous_answer_invalidates_frozen_exclusive_supervision(tmp_path):
    database = tmp_path / 'listening-answers.sqlite3'
    with closing(sqlite3.connect(database)) as db:
        db.execute('CREATE TABLE answers(revision INTEGER PRIMARY KEY, item_id TEXT, choice TEXT)')
        db.execute("INSERT INTO answers VALUES (1,'1','a')")
        db.commit()
        db.row_factory = sqlite3.Row
        history = [dict(row) for row in db.execute('SELECT * FROM answers ORDER BY revision')]
    report, probe = tmp_path / 'review.json', tmp_path / 'probe.json'
    report.write_text('{}', encoding='utf-8')
    probe.write_text('{}', encoding='utf-8')
    selection = {'reviews': [{'directory': str(tmp_path), 'history': history,
                              'report_path': str(report), 'report_sha256': pilot.digest(report),
                              'probe_path': str(probe), 'probe_sha256': pilot.digest(probe)}]}
    pilot.check_answers(selection)
    with closing(sqlite3.connect(database)) as db:
        db.execute("INSERT INTO answers VALUES (2,'1','both')")
        db.commit()
    with pytest.raises(RuntimeError, match='Human answers changed'):
        pilot.check_answers(selection)


def test_scoring_preserves_ssaek_chaek_and_ignores_only_word_boundaries():
    ids = lambda word: [pilot.labels.VOCAB[j] for j in pilot.labels.jamo(word)]
    assert not pilot.scored(ids('책'), ids('쌕'))['exact']
    assert pilot.scored(ids('쌕'), ids('쌕'))['exact']
    assert pilot.scored(ids('뛰어 가요'), ids('뛰어가요'))['exact']
    assert not pilot.scored(ids('귀어가요'), ids('뛰어가요'))['exact']
