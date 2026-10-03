import io
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mic_server as server


class Model:
    ready = True
    error = None
    identity = {'checkpoint_sha256': server.EXPECTED_CHECKPOINT}

    def __init__(self):
        self.inputs = []

    def status(self):
        return {'ready': self.ready, 'error': self.error, **self.identity}

    def transcribe(self, wave, use_cache):
        assert not use_cache
        self.inputs.append(wave.copy())
        return {'text': 'ㅐㄱ', 'jamo': 'ㅐㄱ', 'token_ids': [22, 2]}


@pytest.fixture
def client(monkeypatch, tmp_path):
    model = Model()
    monkeypatch.setattr(server, 'ENGINE', model)
    monkeypatch.setattr(server, 'RECORDINGS', tmp_path)
    # No lifespan: route tests must not allocate a second real GPU model.
    return TestClient(server.app), model


def wav(samples, rate=16000):
    stream = io.BytesIO()
    sf.write(stream, samples, rate, format='WAV', subtype='PCM_16')
    return stream.getvalue()


def test_replay_matches_model_input_and_orphan_jamo_is_not_corrected(client):
    browser, model = client
    original = np.pad(np.linspace(-.3, .3, 1600, dtype=np.float32), (3200, 6400))
    response = browser.post('/api/transcribe', files={'file': ('mic.wav', wav(original), 'audio/wav')})
    assert response.status_code == 200
    record = response.json()
    assert record['prediction']['text'] == 'ㅐㄱ'
    assert record['checkpoint_sha256'] == server.EXPECTED_CHECKPOINT
    replay, rate = sf.read(io.BytesIO(browser.get(record['audio_url']).content), dtype='float32')
    assert rate == 16000 and len(replay) == len(original)
    np.testing.assert_array_equal(replay, model.inputs[0])
    assert not replay[:3200].any() and not replay[-6400:].any()
    assert browser.get('/api/history').json()['items'] == [record]


def test_browser_sample_rate_is_resampled_without_changing_duration(client):
    browser, model = client
    wave = np.sin(np.arange(48000) * .05).astype('float32') * .2
    result = browser.post('/api/transcribe', files={'file': ('mic.wav', wav(wave, 48000))}).json()
    assert result['source_sample_rate'] == 48000
    assert result['seconds'] == 1. and len(model.inputs[0]) == 16000


@pytest.mark.parametrize('payload,status', [(b'not audio', 400), (wav(np.zeros(16000 * 16)), 413)], ids=['invalid_header', 'over_duration'])
def test_invalid_or_long_audio_is_rejected_before_inference(client, payload, status):
    browser, model = client
    assert browser.post('/api/transcribe', files={'file': ('mic.wav', payload)}).status_code == status
    assert not model.inputs


def test_different_checkpoint_is_refused(client):
    browser, model = client
    model.identity = {'checkpoint_sha256': 'another-model'}
    assert browser.post('/api/transcribe', files={'file': ('mic.wav', b'irrelevant')}).status_code == 503
    assert not model.inputs


def test_page_is_html_and_foreign_origin_cannot_record(client):
    browser, _ = client
    page = browser.get('/')
    assert page.status_code == 200 and page.headers['content-type'].startswith('text/html')
    assert '녹음 시작' in page.text
    response = browser.post('/api/transcribe', headers={'Origin': 'https://unrelated.example'}, files={'file': ('mic.wav', b'bad')})
    assert response.status_code == 403
    assert browser.get('/api/audio/invalid').status_code == 404
