"""Server behaviour that a browser pass cannot pin down precisely.

Identity, provenance and refusal paths: the places where the page could
show a confident answer that belongs to something else. The model is
loaded once for the whole module because these assert against the real
checkpoint, not a stub.

    .venv-gpu/Scripts/python.exe -m pytest \
        python/tools/child_jamo_web/test_routes.py -q
"""
from pathlib import Path
import io
import sys

import numpy as np
import pytest
import soundfile as sf

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(WEB.parents[0] / 'child_jamo_finetune'))
import engine as engine_module
import server as server_module
from fastapi.testclient import TestClient


@pytest.fixture(scope='module')
def client():
    with TestClient(server_module.app) as c:
        c.post('/api/warm').raise_for_status()
        yield c


def wav(seconds=1.0, rate=16000, channels=1, amplitude=0.2):
    n = int(seconds * rate)
    tone = (amplitude * np.sin(2 * np.pi * 220 * np.arange(n) / rate)).astype('float32')
    data = tone if channels == 1 else np.stack([tone, tone], axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, data, rate, format='WAV', subtype='FLOAT')
    return buffer.getvalue()


# --- identity --------------------------------------------------------

def test_status_identifies_the_finetuned_checkpoint_not_just_the_base(client):
    model = client.get('/api/status').json()['model']
    assert model['ready']
    assert len(model['checkpoint_sha256']) == 64
    assert model['checkpoint'].endswith('best.pt')
    # The base encoder is identified separately, by all three files.
    assert set(model['encoder_files_sha256']) == {
        'model.safetensors', 'config.json', 'preprocessor_config.json'}
    assert model['checkpoint_sha256'] not in model['encoder_files_sha256'].values()


def test_cache_key_is_keyed_on_the_finetuned_weights(client):
    key = server_module.ENGINE.cache_key('abc')
    assert key.startswith(server_module.ENGINE.identity['checkpoint_sha256'][:16])
    assert 'abc' in key


def test_historical_score_is_only_shown_for_the_checkpoint_it_describes(client):
    model = client.get('/api/status').json()['model']
    assert model['historical_test'] is not None
    assert '일치' in model['historical_provenance']


def test_matmul_tf32_matches_the_training_policy(client):
    assert client.get('/api/status').json()['model']['matmul_allow_tf32'] is False


def test_instance_identity_is_exposed_for_ownership(client):
    instance = client.get('/api/status').json()['instance']
    assert instance['app'] == 'child-jamo-web'
    assert len(instance['token']) == 32 and instance['pid'] > 0


# --- audio provenance ------------------------------------------------

def test_read_row_rejects_audio_that_no_longer_matches_the_audit():
    """A same-length replacement must not be served with the old text."""
    row = dict(server_module.CATALOG.rows['test'][0])
    engine_module.read_row(row)                       # the real one is fine
    row['pcm_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='audited recording'):
        engine_module.read_row(row)


# --- routes ----------------------------------------------------------

def test_unknown_split_is_a_deliberate_404_everywhere(client):
    assert client.get('/api/samples?split=nope').status_code == 404
    assert client.get('/api/random?split=nope').status_code == 404
    assert client.get('/api/batch-scope?split=nope').status_code == 404


def test_unknown_sample_is_404_not_a_crash(client):
    assert client.get('/api/sample/test:missing.wav').status_code == 404
    assert client.get('/api/audio/test:missing.wav').status_code == 404
    assert client.post('/api/transcribe',
                       json={'sample_id': 'nope:x'}).status_code == 404


def test_batch_scope_follows_the_selection_not_the_page(client):
    items = client.get('/api/samples?split=test&size=24').json()['items']
    start = items[5]['id']
    scope = client.get(f'/api/batch-scope?split=test&sample_id={start}&count=10').json()
    assert scope['count'] == 10
    assert scope['items'][0]['id'] == start       # begins where we are


def test_batch_scope_respects_the_active_filter(client):
    scope = client.get('/api/batch-scope?split=test&age=5&count=10').json()
    assert scope['count'] and all(i['age'] == '5' for i in scope['items'])


def test_all_failed_batch_reports_failures_and_no_invented_rate(client):
    out = client.post('/api/batch',
                      json={'sample_ids': ['test:nope-a.wav', 'test:nope-b.wav']})
    body = out.json()
    assert body['all_failed'] and len(body['failures']) == 2
    # None, not 0.0 - nothing was scored.
    assert body['aggregate']['jer'] is None


# --- uploads ---------------------------------------------------------

def test_upload_serves_back_exactly_what_it_transcribed(client):
    out = client.post('/api/upload', files={'file': ('probe.wav', wav(), 'audio/wav')})
    assert out.status_code == 200, out.text
    body = out.json()
    assert body['served_audio_is_model_input']
    served = client.get(body['audio_url'])
    assert served.status_code == 200
    data, rate = sf.read(io.BytesIO(served.content), dtype='float32')
    assert rate == 16000
    # The served bytes decode to the array the prediction was made from.
    assert len(data) == body['prediction']['samples']
    assert engine_module.pcm_sha256(data) == \
        engine_module.pcm_sha256(engine_module.canonical_upload(wav())[0])


def test_upload_handles_non_16k_stereo_float_input(client):
    out = client.post('/api/upload',
                      files={'file': ('probe.wav',
                                      wav(seconds=1.0, rate=44100, channels=2),
                                      'audio/wav')})
    assert out.status_code == 200, out.text
    body = out.json()
    assert body['original_sample_rate'] == 44100
    assert abs(body['seconds'] - 1.0) < 0.05


def test_out_of_range_float_is_resolved_once_before_both_paths(client):
    """Clipping happens in the canonical waveform, not twice, differently."""
    loud = wav(amplitude=2.0)
    final, encoded, _rate = engine_module.canonical_upload(loud)
    again, _ = sf.read(io.BytesIO(encoded), dtype='float32')
    assert np.array_equal(final, again)
    assert float(np.abs(final).max()) <= 1.0


def test_long_upload_is_refused_from_the_header_before_decoding(client):
    out = client.post('/api/upload',
                      files={'file': ('long.wav', wav(seconds=31.0), 'audio/wav')})
    assert out.status_code == 413
    assert '30초' in out.json()['detail']


def test_too_short_upload_is_refused_with_a_reason(client):
    out = client.post('/api/upload',
                      files={'file': ('tiny.wav', wav(seconds=0.02), 'audio/wav')})
    assert out.status_code == 400
    assert '짧습니다' in out.json()['detail']


def test_unsupported_extension_is_refused(client):
    out = client.post('/api/upload',
                      files={'file': ('a.mp3', b'not audio', 'audio/mpeg')})
    assert out.status_code == 400 and '형식만' in out.json()['detail']


def test_upload_audio_token_is_not_a_path(client):
    assert client.get('/api/upload-audio/..%2F..%2Fsecret').status_code == 404
    assert client.get('/api/upload-audio/short').status_code == 404
