"""Check live preview persistence and retry safety without loading model weights."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

import server


class FakeEngine:
    def __init__(self, *args):
        self.ready, self.error, self.device = True, None, 'cpu'
        self.release = {'epoch_equivalents': 2.01}
        self.calls, self.blocked = [], False
        self.text = '책'

    def load(self):
        pass

    def status(self):
        return {'model_id': 'test-model', 'ready': self.ready}

    def transcribe(self, waveform):
        if self.blocked:
            raise BlockingIOError('busy')
        self.calls.append(waveform.copy())
        return {'text': self.text, 'jamo': '책', 'audio_seconds': len(waveform) / 16000,
                'inference_seconds': 0.01}


class LiveApiTests(unittest.TestCase):
    def setUp(self):
        root = (server.ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/web-test/live-qa').resolve()
        assert root.drive.upper() == 'C:' and root.is_relative_to(server.ROOT)
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='api-', dir=root)
        self.records = Path(self.temp.name).resolve()
        assert self.records.is_relative_to(root)
        with patch.object(server, 'Engine', FakeEngine):
            self.app = server.create_app(recordings=self.records)
        self.engine = self.app.state.engine
        self.client = TestClient(self.app)
        self.client.__enter__()
        buffer = io.BytesIO()
        sf.write(buffer, np.zeros(8000, dtype=np.float32), 16000, format='WAV', subtype='PCM_16')
        self.audio = buffer.getvalue()
        self.live = {'source': 'live', 'session_id': 'a' * 32, 'segment_index': '1', 'age': '3', 'reference': '책'}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        assert self.records.is_relative_to(server.ROOT) and self.records.drive.upper() == 'C:'
        self.temp.cleanup()

    def post(self, path='/api/transcribe', data=None, audio=None, **kwargs):
        return self.client.post(path, files={'file': ('audio.wav', self.audio if audio is None else audio, 'audio/wav')},
                                data=data or {}, **kwargs)

    def test_previews_never_create_history(self):
        for _ in range(3):
            response = self.post('/api/preview')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()['provisional'])
        self.assertEqual(list(self.records.iterdir()), [])
        self.assertEqual(self.client.get('/api/history').json()['total'], 0)

    def test_live_retry_reuses_record_and_audio(self):
        first = self.post(data=self.live)
        self.assertEqual(first.status_code, 200, first.text)
        repeat = self.post(data=self.live)
        self.assertEqual(repeat.json(), first.json())
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(len(list(self.records.glob('*.json'))), 1)
        self.assertEqual(len(list(self.records.glob('*.wav'))), 1)
        self.assertEqual(len(list(self.records.glob('*.tmp'))), 0)
        self.assertEqual(self.client.get(first.json()['audio_url']).content, self.audio)
        self.assertTrue(first.json()['score']['exact'])
        conflict = self.post(data={**self.live, 'reference': '다른 정답'})
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(len(self.engine.calls), 1)

    def test_metadata_never_changes_model_input_and_legacy_still_saves(self):
        for age, reference in [('3', '책'), ('adult', '다른 정답')]:
            result = self.post(data={'source': 'upload', 'age': age, 'reference': reference})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()['prediction']['text'], '책')
        np.testing.assert_array_equal(*self.engine.calls)
        self.assertEqual(self.client.get('/api/history').json()['total'], 2)

    def test_invalid_busy_and_cross_origin_requests_do_not_save(self):
        self.assertEqual(self.post(data={**self.live, 'session_id': '../bad'}).status_code, 400)
        self.assertEqual(self.post('/api/preview', audio=b'invalid').status_code, 400)
        self.assertEqual(self.post('/api/preview', headers={'Origin': 'http://example.com'}).status_code, 403)
        self.engine.blocked = True
        self.assertEqual(self.post('/api/preview').status_code, 409)
        self.engine.blocked, self.engine.ready = False, False
        self.assertEqual(self.post('/api/preview').status_code, 503)
        self.assertEqual(list(self.records.iterdir()), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
