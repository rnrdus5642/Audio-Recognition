"""Test real question bindings with disposable databases, never user answers."""
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from server import DEFAULT_DIRECTORY, make_server


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'test.sqlite3'
        self.server = make_server(db_path=self.db, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.state = self.request('/api/state')[1]
        self.item = self.state['items'][0]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path, body=None, headers=None):
        headers = headers or {}
        if body is not None:
            headers = {'Content-Type': 'application/json', **headers}
        request = urllib.request.Request(self.base + path, headers=headers,
                                         data=json.dumps(body).encode() if body is not None else None)
        try:
            response = urllib.request.urlopen(request)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            data = response.read()
            return response.status, json.loads(data) if 'application/json' in response.headers.get('Content-Type', '') else data

    def reveal(self, item=None, heard='before seeing any alternatives'):
        item = item or self.item
        return self.request('/api/reveal', {'id': item['id'], 'binding': item['binding'], 'first_heard': heard})

    def answer(self, item=None, **kwargs):
        item = item or self.item
        return self.request('/api/answer', {'id': item['id'], 'binding': item['binding'], 'revision': 0,
                                          'choice': 'ambiguous', 'note': '', **kwargs})

    def test_blind_state_and_no_model_results_exposed(self):
        self.assertEqual(len(self.state['items']), 20)
        self.assertEqual(self.state['answered'], 0)
        payload = json.dumps(self.state)
        for forbidden in ('reference', 'surface', 'prediction', 'transcript', 'group'):
            self.assertNotIn(forbidden, payload)
        self.assertEqual(self.request('/baseline-predictions.json')[0], 404)
        self.assertEqual(self.request('/evidence.json')[0], 404)

    def test_first_impression_is_preserved_and_requires_reveal(self):
        self.assertEqual(self.answer()[0], 409)
        first = self.reveal()[1]
        self.assertEqual(self.reveal(heard='changed after seeing choices')[1]['reveal'], first['reveal'])
        self.assertEqual(self.answer()[0], 200)

    def test_stale_writes_rejected_and_revisions_preserved(self):
        self.reveal()
        status, result = self.answer()
        self.assertEqual(status, 200)
        self.assertEqual(self.answer()[0], 409)
        self.assertEqual(self.answer(revision=result['revision'], note='changed')[0], 200)
        export = self.request('/api/export')[1]
        self.assertEqual(len(export['answer_history']), 2)
        self.assertEqual(export['answered'], 1)
        self.assertEqual(export['items'][0]['answer']['note'], 'changed')

    def test_wrong_binding_and_invalid_choices_do_not_write(self):
        self.assertEqual(self.reveal({**self.item, 'binding': 'wrong'})[0], 409)
        self.reveal()
        for choice in ('invented', [], None):
            self.assertEqual(self.answer(choice=choice)[0], 400)
        self.assertEqual(self.answer(revision=True)[0], 400)
        self.assertEqual(self.request('/api/state')[1]['answered'], 0)

    def test_approximate_spelling_not_required(self):
        bundle = json.loads((DEFAULT_DIRECTORY / 'review.json').read_text(encoding='utf-8'))
        target = next(i for i in bundle['items'] if i['reference']['kind'] == 'general')
        self.reveal(target, heard='')
        self.assertEqual(self.answer(target, choice='different', note='')[0], 200)

    def test_foreign_origin_and_host_rejected(self):
        body = {'id': self.item['id'], 'binding': self.item['binding']}
        self.assertEqual(self.request('/api/reveal', body, {'Origin': 'https://example.com'})[0], 403)
        self.assertEqual(self.request('/api/state', headers={'Host': 'example.com'})[0], 403)

    def test_audio_ranges_match_actual_payload(self):
        path = '/audio/' + self.item['id'] + '/train'
        status, data = self.request(path)
        self.assertEqual(status, 200)
        self.assertEqual(self.request(path, headers={'Range': 'bytes=0-99'}), (206, data[:100]))
        self.assertEqual(self.request(path, headers={'Range': 'bytes=-20'}), (206, data[-20:]))
        self.assertEqual(self.request(path, headers={'Range': 'bytes=999999999-'} )[0], 416)

    def test_reopen_preserves_answers_and_changed_questions_refused(self):
        self.reveal()
        self.answer()
        reopened = make_server(db_path=self.db, port=0)
        reopened.server_close()
        with closing(sqlite3.connect(self.db)) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM answers').fetchone()[0], 1)
        changed = Path(self.temp.name) / 'changed'
        changed.mkdir()
        shutil.copytree(DEFAULT_DIRECTORY / 'audio', changed / 'audio')
        raw = (DEFAULT_DIRECTORY / 'review.json').read_text(encoding='utf-8')
        (changed / 'review.json').write_text(raw + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'different question set'):
            make_server(changed, port=0, db_path=self.db)
        clip = next((changed / 'audio').glob('*.wav'))
        clip.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Audio hash mismatch'):
            make_server(changed, port=0, db_path=Path(self.temp.name) / 'other.db')


if __name__ == '__main__':
    unittest.main()
