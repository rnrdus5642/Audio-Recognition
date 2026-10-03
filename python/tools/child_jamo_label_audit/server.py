"""Local listening review; append-only answers, frozen questions, no model UI."""
import argparse
from contextlib import closing
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_DIRECTORY = ROOT / '.venv/child-jamo-web-20260916/dataset-audit-20260928'


def make_server(directory=DEFAULT_DIRECTORY, port=8777, db_path=None, page_path=None):
    directory = Path(directory)
    page_path = Path(page_path) if page_path else HERE / 'listen.html'
    page_bytes = page_path.read_bytes()
    raw_review = (directory / 'review.json').read_bytes()
    bundle_hash = hashlib.sha256(raw_review).hexdigest()
    review = json.loads(raw_review)
    items = review['items']
    by_id = {i['id']: i for i in items}
    assert len(by_id) == len(items)
    audio = {}
    for item in items:
        for key in ('train', 'full'):
            clip = item[key]
            path = (directory / 'audio' / clip['file']).resolve()
            assert path.parent == (directory / 'audio').resolve()
            blob = path.read_bytes()
            if hashlib.sha256(blob).hexdigest() != clip['sha256']:
                raise ValueError('Audio hash mismatch: ' + item['id'])
            audio[item['id'] + '/' + key] = blob
    db_path = Path(db_path) if db_path else directory / 'answers.sqlite3'
    db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect():
        db = sqlite3.connect(db_path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    with closing(connect()) as db:
        db.executescript('''
          CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS reveals (
            item_id TEXT PRIMARY KEY, binding TEXT NOT NULL, first_heard TEXT NOT NULL,
            at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
          CREATE TABLE IF NOT EXISTS answers (
            revision INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL,
            binding TEXT NOT NULL, choice TEXT NOT NULL, note TEXT NOT NULL,
            crop_relation TEXT NOT NULL,
            at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
        ''')
        saved = db.execute("SELECT value FROM metadata WHERE key='review_sha256'").fetchone()
        if saved and saved[0] != bundle_hash:
            raise ValueError('This database belongs to a different question set')
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('review_sha256',?)", (bundle_hash,))
        db.commit()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload, kind='application/json; charset=utf-8', extra=None):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def local_request(self):
            allowed = {f'{host}:{self.server.server_port}' for host in ('127.0.0.1', 'localhost')}
            return self.headers.get('Host') in allowed and self.headers.get('Origin') in (
                None, *(f'http://{host}' for host in allowed))

        def state(self):
            with closing(connect()) as db:
                rows = db.execute('SELECT * FROM answers WHERE revision IN '
                                  '(SELECT MAX(revision) FROM answers GROUP BY item_id)').fetchall()
                answers = {r['item_id']: dict(r) for r in rows}
                reveals = {r['item_id']: dict(r) for r in db.execute('SELECT * FROM reveals')}
            public = []
            for item in items:
                answer = answers.get(item['id'])
                public.append({k: item[k] for k in ('id', 'number', 'binding', 'seconds', 'cropped')} |
                              {'answer': answer, 'revision': answer['revision'] if answer else 0,
                               'reveal': reveals.get(item['id'])})
            return {'version': review['version'], 'review_sha256': bundle_hash, 'items': public,
                    'answered': len(answers), 'total': len(items)}

        def do_GET(self):
            if not self.local_request():
                return self.reply(403, {'error': '이 컴퓨터의 청취 페이지에서만 사용할 수 있습니다.'})
            path = urlsplit(self.path).path
            if path == '/':
                return self.reply(200, page_bytes, 'text/html; charset=utf-8')
            if path == '/favicon.ico':
                return self.reply(204, b'')
            if path == '/health':
                return self.reply(200, {'ok': True, 'app': 'training-pronunciation-audit',
                                       'pid': os.getpid(), 'questions': len(items), 'review_sha256': bundle_hash})
            if path == '/api/state':
                return self.reply(200, self.state())
            if path == '/api/export':
                with closing(connect()) as db:
                    histories = [dict(r) for r in db.execute('SELECT * FROM answers ORDER BY revision')]
                return self.reply(200, {**self.state(), 'answer_history': histories})
            if path.startswith('/audio/') and path[7:] in audio:
                blob = audio[path[7:]]
                header = self.headers.get('Range')
                if header:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
                    if not match or not any(match.groups()):
                        return self.reply(416, b'', 'audio/wav', {'Content-Range': f'bytes */{len(blob)}'})
                    start_s, end_s = match.groups()
                    start = int(start_s) if start_s else max(0, len(blob) - int(end_s))
                    end = min(int(end_s), len(blob) - 1) if start_s and end_s else len(blob) - 1
                    if start > end:
                        return self.reply(416, b'', 'audio/wav', {'Content-Range': f'bytes */{len(blob)}'})
                    return self.reply(206, blob[start:end + 1], 'audio/wav',
                                      {'Content-Range': f'bytes {start}-{end}/{len(blob)}', 'Accept-Ranges': 'bytes'})
                return self.reply(200, blob, 'audio/wav', {'Accept-Ranges': 'bytes'})
            return self.reply(404, {'error': '페이지를 찾을 수 없습니다.'})

        def do_POST(self):
            if not self.local_request():
                return self.reply(403, {'error': '청취 페이지에서만 저장할 수 있습니다.'})
            if self.path not in ('/api/reveal', '/api/answer'):
                return self.reply(404, {'error': '페이지를 찾을 수 없습니다.'})
            if self.headers.get_content_type() != 'application/json':
                return self.reply(415, {'error': 'JSON 요청이 필요합니다.'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16000:
                    raise ValueError('답변 크기가 올바르지 않습니다.')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict) or not isinstance(body.get('id'), str):
                    raise ValueError('답변 형식이 올바르지 않습니다.')
                item = by_id.get(body['id'])
                if not item or body.get('binding') != item['binding']:
                    return self.reply(409, {'error': '문항 정보가 다릅니다. 새로고침해 주세요.'})
                if self.path == '/api/reveal':
                    heard = body.get('first_heard', '')
                    if not isinstance(heard, str) or len(heard) > 1000:
                        raise ValueError('메모는 1,000자 이하로 적어 주세요.')
                    with closing(connect()) as db:
                        db.execute('INSERT OR IGNORE INTO reveals(item_id,binding,first_heard) VALUES (?,?,?)',
                                   (item['id'], item['binding'], heard))
                        db.commit()
                        reveal = dict(db.execute('SELECT * FROM reveals WHERE item_id=?', (item['id'],)).fetchone())
                    return self.reply(200, {'reference': item['reference'], 'reveal': reveal})
                choice, note = body.get('choice'), body.get('note', '')
                crop_relation, revision = body.get('crop_relation', ''), body.get('revision')
                allowed = {c['value'] for c in item['reference']['choices']}
                if not isinstance(choice, str) or choice not in allowed:
                    raise ValueError('어떻게 들렸는지 하나 골라 주세요.')
                if (item['reference']['kind'] == 'whole_form' and choice.startswith('whole_form_')
                        and body.get('whole_confirmed') is not True):
                    raise ValueError('문장 전체가 같은 소리로 들렸는지 확인해 주세요. 일부만 확실하면 해당 판단 항목을 골라 주세요.')
                if not isinstance(note, str) or len(note) > 2000:
                    raise ValueError('메모는 2,000자 이하로 적어 주세요.')
                if not isinstance(crop_relation, str) or crop_relation not in ('', 'same', 'cut', 'uncertain'):
                    raise ValueError('원본 비교 답변이 올바르지 않습니다.')
                if crop_relation and not item['cropped']:
                    raise ValueError('이 문항은 학습 구간과 원본이 같습니다.')
                if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
                    raise ValueError('답변 이력이 올바르지 않습니다.')
                with closing(connect()) as db:
                    db.execute('BEGIN IMMEDIATE')
                    if not db.execute('SELECT 1 FROM reveals WHERE item_id=?', (item['id'],)).fetchone():
                        return self.reply(409, {'error': '소리를 듣고 비교 표기를 먼저 열어 주세요.'})
                    current = db.execute('SELECT MAX(revision) FROM answers WHERE item_id=?', (item['id'],)).fetchone()[0] or 0
                    if current != revision:
                        return self.reply(409, {'error': '다른 창에서 답변이 바뀌었습니다. 새로고침 후 확인해 주세요.'})
                    cursor = db.execute('INSERT INTO answers(item_id,binding,choice,note,crop_relation) VALUES (?,?,?,?,?)',
                                        (item['id'], item['binding'], choice, note, crop_relation))
                    db.commit()
                return self.reply(200, {'ok': True, 'revision': cursor.lastrowid})
            except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                return self.reply(400, {'error': str(exc)})
            except sqlite3.Error:
                return self.reply(503, {'error': '저장하지 못했습니다. 잠시 후 다시 눌러 주세요.'})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument('--db', type=Path)
    parser.add_argument('--port', type=int, default=8777)
    parser.add_argument('--page', type=Path)
    args = parser.parse_args()
    server = make_server(args.directory, args.port, args.db, args.page)
    print(f'Ready: http://127.0.0.1:{server.server_port}/ (pid {os.getpid()})', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
