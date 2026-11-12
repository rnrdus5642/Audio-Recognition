"""Local, model-free listening room. Audio/labels are read-only; reviews are separate."""
import argparse
from array import array
from contextlib import closing
from datetime import datetime, timezone
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from urllib.parse import parse_qs, urlsplit
import wave

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
PREPARATION = WORKSPACE / '02_Datasets/학습준비/자모_원문라벨_3-6세'
STATE = WORKSPACE / '02_Datasets/검토/3세_청취'
AUDIO_ROOT = (WORKSPACE / '02_Datasets/AIHUB_108/audio/03세').resolve()
APP = 'age3-listening-room-v1'
CLARITY = {'clear', 'partial', 'unclear', 'hold'}
RELATION = {'', 'same', 'variation', 'different', 'unsure'}


def connect(path):
    db = sqlite3.connect(path, timeout=15)
    db.row_factory = sqlite3.Row
    return db


def audio_path(relative):
    path = (WORKSPACE / relative).resolve()
    if not path.is_relative_to(AUDIO_ROOT) or path.suffix.lower() != '.wav':
        raise ValueError('허용된 3세 녹음 경로가 아닙니다.')
    return path


def prepare_catalog(state=STATE):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    plan = json.loads((PREPARATION / 'dataset-plan.json').read_text(encoding='utf-8'))
    source = PREPARATION / plan['active_catalogs']['AIHUB_108']
    sources = [source / f'{part}.jsonl.gz' for part in ('train', 'dev', 'test', 'review')]
    fingerprint = hashlib.sha256(json.dumps([
        (str(p.relative_to(WORKSPACE)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sources
    ]).encode()).hexdigest()
    catalog = state / 'catalog.sqlite3'
    if catalog.is_file():
        with closing(connect(catalog)) as db:
            previous = db.execute('SELECT value FROM metadata WHERE key=?', ('fingerprint',)).fetchone()
            if previous and previous[0] == fingerprint:
                return catalog
    print('Indexing the four verified catalogues (audio files are not copied)...', flush=True)
    rows = []
    for source_file in sources:
        with gzip.open(source_file, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['age'] != 3:
                    continue
                path = audio_path(row['audio_path'])
                if path.stat().st_size != row['audio_bytes']:
                    raise ValueError('오디오 크기가 기존 검증 기록과 다릅니다: ' + row['record_id'])
                row['_partition'] = source_file.name.split('.')[0]
                rows.append(row)
    if not rows:
        raise ValueError('3세 녹음이 없습니다.')
    rows.sort(key=lambda row: row['record_id'])
    speakers = {key: i + 1 for i, key in enumerate(sorted({r['speaker_key'] for r in rows}))}
    temporary = state / f'catalog-building-{os.getpid()}.sqlite3'
    with closing(connect(temporary)) as db:
        db.executescript('''
          CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE clips (
            id TEXT PRIMARY KEY, number INTEGER UNIQUE NOT NULL, record_id TEXT UNIQUE NOT NULL,
            speaker_key TEXT NOT NULL, speaker_no INTEGER NOT NULL,
            seconds REAL NOT NULL, partition TEXT NOT NULL, assigned_split TEXT NOT NULL,
            label_original TEXT NOT NULL, label_text TEXT NOT NULL,
            audio_path TEXT NOT NULL, audio_sha256 TEXT NOT NULL,
            category TEXT NOT NULL, issues TEXT NOT NULL);
          CREATE INDEX clips_speaker ON clips(speaker_no);
          CREATE INDEX clips_partition ON clips(partition);
        ''')
        for number, row in enumerate(rows, 1):
            issues = list(dict.fromkeys(row.get('review_reasons', []) +
                          row.get('audio_details', {}).get('audio_review_reasons', [])))
            db.execute('INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (
                hashlib.sha256(row['record_id'].encode()).hexdigest()[:24], number,
                row['record_id'], row['speaker_key'], speakers[row['speaker_key']],
                row['audio_details']['audio_seconds'], row['_partition'], row['split'],
                row['label_original'], row['label_text'], row['audio_path'], row['audio_sha256'],
                row.get('category', ''), json.dumps(issues, ensure_ascii=False)))
        db.execute('INSERT INTO metadata VALUES (?,?)', ('fingerprint', fingerprint))
        db.commit()
    os.replace(temporary, catalog)
    print(f'Indexed {len(rows):,} recordings / {len(speakers)} speaker IDs.', flush=True)
    return catalog


class Room:
    def __init__(self, catalog, reviews):
        self.catalog, self.reviews = Path(catalog), Path(reviews)
        self.peaks = {}
        self.reviews.parent.mkdir(parents=True, exist_ok=True)
        with closing(connect(self.reviews)) as db:
            db.executescript('''
              PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS answers (
                clip_id TEXT PRIMARY KEY, audio_sha256 TEXT NOT NULL, revision INTEGER NOT NULL,
                clarity TEXT NOT NULL, relation TEXT NOT NULL, heard TEXT NOT NULL,
                note TEXT NOT NULL, label_seen INTEGER NOT NULL, updated_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS history (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, clip_id TEXT NOT NULL,
                payload TEXT NOT NULL, updated_at TEXT NOT NULL);
            ''')
            db.commit()
        with closing(connect(self.catalog)) as db:
            self.totals = dict(db.execute('SELECT COUNT(*) AS total, COUNT(DISTINCT speaker_key) AS speakers, '
                                         'SUM(seconds) AS seconds FROM clips').fetchone())
            self.speakers = [dict(r) for r in db.execute(
                'SELECT speaker_no AS id, COUNT(*) AS count FROM clips GROUP BY speaker_no ORDER BY speaker_no')]

    def joined(self):
        db = connect(self.catalog)
        db.execute('ATTACH DATABASE ? AS review', (str(self.reviews),))
        db.create_function('shuffle_key', 2, lambda key, seed:
                           int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:15], 16))
        return db

    @staticmethod
    def public(row):
        item = dict(row)
        item.pop('audio_path', None)
        item['issues'] = json.loads(item['issues'])
        item['audio_url'] = '/audio/' + item['id']
        return item

    def clip(self, clip_id):
        with closing(self.joined()) as db:
            row = db.execute('SELECT c.*, r.clarity, r.relation, r.heard, r.note, r.label_seen, '
                             'r.updated_at, COALESCE(r.revision,0) AS revision FROM clips c '
                             'LEFT JOIN review.answers r ON r.clip_id=c.id AND r.audio_sha256=c.audio_sha256 '
                             'WHERE c.id=?', (clip_id,)).fetchone()
        return dict(row) if row else None

    def stats(self):
        with closing(self.joined()) as db:
            counts = {r['clarity']: r['n'] for r in db.execute(
                'SELECT r.clarity, COUNT(*) AS n FROM review.answers r JOIN clips c '
                'ON c.id=r.clip_id AND c.audio_sha256=r.audio_sha256 GROUP BY r.clarity')}
        return {**self.totals, 'reviewed': sum(counts.values()), 'ratings': counts, 'speaker_options': self.speakers}

    def listing(self, params):
        def get(key, default=''):
            return params.get(key, [default])[0]
        limit = max(1, min(60, int(get('limit', '30'))))
        offset = max(0, int(get('offset', '0')))
        seed = int(get('seed', '31')) % 2147483647
        where, args = ['1=1'], []
        if get('q').strip():
            where.append('(instr(c.label_original,?)>0 OR instr(c.label_text,?)>0)')
            args += [get('q').strip()[:200]] * 2
        if get('speaker'):
            where.append('c.speaker_no=?')
            args.append(int(get('speaker')))
        if get('partition') in ('train', 'dev', 'test', 'review'):
            where.append('c.partition=?')
            args.append(get('partition'))
        if get('status') == 'unreviewed':
            where.append('r.clip_id IS NULL')
        elif get('status') == 'reviewed':
            where.append('r.clip_id IS NOT NULL')
        if get('clarity') in CLARITY:
            where.append('r.clarity=?')
            args.append(get('clarity'))
        if get('max_seconds'):
            where.append('c.seconds<=?')
            args.append(float(get('max_seconds')))
        join = (' FROM clips c LEFT JOIN review.answers r '
                'ON r.clip_id=c.id AND r.audio_sha256=c.audio_sha256 WHERE ' + ' AND '.join(where))
        select = ('SELECT c.*,r.clarity,r.relation,r.heard,r.note,r.label_seen,r.updated_at,'
                  'COALESCE(r.revision,0) AS revision')
        sort = get('sort', 'diverse')
        order = {'short': 'seconds,id', 'long': 'seconds DESC,id', 'file': 'number',
                 'random': f'shuffle_key(id,{seed}),id'}.get(sort)
        with closing(self.joined()) as db:
            total = db.execute('SELECT COUNT(*)' + join, args).fetchone()[0]
            if order:
                sql = select + join + ' ORDER BY ' + order + ' LIMIT ? OFFSET ?'
            else:
                sql = ('WITH filtered AS (' + select + join + '), ranked AS (SELECT *, '
                       f'ROW_NUMBER() OVER (PARTITION BY speaker_key ORDER BY shuffle_key(id,{seed})) '
                       'AS speaker_rank FROM filtered) SELECT * FROM ranked ORDER BY speaker_rank,'
                       f'shuffle_key(speaker_key,{seed}),id LIMIT ? OFFSET ?')
            rows = [self.public(r) for r in db.execute(sql, args + [limit, offset])]
        return {'total': total, 'offset': offset, 'limit': limit, 'items': rows}

    def save(self, body):
        item = self.clip(body.get('id', ''))
        if not item or body.get('audio_sha256') != item['audio_sha256']:
            raise ValueError('녹음 정보가 다릅니다. 페이지를 새로고침해 주세요.')
        if body.get('clarity') not in CLARITY or body.get('relation', '') not in RELATION:
            raise ValueError('들리는 정도를 선택해 주세요.')
        for key in ('heard', 'note'):
            if not isinstance(body.get(key, ''), str) or len(body.get(key, '')) > 2000:
                raise ValueError('메모는 2,000자 이하로 입력해 주세요.')
        if type(body.get('revision')) is not int or type(body.get('label_seen')) is not bool:
            raise ValueError('저장 정보 형식이 올바르지 않습니다.')
        if body.get('relation') and not body['label_seen']:
            raise ValueError('라벨을 확인한 후 비교 결과를 선택해 주세요.')
        now = datetime.now(timezone.utc).isoformat()
        with closing(connect(self.reviews)) as db:
            db.execute('BEGIN IMMEDIATE')
            saved = db.execute('SELECT revision FROM answers WHERE clip_id=?', (item['id'],)).fetchone()
            revision = saved[0] if saved else 0
            if revision != body['revision']:
                raise RuntimeError('다른 창에서 이 메모가 변경됐습니다. 새로고침 후 확인해 주세요.')
            values = (item['id'], item['audio_sha256'], revision + 1, body['clarity'],
                      body.get('relation', ''), body.get('heard', '').strip(), body.get('note', '').strip(),
                      int(body['label_seen']), now)
            db.execute('INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?,?,?,?,?)', values)
            db.execute('INSERT INTO history(clip_id,payload,updated_at) VALUES (?,?,?)',
                       (item['id'], json.dumps(body, ensure_ascii=False), now))
            db.commit()
        return {'ok': True, 'revision': revision + 1, 'updated_at': now}

    def waveform(self, item):
        key = item['audio_sha256']
        if key not in self.peaks:
            with wave.open(str(audio_path(item['audio_path'])), 'rb') as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError('파형을 표시할 수 없는 오디오 형식입니다.')
                values = array('h', wav.readframes(wav.getnframes()))
                if sys.byteorder != 'little':
                    values.byteswap()
            count = 120
            peaks = [max((abs(v) for v in values[i*len(values)//count:(i+1)*len(values)//count]), default=0)
                     / 32768 for i in range(count)]
            if len(self.peaks) > 1000:
                self.peaks.clear()
            self.peaks[key] = peaks
        return {'peaks': self.peaks[key], 'seconds': item['seconds']}


def make_server(catalog, reviews, port=8781):
    room = Room(catalog, reviews)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def allowed(self):
            hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            return (self.headers.get('Host') in hosts and
                    self.headers.get('Origin') in (None, *(f'http://{h}' for h in hosts)))

        def reply(self, code, body, content_type='application/json; charset=utf-8', headers=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            for key, value in {'Content-Type': content_type, 'Content-Length': str(len(body)),
                               'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                               **(headers or {})}.items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            try:
                if not self.allowed():
                    return self.reply(403, {'error': '이 컴퓨터의 청취 페이지에서만 접근할 수 있습니다.'})
                parsed = urlsplit(self.path)
                path = parsed.path
                static = {'/': ('index.html', 'text/html; charset=utf-8'),
                          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                          '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if path in static:
                    name, kind = static[path]
                    return self.reply(200, (HERE / 'static' / name).read_bytes(), kind)
                if path == '/favicon.ico':
                    return self.reply(204, b'')
                if path == '/health':
                    return self.reply(200, {'ok': True, 'app': APP, 'pid': os.getpid(), **room.totals})
                if path == '/api/stats':
                    return self.reply(200, room.stats())
                if path == '/api/clips':
                    return self.reply(200, room.listing(parse_qs(parsed.query)))
                if path.startswith(('/api/clip/', '/api/waveform/', '/audio/')):
                    item = room.clip(path.rsplit('/', 1)[-1])
                    if not item:
                        return self.reply(404, {'error': '녹음을 찾을 수 없습니다.'})
                    if path.startswith('/api/clip/'):
                        return self.reply(200, room.public(item))
                    if path.startswith('/api/waveform/'):
                        return self.reply(200, room.waveform(item))
                    source_path = audio_path(item['audio_path'])
                    size = source_path.stat().st_size
                    first, last, code = 0, size - 1, 200
                    if self.headers.get('Range'):
                        match = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers['Range'])
                        if not match or not any(match.groups()):
                            return self.reply(416, b'', 'audio/wav', {'Content-Range': f'bytes */{size}'})
                        a, b = match.groups()
                        first = int(a) if a else max(0, size-int(b))
                        last = min(int(b), size-1) if a and b else size-1
                        if first > last or first >= size:
                            return self.reply(416, b'', 'audio/wav', {'Content-Range': f'bytes */{size}'})
                        code = 206
                    headers = {'Accept-Ranges': 'bytes'}
                    if code == 206:
                        headers['Content-Range'] = f'bytes {first}-{last}/{size}'
                    with source_path.open('rb') as audio:
                        audio.seek(first)
                        return self.reply(code, audio.read(last-first+1), 'audio/wav', headers)
                if path == '/api/export':
                    with closing(room.joined()) as db:
                        rows = [dict(r) for r in db.execute(
                            'SELECT c.record_id,c.speaker_key,c.partition,c.label_original,c.audio_path,r.* '
                            'FROM review.answers r JOIN clips c ON c.id=r.clip_id ORDER BY r.updated_at')]
                        history = [dict(r) for r in db.execute('SELECT * FROM review.history ORDER BY event_id')]
                    return self.reply(200, {'app': APP, 'exported_at': datetime.now(timezone.utc).isoformat(),
                                           'reviews': rows, 'history': history}, headers={
                        'Content-Disposition': 'attachment; filename="age3-listening-reviews.json"'})
                return self.reply(404, {'error': '페이지를 찾을 수 없습니다.'})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except (ValueError, FileNotFoundError) as exc:
                self.reply(400, {'error': str(exc)})
            except Exception as exc:
                print(f'GET failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
                self.reply(500, {'error': '자료를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.'})

        def do_POST(self):
            try:
                if not self.allowed():
                    return self.reply(403, {'error': '이 청취 페이지에서만 저장할 수 있습니다.'})
                if self.path != '/api/review':
                    return self.reply(404, {'error': '페이지를 찾을 수 없습니다.'})
                if self.headers.get_content_type() != 'application/json':
                    return self.reply(415, {'error': 'JSON 요청이 필요합니다.'})
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 20000:
                    raise ValueError('저장 요청 크기가 올바르지 않습니다.')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('저장 요청 형식이 올바르지 않습니다.')
                self.reply(200, room.save(body))
            except RuntimeError as exc:
                self.reply(409, {'error': str(exc)})
            except (ValueError, TypeError, UnicodeDecodeError) as exc:
                self.reply(400, {'error': str(exc)})
            except sqlite3.Error:
                self.reply(503, {'error': '메모를 저장하지 못했습니다. 다시 시도해 주세요.'})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.room = room
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8781)
    parser.add_argument('--state', type=Path, default=STATE)
    parser.add_argument('--reviews-db', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    catalog = prepare_catalog(args.state)
    if args.prepare_only:
        print(catalog, flush=True)
        return
    server = make_server(catalog, args.reviews_db or args.state / 'reviews.sqlite3', args.port)
    print(f'Ready: http://127.0.0.1:{server.server_port}/ (pid {os.getpid()})', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
