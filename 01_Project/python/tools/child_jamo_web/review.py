"""The listening review: which recordings to ask about, and what was said.

The audit can say where the model disagrees with the expected pronunciation.
It cannot say why. Only someone listening can tell a mispronunciation from a
model error, a bad label or a bad cut - so this module's job is to put a
small, deliberately chosen set of recordings in front of a person and keep
exactly what they say about them.

Two rules shape everything here:

  the answer is the person's     A free-text "heard" form is stored as
                                 typed. It is not run through the G2P
                                 rules, not normalised, not corrected
                                 towards the transcript or the model.
  answers are bound to audio     Every annotation records the PCM hash of
                                 what was played and the baseline identity
                                 of the prediction shown. If the audio or
                                 the model changes, the annotation is
                                 marked stale rather than silently reused.

Storage is SQLite with an append-only revision history: editing an answer
adds a revision, and the current answer is the newest revision for that
recording. Nothing is ever overwritten in place, so a mistake made during
review can be traced and undone.
"""
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / 'child_jamo_finetune'))
import audit                                                   # noqa: E402
from common import ROOT                                        # noqa: E402

WORK = audit.WORK
DEFAULT_DB = WORK / 'reviews.sqlite3'

#: What a reviewer can say. `unclear` and `unusable` are answers too - they
#: are what an honest review produces when the audio does not support a
#: judgement, and they are kept out of the accepted-label export.
CHOICES = {
    'expected': '전사대로 들림',
    'prediction': '모델이 쓴 대로 들림',
    'other': '다르게 들림',
    'unclear': '판단하기 어려움',
}
STATUSES = {'answered', 'unclear', 'unusable'}
#: Problems with the recording itself, which a reviewer can report without
#: being made to label a pronunciation.
FLAGS = {
    'boundary': '구간이 잘못 잘림',
    'multiple_speakers': '여러 사람 목소리',
    'noise': '잡음이 심함',
    'silent': '소리가 거의 없음',
    'other_problem': '기타 문제',
}

_local = threading.local()


class Conflict(Exception):
    """A save that would overwrite a newer answer, or answer stale audio."""

    def __init__(self, message, current_revision=None):
        super().__init__(message)
        self.current_revision = current_revision


# --- store ------------------------------------------------------------

class Store:
    """Annotations, with history, in one small SQLite file."""

    def __init__(self, path=None):
        self.path = Path(path or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # A unique token, not id(self): CPython reuses an id once the
        # object behind it is collected, so a later Store could inherit a
        # cached connection to an earlier Store's file.
        self._key = f'db_{uuid.uuid4().hex}'
        with self._connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS revision (
                revision      INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id     TEXT NOT NULL,
                pool          TEXT NOT NULL,
                status        TEXT NOT NULL,
                choice        TEXT,
                heard_text    TEXT,
                note          TEXT,
                flags         TEXT NOT NULL DEFAULT '[]',
                audio_pcm_sha256 TEXT NOT NULL,
                baseline_key  TEXT NOT NULL,
                binding       TEXT NOT NULL DEFAULT '',
                reference_ids TEXT,
                reference_surface TEXT,
                predicted_text    TEXT,
                reviewer      TEXT NOT NULL DEFAULT 'user',
                at            TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS revision_record
                ON revision(record_id, revision);
            """)
            # A store written before bindings existed gains the columns
            # rather than being thrown away.
            have = {row[1] for row in db.execute('PRAGMA table_info(revision)')}
            for column in ('binding', 'reference_ids'):
                if column not in have:
                    db.execute(f'ALTER TABLE revision ADD COLUMN {column} TEXT')

    def _connect(self):
        # One connection per thread: FastAPI runs `def` endpoints on a pool.
        conn = getattr(_local, self._key, None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            setattr(_local, self._key, conn)
        return conn

    def save(self, expected_revision=None, **fields):
        """Append a revision, unless someone else got there first.

        `expected_revision` is the revision the page was showing. Under the
        lock, the newest revision is read back and compared: if it moved,
        this answer was written against a screen that is already out of
        date and is refused rather than allowed to supersede it. That is
        what stops a forgotten tab from overwriting a newer answer.
        """
        fields.setdefault('reviewer', 'user')
        fields['at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        fields['flags'] = json.dumps(fields.get('flags') or [],
                                     ensure_ascii=False)
        fields['reference_ids'] = json.dumps(fields.get('reference_ids') or [],
                                             ensure_ascii=False)
        columns = ('record_id', 'pool', 'status', 'choice', 'heard_text',
                   'note', 'flags', 'audio_pcm_sha256', 'baseline_key',
                   'binding', 'reference_ids', 'reference_surface',
                   'predicted_text', 'reviewer', 'at')
        with self._lock:
            db = self._connect()
            with db:
                newest = db.execute(
                    'SELECT revision FROM revision WHERE record_id = ? '
                    'ORDER BY revision DESC LIMIT 1',
                    (fields['record_id'],)).fetchone()
                seen = newest[0] if newest else 0
                if expected_revision is not None and int(expected_revision) != seen:
                    raise Conflict(
                        f'이 음성에는 더 새로운 답이 저장되어 있습니다 '
                        f'(화면 {expected_revision}, 저장된 것 {seen})',
                        current_revision=seen)
                cursor = db.execute(
                    f"INSERT INTO revision ({','.join(columns)}) VALUES "
                    f"({','.join('?' * len(columns))})",
                    [fields.get(c) for c in columns])
            return self.current(fields['record_id'], revision=cursor.lastrowid)

    def current(self, record_id, revision=None):
        db = self._connect()
        row = db.execute(
            'SELECT * FROM revision WHERE record_id = ? '
            'ORDER BY revision DESC LIMIT 1', (record_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out['flags'] = json.loads(out['flags'])
        out['reference_ids'] = json.loads(out['reference_ids'] or '[]')
        out['revisions'] = db.execute(
            'SELECT COUNT(*) FROM revision WHERE record_id = ?',
            (record_id,)).fetchone()[0]
        return out

    def all_current(self):
        db = self._connect()
        rows = db.execute("""
            SELECT r.* FROM revision r
            JOIN (SELECT record_id, MAX(revision) AS newest
                  FROM revision GROUP BY record_id) latest
              ON r.record_id = latest.record_id AND r.revision = latest.newest
        """).fetchall()
        out = {}
        for row in rows:
            entry = dict(row)
            entry['flags'] = json.loads(entry['flags'])
            entry['reference_ids'] = json.loads(entry['reference_ids'] or '[]')
            out[entry['record_id']] = entry
        return out


    def history(self, record_id):
        db = self._connect()
        rows = db.execute(
            'SELECT * FROM revision WHERE record_id = ? ORDER BY revision',
            (record_id,)).fetchall()
        out = []
        for row in rows:
            entry = dict(row)
            entry['flags'] = json.loads(entry['flags'])
            entry['reference_ids'] = json.loads(entry['reference_ids'] or '[]')
            out.append(entry)
        return out


def is_stale(annotation, record):
    """Why an answer no longer describes the recording it was given for.

    A missing record counts as stale: if the audit no longer holds the
    recording, there is nothing left to say the answer is still about it.
    """
    if annotation is None:
        return None
    if record is None:
        return 'baseline_record_missing'
    if annotation.get('binding') and record.get('binding'):
        if annotation['binding'] != record['binding']:
            return 'binding_changed'
        return None
    # Records or answers written before bindings existed: fall back to the
    # parts that were always stored.
    if annotation.get('audio_pcm_sha256') != record.get('audio_pcm_sha256'):
        return 'audio_changed'
    if annotation.get('baseline_key') != record.get('baseline_key'):
        return 'baseline_changed'
    return None


# --- what to ask about ------------------------------------------------

#: The recordings the earlier diagnosis singled out, kept at the front so
#: the questions that block the next decision get answered first.
KNOWN_CASES = ('personal:child1_18', 'personal:child1_22',
               'personal:child1_17', 'personal:child1_09')
PRIORITY_TARGET = 18          # a first sitting, not a backlog


def _pair_of(entry):
    events = [e for e in entry['consonant_events'] if e['op'] != 'insert']
    if not events:
        return None
    first = events[0]
    return f"{first['reference']}>{first['hypothesis'] or '(없음)'}"


def build_shortlist(per_record, target=PRIORITY_TARGET):
    """A deterministic first sitting: small, varied, and reproducible.

    Chosen, in order: the recordings the diagnosis already singled out; then
    one recording for each distinct consonant confusion, alternating
    between the two children and between the personal clips and the corpus
    so no single speaker or source fills the list; then a few recordings the
    model got right, so the reviewer is not only ever hearing mistakes.

    No randomness: the same baseline always produces the same list.
    """
    entries = sorted(per_record.values(), key=lambda e: e['id'])
    by_id = {e['id']: e for e in entries}
    picked, reasons = [], {}

    def take(entry, reason):
        if entry['id'] in reasons:
            return False
        picked.append(entry['id'])
        reasons[entry['id']] = reason
        return True

    for record_id in KNOWN_CASES:
        if record_id in by_id:
            take(by_id[record_id], '이전 분석에서 지목된 녹음')

    wrong = [e for e in entries if e['consonant_mistakes']]
    # One recording per distinct confusion, with a quota per source so the
    # corpus half cannot be squeezed out by the personal clips, and a quota
    # per speaker so one child cannot fill their half.
    quota = {'personal': 6, 'valid': 4}
    seen_pairs = set()

    def order(pool):
        """Candidates for one source, one per distinct confusion.

        The personal clips are drawn alternately from the two children:
        sorting by id alone would hand the whole quota to whichever child
        happens to sort first, and a review that only ever hears one child
        answers half the question.
        """
        rows = sorted((e for e in wrong if e['pool'] == pool),
                      key=lambda e: (e['consonant_mistakes'], e['id']))
        if pool != 'personal':
            return rows
        buckets = {}
        for entry in rows:
            buckets.setdefault(entry['speaker'], []).append(entry)
        interleaved, index = [], 0
        while any(len(b) > index for b in buckets.values()):
            for speaker in sorted(buckets):
                if len(buckets[speaker]) > index:
                    interleaved.append(buckets[speaker][index])
            index += 1
        return interleaved

    for pool in ('personal', 'valid'):
        taken = 0
        for entry in order(pool):
            if taken >= quota[pool]:
                break
            pair = _pair_of(entry)
            if pair is None or pair in seen_pairs:
                continue
            if take(entry, f'자음 혼동 {pair}'):
                seen_pairs.add(pair)
                taken += 1

    # Controls: recordings with no consonant mistake at all, so the review
    # is not a list of failures only. Alternating between the sources, so a
    # truncation at the end cannot leave one source without any.
    clean = {pool: [e for e in entries
                    if e['pool'] == pool and not e['consonant_mistakes']
                    and e['metrics']['exact']]
             for pool in ('personal', 'valid')}
    for rank in range(2):
        for pool in ('personal', 'valid'):
            if rank < len(clean[pool]):
                take(clean[pool][rank], '대조군 (맞게 인식된 녹음)')

    # If there were fewer distinct confusions than slots, top up with more
    # affected recordings rather than handing back a half-empty sitting.
    # A repeated pair from a different recording is still worth hearing.
    if len(picked) < target:
        for entry in sorted(wrong, key=lambda e: (e['pool'] != 'personal',
                                                  e['id'])):
            if len(picked) >= target:
                break
            pair = _pair_of(entry)
            take(entry, f'자음 혼동 {pair}' if pair else '자음 오류')

    return [{'id': record_id, 'reason': reasons[record_id],
             'pool': by_id[record_id]['pool'],
             'speaker': by_id[record_id]['speaker'],
             'consonant_mistakes': by_id[record_id]['consonant_mistakes']}
            for record_id in picked[:target]]


#: Bumped when `build_shortlist` changes what it would pick. Stored with
#: the list so an old selection is rebuilt rather than silently kept.
SELECTION_VERSION = 3


def index_digest(per_record):
    """Fingerprint of the audit this shortlist belongs to."""
    material = json.dumps(
        {rid: rec.get('binding') or rec.get('audio_pcm_sha256')
         for rid, rec in sorted(per_record.items())},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode()).hexdigest()[:24]


def load_index(path=None, work=None):
    """The audit's per-record view plus the shortlist, for the server.

    Everything is read from, and written under, `work`. Naming an index
    file is naming where its artefacts live: with no `work` given, the
    shortlist is written beside that file, never into the live directory.
    Falling back to the live directory was the defect - a fixture index
    would have rebuilt the reviewer's real shortlist from six fake records.

    A shortlist built from a six-record smoke run must not survive a full
    audit either, so the list records the digest of the index it came from
    and is rebuilt when that moves.
    """
    per_record_path = Path(path) if path else None
    if work is None and per_record_path is not None:
        work = per_record_path.resolve().parent
    work = Path(work or WORK)
    if per_record_path is None:
        per_record_path = work / 'per-record.json'
    if not per_record_path.exists():
        return {'records': {}, 'shortlist': [], 'ready': False,
                'reason': 'per-record.json 이 아직 없습니다 (audit.py --analyse)'}
    per_record = json.loads(per_record_path.read_text(encoding='utf-8'))
    digest = index_digest(per_record)

    shortlist_path = work / 'shortlist.json'
    saved = None
    if shortlist_path.exists():
        try:
            saved = json.loads(shortlist_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            saved = None                     # half-written; rebuild below
    if (isinstance(saved, dict)
            and saved.get('index_digest') == digest
            and saved.get('selection_version') == SELECTION_VERSION
            and all(item['id'] in per_record for item in saved.get('items', []))):
        shortlist = saved['items']
    else:
        shortlist = build_shortlist(per_record)
        payload = {'selection_version': SELECTION_VERSION,
                   'index_digest': digest,
                   'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'items': shortlist}
        # Written whole, then moved into place: a reader never sees half.
        temp = shortlist_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding='utf-8')
        os.replace(temp, shortlist_path)
    return {'records': per_record, 'shortlist': shortlist, 'ready': True,
            'index_digest': digest, 'selection_version': SELECTION_VERSION}


# --- export -----------------------------------------------------------

def export(index, store):
    """Everything a later decision needs, with its provenance attached.

    Only `answered` rows carry a usable label. `unclear` and `unusable` are
    exported too - they are findings - but separately, so nothing downstream
    can mistake "could not tell" for a label. An annotation whose audio or
    baseline no longer matches is marked stale rather than dropped silently.
    """
    records, current = index['records'], store.all_current()
    accepted, set_aside, stale = [], [], []
    for record_id, annotation in sorted(current.items()):
        record = records.get(record_id)
        # What the person meant by the option they picked. For the two
        # bound options it is the text that was on the button; a custom
        # answer is their own words and is never regenerated.
        if annotation['choice'] == 'expected':
            heard = annotation['reference_surface']
            heard_from = 'bound_option_expected_surface'
        elif annotation['choice'] == 'prediction':
            heard = annotation['predicted_text']
            heard_from = 'bound_option_model_prediction'
        else:
            heard = annotation['heard_text']
            heard_from = 'typed_by_reviewer_verbatim'
        entry = {
            'record_id': record_id,
            'pool': annotation['pool'],
            'split': record['pool'] if record else annotation['pool'],
            'speaker': record['speaker'] if record else None,
            'child': record.get('child') if record else None,
            'transcript': record['transcript'] if record else None,
            'expected_surface': annotation['reference_surface'],
            'model_prediction': annotation['predicted_text'],
            'status': annotation['status'],
            'choice': annotation['choice'],
            'choice_label': CHOICES.get(annotation['choice']),
            'heard_form': heard,
            'heard_form_source': heard_from,
            'heard_text': annotation['heard_text'],
            'heard_text_note': '사람이 입력한 그대로입니다. 규칙 변환이나 '
                               '교정을 하지 않았고 앞뒤 공백도 그대로입니다.',
            'note': annotation['note'],
            'flags': annotation['flags'],
            'reviewer': annotation['reviewer'],
            'at': annotation['at'],
            'revision': annotation['revision'],
            'revisions': None,
            'audio_pcm_sha256': annotation['audio_pcm_sha256'],
            'reference_ids': annotation['reference_ids'],
            'baseline_key': annotation['baseline_key'],
            'binding': annotation['binding'],
            # Development and evaluation only. Both pools are excluded from
            # training, and a listening review is not a training label.
            'training_eligible': False,
        }
        reason = is_stale(annotation, record)
        if reason:
            entry['stale_reason'] = {
                'baseline_record_missing': '기준 기록이 더 이상 없습니다',
                'binding_changed': '주석을 단 뒤 음성·목표·기준 예측 중 하나가 '
                                   '바뀌었습니다',
                'audio_changed': '주석을 단 뒤 음성이 바뀌었습니다',
                'baseline_changed': '주석을 단 뒤 기준 예측이 바뀌었습니다',
            }[reason]
            entry['stale_code'] = reason
            stale.append(entry)
        elif annotation['status'] == 'answered':
            accepted.append(entry)
        else:
            set_aside.append(entry)
    return {
        'at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'what_this_is': '사람이 듣고 남긴 검토 결과입니다. 전문가 음성 전사나 '
                        '임상 판정이 아니며, 학습 정답으로 바로 쓸 수 있는 '
                        '자료도 아닙니다.',
        'training_eligible': False,
        'usage': '개발·평가 전용입니다. 직접 녹음 83개와 검증 분할 모두 학습에 '
                 '넣지 않습니다. 이 파일의 모든 항목은 training_eligible=false '
                 '입니다.',
        'counts': {'accepted': len(accepted), 'set_aside': len(set_aside),
                   'stale': len(stale)},
        'accepted': accepted,
        'set_aside': set_aside,
        'stale': stale,
    }
