"""Where the model disagrees with the expected pronunciation, by consonant.

This is an audit of *disagreement with a derived target*, not a measurement
of how a child actually spoke. The reference is the surface form the
project's rules produce from a written transcript; it is not an expert
phonetic annotation, and nothing here should be read as a verdict on
anyone's speech. Which of the disagreements are real mispronunciations,
model errors, label problems or recording problems is exactly what the
listening review exists to find out.

Two steps, separately runnable:

    python audit.py --baseline     # one offline pass, resumable
    python audit.py --analyse      # counting, from the baseline file

The baseline is a durable JSONL: one line per recording, carrying the
identity of everything that decided the prediction (checkpoint, label
contract, decoder, input policy) plus the PCM hash of the audio it ran on.
A reboot in the middle costs only the records not yet written, and an
annotation made later can be checked against that identity instead of
being applied blindly to audio that has since changed.

Counting is done on the reference side. Every consonant in the expected
surface gets a denominator, so a pair that is wrong twice out of three
occurrences is visible next to one that is wrong twice out of nine hundred.
Onset and coda come from decomposing the expected Hangul, never from
guessing at the model's output - an uncomposable predicted jamo has no
syllable position, and is recorded as such.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / 'child_jamo_finetune'))
import compose                                                 # noqa: E402
import engine as engine_module                                 # noqa: E402
import labels                                                  # noqa: E402
import samples as samples_module                               # noqa: E402
from common import ROOT, digest                                # noqa: E402
from python.build.g2p.ko.jamo_ipa import FINALS, INITIALS, MEDIALS  # noqa: E402

WORK = ROOT / '.venv/child-jamo-web-20260916/consonant-review'
BASELINE = WORK / 'baseline.jsonl'
STATE = WORK / 'baseline-state.json'
AUDIT = WORK / 'consonant-audit.json'
SHORTLIST = WORK / 'shortlist.json'

#: The two pools this audit covers. The 2,999-recording test split and the
#: 150k training split are deliberately not here: test is a fixed
#: regression split, and training is where candidates come from, not where
#: errors are counted.
POOLS = ('personal', 'valid')

CONSONANTS = frozenset(INITIALS)
VOWELS = frozenset(MEDIALS)

#: Identity of the computation, recorded with every baseline record.
INPUT_POLICY = ('float32 raw PCM, 16 kHz mono, no amplitude normalisation, '
                'full attention mask, greedy CTC')
#: Bumped when a baseline record's own shape changes. Part of the key, so
#: records written by an older layout are not silently mixed in.
RECORD_VERSION = 2


def identity_of(engine):
    """Everything that decides a prediction, in one dict."""
    return {
        'checkpoint_sha256': engine.identity['checkpoint_sha256'],
        'labels_contract_sha256': engine.identity['labels_contract_sha256'],
        'labels_version': labels.LABELS_VERSION,
        'decode_version': compose.DECODE_VERSION,
        'input_policy': INPUT_POLICY,
        'precision': engine.identity['precision'],
        'record_version': RECORD_VERSION,
    }


def baseline_key(identity):
    """One string that changes whenever a prediction could change.

    Every field is folded in, not just the checkpoint: the same weights
    decoded differently, or fed differently, are a different baseline, and
    an annotation bound to one must not be honoured against the other.
    """
    material = json.dumps({k: identity[k] for k in sorted(identity)
                           if k in IDENTITY_FIELDS}, sort_keys=True,
                          ensure_ascii=False)
    return hashlib.sha256(material.encode()).hexdigest()[:24]


IDENTITY_FIELDS = ('checkpoint_sha256', 'labels_contract_sha256',
                   'labels_version', 'decode_version', 'input_policy',
                   'precision', 'record_version')


def catalog_identity(row):
    """The parts of a catalogue row a prediction is only valid for."""
    return {
        'samples': int(row['samples']),
        'reference_ids': list(row['label_ids'] or []),
        'reference_surface': row['surface'],
        'transcript': row['text'],
        'span_policy': row.get('span_policy'),
    }


def record_binding(record):
    """Identity of one answerable item: this audio, target and prediction.

    Handed to the review page and required back on save. If any of the
    three changed in between, the answer was about something else.
    """
    material = json.dumps({
        'id': record['id'],
        'audio_pcm_sha256': record['audio_pcm_sha256'],
        'reference_ids': record['reference_ids'],
        'hypothesis_ids': record['hypothesis_ids'],
        'baseline_key': record['baseline_key'],
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def validate_record(record, row, audio_pcm_sha256=None):
    """Why a stored prediction no longer describes the current recording.

    The catalogue row's own PCM hash is compared always, not only when the
    caller has paid to read the audio back: a recording that was replaced
    keeps its id, its length and its target, so nothing else here would
    notice. Re-reading the file is the extra, physical check on top.

    An empty list means the record still stands.
    """
    problems = []
    if record.get('record_version') != RECORD_VERSION:
        problems.append('record_version')
    expected = catalog_identity(row)
    for field, value in expected.items():
        if record.get(field) != value:
            problems.append(field)
    catalog_pcm = row.get('pcm_sha256')
    if catalog_pcm and record.get('audio_pcm_sha256') != catalog_pcm:
        problems.append('catalog_pcm_sha256')
    if audio_pcm_sha256 is not None and \
            record.get('audio_pcm_sha256') != audio_pcm_sha256:
        problems.append('audio_pcm_sha256')
    return problems


def _jamo_roles(surface):
    """(jamo, role) for the expected surface, in scored-token order.

    Roles come from decomposing the Hangul: initial -> onset, medial ->
    vowel, final -> coda. Spaces become boundaries, which this project's
    metric excludes, so they are dropped here too.
    """
    out = []
    for char in surface:
        if char == ' ':
            continue
        if not ('가' <= char <= '힣'):
            return None
        n = ord(char) - 0xAC00
        out.append((INITIALS[n // 588], 'onset'))
        out.append((MEDIALS[n // 28 % 21], 'vowel'))
        final = FINALS[n % 28]
        if final:
            out.append((final, 'coda'))
    return out


def reference_roles(row):
    """Roles for a manifest row, or None if they cannot be trusted.

    The decomposition is checked against the label ids the row actually
    carries. If they disagree the row is not silently annotated with a
    guess - it is reported with unknown roles.
    """
    scored = [labels.TOKENS[i] for i in row['label_ids']
              if i != labels.BOUNDARY]
    roles = _jamo_roles(row['surface'] or '')
    if roles is None or [j for j, _ in roles] != scored:
        return [(j, 'unknown') for j in scored], False
    return roles, True


def opcodes(reference_ids, hypothesis_ids):
    """Deterministic edit alignment over scored tokens, with indices.

    rapidfuzz returns one optimal alignment. Where a token repeats, several
    alignments can share the same minimum cost, so which position is called
    "substituted" and which "deleted" is one valid reading among equals.
    The totals are unaffected; the per-position attribution is not always
    unique, and the audit says so.
    """
    from rapidfuzz.distance import Levenshtein
    reference = compose.scored_ids(reference_ids)
    hypothesis = compose.scored_ids(hypothesis_ids)
    return reference, hypothesis, Levenshtein.opcodes(reference, hypothesis)


# --- baseline ---------------------------------------------------------

def load_baseline(expect_key=None, path=None, repair=False):
    """Existing records by id, plus an honest account of the file.

    A crash can leave the final line half-written. That tail is repairable
    - it is provably incomplete - and `repair=True` truncates the file back
    to the last whole record, because appending after it would otherwise
    glue the next record onto the fragment and lose both.

    A *complete* final record with no newline after it is repairable too,
    and just as dangerous: the next append would be glued onto its tail and
    both records would be lost. `repair=True` terminates it.

    A line that fails to parse anywhere *other* than at the end is a
    different thing: something corrupted a finished record. That is
    reported and the file is left exactly as it is - repairing around
    damage nobody has explained yet would write over the evidence.
    """
    path = Path(path or BASELINE)
    report = {'lines': 0, 'kept': 0, 'wrong_key': 0, 'corrupt_middle': [],
              'torn_tail': False, 'missing_final_newline': False,
              'repaired_bytes': 0, 'added_final_newline': False,
              'duplicates': 0}
    records = {}
    if not path.exists():
        return records, report

    raw = path.read_bytes()
    if not raw:
        return records, report
    ends_clean = raw.endswith(b'\n')
    lines = raw.decode('utf-8', errors='replace').split('\n')
    if lines and lines[-1] == '':
        lines.pop()

    good_bytes = 0
    for index, line in enumerate(lines):
        is_last = index == len(lines) - 1
        report['lines'] += 1
        if not line.strip():
            good_bytes += len(line.encode('utf-8')) + 1
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if is_last and not ends_clean:
                # Provably incomplete: nothing was written after it.
                report['torn_tail'] = True
                break
            report['corrupt_middle'].append(index + 1)
            good_bytes += len(line.encode('utf-8')) + 1
            continue
        good_bytes += len(line.encode('utf-8')) + 1
        if expect_key and row.get('baseline_key') != expect_key:
            report['wrong_key'] += 1
            continue
        if row.get('id') in records:
            report['duplicates'] += 1
        records[row['id']] = row
        report['kept'] += 1

    if not ends_clean and not report['torn_tail']:
        # A complete record with no trailing newline: appending would join
        # the next one to it.
        report['missing_final_newline'] = True

    if repair and not report['corrupt_middle']:
        if report['torn_tail']:
            with path.open('r+b') as f:
                f.truncate(good_bytes)
                f.flush()
                os.fsync(f.fileno())
            report['repaired_bytes'] = len(raw) - good_bytes
        elif report['missing_final_newline']:
            with path.open('ab') as f:
                f.write(b'\n')
                f.flush()
                os.fsync(f.fileno())
            report['added_final_newline'] = True
            report['missing_final_newline'] = False
    return records, report


def write_state(payload, path=None):
    """Replace the state file whole, so a reader never sees half of it.

    Progress is written while a job is running, which means this happens
    repeatedly and can be interrupted at any point. Writing in place would
    leave a truncated JSON file behind; writing beside it and renaming
    leaves either the old state or the new one.
    """
    path = Path(path or STATE)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding='utf-8')
    os.replace(temp, path)


def pool_coverage(records, catalog, pools=POOLS):
    """How much of each pool is covered by records that still stand.

    Not "how many stored records name this pool": the question is how many
    of the recordings the catalogue holds *now* have a prediction that is
    still about them. A row whose audio or target moved on, and whose
    recomputation then failed, leaves its old record in the file - counting
    that as covered would report a finished pool that is not finished.
    """
    coverage = {}
    for pool in pools:
        rows = catalog.rows[pool]
        have = stale = 0
        for row in rows:
            record = records.get(row['id'])
            if record is None:
                continue
            if validate_record(record, row):
                stale += 1
                continue
            have += 1
        coverage[pool] = {'expected': len(rows), 'have': have,
                          'stale_or_unusable': stale,
                          'complete': have == len(rows)}
    return coverage


def run_baseline(limit=None, pools=POOLS, verify_audio=False):
    """One offline pass over the audited pools, resumable and counted.

    Resuming is not "the id is already there". A stored prediction only
    describes the recording it was made from, so before it is kept the
    row's sample count, target, transcript, crop policy *and* the PCM hash
    the catalogue carries are compared with the catalogue as it stands now.
    `--verify-audio` adds the physical check on top: the file is read back
    and hashed, which catches audio that changed underneath a manifest that
    did not. Anything that no longer matches is recomputed.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    # Before a model is loaded and before any repair touches the file: a
    # finished record that no longer parses is not something to append
    # after, and not something to quietly rewrite around either.
    _, precheck = load_baseline()
    if precheck['corrupt_middle']:
        raise SystemExit(
            f"baseline.jsonl 의 {precheck['corrupt_middle'][:5]} 번째 줄이 "
            f"손상되었습니다. 파일을 고치지도, 덧붙이지도 않고 멈춥니다.")

    catalog = samples_module.Catalog()
    engine = engine_module.Engine()
    engine.load()
    identity = identity_of(engine)
    key = baseline_key(identity)
    done, file_report = load_baseline(key, repair=True)

    wanted = []
    for pool in pools:
        rows = catalog.rows[pool]
        wanted.extend((pool, row) for row in (rows[:limit] if limit else rows))
    counts = {'requested': len(wanted), 'already_done': 0, 'attempted': 0,
              'succeeded': 0, 'failed': 0,
              'wrong_identity_dropped': file_report['wrong_key'],
              'revalidated': 0, 'stale_recomputed': 0}
    failures, invalidated = [], []
    started = time.monotonic()

    def snapshot(status, index=0):
        """The state file as it stands right now.

        Written before the first recording and again as the job goes, so
        the file on disk says "running" while it is running. Leaving the
        previous run's finished state in place would make a crashed or
        half-finished job look complete to anything that reads it.
        """
        return {'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'status': status,
                'pools': list(pools), 'limit': limit,
                'verify_audio': verify_audio, 'baseline_key': key, **identity,
                'counts': dict(counts), 'failures': failures,
                'invalidated_on_resume': invalidated,
                'file_report': file_report,
                'progress': {'processed': index, 'requested': len(wanted)}}

    write_state(snapshot('running'))
    try:
        with BASELINE.open('a', encoding='utf-8') as sink:
            for index, (pool, row) in enumerate(wanted, 1):
                existing = done.get(row['id'])
                if existing is not None:
                    audio_sha = None
                    if verify_audio:
                        try:
                            audio_sha = engine_module.pcm_sha256(
                                engine_module.read_row(row))
                        except Exception as exc:
                            audio_sha = f'unreadable: {exc}'
                    problems = validate_record(existing, row, audio_sha)
                    counts['revalidated'] += 1
                    if not problems:
                        counts['already_done'] += 1
                        continue
                    # The recording or its target moved on; the old
                    # prediction is not evidence about the new one.
                    invalidated.append({'id': row['id'], 'problems': problems})
                    counts['stale_recomputed'] += 1
                counts['attempted'] += 1
                try:
                    waveform = engine_module.read_row(row)
                    pcm = engine_module.pcm_sha256(waveform)
                    out = engine.transcribe(waveform, pcm_sha256=None,
                                            use_cache=False)
                    record = {
                        'id': row['id'], 'pool': pool, 'split': row['split'],
                        'wav': row['wav'],
                        'speaker': row['speaker'],
                        'child': row.get('child'),
                        'age': row['age'], 'style': row['style'],
                        'seconds': round(row['seconds'], 3),
                        'audio_pcm_sha256': pcm,
                        **catalog_identity(row),
                        'hypothesis_ids': out['token_ids'],
                        'predicted_text': out['text'],
                        'predicted_jamo': out['jamo'],
                        'orphan_jamo': out['orphan_jamo'],
                        'baseline_key': key, **identity,
                        'at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    record['metrics'] = compose.compare(row['label_ids'],
                                                        out['token_ids'])
                    record['binding'] = record_binding(record)
                    sink.write(json.dumps(record, ensure_ascii=False) + '\n')
                    sink.flush()
                    os.fsync(sink.fileno())      # survive a power cut
                    done[row['id']] = record
                    counts['succeeded'] += 1
                except Exception as exc:
                    counts['failed'] += 1
                    failures.append({'id': row['id'],
                                     'reason': f'{type(exc).__name__}: {exc}'})
                if index % 100 == 0:
                    write_state(snapshot('running', index))
                    print(f'  {index}/{len(wanted)}'
                          f' (새로 {counts["succeeded"]}, 실패 {counts["failed"]})',
                          flush=True)
    except BaseException as exc:
        # A Ctrl-C or a crash leaves the records already written on disk,
        # and says so in the state file rather than letting the previous
        # run's "complete" stand for work that stopped in the middle.
        counts['elapsed_seconds'] = round(time.monotonic() - started, 1)
        counts['job_complete'] = False
        interrupted = snapshot(
            'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed')
        interrupted['stopped_by'] = f'{type(exc).__name__}: {exc}'
        write_state(interrupted)
        raise
    finally:
        engine.model = engine.head = None
        import torch
        torch.cuda.empty_cache()

    counts['elapsed_seconds'] = round(time.monotonic() - started, 1)
    # Did *this job* finish what it asked for?
    counts['job_complete'] = (counts['failed'] == 0 and
                              counts['already_done'] + counts['succeeded']
                              == counts['requested'])

    # Separately: is the pool actually covered? A `--limit 6` smoke run
    # finishes its job and covers almost nothing, and the difference has to
    # survive into the report and the page.
    final, _ = load_baseline(key)
    coverage = pool_coverage(final, catalog, pools)
    state = snapshot('complete', len(wanted))
    state.update({
        'pool_coverage': coverage,
        'pools_complete': all(c['complete'] for c in coverage.values()),
        'note': 'counts.job_complete 는 이번 실행이 요청한 만큼 끝났는가, '
                'pools_complete 는 풀 전체가 덮였는가입니다. --limit 로 '
                '일부만 돌린 실행은 job 은 끝나도 풀은 덮이지 않습니다. '
                'pool_coverage.have 는 지금 카탈로그에 있는 녹음 중 아직 '
                '유효한 기록을 가진 것만 셉니다.'})
    write_state(state)
    print(f"요청 {counts['requested']} / 기존 {counts['already_done']} / "
          f"시도 {counts['attempted']} / 성공 {counts['succeeded']} / "
          f"실패 {counts['failed']} / 이번 작업 완료 {counts['job_complete']}")
    for pool, entry in coverage.items():
        print(f"  {pool}: {entry['have']}/{entry['expected']} "
              f"{'덮음' if entry['complete'] else '미완'}")
    if invalidated:
        print(f"  기준이 달라져 다시 계산한 기록 {len(invalidated)}개:",
              invalidated[:3])
    if file_report['corrupt_middle']:
        print('  손상된 줄(끝이 아닌 곳):', file_report['corrupt_middle'][:5])
    if failures:
        for f in failures[:5]:
            print('  실패:', f)
    return state


def migrate_baseline(verify_audio=True):
    """Bring existing records onto the current record shape, without rerunning.

    The predictions are valid; what they were missing was a complete
    identity. So each record is checked against the catalogue as it stands
    - sample count, target, transcript, crop policy, and by default the PCM
    hash read back off disk - and only then restamped with the full key and
    an item binding. Anything that fails is left out and reported, to be
    recomputed by a normal `--baseline` run.

    Restamping is the dangerous part, so it is never done to a record that
    would be *changed* by it. A record only gains identity fields it does
    not already carry; one that carries a different labels version, decoder,
    input policy or precision was computed another way, and giving it this
    run's identity would be inventing a provenance. Whole-file refusal,
    because a file holding two computations is not a baseline.

    No model is loaded and no inference happens here.
    """
    records, file_report = load_baseline(repair=True)
    if file_report['corrupt_middle']:
        return {'migrated': 0, 'refused': True,
                'reason': f"baseline.jsonl 의 {file_report['corrupt_middle'][:5]} "
                          f"번째 줄이 손상되었습니다"}
    if not records:
        return {'migrated': 0, 'note': 'baseline.jsonl 이 비어 있습니다'}
    catalog = samples_module.Catalog()

    #: The identity a migrated record will carry. Everything that can be
    #: read off disk or off the current code is taken from there rather
    #: than from whichever record happened to be read first; `precision`
    #: has no source outside the records, so it must be unanimous.
    identity = {
        'checkpoint_sha256': digest(engine_module.CHECKPOINT),
        'labels_contract_sha256': labels.contract_sha256(),
        'labels_version': labels.LABELS_VERSION,
        'decode_version': compose.DECODE_VERSION,
        'input_policy': INPUT_POLICY,
        'record_version': RECORD_VERSION,
    }
    recorded = {r.get('checkpoint_sha256') for r in records.values()}
    if recorded != {identity['checkpoint_sha256']}:
        return {'migrated': 0, 'refused': True,
                'reason': f'기록된 체크포인트가 현재 파일'
                          f"({identity['checkpoint_sha256'][:16]})과 맞지 "
                          f'않습니다: {sorted(str(c)[:16] for c in recorded)}'}
    precisions = {r['precision'] for r in records.values() if r.get('precision')}
    if len(precisions) > 1:
        return {'migrated': 0, 'refused': True,
                'reason': f'계산 정밀도가 섞여 있습니다: {sorted(precisions)}'}
    if not precisions:
        return {'migrated': 0, 'refused': True,
                'reason': '기록에 계산 정밀도가 없어 신원을 완성할 수 없습니다'}
    identity['precision'] = precisions.pop()

    # Every record must have been computed this same way. A field a legacy
    # record simply does not carry can be filled in; a field it carries
    # with a different value cannot be overwritten.
    conflicting = []
    for record_id, record in sorted(records.items()):
        differing = [field for field, value in identity.items()
                     if field != 'record_version'
                     and field in record and record[field] != value]
        version = record.get('record_version')
        if version is not None and version > RECORD_VERSION:
            differing.append('record_version')
        if differing:
            conflicting.append({'id': record_id, 'problems': differing})
    if conflicting:
        return {'migrated': 0, 'refused': True,
                'conflicting': conflicting[:10],
                'reason': f'계산 신원이 다른 기록이 {len(conflicting)}건 섞여 '
                          f'있습니다 (예: {conflicting[0]})'}

    key = baseline_key(identity)
    migrated, rejected = [], []
    for record_id, record in sorted(records.items()):
        row = catalog.get(record_id)
        if row is None:
            rejected.append({'id': record_id, 'problems': ['not_in_catalog']})
            continue
        audio_sha = None
        if verify_audio:
            try:
                audio_sha = engine_module.pcm_sha256(engine_module.read_row(row))
            except Exception as exc:
                rejected.append({'id': record_id,
                                 'problems': [f'unreadable: {exc}']})
                continue
        expected = catalog_identity(row)
        # The catalogue fields are filled in where a legacy record lacks
        # them, never corrected where it has them: a record whose stored
        # target, transcript, length or crop policy disagrees with the row
        # was scored against something else, and overwriting the field
        # would make the disagreement disappear instead of reporting it.
        drifted = [field for field, value in expected.items()
                   if field in record and record[field] != value]
        if drifted:
            rejected.append({'id': record_id, 'problems': drifted})
            continue
        updated = {**record, **expected, **identity, 'baseline_key': key}
        problems = validate_record(updated, row, audio_sha)
        # The stored prediction must have been made from this audio.
        if audio_sha is not None and record.get('audio_pcm_sha256') != audio_sha:
            problems.append('audio_pcm_sha256_changed')
        if problems:
            rejected.append({'id': record_id, 'problems': problems})
            continue
        updated['binding'] = record_binding(updated)
        updated['migrated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        migrated.append(updated)

    backup = BASELINE.with_suffix('.jsonl.pre-migration')
    if BASELINE.exists() and not backup.exists():
        backup.write_bytes(BASELINE.read_bytes())
    temp = BASELINE.with_suffix('.jsonl.tmp')
    with temp.open('w', encoding='utf-8') as sink:
        for record in migrated:
            sink.write(json.dumps(record, ensure_ascii=False) + '\n')
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temp, BASELINE)

    coverage = pool_coverage({r['id']: r for r in migrated}, catalog, POOLS)
    state = {'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'status': 'complete',
             'pools': list(POOLS), 'limit': None, 'verify_audio': verify_audio,
             'baseline_key': key, **identity,
             'counts': {'requested': len(records), 'already_done': len(migrated),
                        'attempted': 0, 'succeeded': 0,
                        'failed': len(rejected), 'revalidated': len(records),
                        'stale_recomputed': 0, 'wrong_identity_dropped': 0,
                        'job_complete': not rejected},
             'failures': rejected, 'invalidated_on_resume': [],
             'file_report': file_report, 'pool_coverage': coverage,
             'pools_complete': all(c['complete'] for c in coverage.values()),
             'migrated_from': str(backup),
             'note': '추론을 다시 돌리지 않고, 기존 예측을 현재 카탈로그·음성과 '
                     '대조해 신원만 갱신했습니다.'}
    write_state(state)
    return {'migrated': len(migrated), 'rejected': rejected,
            'baseline_key': key, 'coverage': coverage,
            'verify_audio': verify_audio, 'backup': str(backup)}


def current_key():
    """The baseline key the state file says the records were written with."""
    if not STATE.exists():
        return None
    return json.loads(STATE.read_text(encoding='utf-8')).get('baseline_key')


def cross_check_personal():
    """Do the recorded personal predictions match the earlier run?

    Both the text and the identity are compared. Matching text produced by
    a different checkpoint or decoder is not a reproduction, and zero
    records compared is not a success - it is a missing check.
    """
    prior = ROOT / '.venv/child-jamo-web-20260916/personal-recordings/validation.json'
    if not prior.exists():
        return {'checked': 0, 'reproduces': False,
                'note': 'validation.json 이 없어 확인하지 못했습니다'}
    saved = json.loads(prior.read_text(encoding='utf-8'))
    key = current_key()
    records, report = load_baseline(key)
    same_identity = (
        saved.get('checkpoint_sha256') ==
        next(iter(records.values()), {}).get('checkpoint_sha256')
        and saved.get('decode_version') ==
        next(iter(records.values()), {}).get('decode_version')
        and saved.get('labels_contract_sha256') ==
        next(iter(records.values()), {}).get('labels_contract_sha256'))
    agree, differ, missing = 0, [], []
    for clip in saved['clips']['results']:
        record = records.get(clip['id'])
        if record is None:
            missing.append(clip['id'])
            continue
        if record['predicted_jamo'] == clip['predicted_jamo']:
            agree += 1
        else:
            differ.append({'id': clip['id'], 'now': record['predicted_jamo'],
                           'before': clip['predicted_jamo']})
    checked = agree + len(differ)
    return {
        'source': str(prior),
        'baseline_key': key,
        'saved_checkpoint': saved['checkpoint_sha256'],
        'saved_decode_version': saved['decode_version'],
        'identity_matches': bool(same_identity),
        'expected': len(saved['clips']['results']),
        'checked': checked, 'agree': agree, 'differ': differ,
        'missing_from_baseline': missing,
        # All three must hold: same computation, everything compared, and
        # nothing different.
        'reproduces': bool(same_identity and checked == len(
            saved['clips']['results']) and not differ),
    }


# --- analysis ---------------------------------------------------------

def analyse_record(record):
    """Per-record consonant events, with reference-side syllable roles."""
    roles, roles_trusted = reference_roles({
        'label_ids': record['reference_ids'],
        'surface': record['reference_surface']})
    reference, hypothesis, ops = opcodes(record['reference_ids'],
                                         record['hypothesis_ids'])
    events = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'equal':
            for offset in range(i2 - i1):
                jamo, role = roles[i1 + offset]
                if jamo in CONSONANTS:
                    events.append({'op': 'equal', 'reference': jamo,
                                   'hypothesis': jamo, 'role': role})
            continue
        if tag in ('replace', 'delete'):
            for offset in range(i2 - i1):
                jamo, role = roles[i1 + offset]
                if jamo not in CONSONANTS:
                    # A consonant predicted where the target has a vowel is
                    # a real mistake, but it has no reference consonant, so
                    # it cannot share this audit's denominator. Counted on
                    # its own rather than dropped or quietly folded in.
                    if tag == 'replace' and j1 + offset < j2:
                        got = labels.TOKENS[hypothesis[j1 + offset]]
                        if got in CONSONANTS:
                            events.append({'op': 'consonant_for_vowel',
                                           'reference': jamo,
                                           'hypothesis': got, 'role': role})
                    continue
                if tag == 'replace' and j1 + offset < j2:
                    got = labels.TOKENS[hypothesis[j1 + offset]]
                    events.append({'op': 'substitute', 'reference': jamo,
                                   'hypothesis': got, 'role': role,
                                   'hypothesis_is_consonant': got in CONSONANTS})
                else:
                    events.append({'op': 'delete', 'reference': jamo,
                                   'hypothesis': None, 'role': role})
        if tag in ('replace', 'insert'):
            start = j1 + (i2 - i1) if tag == 'replace' else j1
            for position in range(max(start, j1), j2):
                got = labels.TOKENS[hypothesis[position]]
                if got in CONSONANTS:
                    # An inserted jamo has no place in the expected word, so
                    # it has no onset/coda role. Guessing one from the
                    # model's own output would invent structure.
                    events.append({'op': 'insert', 'reference': None,
                                   'hypothesis': got, 'role': 'none'})
    return events, roles_trusted


def analyse(records):
    pools = defaultdict(lambda: {
        'recordings': 0, 'roles_untrusted': 0,
        'reference_consonants': 0, 'correct': 0,
        'substitutions': 0, 'deletions': 0, 'insertions': 0,
        'consonant_for_vowel_substitutions': 0,
        'errors': 0, 'reference_jamo': 0, 'exact': 0,
        'per_consonant': defaultdict(lambda: {
            'reference_total': 0, 'correct': 0, 'substituted': 0,
            'deleted': 0,
            # Denominators per syllable position, not just error counts:
            # an onset that occurs 200 times and a coda that occurs 46
            # cannot be compared on raw counts.
            'by_role': defaultdict(lambda: {
                'reference_total': 0, 'correct': 0, 'substituted': 0,
                'deleted': 0}),
            'to': defaultdict(int)}),
        'by_role': defaultdict(lambda: {
            'reference_total': 0, 'correct': 0, 'substituted': 0,
            'deleted': 0}),
        'pairs': defaultdict(int),
        'pairs_by_role': defaultdict(int),
        'inserted': defaultdict(int),
        'recordings_with_consonant_error': 0,
        'by_speaker': defaultdict(lambda: {'recordings': 0, 'errors': 0}),
    })
    per_record = {}
    for record in records.values():
        pool = pools[record['pool']]
        events, trusted = analyse_record(record)
        pool['recordings'] += 1
        pool['roles_untrusted'] += 0 if trusted else 1
        pool['errors'] += record['metrics']['errors']
        pool['reference_jamo'] += record['metrics']['reference_jamo']
        pool['exact'] += 1 if record['metrics']['exact'] else 0
        speaker = pool['by_speaker'][record['speaker']]
        speaker['recordings'] += 1

        mistakes = 0
        for event in events:
            stats = role_stats = None
            if event['op'] == 'consonant_for_vowel':
                pool['consonant_for_vowel_substitutions'] += 1
                mistakes += 1
                continue
            if event['reference']:
                stats = pool['per_consonant'][event['reference']]
                role_stats = stats['by_role'][event['role']]
                stats['reference_total'] += 1
                role_stats['reference_total'] += 1
                pool['by_role'][event['role']]['reference_total'] += 1
                pool['reference_consonants'] += 1
            if event['op'] == 'equal':
                stats['correct'] += 1
                role_stats['correct'] += 1
                pool['by_role'][event['role']]['correct'] += 1
                pool['correct'] += 1
                continue
            mistakes += 1
            if event['op'] == 'substitute':
                pool['substitutions'] += 1
                stats['substituted'] += 1
                role_stats['substituted'] += 1
                pool['by_role'][event['role']]['substituted'] += 1
                stats['to'][event['hypothesis']] += 1
                pool['pairs'][f"{event['reference']}>{event['hypothesis']}"] += 1
                pool['pairs_by_role'][
                    f"{event['reference']}>{event['hypothesis']}@{event['role']}"] += 1
            elif event['op'] == 'delete':
                pool['deletions'] += 1
                stats['deleted'] += 1
                role_stats['deleted'] += 1
                pool['by_role'][event['role']]['deleted'] += 1
                pool['pairs'][f"{event['reference']}>(없음)"] += 1
                pool['pairs_by_role'][
                    f"{event['reference']}>(없음)@{event['role']}"] += 1
            else:
                pool['insertions'] += 1
                pool['inserted'][event['hypothesis']] += 1
        if mistakes:
            pool['recordings_with_consonant_error'] += 1
            speaker['errors'] += mistakes
        per_record[record['id']] = {
            'id': record['id'], 'pool': record['pool'],
            'speaker': record['speaker'], 'child': record.get('child'),
            'seconds': record['seconds'],
            'transcript': record['transcript'],
            'reference_surface': record['reference_surface'],
            'predicted_text': record['predicted_text'],
            'consonant_events': [e for e in events if e['op'] != 'equal'],
            'consonant_mistakes': mistakes,
            'metrics': record['metrics'],
            'roles_trusted': trusted,
            # Enough identity for the review to re-verify an item against
            # the catalogue before it is played, answered or exported.
            'samples': record.get('samples'),
            'reference_ids': record.get('reference_ids'),
            'span_policy': record.get('span_policy'),
            'record_version': record.get('record_version'),
            'audio_pcm_sha256': record['audio_pcm_sha256'],
            'baseline_key': record['baseline_key'],
            'binding': record.get('binding'),
        }
    return pools, per_record


def _plain(obj):
    if isinstance(obj, defaultdict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    return obj


def run_analysis():
    """Count, from one coherent baseline only.

    The key is taken from the state file and enforced: a file holding two
    runs from different checkpoints would otherwise average them into a
    single number that describes no model at all.
    """
    key = current_key()
    if key is None:
        raise SystemExit('baseline-state.json 이 없습니다. --baseline 먼저 실행')
    records, file_report = load_baseline(key)
    if not records:
        raise SystemExit('이 기준선에 해당하는 기록이 없습니다. --baseline 실행')
    if file_report['corrupt_middle']:
        raise SystemExit(
            f"baseline.jsonl 의 {file_report['corrupt_middle'][:5]} 번째 줄이 "
            f"손상되었습니다. 집계를 멈춥니다.")
    state = json.loads(STATE.read_text(encoding='utf-8'))
    # Only the pools this audit is about, as the catalogue defines them now.
    catalog = samples_module.Catalog()
    known = {row['id'] for pool in POOLS for row in catalog.rows[pool]}
    scoped = {rid: rec for rid, rec in records.items()
              if rid in known and rec.get('pool') in POOLS}
    dropped = len(records) - len(scoped)

    pools, per_record = analyse(scoped)
    report = {
        'at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'baseline_key': key,
        'baseline_records': len(scoped),
        'baseline_records_outside_scope': dropped,
        'baseline_file_report': file_report,
        'pool_coverage': state.get('pool_coverage'),
        'pools_complete': state.get('pools_complete'),
        'baseline_state': state,
        'convention': {
            'jer': '학습과 같은 방식: 띄어쓰기 토큰 제외, 편집 횟수를 기대 '
                   '발음 자모 수로 나눔. 100%를 넘을 수 있고 자르지 않음.',
            'roles': '초성/받침은 기대 발음 한글을 분해해서 얻습니다. 초성은 '
                     '단어 첫소리(어두)가 아니라 모든 음절의 첫 자리입니다. '
                     '모델이 낸 자모로 위치를 추측하지 않으며, 덧붙은 자모는 '
                     '원래 자리가 없어 none 입니다.',
            'empty_onset': '초성 ㅇ 은 소리가 없는 빈 초성 표시입니다. 라벨 '
                           '체계상 하나의 토큰이라 여기 집계에도 들어가지만, '
                           '초성 ㅇ 오류를 "자음을 못 냈다" 로 읽으면 안 됩니다. '
                           '받침 ㅇ 은 실제 소리라 다릅니다.',
            'alignment': '편집 정렬은 결정적입니다. 같은 자모가 반복되면 같은 '
                         '비용의 정렬이 여럿일 수 있는데, 이때 총 편집 거리는 '
                         '같지만 **치환/삭제/삽입의 배분과 자음·위치 귀속은 '
                         '달라질 수 있습니다.**',
            'rates': 'consonant_sub_del_rate 는 (치환+삭제)을 기준 자음 수로 '
                     '나눈 값이며 덧붙음은 들어 있지 않습니다. 덧붙음은 기준 '
                     '쪽에 자리가 없어 같은 분모를 쓸 수 없으므로 따로 '
                     '셉니다(consonant_mistakes_including_insertions).',
            'scope': '기준 자음 감사입니다. 모음 자리에 자음이 나온 치환은 '
                     '기준 쪽이 자음이 아니므로 여기 분모에 없고, '
                     'consonant_for_vowel_substitutions 로 따로 셉니다.',
            'phonology': '기준은 전사에 음운 규칙을 적용한 기대 발음입니다. '
                         "'씩 넣고 -> 씽 너코' 같은 변동은 기준 자체가 그러하므로 "
                         '오류로 세지 않습니다.',
            'meaning': '이것은 기대 발음과의 불일치 감사이지, 아이 발음의 '
                       '정확성 판정이 아닙니다.',
        },
        'pools': {},
    }
    def rate(stats):
        total = stats['reference_total']
        return (round((stats['substituted'] + stats['deleted']) / total, 4)
                if total else None)

    for name, pool in pools.items():
        plain = _plain(pool)
        for consonant, stats in plain['per_consonant'].items():
            stats['sub_del_rate'] = rate(stats)
            for role, role_stats in stats['by_role'].items():
                role_stats['sub_del_rate'] = rate(role_stats)
        for role, role_stats in plain['by_role'].items():
            role_stats['sub_del_rate'] = rate(role_stats)
        plain['jer'] = (round(plain['errors'] / plain['reference_jamo'], 5)
                        if plain['reference_jamo'] else None)
        # Named for exactly what it divides: substitutions and deletions of
        # reference consonants, over reference consonants. Insertions are
        # not in it - they have no reference slot to be counted against.
        plain['consonant_sub_del_rate'] = (
            round((plain['substitutions'] + plain['deletions'])
                  / plain['reference_consonants'], 5)
            if plain['reference_consonants'] else None)
        plain['consonant_mistakes_including_insertions'] = (
            plain['substitutions'] + plain['deletions'] + plain['insertions']
            + plain['consonant_for_vowel_substitutions'])
        plain['rate_note'] = (
            'consonant_sub_del_rate 의 분모는 기준 자음 수입니다. 덧붙은 자음과 '
            '모음 자리에 나온 자음은 분모가 없어 따로 셉니다.')
        plain['top_pairs'] = sorted(plain['pairs'].items(),
                                    key=lambda kv: -kv[1])[:25]
        plain['top_pairs_by_role'] = sorted(plain['pairs_by_role'].items(),
                                            key=lambda kv: -kv[1])[:25]
        # Rate as well as count: a pair wrong twice out of three matters
        # differently from one wrong twice out of nine hundred.
        plain['worst_by_rate'] = sorted(
            ((c, s['sub_del_rate'], s['reference_total'])
             for c, s in plain['per_consonant'].items()
             if s['reference_total'] >= 5 and s['sub_del_rate']),
            key=lambda item: -item[1])[:15]
        plain['worst_by_rate_note'] = (
            '적은 횟수에서 나온 비율은 순위로 읽지 마시고 조사 후보로만 '
            '보십시오. reference_total 을 함께 적어 둡니다.')
        report['pools'][name] = plain

    AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                     encoding='utf-8')
    (WORK / 'per-record.json').write_text(
        json.dumps(per_record, ensure_ascii=False, indent=2), encoding='utf-8')

    for name, pool in report['pools'].items():
        coverage = (report['pool_coverage'] or {}).get(name, {})
        print(f"\n[{name}] 녹음 {pool['recordings']}개"
              f" (풀 {coverage.get('have', '?')}/{coverage.get('expected', '?')},"
              f" 덮음={coverage.get('complete')})"
              f", 자모 오류율 {(pool['jer'] or 0) * 100:.2f}%,"
              f" 완전 일치 {pool['exact']}개")
        print(f"  기준 자음 {pool['reference_consonants']}개 중 맞음 "
              f"{pool['correct']} / 바뀜 {pool['substitutions']} / 빠짐 "
              f"{pool['deletions']} "
              f"→ 치환+삭제 비율 {(pool['consonant_sub_del_rate'] or 0) * 100:.2f}%")
        print(f"  분모 밖: 덧붙음 {pool['insertions']}, 모음 자리에 자음 "
              f"{pool['consonant_for_vowel_substitutions']}")
        for role, stats in sorted(pool['by_role'].items()):
            if stats['reference_total']:
                print(f"    {role}: 기준 {stats['reference_total']} 중 "
                      f"치환 {stats['substituted']} / 삭제 {stats['deleted']} "
                      f"= {(stats['sub_del_rate'] or 0) * 100:.2f}%")
        print(f"  자음 오류가 있는 녹음 {pool['recordings_with_consonant_error']}개")
        print('  많이 나온 짝:', pool['top_pairs'][:8])
        print('  비율 높은 자음 (조사 후보):', pool['worst_by_rate'][:6])
    print('\n기록:', AUDIT)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--analyse', action='store_true')
    parser.add_argument('--cross-check', action='store_true')
    parser.add_argument('--migrate', action='store_true',
                        help='기존 예측을 재검증해 신원만 갱신 (추론 없음)')
    parser.add_argument('--limit', type=int, default=None,
                        help='각 풀에서 앞 N개만 (점검용)')
    parser.add_argument('--pools', default=','.join(POOLS))
    parser.add_argument('--verify-audio', action='store_true',
                        help='이어서 돌 때 저장된 기록의 PCM 해시까지 대조')
    parser.add_argument('--no-verify-audio', action='store_true',
                        help='--migrate 에서 음성 대조를 건너뜁니다 (권장하지 않음)')
    args = parser.parse_args()
    if not (args.baseline or args.analyse or args.cross_check or args.migrate):
        parser.error('--baseline / --migrate / --analyse / --cross-check 중 하나')
    if args.migrate:
        out = migrate_baseline(verify_audio=not args.no_verify_audio)
        print('이관:', out.get('migrated'), '건,',
              '거부', len(out.get('rejected') or []), '건',
              '(음성 대조', out.get('verify_audio'), ')')
        for entry in (out.get('rejected') or [])[:5]:
            print('  거부:', entry)
        if out.get('refused'):
            raise SystemExit(f"이관하지 않았습니다: {out['reason']}")
        print('  풀 덮음:', out.get('coverage'))
    if args.baseline:
        run_baseline(limit=args.limit, pools=tuple(args.pools.split(',')),
                     verify_audio=args.verify_audio)
    if args.cross_check:
        check = cross_check_personal()
        (WORK / 'personal-cross-check.json').write_text(
            json.dumps(check, ensure_ascii=False, indent=2), encoding='utf-8')
        print('직접 녹음 재현 확인:', check['agree'], '/', check['checked'],
              '일치, 재현', check['reproduces'])
    if args.analyse:
        run_analysis()


if __name__ == '__main__':
    main()
