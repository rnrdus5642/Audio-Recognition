"""The personally recorded child audio, as two browsable datasets.

Two things live on the Desktop under one source folder, and they are very
different kinds of evidence, so they are kept as two separate datasets
rather than one mixed list:

  personal   83 word-level clips that were cut from two of the recordings
             and labelled from the supplied HWP transcript. These have a
             human-written reference, so they can be scored.
  original   the four recordings as delivered, whole. There is no
             time-aligned transcript for them, so they are transcribed and
             shown, and nothing is scored.

Everything here is read-only with respect to the Desktop folder. The only
files written are a canonical 16 kHz WAV per original recording, under the
app's own work directory, so that playback and inference are literally the
same array.

The reference for a clip is the transcript `text`. `expected_answer` is the
answer the elicitation was fishing for and is recorded on eight rows only;
it is carried through as a separate annotation and never substituted for
what the child was transcribed as saying.
"""
from math import ceil
from pathlib import Path
import json
import os
import sys
import threading

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'child_jamo_finetune'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels
from common import ROOT, digest
from workspace_paths import RECORDINGS, physical_path
#: The same hash the archive reader and the result cache are keyed on.
#: Defining a second one here would let the two drift apart silently.
from engine import pcm_sha256

sys.path.insert(0, str(ROOT))
from python.runtime.audio import load_audio_16k_mono

SAMPLE_RATE = 16000

#: The source folder, overridable so this is not a path literal wired into
#: the app. Nothing outside it is ever opened, and nothing in it is written.
SOURCE_ROOT = physical_path(os.environ.get(
    'CHILD_PERSONAL_AUDIO_ROOT', RECORDINGS / '아동오디오'))
#: The labelling run that produced the clips. One directory name, not 83
#: file paths: the clip list comes from that run's own labels.jsonl.
LABELLED_DIR = '단어별_라벨링_20260906'

WORK = ROOT / '.venv/child-jamo-web-20260916/personal-recordings'
CANONICAL = WORK / 'canonical'

#: Bounded chunks for the long recordings. A whole 115 s recording as one
#: tensor is not something this model should be asked to hold, and the
#: point of the page is to see what it makes of every second of it, so the
#: recording is cut into chunks that together cover every sample exactly
#: once - never sampled, never trimmed.
CHUNK_TARGET_SECONDS = 12.0
CHUNK_MAX_SECONDS = 15.0
#: How far a chunk boundary may be nudged to land in a quiet spot.
BOUNDARY_SEARCH_SECONDS = 1.0
BOUNDARY_FRAME = 320                      # 20 ms
#: Above this, the chosen boundary is not silence and may sit inside a
#: word. That is reported rather than hidden - the segment text on either
#: side of such a cut can be a fragment.
BOUNDARY_QUIET_DBFS = -45.0

_lock = threading.Lock()


def available():
    """True when the source folder is where this expects it."""
    return (SOURCE_ROOT / LABELLED_DIR / 'labels.jsonl').exists()


# --- provenance ------------------------------------------------------

def _within(path, root):
    """Resolve a manifest-relative path and refuse anything outside root."""
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'{path}: 원본 폴더 밖을 가리킵니다')
    return resolved


# --- the 83 labelled clips -------------------------------------------

def _clip_row(raw, root):
    path = _within(raw['audio'], root)
    data, rate = sf.read(path, dtype='float32', always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(f'{raw["id"]}: {rate}Hz, 16000Hz 가 아닙니다')
    if data.ndim == 2:
        data = data.mean(axis=1)
    waveform = np.ascontiguousarray(data, dtype=np.float32)
    if len(waveform) != raw['num_samples']:
        raise ValueError(f'{raw["id"]}: {len(waveform)} samples, 라벨은 '
                         f'{raw["num_samples"]}')
    file_sha = digest(path)
    if file_sha != raw['audio_sha256']:
        raise ValueError(f'{raw["id"]}: 파일 해시가 라벨과 다릅니다')

    # The transcript label, put through the same target conversion the
    # training data used. A row it cannot express is kept and marked, not
    # dropped and not rewritten.
    text, surface, ids = labels.targets(raw['text'])
    state, reason = 'transcript', None
    if ids is None:
        state, reason, surface, text = 'unsupported', text, None, raw['text']

    return {
        'id': f'personal:{raw["id"]}',
        'split': 'personal',
        'kind': 'personal_clip',
        'reader': 'personal',
        'wav': raw['audio'],
        'path': str(path),
        'speaker': raw['speaker'],
        'child': raw['speaker'],
        # Nobody recorded these children's ages, so none is shown.
        'age': 'unknown',
        'style': 'word',
        'text': text,
        'transcript_raw': raw['text'],
        'surface': surface,
        'label_ids': ids,
        'reference_state': state,
        'reference_reason': reason,
        'samples': int(len(waveform)),
        'seconds': len(waveform) / SAMPLE_RATE,
        'pcm_sha256': pcm_sha256(waveform),
        'file_sha256': file_sha,
        'source': {
            'dataset': '직접 녹음 · 단어별 라벨링',
            'file': raw['source_file'],
            'start_s': raw['start_s'],
            'end_s': raw['end_s'],
            'label_source': raw.get('label_source'),
            'boundary_method': raw.get('boundary_method'),
            'human_reviewed': bool(raw.get('human_reviewed')),
            'priority_review': bool(raw.get('priority_review')),
            'expected_answer': raw.get('expected_answer'),
            'note': raw.get('note') or '',
            'quality_flags': raw.get('quality_flags') or [],
        },
    }


def clip_rows():
    root = SOURCE_ROOT / LABELLED_DIR
    manifest = root / 'labels.jsonl'
    rows, seen = [], set()
    with manifest.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            row = _clip_row(raw, root)
            if row['id'] in seen:
                raise ValueError(f'{row["id"]}: 중복된 id')
            seen.add(row['id'])
            rows.append(row)
    return rows


# --- the four recordings as delivered --------------------------------

def _canonical(source):
    """One 16 kHz mono waveform per recording, decoded once and kept.

    Decoded with the project's own `load_audio_16k_mono` - the same loader
    the 83 clips were cut with - then written as 16-bit PCM and read back,
    so the array the model sees is exactly the array the browser plays,
    quantisation included.
    """
    CANONICAL.mkdir(parents=True, exist_ok=True)
    source_sha = digest(source)
    cached = CANONICAL / f'{source_sha[:16]}.wav'
    if not cached.exists():
        decoded = load_audio_16k_mono(source)
        if not np.isfinite(decoded).all():
            raise ValueError(f'{source.name}: 유효하지 않은 샘플')
        temp = cached.with_name(cached.name + '.tmp')
        sf.write(temp, decoded, SAMPLE_RATE, subtype='PCM_16', format='WAV')
        os.replace(temp, cached)
    waveform, rate = sf.read(cached, dtype='float32', always_2d=False)
    assert rate == SAMPLE_RATE
    return np.ascontiguousarray(waveform, dtype=np.float32), cached, source_sha


def original_rows():
    rows = []
    for source in sorted(SOURCE_ROOT.glob('*.mp3'), key=lambda p: p.name):
        waveform, cached, source_sha = _canonical(source)
        free = '자유대화' in source.name
        child = 'child2' if source.name.startswith('아동2') else 'child1'
        rows.append({
            'id': f'original:{source_sha[:12]}',
            'split': 'original',
            'kind': 'personal_original',
            'reader': 'personal',
            'wav': source.name,
            'path': str(cached),
            'speaker': child,
            'child': child,
            'age': 'unknown',
            'style': 'free' if free else 'word',
            # No time-aligned transcript exists for the whole recording, so
            # there is no reference here and nothing is scored.
            'text': '',
            'surface': None,
            'label_ids': None,
            'reference_state': 'none',
            'reference_reason': '이 원본 전체에 맞춰진 전사가 없어 비교하지 '
                                '않습니다',
            'samples': int(len(waveform)),
            'seconds': len(waveform) / SAMPLE_RATE,
            'pcm_sha256': pcm_sha256(waveform),
            'file_sha256': source_sha,
            'segment_plan': plan_segments(len(waveform), waveform),
            'source': {
                'dataset': '직접 녹음 · 원본 전체',
                'file': source.name,
                'mp3_sha256': source_sha,
                'canonical': cached.name,
                'kind': '자유대화' if free else '어휘 리스트',
                'human_reviewed': False,
                'note': '받은 MP3 전체입니다. 잘라내지 않았습니다.',
            },
        })
    return rows


# --- chunking --------------------------------------------------------

def _frame_dbfs(waveform):
    """RMS per 20 ms frame, in dBFS, for boundary placement."""
    usable = len(waveform) // BOUNDARY_FRAME * BOUNDARY_FRAME
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    frames = waveform[:usable].reshape(-1, BOUNDARY_FRAME)
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-9))


def plan_segments(total, waveform=None):
    """Contiguous chunks covering every sample exactly once.

    An even split first, so no chunk can grow past the bound, then each
    interior boundary is nudged to the quietest 20 ms frame within a second
    of where it fell. Nudging only moves a cut; it never drops, repeats or
    overlaps a sample, and the model output for each chunk is whatever it
    is - nothing is merged or de-duplicated afterwards.
    """
    if total <= 0:
        return []
    count = max(1, ceil(total / (CHUNK_TARGET_SECONDS * SAMPLE_RATE)))
    edges = [round(i * total / count) for i in range(count + 1)]
    edges[0], edges[-1] = 0, total

    dbfs = _frame_dbfs(waveform) if waveform is not None else np.zeros(0)
    quiet = {}
    window = int(BOUNDARY_SEARCH_SECONDS * SAMPLE_RATE)
    for i in range(1, count):
        target = edges[i]
        if len(dbfs):
            lo = max(edges[i - 1] + BOUNDARY_FRAME, target - window)
            hi = min(edges[i + 1] - BOUNDARY_FRAME, target + window)
            first, last = lo // BOUNDARY_FRAME, hi // BOUNDARY_FRAME
            if last > first:
                band = dbfs[first:min(last, len(dbfs))]
                if len(band):
                    chosen = first + int(np.argmin(band))
                    edges[i] = chosen * BOUNDARY_FRAME + BOUNDARY_FRAME // 2
                    quiet[i] = float(band.min())
    segments = []
    for i in range(count):
        start, end = edges[i], edges[i + 1]
        level = quiet.get(i + 1)
        segments.append({
            'index': i,
            'start': int(start), 'end': int(end),
            'samples': int(end - start),
            'start_s': round(start / SAMPLE_RATE, 3),
            'end_s': round(end / SAMPLE_RATE, 3),
            'seconds': round((end - start) / SAMPLE_RATE, 3),
            'boundary_dbfs': None if level is None else round(level, 1),
            # True when the cut after this chunk did not land in silence.
            'boundary_cuts_speech': bool(
                level is not None and level > BOUNDARY_QUIET_DBFS),
        })
    covered = sum(s['samples'] for s in segments)
    if covered != total or segments[0]['start'] != 0 \
            or segments[-1]['end'] != total:
        raise ValueError(f'분할이 전체를 덮지 못했습니다 ({covered}/{total})')
    if any(s['seconds'] > CHUNK_MAX_SECONDS for s in segments):
        raise ValueError('분할 조각이 상한을 넘었습니다')
    return segments


# --- reading ---------------------------------------------------------

_manifest_cache = {}


def _labelled_manifest():
    """The labelling run's own rows, by clip id. Read-only."""
    if not _manifest_cache:
        path = SOURCE_ROOT / LABELLED_DIR / 'labels.jsonl'
        with path.open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    raw = json.loads(line)
                    _manifest_cache[raw['id']] = raw
    return _manifest_cache


def clip_context(row, seconds=1.0):
    """The original recording either side of a clip, for listening only.

    Returned so a reviewer can hear a word in the place it came from. It is
    not a new cut and nothing is written: the canonical original is read,
    a window around the clip's own sample range is taken, and the offset of
    the clip inside that window is reported so the page can say where it
    starts.
    """
    raw = _labelled_manifest().get(str(row['id']).partition(':')[2])
    if raw is None:
        return None
    source = SOURCE_ROOT / raw['source_file']
    if not source.exists():
        return None
    waveform, _cached, _sha = _canonical(source)
    start = raw['source_start_sample_16k']
    end = raw['source_end_sample_16k_exclusive']
    pad = int(seconds * SAMPLE_RATE)
    lo, hi = max(0, start - pad), min(len(waveform), end + pad)
    return {
        'waveform': np.ascontiguousarray(waveform[lo:hi], dtype=np.float32),
        'seconds': round((hi - lo) / SAMPLE_RATE, 3),
        'clip_starts_at_s': round((start - lo) / SAMPLE_RATE, 3),
        'clip_seconds': round((end - start) / SAMPLE_RATE, 3),
        'source_file': raw['source_file'],
        'source_span_s': [raw['start_s'], raw['end_s']],
    }


def read(row):
    """The waveform for a personal row, checked against what was catalogued.

    Same contract as the archive reader: sample count and PCM hash both
    have to match, so a replaced file is an error rather than old text
    served over new audio.
    """
    path = physical_path(row['path'])
    with _lock:
        data, rate = sf.read(path, dtype='float32', always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(f'{row["wav"]}: {rate}Hz')
    if data.ndim == 2:
        data = data.mean(axis=1)
    waveform = np.ascontiguousarray(data, dtype=np.float32)
    if len(waveform) != row['samples']:
        raise ValueError(f'{row["wav"]}: {len(waveform)} samples, 카탈로그는 '
                         f'{row["samples"]}')
    actual = pcm_sha256(waveform)
    if actual != row['pcm_sha256']:
        raise ValueError(f'{row["wav"]}: 음성이 카탈로그와 달라졌습니다 '
                         f'({actual[:16]} vs {row["pcm_sha256"][:16]})')
    return waveform
