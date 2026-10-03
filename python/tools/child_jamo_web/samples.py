"""The audited manifests, searchable, without shipping them to the browser.

Samples are addressed by an opaque id built from the split and the
recording's own name. Nothing here takes a filesystem path from a
request, and the archives are only ever read through the offsets the
audit recorded.

The train manifest is 110 MB, so it is parsed on first use in the
background rather than at startup, and only the fields the page needs are
kept.
"""
from pathlib import Path
import json
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'child_jamo_finetune'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import personal
from common import WORK as TRAIN_WORK

#: The two places recordings come from, each with its own page.
#:
#: They are different audio, gathered differently, and only one of them is
#: what the recorded test score describes - so they are not two entries in
#: one dropdown but two pages. `default` is what a page opens on, and is
#: also what a request for the other source's split falls back to, so
#: `/aihub?split=personal` can never show the personal recordings.
SOURCES = {
    'aihub': {
        'path': '/aihub', 'label': 'AIHub 데이터',
        'title': '말뭉치(AIHub) 음성 인식 확인',
        'blurb': '학습에 쓴 아동 음성 말뭉치입니다. 이 모델의 기록된 전체 '
                 '성적은 이 데이터의 테스트 세트에 대한 것입니다.',
        'default': 'test',
    },
    'recordings': {
        'path': '/recordings', 'label': '직접 녹음한 아동 음성',
        'title': '직접 녹음한 아동 음성 인식 확인',
        'blurb': '직접 녹음해 오신 아동 음성입니다. 말뭉치 성적과는 별개이며, '
                 '여기 결과는 그 점수에 들어가지 않습니다.',
        'default': 'personal',
    },
}

#: Held-out first: it is the split the recorded test score describes.
#: Short enough that the recording count beside them is not truncated.
#: `group` labels them inside their own page's picker.
SPLITS = {
    'test': {'label': '테스트', 'hint': '학습에 쓰이지 않은 음성입니다.',
             'eager': True, 'source': 'aihub', 'group': '말뭉치 (AIHub)',
             'scored': True},
    'valid': {'label': '검증', 'hint': '모델을 고를 때 사용한 음성입니다.',
              'eager': True, 'source': 'aihub', 'group': '말뭉치 (AIHub)',
              'scored': True},
    'train': {'label': '학습', 'hint': '모델이 학습한 음성입니다. 목록 준비에 '
                                       '시간이 걸립니다.', 'eager': False,
              'source': 'aihub', 'group': '말뭉치 (AIHub)', 'scored': True},
}

if personal.available():
    SPLITS['personal'] = {
        'label': '직접 녹음 · 단어별',
        'hint': '직접 녹음한 아동 음성을 전사에 따라 단어별로 자른 것입니다. '
                '전사 라벨이 있어 비교할 수 있습니다.',
        'eager': True, 'source': 'recordings',
        'group': '직접 녹음한 아동 음성', 'scored': True,
        'builder': personal.clip_rows,
    }
    SPLITS['original'] = {
        'label': '직접 녹음 · 원본 전체',
        'hint': '받은 녹음 파일 전체입니다. 맞춰진 전사가 없어 비교하지 않고, '
                '인식 결과만 구간별로 보여 줍니다.',
        'eager': True, 'source': 'recordings',
        'group': '직접 녹음한 아동 음성', 'scored': False,
        # A minute or two each: they are recognised one at a time, in
        # bounded chunks, not ten at a time.
        'batchable': False,
        'builder': personal.original_rows,
    }
KEEP = ('wav', 'archive', 'offset', 'size', 'speaker', 'age', 'style',
        'split', 'text', 'surface', 'label_ids', 'samples', 'seconds',
        'speech_start', 'speech_end', 'span_policy', 'pcm_sha256')


def splits_of(source):
    """The split names one source's page is allowed to show."""
    return [name for name, meta in SPLITS.items() if meta['source'] == source]


def _duration_bucket(seconds):
    if seconds < 2.0:
        return 'short'
    if seconds < 5.0:
        return 'medium'
    return 'long'


class Catalog:
    def __init__(self):
        self.rows = {}
        self.index = {}
        self.loading = set()
        self.lock = threading.Lock()
        for split, meta in SPLITS.items():
            if meta['eager']:
                self._load(split)

    def _load(self, split):
        builder = SPLITS[split].get('builder')
        rows, index = [], {}
        if builder is not None:
            # Already whole rows, built and provenance-checked by `personal`.
            for row in builder():
                row['duration_bucket'] = _duration_bucket(row['seconds'])
                index[row['id']] = len(rows)
                rows.append(row)
        else:
            path = TRAIN_WORK / f'manifest-{split}.jsonl'
            with path.open(encoding='utf-8') as f:
                for line in f:
                    raw = json.loads(line)
                    row = {k: raw[k] for k in KEEP if k in raw}
                    row['id'] = f'{split}:{row["wav"]}'
                    row['duration_bucket'] = _duration_bucket(row['seconds'])
                    index[row['id']] = len(rows)
                    rows.append(row)
        with self.lock:
            self.rows[split] = rows
            self.index[split] = index
            self.loading.discard(split)

    def ensure(self, split):
        """Start a lazy split loading; returns True once it is available."""
        if split not in SPLITS:
            raise KeyError(split)
        if split in self.rows:
            return True
        with self.lock:
            if split in self.loading:
                return False
            self.loading.add(split)
        threading.Thread(target=self._load, args=(split,), daemon=True).start()
        return False

    def available(self):
        return {split: {'label': meta['label'], 'hint': meta['hint'],
                        'source': meta['source'],
                        'group': meta['group'], 'scored': meta['scored'],
                        'batchable': meta.get('batchable', True),
                        'loaded': split in self.rows,
                        'loading': split in self.loading,
                        'recordings': len(self.rows.get(split, ()))}
                for split, meta in SPLITS.items()}

    def get(self, sample_id):
        split, _, _name = str(sample_id).partition(':')
        if split not in SPLITS or split not in self.rows:
            return None
        position = self.index[split].get(sample_id)
        return self.rows[split][position] if position is not None else None

    def search(self, split, query='', age='', bucket='', style='', child='',
               page=1, size=24):
        if split not in self.rows:
            return None
        rows = self._filtered(split, query, age, bucket, style, child)
        total = len(rows)
        size = max(1, min(int(size), 100))
        page = max(1, int(page))
        start = (page - 1) * size
        return {
            'total': total, 'page': page, 'size': size,
            'pages': max(1, (total + size - 1) // size),
            'items': [summary(r) for r in rows[start:start + size]],
        }

    def pick(self, split, seed=None, **filters):
        """A random sample from the current filter, for the shuffle button."""
        import random
        if split not in self.rows:
            return None
        found = self.search(split, page=1, size=100, **filters)
        if not found or not found['total']:
            return None
        rng = random.Random(seed)
        position = rng.randrange(found['total'])
        page = position // 24 + 1
        page_found = self.search(split, page=page, size=24, **filters)
        items = page_found['items']
        return items[position % 24] if position % 24 < len(items) else items[0]

    def following(self, split, sample_id, count, **filters):
        """`sample_id` and the next ones after it, in the current filter.

        The batch button promises the next N from where you are. Taking
        the first N of whatever page happens to be shown would quietly
        run something else whenever the selection came from the random
        button or from paging.
        """
        if split not in self.rows:
            return []
        matched = self._filtered(split, **filters)
        position = next((i for i, r in enumerate(matched)
                         if r['id'] == sample_id), None)
        if position is None:
            position = 0
        return [summary(r) for r in matched[position:position + count]]

    def _filtered(self, split, query='', age='', bucket='', style='',
                  child=''):
        rows = self.rows[split]
        query = (query or '').strip().lower()
        if not (query or age or bucket or style or child):
            return rows
        return [r for r in rows
                if (not age or r['age'] == age)
                and (not bucket or r['duration_bucket'] == bucket)
                and (not style or r['style'] == style)
                and (not child or r.get('child') == child)
                and (not query
                     or query in (r['text'] or '').lower()
                     or query in (r['surface'] or '').lower()
                     or query in r['wav'].lower()
                     or query in r['speaker'].lower())]

    def neighbour(self, sample_id, step):
        row = self.get(sample_id)
        if row is None:
            return None
        split = row['split']
        position = self.index[split][sample_id] + step
        rows = self.rows[split]
        if not 0 <= position < len(rows):
            return None
        return summary(rows[position])


def summary(row):
    """What a list item needs; never the label ids, paths or archive offsets.

    `reference_state` travels with every row so the page can say whether a
    recording has something to be compared against, instead of inferring it
    from an empty transcript.
    """
    out = {
        'id': row['id'], 'wav': row['wav'], 'split': row['split'],
        'age': row['age'], 'style': row['style'],
        'seconds': round(row['seconds'], 3),
        'duration_bucket': row['duration_bucket'],
        'speaker': row['speaker'], 'text': row['text'],
        'reference_state': row.get('reference_state', 'transcript'),
    }
    if 'child' in row:
        out['child'] = row['child']
    if 'source' in row:
        out['source'] = row['source']
    if row.get('kind'):
        out['kind'] = row['kind']
    return out
