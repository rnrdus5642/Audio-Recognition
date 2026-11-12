"""Compare completed book-label observations with the unchanged baseline.

Only audio enters inference. Choices and corpus targets are compared after
prediction; neither is treated as independently verified pronunciation truth.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import soundfile as sf
import torch

from analyze_phonetic import read_snapshot
from prepare_book_review import DEST, ROOT, WORK
from common import digest, dump, load

sys.path.insert(0, str(ROOT / 'python/tools/child_jamo_web'))
from engine import Engine, pcm_sha256

EXPECTED_CHECKPOINT = '5a330f1bea3a17d1d063fdfd71a0cc25394695b0b797de01b293f6f688257aa7'


def main():
    snapshot = read_snapshot(DEST)
    if snapshot['answered'] != snapshot['total']:
        raise RuntimeError('Finish the frozen review before comparing model outputs.')
    revision = snapshot['latest_revision']
    snapshot_path = DEST / f'answer-snapshot-revision-{revision:03d}.json'
    assert load(snapshot_path) == snapshot
    output = DEST / f'baseline-predictions-revision-{revision:03d}.json'
    if output.exists():
        raise RuntimeError(f'Keep the existing predictions: {output}')
    preparation = load(DEST / 'preparation.json')
    for filename, key in [('selection.json', 'selection_sha256'), ('review.json', 'review_sha256'),
                          ('evidence.json', 'evidence_sha256')]:
        assert digest(DEST / filename) == preparation[key]
    plan = load(DEST / 'selection.json')
    assert digest(WORK / 'manifest-train.jsonl') == plan['original_manifest_sha256']
    evidence = load(DEST / 'evidence.json')
    replies = {r['id']: r for r in snapshot['items']}

    torch.set_num_threads(4)
    model = Engine()
    print('Loading the unchanged baseline; audio-only inference, no training.', flush=True)
    model.load()
    assert model.identity['checkpoint_sha256'] == EXPECTED_CHECKPOINT
    rows = []
    for entry in evidence:
        number = entry['number']
        reply = replies[str(number)]
        assert reply['binding'] == entry['binding']
        source = DEST / 'audio' / entry['served_train']['file']
        assert digest(source) == entry['served_train']['sha256']
        wave, rate = sf.read(source, dtype='float32')
        assert rate == 16000 and len(wave) == entry['row']['samples']
        assert pcm_sha256(wave) == entry['row']['pcm_sha256']
        prediction = model.transcribe(wave, use_cache=False)
        candidate = entry['candidates'].get(reply['answer']['choice'])
        rows.append({'number': number, 'wav': entry['row']['wav'], 'binding': entry['binding'],
                     'pcm_sha256': entry['row']['pcm_sha256'],
                     'first_impression': reply['reveal']['first_heard'],
                     'choice': reply['answer']['choice'],
                     'selected_surface': candidate['surface'] if candidate else None,
                     'training_surface': entry['row']['surface'],
                     'prediction': prediction,
                     'matches_training_label': prediction['token_ids'] == entry['row']['label_ids'],
                     'matches_selected_form': None if not candidate else prediction['token_ids'] == candidate['label_ids']})
        print(f"{number:02d}: {prediction['text']}", flush=True)
    assert read_snapshot(DEST) == snapshot
    assert digest(DEST / 'selection.json') == preparation['selection_sha256']
    report = {'analyzed_at_utc': datetime.now(timezone.utc).isoformat(),
              'answer_revision': revision, 'answer_snapshot_sha256': digest(snapshot_path),
              'selection_sha256': preparation['selection_sha256'],
              'review_sha256': snapshot['review_sha256'],
              'evidence_sha256': preparation['evidence_sha256'], 'script_sha256': digest(Path(__file__)),
              'identity': model.identity, 'recordings': len(rows),
              'training_label_agreements': sum(r['matches_training_label'] for r in rows),
              'selected_form_count': sum(r['selected_surface'] is not None for r in rows),
              'selected_form_agreements': sum(r['matches_selected_form'] is True for r in rows),
              'items': rows, 'training_started': False, 'model_replaced': False,
              'references_used_in_inference': False,
              'limits': ['These are previously trained recordings and single-listener observations, not an independent accuracy benchmark.',
                         'Model agreement with its training label does not verify the pronunciation or prove why a different recording was misrecognized.',
                         'Free-text first impressions are retained, never converted into scoring targets.']}
    dump(output, report)
    dump(ROOT / 'docs/data/child-jamo-book-predictions-20261003.json', report)
    print(json.dumps({k: report[k] for k in ('recordings', 'training_label_agreements',
                                           'selected_form_count', 'selected_form_agreements')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
