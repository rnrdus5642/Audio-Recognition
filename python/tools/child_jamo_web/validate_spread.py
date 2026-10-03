"""Run the served model over a fixed spread of held-out audio, and record it.

Two questions, both answered against the real checkpoint:

  spread     what the model actually does on a deterministic slice of the
             test split - every age and every length bucket, picked by a
             fixed rule rather than by how well it scores
  agreement  whether this viewer's single-utterance path decodes the same
             tokens as the batched path `train.evaluate` used, so the
             numbers on screen belong to the same computation the
             recorded score came from

Results go to the app work directory. The recorded final test score is
never recomputed or overwritten here.

    python python/tools/child_jamo_web/validate_spread.py
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch

WEB = Path(__file__).resolve().parent
FINETUNE = WEB.parents[0] / 'child_jamo_finetune'
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(FINETUNE))
import compose
import engine as engine_module
import labels
import samples as samples_module
from common import ROOT, dump

WORK = ROOT / '.venv/child-jamo-web-20260916'
PER_CELL = 2                 # per (age, length bucket) cell -> 18 recordings


def spread(catalog):
    """A fixed slice: the first PER_CELL of every age x length cell.

    Deterministic and blind to the model - the manifest order decides,
    not the score, so this cannot become a list of successes.
    """
    chosen = []
    for age in ['5', '6', '7']:
        for bucket in ['short', 'medium', 'long']:
            found = catalog.search('test', age=age, bucket=bucket,
                                   page=1, size=PER_CELL)
            for item in found['items']:
                chosen.append(catalog.get(item['id']))
    return chosen


def agreement(engine, rows, limit=4):
    """Decode a few rows the way training batches them, and compare."""
    sys.path.insert(0, str(FINETUNE))
    import data as train_data
    import train as train_module

    picked = rows[:limit]
    batch = train_data.collate(picked, list(range(len(picked))), 'cuda')
    with torch.inference_mode():
        logits, lengths = train_module.forward(
            engine.model, engine.head, batch, grad=False, autocast=False)
        argmax = logits.argmax(-1).cpu().numpy()
    out = []
    for b, row in enumerate(picked):
        width = int(lengths[b])
        batched = train_module.collapse(argmax[b, :width].tolist())
        single = engine.transcribe(engine_module.read_row(row),
                                   pcm_sha256=row['pcm_sha256'],
                                   use_cache=False)
        out.append({
            'wav': row['wav'],
            'batched_tokens': ''.join(labels.TOKENS[t] for t in batched),
            'single_tokens': ''.join(labels.TOKENS[t]
                                     for t in single['token_ids']),
            'identical': batched == single['token_ids'],
        })
    return out


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    catalog = samples_module.Catalog()
    engine = engine_module.Engine()
    engine.load()

    rows = spread(catalog)
    records, latencies = [], []
    for row in rows:
        waveform = engine_module.read_row(row)
        started = time.monotonic()
        prediction = engine.transcribe(waveform, pcm_sha256=row['pcm_sha256'],
                                       use_cache=False)
        latencies.append(time.monotonic() - started)
        metrics = compose.compare(row['label_ids'], prediction['token_ids'])
        records.append({
            'wav': row['wav'], 'age': row['age'], 'style': row['style'],
            'seconds': round(row['seconds'], 3),
            'bucket': row['duration_bucket'],
            'transcript': row['text'], 'expected_surface': row['surface'],
            'predicted_text': prediction['text'],
            'expected_jamo': ''.join(labels.TOKENS[i] for i in row['label_ids']
                                     if i != labels.BOUNDARY),
            'predicted_jamo': prediction['jamo'],
            'orphan_jamo': prediction['orphan_jamo'],
            'jer': metrics['jer'], 'errors': metrics['errors'],
            'reference_jamo': metrics['reference_jamo'],
            'exact': metrics['exact'],
            'inference_seconds': round(prediction['inference_seconds'], 3),
        })
        print(f'{row["age"]}세 {row["duration_bucket"]:<6} '
              f'{row["seconds"]:5.2f}s  jer={metrics["jer"]:.3f}  '
              f'{row["surface"]}  ->  {prediction["text"]}')

    errors = sum(r['errors'] for r in records)
    reference = sum(r['reference_jamo'] for r in records)
    report = {
        'note': 'Interactive exploration on a fixed slice of the test split. '
                'Separate from the recorded final test score, which was not '
                'recomputed or overwritten.',
        'checkpoint': engine.status()['checkpoint'],
        'historical_full_test': engine.status()['historical_test'],
        'selection_rule': f'first {PER_CELL} recordings of every age x length '
                          f'bucket cell, in manifest order',
        'recordings': len(records),
        'selection_jer': errors / reference if reference else None,
        'selection_errors': errors, 'selection_reference_jamo': reference,
        'selection_exact': sum(1 for r in records if r['exact']),
        'latency_seconds': {
            'median': float(np.median(latencies)),
            'min': float(np.min(latencies)), 'max': float(np.max(latencies)),
        },
        'gpu': engine.status()['device'],
        'samples': records,
        'decoder_agreement_with_training_batched_path': agreement(engine, rows),
        'at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    report['decoder_agreement_all_identical'] = all(
        a['identical'] for a in report['decoder_agreement_with_training_batched_path'])
    dump(WORK / 'spread-validation.json', report)
    print()
    print(f'{len(records)} recordings | selection JER '
          f'{report["selection_jer"]:.4f} | exact {report["selection_exact"]}'
          f'/{len(records)} | median {report["latency_seconds"]["median"]:.2f}s')
    print('decoder agreement with training batched path:',
          report['decoder_agreement_all_identical'])
    print('written', WORK / 'spread-validation.json')


if __name__ == '__main__':
    main()
