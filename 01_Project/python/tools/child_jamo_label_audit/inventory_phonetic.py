"""Find same-prompt, non-training speakers without consulting predictions/audio."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/tools/child_jamo_finetune'))
import labels
from common import INDEX, SPLITS, WORK, digest, dump, load

WEB = ROOT / '.venv/child-jamo-web-20260916'
OLD = WEB / 'dataset-audit-20260928'
DEST = WEB / 'phonetic-labels-20260929'
SEED = 'matched-prompt-phonetic-labels-20260929-v1'
NUMBERS = (3, 4, 6, 10, 15)


def main():
    DEST.mkdir(exist_ok=True)
    audit = load(OLD / 'evidence.json')
    anchors = {str(e['number']): e for e in audit if e['number'] in NUMBERS}
    splits = load(SPLITS)['speakers']
    sets, selected_names = {}, set()
    for split in ('train', 'valid', 'test'):
        with (WORK / f'manifest-{split}.jsonl').open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        sets[split] = {r['speaker'] for r in rows}
        selected_names.update(r['wav'] for r in rows)
    codes = {a['row']['wav'].split('-')[0][:5]: number for number, a in anchors.items()}
    pool, counts, duplicates = [], Counter(), Counter()
    with INDEX.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            number = codes.get(row['wav'].split('-')[0][:5])
            if number is None or row['age'] not in ('5', '6', '7'):
                continue
            split = splits.get(row['speaker'], {}).get('split')
            text = labels.normalize(row['text'])
            anchor_text = anchors[number]['row']['text']
            record = {'case': int(number), 'index': row, 'speaker_split': split,
                      'normalized_text': text,
                      'exact_source_transcript': text == anchor_text,
                      'has_inline_notation': any(ch in row['text'] for ch in '()[]'),
                      'speaker_used_in_baseline_training': row['speaker'] in sets['train'],
                      'speaker_used_in_baseline_validation': row['speaker'] in sets['valid'],
                      'recording_used_in_baseline_any_split': row['wav'] in selected_names,
                      'sort_key': hashlib.sha256(f'{SEED}:{number}:{row["wav"]}'.encode()).hexdigest()}
            pool.append(record)
            duplicates[row['wav']] += 1
            counts[(number, split, record['exact_source_transcript'], record['has_inline_notation'])] += 1
    assert max(duplicates.values()) == 1
    report = {'selection_seed': SEED, 'index_sha256': digest(INDEX),
              'speaker_split_sha256': digest(SPLITS), 'baseline_split_speaker_counts': {k: len(v) for k, v in sets.items()},
              'eligible_valid_definition': 'Source split valid; speaker absent from baseline train; recording absent from train/valid/test manifests. Same prompt code is a retrieval hint, not evidence of actual spoken words.',
              'used_audio': False, 'used_model_predictions': False, 'anchors': anchors, 'pool': pool}
    dump(DEST / 'inventory.json', report)
    for number in NUMBERS:
        rows = [r for r in pool if r['case'] == number and r['speaker_split'] == 'valid'
                and not r['speaker_used_in_baseline_training'] and not r['recording_used_in_baseline_any_split']]
        print(json.dumps({'case': number, 'valid_remaining': len(rows),
              'unique_speakers': len({r['index']['speaker'] for r in rows}),
              'speakers_not_in_baseline_valid': len({r['index']['speaker'] for r in rows if not r['speaker_used_in_baseline_validation']}),
              'label_texts': Counter(r['normalized_text'] for r in rows),
              'first': [{k:r[k] for k in ('normalized_text','speaker_used_in_baseline_validation')} | {'wav':r['index']['wav']} for r in sorted(rows,key=lambda r:r['sort_key'])[:4]]},ensure_ascii=False),flush=True)
    print('Inventory complete',len(pool),flush=True)


if __name__ == '__main__':
    main()
