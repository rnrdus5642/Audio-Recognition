"""Find unused development recordings for a possible book-label comparison.

Corpus text is a retrieval key, never evidence of the sound produced. No
audio, model prediction, answer, training label or checkpoint is changed.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json

from inventory_phonetic import INDEX, ROOT, SPLITS, WEB, WORK
from prepare import collect_wavs, existing_reviews
from common import digest, dump, load
import labels

DEST = WEB / 'book-label-review-20261003'
VERSION = 'book-comparison-inventory-20261003-v1'


def main():
    splits = load(SPLITS)['speakers']
    used_names, split_speakers, manifest_hashes = set(), {}, {}
    for split in ('train', 'valid', 'test'):
        path = WORK / f'manifest-{split}.jsonl'
        manifest_hashes[split] = digest(path)
        with path.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        split_speakers[split] = {r['speaker'] for r in rows}
        used_names.update(r['wav'] for r in rows)
    previous, _ = existing_reviews()
    for directory in ('dataset-audit-20260928', 'phonetic-labels-20260929',
                      'phonetic-followup-20261003', 'book-label-review-20261003'):
        previous.update(collect_wavs(load(WEB / directory / 'evidence.json')))
    pool, exact_counts, other_jjaek = [], Counter(), Counter()
    index_rows = 0
    with INDEX.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            index_rows += 1
            if row['age'] not in ('5', '6', '7') or not any(s in row['text'] for s in ('책', '짹')):
                continue
            text = labels.normalize(row['text'])
            if text not in ('책', '짹'):
                if '짹' in text:
                    other_jjaek[text] += 1
                continue
            split = splits.get(row['speaker'], {}).get('split')
            exact_counts[(text, split)] += 1
            eligible = (split == 'valid' and row['speaker'] not in split_speakers['train']
                        and row['wav'] not in used_names and row['wav'] not in previous)
            pool.append({'index': row, 'normalized_transcript': text, 'speaker_split': split,
                         'speaker_used_in_baseline_training': row['speaker'] in split_speakers['train'],
                         'speaker_used_in_baseline_validation': row['speaker'] in split_speakers['valid'],
                         'recording_used_in_baseline_any_split': row['wav'] in used_names,
                         'previously_presented': row['wav'] in previous, 'eligible_development_candidate': eligible,
                         'sort_key': hashlib.sha256(f'{VERSION}:{row["wav"]}'.encode()).hexdigest()})
    assert len({r['index']['wav'] for r in pool}) == len(pool)
    summary = {}
    for word in ('책', '짹'):
        rows = [r for r in pool if r['normalized_transcript'] == word and r['eligible_development_candidate']]
        summary[word] = {'recordings': len(rows), 'speakers': len({r['index']['speaker'] for r in rows}),
                         'seconds_from_index': round(sum(float(r['index']['seconds']) for r in rows), 3),
                         'speakers_used_in_baseline_validation': len({r['index']['speaker'] for r in rows
                                                                     if r['speaker_used_in_baseline_validation']})}
    report = {'version': VERSION, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'index_rows': index_rows, 'index_sha256': digest(INDEX), 'splits_sha256': digest(SPLITS),
              'baseline_manifest_sha256': manifest_hashes,
              'exact_transcript_counts': [{'transcript': word, 'speaker_split': split, 'recordings': count}
                                         for (word, split), count in sorted(exact_counts.items(), key=lambda x: str(x[0]))],
              'eligible_candidates_by_transcript': summary,
              'other_transcripts_containing_jjaek': dict(other_jjaek.most_common(20)),
              'pool': sorted(pool, key=lambda r: r['sort_key']),
              'audio_read': False, 'model_predictions_used': False, 'training_started': False,
              'limits': ['Metadata-only retrieval in the existing ages 5-7 cohort. Source audio and labels have not yet been reverified for these new candidates.',
                         'Same transcript does not establish a 책 or 짹 sound; a lack of 짹 text does not establish a lack of that pronunciation.',
                         'Development speakers are excluded from baseline training but may occur in baseline validation. Final test speakers are not recruited.',
                         'Candidate availability is not evaluation readiness or permission to treat these transcripts as verified pronunciation targets.']}
    dump(DEST / 'comparison-inventory.json', report)
    dump(ROOT / 'docs/data/child-jamo-book-comparison-inventory-20261003.json', report)
    print(json.dumps({k: report[k] for k in ('index_rows', 'exact_transcript_counts',
                     'eligible_candidates_by_transcript', 'other_transcripts_containing_jjaek')}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
