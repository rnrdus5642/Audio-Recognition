"""Training-split recordings that contain the confusions we are chasing.

Metadata only. Nothing is copied, decoded or written near the archives -
each entry is a pointer into the existing train manifest, so the pool can
be rebuilt or thrown away at no cost.

Two boundaries this file keeps:

  train only        Candidates come from the 150k training split. The 83
                    personal clips and the validation split stay what they
                    are - development and evaluation material. Putting an
                    evaluated recording into training would leave nothing
                    honest to measure against.
  candidates only   A recording is here because its *expected surface*
                    contains one side of a confusion pair. That is a
                    transcript-derived label, not a verified statement
                    about how the speaker actually said it, and it is not
                    evidence that any particular jamo was pronounced well.

Which pairs matter, and whether the fix is data or something else, is not
settled until the listening review comes back.

    python candidates.py            # writes train-candidates.json
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / 'child_jamo_finetune'))
import audit                                                   # noqa: E402
import labels                                                  # noqa: E402
from common import WORK as TRAIN_WORK, digest                  # noqa: E402

OUT = audit.WORK / 'train-candidates.json'

#: The confusions the audit and the earlier diagnosis actually turned up.
#: Both sides of each pair are collected: a model that says ㅅ for ㅊ needs
#: to hear both, not just more ㅊ.
PAIRS = [('ㅊ', 'ㅅ'), ('ㄱ', 'ㅂ'), ('ㄸ', 'ㄱ'), ('ㅊ', 'ㅌ'),
         ('ㅊ', 'ㅋ'), ('ㄴ', 'ㅇ'), ('ㅂ', 'ㄹ'), ('ㅎ', 'ㅇ')]
#: Bounded on purpose: a diverse pool to choose from, not a copy of the
#: split. Per jamo, per syllable position, and per speaker.
PER_JAMO_ROLE = 60
PER_SPEAKER = 4


def _surface_words(row):
    """Each written word beside the surface form it becomes, with roles.

    Matching on spelling was wrong twice over: the sound in question may
    only exist after the rules run (`익은` -> `이근`), and a spelling match
    says nothing about whether it landed in an onset or a coda. So the
    surface is split the same way the transcript is and the two are zipped,
    and each surface word is decomposed into (jamo, role) pairs.
    """
    written = row['text'].split()
    surface = (row['surface'] or '').split()
    if len(written) != len(surface):
        return []
    out = []
    for word, form in zip(written, surface):
        roles = audit._jamo_roles(form)
        if roles is None:
            continue
        out.append((word, form, roles))
    return out


def collect(limit_per_role=PER_JAMO_ROLE, per_speaker=PER_SPEAKER):
    wanted = {jamo for pair in PAIRS for jamo in pair}
    buckets = defaultdict(list)          # (jamo, role) -> rows
    speaker_counts = defaultdict(lambda: defaultdict(int))
    scanned = 0
    totals = defaultdict(int)
    corpus_words = defaultdict(lambda: defaultdict(int))
    untrusted = 0

    with (TRAIN_WORK / 'manifest-train.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            scanned += 1
            roles, trusted = audit.reference_roles(row)
            if not trusted:
                untrusted += 1
                continue
            words = _surface_words(row)
            seen_here = set()
            for index, (jamo, role) in enumerate(roles):
                if jamo not in wanted or role not in ('onset', 'coda'):
                    continue
                key_name = f'{jamo}@{role}'
                totals[key_name] += 1
                key = (jamo, role)
                if key in seen_here:
                    continue
                seen_here.add(key)
                # The surface word carrying this jamo *in this role*.
                carrier = next(
                    ((written, form) for written, form, word_roles in words
                     if (jamo, role) in word_roles), None)
                if carrier:
                    corpus_words[key_name][carrier[1]] += 1
                if len(buckets[key]) >= limit_per_role:
                    continue
                if speaker_counts[key][row['speaker']] >= per_speaker:
                    continue
                speaker_counts[key][row['speaker']] += 1
                buckets[key].append({
                    'id': f'train:{row["wav"]}',
                    'wav': row['wav'],
                    # Where it lives, so the pool can be rebuilt without a
                    # copy of the audio, plus the immutable identity the
                    # manifest already holds - no decoding needed.
                    'archive': row['archive'], 'offset': row['offset'],
                    'size': row['size'],
                    'samples': row.get('samples'),
                    'pcm_sha256': row.get('pcm_sha256'),
                    'span_policy': row.get('span_policy'),
                    'speech_start': row.get('speech_start'),
                    'speech_end': row.get('speech_end'),
                    'label_ids': row.get('label_ids'),
                    'speaker': row['speaker'], 'age': row['age'],
                    'style': row['style'],
                    'seconds': round(row['seconds'], 3),
                    'text': row['text'], 'surface': row['surface'],
                    'jamo_position': index,
                    'carrier_written': carrier[0] if carrier else None,
                    'carrier_surface': carrier[1] if carrier else None,
                })
    return buckets, totals, corpus_words, scanned, untrusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-role', type=int, default=PER_JAMO_ROLE)
    parser.add_argument('--per-speaker', type=int, default=PER_SPEAKER)
    args = parser.parse_args()

    started = time.monotonic()
    buckets, totals, corpus_words, scanned, untrusted = collect(
        args.per_role, args.per_speaker)

    # The frozen splits, so train-only membership is checked rather than
    # assumed, without decoding anything.
    held_out_ids, held_out_speakers = set(), set()
    held_out_pcm = set()
    for split in ('valid', 'test'):
        with (TRAIN_WORK / f'manifest-{split}.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                held_out_ids.add(row['wav'])
                held_out_speakers.add(row['speaker'])
                if row.get('pcm_sha256'):
                    held_out_pcm.add(row['pcm_sha256'])

    pool, overlaps = {}, {'ids': [], 'pcm': [], 'speakers': []}
    for (jamo, role), rows in sorted(buckets.items()):
        key = f'{jamo}@{role}'
        speakers = {r['speaker'] for r in rows}
        for record in rows:
            if record['wav'] in held_out_ids:
                overlaps['ids'].append(record['id'])
            if record.get('pcm_sha256') and record['pcm_sha256'] in held_out_pcm:
                overlaps['pcm'].append(record['id'])
        shared_speakers = speakers & held_out_speakers
        if shared_speakers:
            overlaps['speakers'].extend(sorted(shared_speakers))
        # Diversity of the 60 actually selected, kept apart from the
        # corpus-wide totals they used to sit beside.
        sampled_words = Counter(r['carrier_surface'] for r in rows
                                if r['carrier_surface'])
        pool[key] = {
            'jamo': jamo, 'role': role,
            'occurrences_in_whole_train_split': totals[key],
            'sampled': len(rows),
            'sampled_distinct_speakers': len(speakers),
            'sampled_distinct_surface_words': len(sampled_words),
            'sampled_example_surface_words': [
                w for w, _ in sampled_words.most_common(12)],
            'whole_split_distinct_surface_words': len(corpus_words[key]),
            'sampled_speakers_also_in_valid_or_test': sorted(shared_speakers),
            'records': rows,
        }

    unique = {r['id'] for entry in pool.values() for r in entry['records']}
    report = {
        'at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'split': 'train',
        'manifest': str(TRAIN_WORK / 'manifest-train.jsonl'),
        'manifest_sha256': digest(TRAIN_WORK / 'manifest-train.jsonl'),
        'labels_contract_sha256': labels.contract_sha256(),
        'labels_version': labels.LABELS_VERSION,
        'scanned_records': scanned,
        'rows_with_untrusted_roles': untrusted,
        'pairs': [list(pair) for pair in PAIRS],
        'caps': {'per_jamo_role': args.per_role, 'per_speaker': args.per_speaker},
        # One recording can serve several buckets, so the bucket sizes do
        # not add up to the pool.
        'unique_recordings': len(unique),
        'bucket_memberships': sum(entry['sampled'] for entry in pool.values()),
        'held_out_check': {
            'compared_against': ['manifest-valid.jsonl', 'manifest-test.jsonl'],
            'recording_id_overlap': overlaps['ids'],
            'pcm_hash_overlap': overlaps['pcm'],
            'shared_speakers': sorted(set(overlaps['speakers'])),
            'train_only': not overlaps['ids'] and not overlaps['pcm'],
            'note': '화자는 분할 간에 겹칠 수 있습니다. 녹음 자체가 겹치지 '
                    '않는다는 것과는 다른 이야기라 따로 적습니다.',
        },
        'elapsed_seconds': round(time.monotonic() - started, 1),
        'what_this_is': '학습 분할에서 해당 자음이 기대 발음에 들어 있는 녹음의 '
                        '목록입니다. 오디오를 복사하지 않고 매니페스트 위치만 '
                        '가리킵니다.',
        'what_this_is_not': [
            '그 자음이 실제로 어떻게 발음되었는지 확인된 자료가 아닙니다. '
            '기대 발음은 전사에서 유도한 것입니다.',
            '학습에 바로 쓰기로 결정된 자료가 아닙니다. 어떤 짝이 문제인지, '
            '원인이 데이터인지 다른 것인지는 청취 검토 결과가 나와야 '
            '정해집니다.',
            '평가용 자료가 섞여 있지 않습니다. 직접 녹음 83개와 검증/테스트 '
            '분할은 여기에 들어 있지 않습니다.',
        ],
        'pool': pool,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding='utf-8')

    print(f'학습 {scanned:,}개를 훑어 {len(pool)}개 (자음×위치) 후보군을 '
          f'만들었습니다 ({report["elapsed_seconds"]}초)')
    print(f"  고유 녹음 {report['unique_recordings']}개 "
          f"(버킷 소속 합계는 {report['bucket_memberships']}개, 한 녹음이 여러 "
          f"버킷에 들어갑니다)")
    print(f"  학습 전용 확인: {report['held_out_check']['train_only']} "
          f"(검증/테스트와 녹음 겹침 {len(overlaps['ids'])}건, "
          f"PCM 겹침 {len(overlaps['pcm'])}건, "
          f"공유 화자 {len(set(overlaps['speakers']))}명)")
    for key, entry in sorted(pool.items()):
        print(f"  {key:<8} 학습 전체 {entry['occurrences_in_whole_train_split']:>7,}회 · "
              f"표본 {entry['sampled']:>3}개 / 화자 {entry['sampled_distinct_speakers']:>3}명 / "
              f"표본 단어 {entry['sampled_distinct_surface_words']:>3}종 · 예: "
              f"{', '.join(entry['sampled_example_surface_words'][:4])}")
    print('기록:', OUT)


if __name__ == '__main__':
    main()
