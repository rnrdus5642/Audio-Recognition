"""Recognise every personally recorded clip and every whole recording.

Not a benchmark and not a pass mark. It runs the real model over all 83
labelled clips and all four recordings and writes down exactly what came
out, so that what the page shows can be checked against a record rather
than against a memory - including the clips it got wrong, unfiltered.

The jamo error rate here is measured against the expected pronunciation
derived from the supplied transcript. Those boundaries were placed by
energy and have not all been listened to, so this is a record of what the
model produced on this audio, not an assessment of the children's speech.

    .venv-gpu/Scripts/python.exe python/tools/child_jamo_web/validate_personal.py
"""
from pathlib import Path
import json
import sys
import time

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(WEB.parents[0] / 'child_jamo_finetune'))
import compose
import engine as engine_module
import personal
import samples as samples_module

OUT = personal.WORK / 'validation.json'


def main():
    if not personal.available():
        raise SystemExit(f'원본 폴더를 찾지 못했습니다: {personal.SOURCE_ROOT}')
    started = time.monotonic()
    catalog = samples_module.Catalog()
    engine = engine_module.Engine()
    engine.load()

    report = {
        'at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source_root': str(personal.SOURCE_ROOT),
        'checkpoint_sha256': engine.identity['checkpoint_sha256'],
        'labels_contract_sha256': engine.identity['labels_contract_sha256'],
        'decode_version': compose.DECODE_VERSION,
        'device': engine.identity['device'],
        'scope': '직접 녹음한 음성 전체. 말뭉치(AIHub) 테스트 성적과는 별개이며 '
                 '거기에 합산되지 않습니다.',
        'caveat': '비교 대상은 전사에서 유도한 예상 발음 표기입니다. 구간 경계는 '
                  '음량 기반이고 전부 청취 검수된 자료가 아니므로, 발음 평가나 '
                  '합격 판정으로 읽으면 안 됩니다.',
    }

    # --- the 83 labelled clips, every one of them --------------------
    rows = catalog.rows['personal']
    clips, failures = [], []
    for row in rows:
        try:
            waveform = engine_module.read_row(row)
            prediction = engine.transcribe(waveform,
                                           pcm_sha256=row['pcm_sha256'],
                                           use_cache=False)
            metrics = compose.compare(row['label_ids'],
                                      prediction['token_ids'])
            clips.append({
                'id': row['id'], 'child': row['child'],
                'transcript': row['text'], 'surface': row['surface'],
                'predicted': prediction['text'],
                'predicted_jamo': prediction['jamo'],
                'jer': metrics['jer'], 'exact': metrics['exact'],
                'errors': metrics['errors'],
                'reference_jamo': metrics['reference_jamo'],
                'seconds': round(prediction['seconds'], 3),
                'priority_review': row['source']['priority_review'],
                'expected_answer': row['source']['expected_answer'],
            })
        except Exception as exc:
            failures.append({'id': row['id'],
                             'reason': f'{type(exc).__name__}: {exc}'})

    errors = sum(c['errors'] for c in clips)
    reference = sum(c['reference_jamo'] for c in clips)
    report['clips'] = {
        'catalogued': len(rows),
        'attempted': len(rows),
        'recognised': len(clips),
        'failures': failures,
        'selection': '필터 없이 personal 데이터셋 전체를 순서대로 실행했습니다.',
        'by_child': {
            child: {
                'recordings': sum(1 for c in clips if c['child'] == child),
                'exact': sum(1 for c in clips
                             if c['child'] == child and c['exact']),
                'jer': (sum(c['errors'] for c in clips if c['child'] == child)
                        / sum(c['reference_jamo'] for c in clips
                              if c['child'] == child))
                if any(c['child'] == child for c in clips) else None,
            } for child in ('child1', 'child2')
        },
        'exact': sum(1 for c in clips if c['exact']),
        'errors': errors, 'reference_jamo': reference,
        'jer': errors / reference if reference else None,
        # Every clip, including the ones it got wrong.
        'results': clips,
        'not_exact': [c['id'] for c in clips if not c['exact']],
    }

    # --- the four whole recordings, every sample of them --------------
    originals = []
    for row in catalog.rows['original']:
        waveform = engine_module.read_row(row)
        plan = row['segment_plan']
        segments, spent = [], 0.0
        for piece in plan:
            chunk = waveform[piece['start']:piece['end']]
            prediction = engine.transcribe(
                chunk, pcm_sha256=engine_module.pcm_sha256(chunk),
                use_cache=False)
            spent += prediction['inference_seconds']
            segments.append({
                'index': piece['index'],
                'start_s': piece['start_s'], 'end_s': piece['end_s'],
                'seconds': piece['seconds'], 'samples': piece['samples'],
                'boundary_dbfs': piece['boundary_dbfs'],
                'boundary_cuts_speech': piece['boundary_cuts_speech'],
                'predicted': prediction['text'],
            })
        covered = sum(s['samples'] for s in segments)
        originals.append({
            'id': row['id'], 'file': row['wav'], 'child': row['child'],
            'kind': row['source']['kind'],
            'mp3_sha256': row['file_sha256'],
            'seconds': round(row['seconds'], 3),
            'samples': row['samples'],
            'covered_samples': covered,
            'complete': covered == row['samples'],
            'segments': len(segments),
            'max_segment_seconds': max(s['seconds'] for s in segments),
            'boundaries_in_speech': sum(s['boundary_cuts_speech']
                                        for s in segments),
            'inference_seconds': round(spent, 2),
            'realtime_factor': round(row['seconds'] / spent, 1) if spent else None,
            'reference': None,
            'reference_state': 'none',
            'results': segments,
        })
    report['originals'] = {
        'recordings': len(originals),
        'audio_seconds': round(sum(o['seconds'] for o in originals), 2),
        'all_complete': all(o['complete'] for o in originals),
        'reference': '원본 전체에 맞춰진 전사가 없어 비교 지표를 내지 않습니다.',
        'results': originals,
    }
    report['elapsed_seconds'] = round(time.monotonic() - started, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding='utf-8')

    clip_stats = report['clips']
    print(f'단어별 클립 {clip_stats["recognised"]}/{clip_stats["catalogued"]}개 인식, '
          f'완전 일치 {clip_stats["exact"]}개, '
          f'자모 오류율 {clip_stats["jer"] * 100:.2f}%')
    for child, stats in clip_stats['by_child'].items():
        print(f'  {child}: {stats["recordings"]}개, 완전 일치 {stats["exact"]}개, '
              f'자모 오류율 {stats["jer"] * 100:.2f}%')
    if clip_stats['failures']:
        print('  실패:', clip_stats['failures'])
    for item in originals:
        print(f'{item["file"]}: {item["seconds"]}초 → {item["segments"]}구간, '
              f'전체 포함 {item["complete"]}, '
              f'경계가 말 중간 {item["boundaries_in_speech"]}개, '
              f'추론 {item["inference_seconds"]}초 '
              f'(실시간 대비 {item["realtime_factor"]}배)')
    print('기록:', OUT)
    return 0 if not clip_stats['failures'] \
        and report['originals']['all_complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
