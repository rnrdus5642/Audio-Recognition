"""Summarize completed stress evaluation without changing the tested policy."""
from collections import defaultdict
import json
from pathlib import Path

from stress_postprocessing import OUT


def main():
    completed = json.loads((OUT / 'complete.json').read_text(encoding='utf-8'))
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    totals = summary['totals']
    names = {'alone': '단독 발화', 'speech_before': '앞말 바로 뒤', 'speech_before_gap': '앞말 뒤 0.25초 간격',
             'speech_after': '정답 뒤에 다른 말', 'repeat_joined': '같은 음성 바로 반복',
             'repeat_gap': '0.25초 간격으로 반복', 'noise_before': '잡음 뒤에 정답',
             'noise_20db': '정답과 잡음 동시(SNR 20 dB)', 'noise_10db': '정답과 잡음 동시(SNR 10 dB)'}
    lines = ['후처리 추가 검사 — 2026-10-30', '',
             '현재 v4와 직전 v3에 같은 모델 출력을 적용해 비교했다.',
             '실제 음성은 GPU 모델의 2.5초 입력을 0.5초 간격으로 검사했다.',
             '임계값 85점, 확인 2회, 시간 제한 없는 현재 설정이다.', '',
             '1. 정확한 텍스트를 넣었을 때의 후처리 동작 (음성 정확도가 아님)']
    for key, value in totals.items():
        if key.startswith('ideal_'):
            lines.append(f"{key}: {value['before_accepted']}/{value['trials']} -> {value['after_accepted']}/{value['trials']}")
    for phase, title in [('paired_single', '2. 실제 녹음 58개, 단일 정답'),
                         ('paired_stage', '3. 같은 녹음, 현재 활동의 여러 정답 후보')]:
        lines += ['', title]
        for condition, label in names.items():
            v = totals[f'{phase}/{condition}']
            lines.append(f"{label}: {v['before_accepted']}/{v['trials']} -> {v['after_accepted']}/{v['trials']}; "
                         f"v4 정답 녹음 시작 전 통과 {v['after_premature']}건; "
                         f"정답 표현이 출력됐어도 실패 {v['after_missed_with_exact_word']}건 "
                         f"(정확한 표현이 2회 이상 출력됐는데 실패 {v['after_missed_with_two_exact_checks']}건)")
    lines += ['', '4. 자연 문장 / 새 오답 녹음 / 60초 연속 듣기']
    for key, v in totals.items():
        if key.split('/')[0] in ('natural_context', 'new_distractor', 'long_negative'):
            lines.append(f"{key}: {v['before_accepted']}/{v['trials']} -> {v['after_accepted']}/{v['trials']}; "
                         f"정확한 표현 2회 이상인데 실패 {v['after_missed_with_two_exact_checks']}건")
    selected, examples = defaultdict(list), []
    metadata = defaultdict(set)
    with (OUT / 'trials.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('record_id'):
                metadata[row['phase'] + '_recordings'].add(row['record_id'])
            if row.get('speaker_key'):
                metadata[row['phase'] + '_speakers'].add(row['speaker_key'])
            key = None
            if row['phase'] in ('paired_single', 'natural_context') and row['positive'] and not row['after']['accepted'] and row['exact_word_checks'] >= 2:
                key = 'correct_text_missed/' + row['condition']
            elif row['phase'] in ('new_distractor', 'long_negative') and row['after']['accepted']:
                key = 'false_accept/' + row['phase']
            elif row['phase'] == 'ideal_positive' and row['condition'] == 'repeat_joined' and not row['after']['accepted']:
                key = 'one_syllable_repetition'
            elif row['phase'].startswith('paired') and row['after'].get('before_target_started'):
                key = 'premature_accept'
            elif row['phase'] in ('paired_single', 'paired_stage') and row['before']['accepted'] and not row['after']['accepted']:
                key = 'regression/' + row['phase']
            if key and len(selected[key]) < 8:
                selected[key].append(row['id'])
                trace = row['after']['trace']
                examples.append({'category': key, **{k: row.get(k) for k in ('id', 'phase', 'condition', 'reference', 'targets', 'audio_path', 'exact_word_times')},
                                 'accepted': row['after']['accepted'], 'accepted_target': row['after']['target'],
                                 'frames': [{k: frame.get(k) for k in ('elapsed_seconds', 'text', 'comparison_text', 'candidate', 'similarity', 'passed', 'streak', 'decision_reason', 'blocked_by')}
                                            for frame in trace if frame['text']]})
    lines += ['', '5. 데이터와 해석 범위',
              '실제 음성 추가 추론과, 모델 오인식을 제거한 합성 텍스트 검사를 분리했다.',
              '동일 녹음에 여러 조건/정답 목록을 적용했으므로 검사 건수는 독립 녹음 수와 다르다.',
              '연결/반복/백색잡음은 합성 조건이다. 실제 생활 소음 전체를 대표하지 않는다.',
              '문장 검사는 원문 라벨에서 정답 표현을 찾은 것으로, 조사 등이 붙은 경우도 포함한다.',
              '새로 사람이 청취해 발음을 확정한 라벨은 아니며, 일부 로컬 녹음은 연령 미상이다.',
              '오답 검사는 의도적으로 고른 혼동 상황이다. 실제 서비스 오통과율로 해석하면 안 된다.',
              '정답 표현 출력 여부는 띄어쓰기 경계와 완전한 반복을 검사한다. 단어 내부 부분문자열은 별도 기록했다.',
              '원본/모델/현재 판정 규칙 및 학습 일시정지 상태를 유지했다.',
              '검사 출처 수: ' + json.dumps({k: len(v) for k, v in metadata.items()}, ensure_ascii=False),
              f"전체 검사 {completed['trials']}건. 완료 시각 {completed['finished_at']}",
              '상세 실패 사례와 음성 경로는 examples.json, 전체 내역은 trials.jsonl에 있다.']
    (OUT / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
