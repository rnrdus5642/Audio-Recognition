"""Summarize the saved length-window experiment without running inference."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    out = args.directory.resolve()
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    complete = json.loads((out / 'complete.json').read_text(encoding='utf-8'))
    with gzip.open(out / 'results.jsonl.gz', 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    groups = summary['groups']
    probes = [('일반 단독 발화', 'general/single'),
              ('58개 녹음: 단독', 'paired_single/alone'),
              ('58개 녹음: 앞말 바로 뒤', 'paired_single/speech_before'),
              ('58개 녹음: 앞말 뒤 0.25초 간격', 'paired_single/speech_before_gap'),
              ('58개 녹음: 정답 뒤에 다른 말', 'paired_single/speech_after'),
              ('여러 정답 후보: 정답 뒤에 다른 말', 'paired_stage/speech_after'),
              ('정답이 들어간 자연 문장', 'natural_context/sentence'),
              ('오답 문장: 잘못 통과한 수', 'new_distractor/unrelated_sentence'),
              ('다른 활동의 단어: 잘못 통과한 수', 'curriculum/negative'),
              ('60초 다른 말 듣기: 잘못 통과한 수', 'long_negative/unrelated_speech_60s')]
    lines = ['정답 길이 1.2~1.5배 비교 실험',
             f"완료: {complete['finished_at']}", '',
             '결론',
             '- 발음기호 기준 1.2배는 기존에 통과하던 발화도 누락시켰다.',
             '- 1.4배, 1.5배, 문장 전체의 단어 구간을 검사하되 길이를 제한하지 않은 방식은 이번 자료에서 판정·후보·확정시각이 같았다.',
             '- 개선은 문장 끝에만 있던 검색을 중간의 완전한 단어까지 확장한 효과였다. 배수 제한 자체의 추가 개선은 확인하지 못했다.',
             '- 정답 뒤의 말에 대한 검출은 개선됐지만, 오답 문장 통과도 17/1080에서 19/1080으로 늘었다.',
             '- 실험 후보로는 1.4~1.5배를 남길 수 있으나, 길이만으로 해결되지 않는 오통과가 있어 제품 적용은 하지 않았다.', '',
             '실험 정의',
             '- 최대 비교 길이 = floor(정답 길이 × 배수). 소수점 이하를 버렸다. 별도 최소 길이 제한은 추가하지 않았다.',
             '- 한글 글자 수와 G2P 발음기호 개수를 각각 측정했다. 모델 입력 오디오 2.5초는 변경하지 않았다.',
             '- 문장 안의 띄어쓰기 경계로 구분된 연속 구간을 후보별로 비교했다. 단어 내부를 임의로 자르지 않았다.',
             '- 기존 반복 단어 정리를 적용한 뒤 길이를 검사했다. 원본 인식 텍스트는 보존했다.',
             '- 음량 검사, 85점, 두 번 확인, 한 번의 불확실한 검사 허용, 동점 거절, 다른 동작 차단은 현재 v4와 동일하다.',
             '- 현재 v4 / 문장 끝만 검사하며 1.2·1.5배 / 문장 전체 구간의 무제한 검사 / 두 길이 기준 × 네 배수 = 12개 설정.', '',
             f"자료: {summary['trials']:,}개 판정 사례 × {summary['variants']}개 설정 = {complete['policy_runs']:,}회 정책 실행.",
             f"기존 v4 결과 재현: {summary['baseline_parity']:,}개 모두 통과·선택 후보·종료시각 일치.",
             '기존 실제 녹음에서 저장한 모델 출력을 재사용했다. 이번에 새 오디오 추론이나 GPU 사용은 없었다.',
             '같은 녹음을 여러 조건과 정답 목록에서 재사용했으므로 검사 수는 독립 녹음 수가 아니다.',
             '이 자료는 개발용 자료이며, 실제 서비스 정확도나 오통과율을 추정하는 독립 평가셋은 아니다.', '']
    for unit, title in [('phoneme', '발음기호 길이 기준'), ('syllable', '한글 글자 수 기준')]:
        lines += [title, '조건 | 현재 | 1.2배 | 1.3배 | 1.4배 | 1.5배']
        keys = ['current_v4'] + [f'whole_any_{unit}_{r}' for r in ('1.2', '1.3', '1.4', '1.5')]
        for name, group in probes:
            values = groups[group]
            lines.append(name + ' | ' + ' | '.join(f"{values[k]['accepted']}/{values[k]['trials']}" for k in keys))
        lines.append('')
    pairs = [('whole_any_phoneme_1.4', 'whole_any_phoneme_1.5'),
             ('whole_any_phoneme_1.5', 'whole_any_unlimited'),
             ('current_v4', 'suffix_phoneme_1.5')]
    equalities = []
    lines += ['원인 분리 검사']
    for a, b in pairs:
        diff = [r['id'] for r in rows if tuple(r['results'][a][k] for k in ('accepted', 'target', 'seconds')) !=
                tuple(r['results'][b][k] for k in ('accepted', 'target', 'seconds'))]
        equalities.append({'a': a, 'b': b, 'different_decisions': len(diff), 'ids': diff})
        lines.append(f'- {a} / {b}: 판정·후보·시각이 다른 사례 {len(diff)}개')
    lines += ['', '해석에 도움이 되는 사례',
              '- 들어(발음기호 4개)와 들여(5개): 1.2배는 최대 4개이므로 85점인 결과도 비교 대상에서 제외했다. 1.3배부터 허용된다.',
              '- 놓아(3개): 노아 → 노와 → 노아에서 중간의 노와(4개)가 1.2배에서 제외돼 기존의 한 번 유예 처리가 끊겼다. 1.4배는 이를 보존했다.',
              '- 두드려 고양이: 문장 중간의 두드려를 별도로 비교해 기존 미확정을 확정으로 바꿨다.',
              '- 안녕하세요들어처럼 띄어쓰기 없이 붙은 경우는 이번 방식으로 해결되지 않았다(합성 텍스트 0/199).',
              '- 책상→책 등 단어 내부 잘라내기 검사 8개에서는 모든 설정이 오통과 0건이었다.', '',
              '새로 생긴 오답 통과']
    new_false = []
    for row in rows:
        a, b = row['results']['current_v4'], row['results']['whole_any_phoneme_1.4']
        if row['phase'] in ('new_distractor', 'long_negative') and not a['accepted'] and b['accepted']:
            item = {k: row.get(k) for k in ('id', 'reference', 'targets', 'audio_path')} | {'result': b}
            new_false.append(item)
            lines.append(f"- {row['id']}: 인식 [{b['text']}]의 [{b['comparison_text']}] → 정답 [{b['target']}], {b['similarity'] * 100:.2f}점")
    lines += ['', '저장된 사용자 기록 주의 사례']
    user_probe = next(row for row in rows if row['id'] == 'user:35ec27b8ae315273aaefeb9af8001ddb')
    value = user_probe['results']['whole_any_phoneme_1.4']
    lines += [f"- [{value['comparison_text']}]를 [{value['target']}]로 {value['similarity'] * 100:.2f}점에 통과시켰다.",
              '- 사용자 기록 3/4 → 4/4 통과를 성능 향상으로 계산하면 안 된다. 정답 라벨이 새로 확인된 평가자료가 아니며 위처럼 다른 표현도 통과한다.', '',
              '산출물과 적용 상태',
              '- plan.json: 비교 정의·입력 및 소스 해시',
              '- inputs.jsonl.gz / results.jsonl.gz: 재현용 입력과 모든 설정의 결과',
              '- changes.jsonl.gz: 결과가 달라진 사례의 프레임별 원인',
              '- summary.json / analysis.json: 집계와 추가 비교',
              '- 현재 웹사이트, Unity 패키지, 모델, 학습 일시정지 상태는 변경하지 않았다.',
              f"- 보존 확인: {complete['production_and_pause_unchanged']}"]
    (out / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'analysis.json').write_text(json.dumps({'decision_equalities': equalities,
                                                 'new_false_accepts': new_false,
                                                 'user_probe': user_probe,
                                                 'phase_counts': dict(Counter(r['phase'] for r in rows))},
                                                ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(out / 'report.txt'), 'rows': len(rows),
                      'new_false_cases': len(new_false)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
