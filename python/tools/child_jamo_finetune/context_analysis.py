"""Analyze complete paired context experiments without choosing checkpoints."""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_augmentation as aug
import robustness
from common import ROOT, WORK, digest, dump, load


def cluster_interval(parent_rows, candidate_rows, conditions):
    """Paired bootstrap by speaker; all variants of a clip stay together."""
    candidate = {r['id']: r for r in candidate_rows}
    grouped = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    for p in parent_rows:
        c = candidate[p['id']]
        assert p['speaker'] == c['speaker']
        for condition in conditions:
            pv, cv = p['variants'][condition], c['variants'][condition]
            assert pv['reference_jamo'] == cv['reference_jamo']
            grouped[p['speaker']] += [pv['errors'], cv['errors'], pv['reference_jamo']]
    values = np.stack(list(grouped.values()))
    rng = np.random.default_rng(92831)
    intervals = []
    for _ in range(100):
        indices = rng.integers(0, len(values), size=(100, len(values)))
        total = values[indices].sum(axis=1)
        intervals.extend(((total[:, 1] - total[:, 0]) / total[:, 2]).tolist())
    total = values.sum(axis=0)
    return {'candidate_minus_comparator_jer': (total[1] - total[0]) / total[2],
            'speaker_cluster_bootstrap_95ci': np.quantile(intervals, [.025, .975]).tolist(),
            'speakers': len(values), 'resamples': 10000,
            'scope': 'Exploratory interval on the fixed validation subset; repeated variants are not independent samples.'}


def personal(evaluation):
    return {r['id']: r for r in evaluation['human'] if r['cohort'] == 'personal_diagnostic'}


def summarize_comparison(path):
    comparison = load(path)
    assert comparison['complete'] and comparison['paired_input_schedule_verified']
    parent = comparison['parent']
    control = comparison['arms']['control']['evaluation']
    combined = comparison['arms']['combined']['evaluation']
    conditions = [k for k in aug.EVAL if k != 'original']
    p_human = {r['id']: r for r in parent['human']}
    a_human = {r['id']: r for r in combined['human']}
    p_personal, c_personal, a_personal = map(personal, (parent, control, combined))
    p_correct = {key for key, row in p_human.items() if row['variants']['original']['exact']}
    a_correct = {key for key, row in a_human.items() if row['variants']['original']['exact']}
    ss = next(row for row in a_human.values() if row['human_target'] == '쌕')
    stress = {name: evaluation['stress_summary']['transformed_total'] for name, evaluation in
              [('parent', parent), ('control', control), ('combined', combined)]}
    exact = lambda rows: sum(r['variants']['original']['exact'] for r in rows.values())
    gates = {
        'stress_at_least_10pct_fewer_errors_than_parent': stress['combined']['errors'] <= .9 * stress['parent']['errors'],
        'stress_at_least_10pct_fewer_errors_than_control': stress['combined']['errors'] <= .9 * stress['control']['errors'],
        'clean_regression_within_0_1_percentage_point': combined['clean_valid']['jer'] <= parent['clean_valid']['jer'] + .001,
        'personal_original_at_least_one_more_exact_than_parent_and_control': exact(a_personal) >= max(exact(p_personal), exact(c_personal)) + 1,
        'preserves_all_previously_correct_human_original': p_correct <= a_correct,
        'ssaek_never_rewritten_to_chaek_in_seven_conditions': all(r['text'] != '책' for r in ss['variants'].values()),
    }
    summary = {'seed': comparison['seed'], 'source_sha256': digest(path),
               'paired_input_schedule_verified': True, 'gates': gates,
               'all_gates_passed': all(gates.values()),
               'clean_valid': {name: e['clean_valid'] for name, e in [('parent', parent), ('control', control), ('combined', combined)]},
               'stress': {name: e['stress_summary'] for name, e in [('parent', parent), ('control', control), ('combined', combined)]},
               'human': {name: e['human_cohorts'] for name, e in [('parent', parent), ('control', control), ('combined', combined)]},
               'fit': {name: e['fit_summary'] for name, e in [('parent', parent), ('control', control), ('combined', combined)]},
               'new_correct_human_original': sorted(a_correct - p_correct),
               'lost_correct_human_original': sorted(p_correct - a_correct),
               'stress_uncertainty': {'versus_parent': cluster_interval(parent['stress'], combined['stress'], conditions),
                                      'versus_control': cluster_interval(control['stress'], combined['stress'], conditions)},
               'personal_predictions': [
                   {'id': key, 'human_target': p_personal[key]['human_target'],
                    'parent': p_personal[key]['variants']['original']['text'],
                    'control': c_personal[key]['variants']['original']['text'],
                    'combined': a_personal[key]['variants']['original']['text']}
                   for key in p_personal],
               'personal_condition_outputs': [
                   {'id': key, 'human_target': p_personal[key]['human_target'],
                    'conditions': {condition: {'parent': p_personal[key]['variants'][condition]['text'],
                                                'control': c_personal[key]['variants'][condition]['text'],
                                                'combined': a_personal[key]['variants'][condition]['text']}
                                   for condition in aug.EVAL}}
                   for key in p_personal],
               'training': {arm: {'seconds': value['state']['training_seconds'],
                                  'updates': value['state']['update'], 'exposures': value['state']['exposures'],
                                  'transform_counts': value['state']['counts'], 'gain_capped': value['state']['gain_capped']}
                            for arm, value in comparison['arms'].items()},
               'deployment': False}
    return summary


def main(name):
    directory = WORK / name
    documents = {
        'context-variation-20260927': 'CHILD_JAMO_CONTEXT_VARIATION_2026-09-27.md',
        'context-full-20260927': 'CHILD_JAMO_CONTEXT_FULL_2026-09-27.md',
        'context-span-20260927': 'CHILD_JAMO_CONTEXT_SPAN_2026-09-27.md',
    }
    assert name in documents
    selection = load(directory / 'selection.json')
    summaries = [summarize_comparison(path) for path in sorted(directory.glob('comparison-*.json'))]
    assert summaries, 'No completed comparisons'
    report = {'version': 'context-variation-analysis-v1', 'experiment': name,
              'plan_sha256': digest(directory / 'plan.json'),
              'selection_sha256': digest(directory / 'selection.json'),
              'seeds_complete': len(summaries), 'results': summaries,
              'training_selection': {'recordings': selection['count'], 'speakers': selection['training_speakers'],
                                     'audio_hours': selection['training_hours'],
                                     'single_word_recordings': selection['single_word_allocation'],
                                     'under_1_5_seconds': sum(r['seconds'] < 1.5 for r in selection['train_rows']),
                                     'jamo_types': len({t for r in selection['train_rows'] for t in r['label_ids'] if t >= 2}),
                                     'human_clips_used_for_training': selection['human_examples_used_for_training']},
              'original_model_sha256': digest(robustness.PARENT),
              'original_model_unchanged': digest(robustness.PARENT) == robustness.PARENT_SHA,
              'limitations': ['Corpus scores compare transcript-derived jamo, not expert phonetic labels.',
                              'Twenty-seven human diagnoses are selected development cases and single-listener spellings.',
                              'Seed replication reuses the same validation speakers; it is not a new independent test.',
                              'No original problem is declared solved based only on stable or improved stress predictions.']}
    dump(directory / 'analysis.json', report)
    output_name = 'child-jamo-' + name + '.json'
    dump(ROOT / 'docs/data' / output_name, report)
    document = ROOT / 'docs' / documents[name]
    introduction = document.read_text(encoding='utf-8').split('\n## 학습 결과')[0].rstrip()
    lines = ['\n\n## 학습 결과', '',
             '아래 결과는 고정된 각 2회 학습 종료 모델이다. 변형 출력 변화는 각 모델의 원본 출력과 비교하며, 적을수록 일관적이라는 뜻이다. 정확성은 별도의 오류와 청취 점수로 판단한다.', '',
             '| 모델 | 원본 검증 오류 / 33,182 | 변형 검증 오류 / 34,650 | 변형 출력 변화 / 1,536 | 개인 원본 일치 / 11 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for index, summary in enumerate(summaries):
        for name in (('parent', 'control', 'combined') if index == 0 else ('control', 'combined')):
            model = '기존 모델' if name == 'parent' else f'{summary["seed"]} / ' + ('원본 추가 학습' if name == 'control' else '무음·음량 추가 학습')
            stress = summary['stress'][name]['transformed_total']
            lines.append(f'| {model} | {summary["clean_valid"][name]["errors"]} | {stress["errors"]} | {stress["changed_from_original"]} | {summary["human"][name]["personal_diagnostic"]["original"]["exact"]} |')
    for summary in summaries:
        lines.extend(['', f'### 시드 {summary["seed"]}', '',
                      '| 청취 기준 | 기존 모델 | 원본 추가 학습 | 무음·음량 추가 학습 |',
                      '| --- | --- | --- | --- |'])
        for row in summary['personal_predictions']:
            lines.append(f'| {row["human_target"]} | {row["parent"]} | {row["control"]} | {row["combined"]} |')
        lines.extend(['', '| 사전 판정 조건 | 통과 |', '| --- | --- |'])
        names = ['변형 오류가 기존 모델보다 10% 이상 적음', '변형 오류가 추가 학습 대조군보다 10% 이상 적음',
                 '원본 검증 악화 0.1%p 이내', '개인 원본 일치가 양쪽보다 최소 1개 증가',
                 '기존에 맞히던 청취 원본 유지', '일곱 조건 모두에서 쌕을 책으로 바꾸지 않음']
        for label, passed in zip(names, summary['gates'].values(), strict=True):
            lines.append(f'| {label} | {"예" if passed else "아니오"} |')
        for name, effect in summary['stress_uncertainty'].items():
            low, high = effect['speaker_cluster_bootstrap_95ci']
            label = '기존 모델' if name == 'versus_parent' else '추가 학습 대조군'
            lines.extend(['', f'변형 검증에서 {label} 대비 자모 오류율 차이: {effect["candidate_minus_comparator_jer"] * 100:+.3f}%p, '
                          f'화자 단위 95% bootstrap 구간 [{low * 100:+.3f}, {high * 100:+.3f}]%p. 음수는 변형 학습에 유리하다.'])
    lines.extend(['', f'[개별 지표·판정·해시](data/{output_name})', '',
                  '전사 기반 오류율은 실제 발음 정확도와 같지 않으며, 청취 27개는 선별한 개발 사례다. '
                  '시드 반복은 같은 평가 화자를 사용하므로 새 독립 테스트로 세지 않는다. 원본 모델은 교체하지 않았다.', ''])
    document.write_text(introduction + '\n'.join(lines), encoding='utf-8')
    for summary in summaries:
        print(json.dumps({'seed': summary['seed'], 'gates': summary['gates'],
                          'clean': {k: v['errors'] for k, v in summary['clean_valid'].items()},
                          'stress_errors': {k: v['transformed_total']['errors'] for k, v in summary['stress'].items()},
                          'stress_flips': {k: v['transformed_total']['changed_from_original'] for k, v in summary['stress'].items()},
                          'personal_exact': {k: v['personal_diagnostic']['original']['exact'] for k, v in summary['human'].items()},
                          'uncertainty': summary['stress_uncertainty'], 'personal_predictions': summary['personal_predictions']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='context-variation-20260927')
    main(parser.parse_args().name)
