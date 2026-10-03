"""Secondary evaluation of existing-zero shortening on fixed development inputs.

Primary gates are unchanged. Every nonzero sample and the original convolution
phase are preserved, with the same minimum-length fallback used during training.
Compare all completed paired models, never select a checkpoint from this probe.
"""
from pathlib import Path
import gc
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augmentation
import context_analysis as analysis
import context_pilot as pilot
import context_span_augmentation as policy
import data
import robustness
from common import ROOT, WORK, digest, dump, load
from ownership import Owned

DIRECTORY = WORK / 'context-span-20260927'
SCREEN = WORK / 'context-variation-20260927'
CONDITIONS = {'original': None, 'shortened': None}


def inputs(selection):
    wanted = set(selection['stress_wavs'])
    result = [(dict(id=r['wav'], speaker=r['speaker'], surface=r['surface'], cohort='validation'),
               robustness.verify_read(r), r['label_ids']) for r in data.rows('valid') if r['wav'] in wanted]
    assert len(result) == len(wanted) == 256
    result.extend(pilot.human_inputs())
    assert len(result) == 283 and len({r[0]['id'] for r in result}) == 283
    prepared = []
    for item, wave, target in result:
        changed, meta = policy.apply(wave, policy.TrainingMargin(0, augmentation.minimum_ctc_frames(target), 'none'))
        lo, right = meta['removed_left_zero_samples'], meta['removed_right_zero_samples']
        hi = len(wave) - right
        np.testing.assert_array_equal(changed, wave[lo:hi])
        assert not wave[:lo].any() and not wave[hi:].any()
        assert lo % 320 == right % 320 == 0
        prepared.append((item, wave, target, changed, meta))
    return prepared


def sources(plan):
    models = {'parent': {'checkpoint': str(robustness.PARENT), 'checkpoint_sha256': robustness.PARENT_SHA,
                         'evaluation': str(DIRECTORY / 'baseline-evaluation.json')}}
    for seed in plan['seeds']:
        for family, directory, arm in [('control', SCREEN, 'control'),
                                        ('added_silence', SCREEN, 'combined'),
                                        ('existing_margin', DIRECTORY, 'combined')]:
            result_path = directory / f'seed-{seed}/{arm}/result.json'
            result = load(result_path)
            assert result['complete']
            checkpoint = result_path.parent / 'model.pt'
            models[f'{seed}/{family}'] = {
                'checkpoint': str(checkpoint), 'checkpoint_sha256': result['checkpoint_sha256'],
                'result_sha256': digest(result_path),
                'evaluation': str(result_path.parent / 'final-evaluation.json')}
    for value in models.values():
        assert digest(value['checkpoint']) == value['checkpoint_sha256']
        assert load(value['evaluation'])['complete']
        value['evaluation_sha256'] = digest(value['evaluation'])
    return models


def summarize(rows):
    previous = pilot.aug.EVAL
    try:
        pilot.aug.EVAL = CONDITIONS
        return pilot.summarize_variants(rows)
    finally:
        pilot.aug.EVAL = previous


def main():
    with Owned(DIRECTORY, WORK):
        training = load(DIRECTORY / 'plan.json')
        selection = load(DIRECTORY / 'selection.json')
        assert all(load(DIRECTORY / f'comparison-{seed}.json')['complete'] for seed in training['seeds'])
        plan = {'type': 'secondary development diagnostic; primary gates unchanged',
                'training_plan_sha256': digest(DIRECTORY / 'plan.json'),
                'selection_sha256': digest(DIRECTORY / 'selection.json'),
                'code_sha256': digest(__file__), 'policy_sha256': digest(policy.__file__),
                'conditions': list(CONDITIONS), 'models': sources(training),
                'input_policy': 'Retain >=80ms of available exact-zero edges; remove 320-sample multiples only; unchanged CTC target with minimum-frame fallback.',
                'raw_prediction_policy': 'Reuse source- and checkpoint-hashed primary raw evaluations; additionally rerun all 27 human raw clips to verify checkpoint reload parity.',
                'unchanged_inputs': 'When no samples are removed, reuse the identical raw prediction; every newly shortened waveform is evaluated directly.',
                'extra_training_updates': 0}
        plan_path = DIRECTORY / 'span-probe-plan.json'
        if plan_path.exists():
            assert load(plan_path) == plan
        else:
            dump(plan_path, plan)
        prepared = inputs(selection)
        cohorts = sorted({r[0]['cohort'] for r in prepared})
        audit = {cohort: {'clips': sum(r[0]['cohort'] == cohort for r in prepared),
                          'shortened_clips': sum(r[0]['cohort'] == cohort and len(r[1]) != len(r[3]) for r in prepared),
                          'original_under_1_5s': sum(r[0]['cohort'] == cohort and len(r[1]) < 24000 for r in prepared),
                          'shortened_under_1_5s': sum(r[0]['cohort'] == cohort and len(r[3]) < 24000 for r in prepared),
                          'ctc_length_fallbacks': sum(r[0]['cohort'] == cohort and r[4]['ctc_length_fallback'] for r in prepared)}
                 for cohort in cohorts}
        output = DIRECTORY / 'span-probe.json'
        report = load(output) if output.exists() else {'complete': False, 'plan': plan, 'input_audit': audit, 'models': {}}
        assert report['plan'] == plan and report['input_audit'] == audit
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        pilot.status(status='context_span_probe_running', current_training_running=False,
                     automatic_diagnostics_running=True, active_experiment=str(Path(__file__).relative_to(ROOT)),
                     next_step_status='Comparing safe existing-zero shortening across all seven fixed models on 256 validation and 27 human clips.')
        started = time.monotonic()
        for name, source in plan['models'].items():
            if name in report['models']:
                assert report['models'][name]['reload_parity']
                continue
            model, head = pilot.build_model(training, training['seeds'][0])
            if name != 'parent':
                saved = torch.load(source['checkpoint'], map_location='cpu', weights_only=False)
                model.load_state_dict(saved['model'], strict=True)
                head.load_state_dict(saved['head'], strict=True)
                del saved
            model.eval()
            head.eval()
            expected = load(source['evaluation'])
            expected = {r['id']: r['variants']['original']['ids'] for group in ('stress', 'human') for r in expected[group]}
            rows = []
            parity_count = 0
            for item, wave, target, changed, meta in prepared:
                original_ids = expected[item['id']]
                if item['cohort'] != 'validation':
                    reloaded_ids = robustness.prediction(model, head, wave)
                    assert reloaded_ids == original_ids, f'Reload parity failed: {name} / {item["id"]}'
                    parity_count += 1
                same_input = len(wave) == len(changed)
                changed_ids = original_ids if same_input else robustness.prediction(model, head, changed)
                rows.append({**item, 'input_shortened': not same_input,
                             'variants': {'original': pilot.scores(original_ids, target),
                                          'shortened': {**pilot.scores(changed_ids, target), 'transform': meta}}})
                if len(rows) % 64 == 0:
                    print('SPAN_PROBE', name, len(rows), '/', len(prepared), flush=True)
            summary = {cohort: summarize([r for r in rows if r['cohort'] == cohort]) for cohort in cohorts}
            changed_validation = [r for r in rows if r['cohort'] == 'validation' and r['input_shortened']]
            assert parity_count == 27
            report['models'][name] = {'checkpoint_sha256': source['checkpoint_sha256'], 'reload_parity': True,
                                      'reload_parity_clips': parity_count,
                                      'rows': rows, 'cohorts': summary,
                                      'changed_validation_only': summarize(changed_validation)}
            dump(output, report)
            print('SPAN_PROBE_MODEL', name, 'validation', summary['validation'],
                  'personal', summary['personal_diagnostic'], flush=True)
            del model, head
            gc.collect()
            torch.cuda.empty_cache()
        uncertainty = {}
        for seed in training['seeds']:
            candidate = report['models'][f'{seed}/existing_margin']
            for comparator in ('parent', f'{seed}/control', f'{seed}/added_silence'):
                reference = report['models'][comparator]
                for changed_only in (False, True):
                    choose = lambda r: r['cohort'] == 'validation' and (not changed_only or r['input_shortened'])
                    left, right = [[r for r in entry['rows'] if choose(r)] for entry in (reference, candidate)]
                    uncertainty[f'{seed}_versus_{comparator}_' + ('changed_only' if changed_only else 'all')] = (
                        analysis.cluster_interval(left, right, ['shortened']))
        report.update(complete=True, uncertainty=uncertainty,
                      elapsed_seconds=round(report.get('elapsed_seconds', 0.) + time.monotonic() - started, 2),
                      parent_unchanged=digest(robustness.PARENT) == robustness.PARENT_SHA,
                      note='Exploratory development probe. Human spellings remain provisional; no checkpoint selection or deployment.')
        dump(output, report)
        public = ROOT / 'docs/data/child-jamo-context-span-probe-20260927.json'
        dump(public, report)
        pilot.status(status='context_span_probe_complete', current_training_running=False,
                     automatic_diagnostics_running=False, active_experiment=None,
                     latest_context_span_probe=str(public.relative_to(ROOT)))
        print('SPAN_PROBE_COMPLETE', report['input_audit'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        pilot.status(status='context_span_probe_failed', current_training_running=False,
                     automatic_diagnostics_running=False, active_experiment=None, latest_diagnostic_error=repr(exc))
        raise
