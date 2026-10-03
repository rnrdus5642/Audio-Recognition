"""Fixed paired fine-tuning on identical audio with canonical/realized labels.

This consumes a frozen, source-bound human review experiment under WORK.
The source checkpoint and production manifests are never overwritten.
"""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gc
import hashlib
import math
import sqlite3
import sys
import time

import numpy as np
import soundfile as sf
import torch
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio
import data
import integrity
import labels
import robustness
import train
from common import ROOT, WORK, digest, dump, load
from workspace_paths import physical_path
from ownership import Owned

OUT = ROOT / '.venv/child-jamo-web-20260916/cause-analysis-20260927'
PARENT = WORK / 'main/best.pt'


def status(**changes):
    current = load(OUT / 'work-state.json')
    current.update(changes)
    current['status_updated_at'] = datetime.now().astimezone().isoformat()
    dump(OUT / 'work-state.json', current)


def check_answers(selection):
    for source in selection['reviews']:
        directory = physical_path(source['directory'])
        with closing(sqlite3.connect((directory / 'listening-answers.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            history = [dict(r) for r in db.execute('SELECT * FROM answers ORDER BY revision')]
        if history != source['history']:
            raise RuntimeError('Human answers changed; stop before using stale supervision.')
        if digest(source['report_path']) != source['report_sha256']:
            raise RuntimeError('Human review report changed.')
        if digest(source['probe_path']) != source['probe_sha256']:
            raise RuntimeError('Baseline review probe changed.')


def reviewed_wave(row):
    if digest(row['path']) != row['file_sha256']:
        raise RuntimeError('Reviewed audio file changed: ' + row['id'])
    wave, rate = sf.read(physical_path(row['path']), dtype='float32')
    if rate != 16000 or wave.ndim != 1 or len(wave) != row['samples']:
        raise RuntimeError('Reviewed audio shape changed: ' + row['id'])
    if hashlib.sha256(wave.tobytes()).hexdigest() != row['pcm_sha256']:
        raise RuntimeError('Reviewed PCM changed: ' + row['id'])
    return wave


def rng_digest():
    h = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    for state in torch.cuda.get_rng_state_all():
        h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def scored(ids, target):
    hypothesis, reference = robustness.compact(ids), robustness.compact(target)
    errors = Levenshtein.distance(reference, hypothesis)
    return {'errors': errors, 'reference_jamo': len(reference), 'exact': errors == 0}


def summarize(rows):
    errors = sum(row['errors'] for row in rows)
    denominator = sum(row['reference_jamo'] for row in rows)
    return {'recordings': len(rows), 'exact': sum(row['exact'] for row in rows),
            'errors': errors, 'reference_jamo': denominator, 'jer': errors / denominator if denominator else None}


def evaluate(model, head, selection, config, directory, name):
    started = time.monotonic()
    model.eval()
    head.eval()
    result = {'name': name, 'complete': False, 'selection_sha256': config['selection_sha256'],
              'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
              'reviewed_train': [], 'reviewed_valid': [], 'diagnostics': [], 'test_scored': False}
    for group in ('train', 'valid'):
        for row in selection[group + '_reviewed']:
            ids = robustness.prediction(model, head, reviewed_wave(row))
            if name == 'baseline':
                assert ids == row['baseline_prediction_ids'], ('Parent parity', row['id'])
            result['reviewed_' + group].append({
                'id': row['id'], 'wav': row['wav'], 'speaker': row['speaker'],
                'human_target': row['realized_surface'], 'canonical': row['canonical_surface'],
                **robustness.describe_ids(ids), **scored(ids, row['realized_ids']),
                'canonical_rewrite': row['label_differs'] and robustness.compact(ids) == robustness.compact(row['canonical_ids'])})
        result[group + '_summary'] = summarize(result['reviewed_' + group])
    print('EVALUATE', name, 'heldout', result['valid_summary'], flush=True)
    valid = data.rows('valid')
    result['clean_valid'] = train.evaluate(model, head, valid, data.plan(valid, 640000, 16, 927, 'robustness-valid'))
    sys.path.insert(0, str(ROOT / 'python/tools/child_jamo_web'))
    import engine
    for row in selection['diagnostic_entries']:
        wave = engine.read_row(row['row'])
        assert engine.pcm_sha256(wave) == row['variants']['baseline']['pcm_sha256']
        ids = robustness.prediction(model, head, wave)
        if name == 'baseline':
            assert ids == row['variants']['baseline']['ids'], ('Diagnostic parity', row['id'])
        target = [labels.VOCAB[j] for j in labels.jamo(row['human_target'])] if row['human_target'] else None
        result['diagnostics'].append({'id': row['id'], 'cohort': row['cohort'], 'human_target': row['human_target'],
                                      **robustness.describe_ids(ids),
                                      'human_exact': robustness.compact(ids) == robustness.compact(target) if target else None})
    result['personal_exact'] = sum(row['human_exact'] is True for row in result['diagnostics'] if row['cohort'] == 'personal_diagnostic')
    result['personal_judged'] = sum(row['human_exact'] is not None for row in result['diagnostics'] if row['cohort'] == 'personal_diagnostic')
    result['complete'] = True
    result['elapsed_seconds'] = round(time.monotonic() - started, 2)
    dump(directory / f'{name}-evaluation.json', result)
    print('EVALUATED', name, 'clean JER', result['clean_valid']['jer'],
          'personal', result['personal_exact'], '/', result['personal_judged'], flush=True)
    return result


def run_arm(arm, root, selection, plan, common_config):
    run = root / arm
    run.mkdir(exist_ok=True)
    config = {**common_config, 'arm': arm}
    if (run / 'config.json').exists():
        assert load(run / 'config.json') == config, 'Configuration drift'
    else:
        dump(run / 'config.json', config)
    if (run / 'result.json').exists():
        result = load(run / 'result.json')
        assert result['complete'] and digest(run / 'model.pt') == result['checkpoint_sha256']
        return result
    check_answers(selection)
    status(status='paired_label_comparison_loading', current_training_running=False,
           automatic_diagnostics_running=True, trial_models_training=False,
           active_experiment=str(Path(__file__).relative_to(ROOT)), label_comparison_arm=arm,
           next_step_status=f'Loading the original checkpoint for the {arm} arm of the fixed label comparison.')
    args = argparse.Namespace(seed=plan['seed'], attention='sdpa', mask_time_prob=0.0, mask_feature_prob=0.0)
    model, head = train.build(args)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    model.load_state_dict(parent['model'], strict=True)
    head.load_state_dict(parent['head'], strict=True)
    del parent
    model.config.apply_spec_augment = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.encoder.layers[-plan['last_layers']:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    encoder_parameters = [p for p in model.parameters() if p.requires_grad]
    parameters = encoder_parameters + list(head.parameters())
    optimizer = torch.optim.AdamW([{'params': encoder_parameters, 'lr': plan['encoder_lr']},
                                   {'params': head.parameters(), 'lr': plan['head_lr']}],
                                  weight_decay=plan['weight_decay'], foreach=False)
    state = {'update': 0, 'training_seconds': 0.0, 'trace': [],
             'trainable_parameters': sum(p.numel() for p in parameters)}
    baseline_path = root / 'baseline-evaluation.json'
    if not baseline_path.exists():
        status(status='paired_label_comparison_baseline_evaluation', current_training_running=False,
               next_step_status='Reproducing ten reviewed and nineteen prior diagnostic predictions and scoring 1500 validation clips before learning.')
        evaluate(model, head, selection, config, root, 'baseline')
    else:
        baseline = load(baseline_path)
        assert baseline['complete'] and baseline['selection_sha256'] == config['selection_sha256']
        assert baseline['parent_checkpoint_sha256'] == config['parent_checkpoint_sha256']
    if (run / 'last.pt').exists():
        saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['config'] == config
        model.load_state_dict(saved['model'], strict=True)
        head.load_state_dict(saved['head'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        state = saved['state']
        train.set_rng_state(saved['rng'])
        del saved
        print('RESUME', arm, state['update'], flush=True)
    else:
        train.seed_everything(plan['seed'])
    reviewed = {row['id']: row for row in selection['train_reviewed']}
    replay = {row['wav']: row for row in selection['replay_rows']}
    cache = {key: reviewed_wave(row) for key, row in reviewed.items()}
    cache.update({key: robustness.verify_read(row) for key, row in replay.items()})
    elapsed = state['training_seconds']
    clock = time.monotonic()
    status(status='paired_label_comparison_training', current_training_running=True,
           trial_models_training=True, training_started=True,
           next_step_status=f'Training {arm}: 72 fixed updates, same six reviewed clips and 216 unchanged replay clips in both arms.')
    def persist(stage):
        state['training_seconds'] = elapsed + time.monotonic() - clock
        state['status'] = stage
        train.save(run / 'last.pt', {'model': model.state_dict(), 'head': head.state_dict(),
                   'optimizer': optimizer.state_dict(), 'state': state, 'rng': train.rng_state(), 'config': config})
        dump(run / 'progress.json', {k: v for k, v in state.items() if k != 'trace'})
    model.train()
    head.train()
    for step_index in range(state['update'], plan['updates_each']):
        step = selection['updates'][step_index]
        human_row = reviewed[step['reviewed_id']]
        rows = [dict(human_row, label_ids=human_row[arm + '_ids'])] + [replay[name] for name in step['replay_wavs']]
        keys = [human_row['id']] + step['replay_wavs']
        assert len(rows) == plan['utterances_per_update']
        trace = {'update': step_index + 1, 'audio_ids': keys, 'rng_before_update_sha256': rng_digest(),
                 'pcm_sha256': [row['pcm_sha256'] for row in rows], 'label_ids': [row['label_ids'] for row in rows]}
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for key, row in zip(keys, rows, strict=True):
            batch = robustness.pack([cache[key]], [row])
            logits, lengths = train.forward(model, head, batch, grad=True, autocast=True)
            loss = train.ctc_sum(logits, lengths, batch)
            assert torch.isfinite(loss)
            (loss / len(rows)).backward()
            losses.append(float(loss.detach()))
        update = step_index + 1
        factor = min(1.0, update / plan['warmup_updates']) * (.1 + .9 * (1 + math.cos(math.pi * update / plan['updates_each'])) / 2)
        optimizer.param_groups[0]['lr'] = plan['encoder_lr'] * factor
        optimizer.param_groups[1]['lr'] = plan['head_lr'] * factor
        norm = torch.nn.utils.clip_grad_norm_(parameters, plan['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state.update(update=update, mean_loss=float(np.mean(losses)), grad_norm=float(norm))
        state['trace'].append(trace)
        if update == 1 or update % 6 == 0:
            state.update(training_seconds=elapsed + time.monotonic() - clock, status='training',
                         cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            dump(run / 'progress.json', {k: v for k, v in state.items() if k != 'trace'})
            print('TRAIN', arm, update, '/', plan['updates_each'], 'loss', round(state['mean_loss'], 4),
                  'seconds', round(state['training_seconds'], 1), flush=True)
        if update % 36 == 0:
            check_answers(selection)
            persist('training' if update < plan['updates_each'] else 'evaluating')
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    status(status='paired_label_comparison_evaluating', current_training_running=False,
           trial_models_training=False, next_step_status=f'{arm} completed 72 updates; evaluating held-out pronunciation choices, 1500 general validation recordings and original diagnostics.')
    evaluation = evaluate(model, head, selection, config, run, 'final')
    train.save(run / 'model.pt', {'model': model.state_dict(), 'head': head.state_dict(),
                                 'config': config, 'valid': evaluation['clean_valid']})
    state['status'] = 'complete'
    dump(run / 'progress.json', {k: v for k, v in state.items() if k != 'trace'})
    result = {'complete': True, 'config': config, 'state': state, 'evaluation': evaluation,
              'checkpoint_sha256': digest(run / 'model.pt'), 'deployed': False}
    dump(run / 'result.json', result)
    print('ARM_COMPLETE', arm, flush=True)
    del model, head, encoder_parameters, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compare(root, selection, plan, arms):
    canonical, realized = arms['canonical'], arms['realized']
    configs = [{k: v for k, v in r['config'].items() if k != 'arm'} for r in (canonical, realized)]
    assert configs[0] == configs[1]
    assert len(canonical['state']['trace']) == len(realized['state']['trace']) == plan['updates_each']
    differing_target_exposures = 0
    for a, b in zip(canonical['state']['trace'], realized['state']['trace'], strict=True):
        assert {k: v for k, v in a.items() if k != 'label_ids'} == {k: v for k, v in b.items() if k != 'label_ids'}
        differing_target_exposures += sum(x != y for x, y in zip(a['label_ids'], b['label_ids'], strict=True))
        assert a['label_ids'][1:] == b['label_ids'][1:], 'Replay supervision differs'
    assert differing_target_exposures == 48
    evaluations = {'parent': load(root / 'baseline-evaluation.json'),
                   'canonical': canonical['evaluation'], 'realized': realized['evaluation']}
    parent, after = evaluations['parent'], evaluations['realized']
    before_diag = {r['id']: r for r in parent['diagnostics']}
    after_diag = {r['id']: r for r in after['diagnostics']}
    improved = [key for key in before_diag if before_diag[key]['human_exact'] is False and after_diag[key]['human_exact'] is True]
    regressed = [key for key in before_diag if before_diag[key]['human_exact'] is True and after_diag[key]['human_exact'] is False]
    heldout_before = {r['id']: r for r in parent['reviewed_valid']}
    heldout_after = {r['id']: r for r in after['reviewed_valid']}
    heldout_improved = [key for key in heldout_before if not heldout_before[key]['exact'] and heldout_after[key]['exact']]
    heldout_regressed = [key for key in heldout_before if heldout_before[key]['exact'] and not heldout_after[key]['exact']]
    rewrites = {name: sum(row['canonical_rewrite'] for row in evaluation['reviewed_valid']) for name, evaluation in evaluations.items()}
    checks = {
        'heldout_jamo_error_below_parent_and_canonical': after['valid_summary']['jer'] < min(evaluations[name]['valid_summary']['jer'] for name in ('parent', 'canonical')),
        'heldout_canonical_rewrites_below_parent_and_canonical': rewrites['realized'] < min(rewrites['parent'], rewrites['canonical']),
        'additional_exact_heldout_without_regression': bool(heldout_improved) and not heldout_regressed,
        'clean_validation_within_tolerance': after['clean_valid']['jer'] <= parent['clean_valid']['jer'] + .001,
        'preserves_previously_correct_diagnostics': not regressed,
        'does_not_turn_human_ssaek_into_chaek': after_diag['personal:child1_17']['text'] != '책'}
    report = {'complete': True, 'plan': plan, 'paired_config_audio_order_rng_verified': True,
              'utterance_exposures_each': plan['updates_each'] * plan['utterances_per_update'],
              'changed_target_exposures_each': differing_target_exposures,
              'checks': checks, 'all_expansion_checks_passed': all(checks.values()),
              'heldout_improved': heldout_improved, 'heldout_regressed': heldout_regressed,
              'diagnostic_improved': improved, 'diagnostic_regressed': regressed,
              'canonical_rewrites_on_four_heldout': rewrites,
              'evaluations': evaluations,
              'checkpoint_sha256': {name: r['checkpoint_sha256'] for name, r in arms.items()},
              'training_seconds': {name: r['state']['training_seconds'] for name, r in arms.items()},
              'parent_checkpoint_preserved': digest(PARENT) == plan['parent_checkpoint_sha256'],
              'deployed': False, 'test_scored': False,
              'original_problem_solved': False,
              'limitation': 'One small fixed pilot with single-listener targets; training fit and four-utterance held-out behavior do not establish broad accuracy or the original error cause.'}
    assert report['parent_checkpoint_preserved']
    check_answers(selection)
    dump(root / 'comparison.json', report)
    dump(ROOT / 'docs/data/child-jamo-label-comparison-20260927.json', report)
    status(status='paired_label_comparison_complete', automatic_diagnostics_running=False,
           current_training_running=False, trial_models_training=False, trial_models_trained=True,
           active_experiment=None, label_comparison_arm=None,
           latest_trial_summary='docs/data/child-jamo-label-comparison-20260927.json',
           label_comparison_result=str(root / 'comparison.json'),
           next_step_status='Both fixed 72-update label arms and all evaluations completed. Interpret held-out effects, original diagnostic regressions and the small-sample limits; no model is deployed and no new listening batch is pending.')
    print('COMPARISON_COMPLETE', {name: {'heldout': r['valid_summary'], 'clean': r['clean_valid']['jer'], 'personal': r['personal_exact']} for name, r in evaluations.items()}, flush=True)
    print('CHECKS', checks, flush=True)


def main(name):
    root = (WORK / name).resolve()
    if root.parent != WORK.resolve() or not name.startswith('label-comparison-'):
        raise ValueError('Unexpected run path')
    with Owned(root, WORK):
        selection, plan = load(root / 'selection.json'), load(root / 'plan.json')
        assert plan['selection_sha256'] == digest(root / 'selection.json')
        assert digest(PARENT) == plan['parent_checkpoint_sha256']
        assert integrity.contract({'train': 'train-repaired-20260927', 'valid': 'valid'}) == selection['data_contract']
        check_answers(selection)
        common_config = {**plan, 'data_contract': selection['data_contract'],
                         'source_code_sha256': {name: digest(Path(__file__).parent / name) for name in
                                                ('label_pilot.py', 'robustness.py', 'train.py', 'audio.py', 'data.py', 'labels.py', 'common.py')},
                         'torch_version': torch.__version__, 'numpy_version': np.__version__}
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            arms = {arm: run_arm(arm, root, selection, plan, common_config) for arm in plan['arms']}
            compare(root, selection, plan, arms)
        except BaseException as exc:
            dump(root / 'failure.json', {'error': f'{type(exc).__name__}: {exc}',
                                        'at': datetime.now(timezone.utc).isoformat()})
            status(status='paired_label_comparison_failed', automatic_diagnostics_running=False,
                   current_training_running=False, trial_models_training=False, active_experiment=None,
                   next_step_status=f'Label comparison stopped: {type(exc).__name__}: {exc}. Inspect isolated run logs before continuing.')
            raise
        finally:
            audio.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='label-comparison-20260927')
    main(parser.parse_args().name)
