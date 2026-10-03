"""Paired, resumable 20-ms fine-tuning for silence/gain robustness.

Corpora and human diagnoses are frozen before learning. Both arms process
the same complete epochs and minibatches. Human evaluation clips never
enter this training selection. The production checkpoint is never changed.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_augmentation as aug
import data
import integrity
import label_pilot
import labels
import reviewed
import robustness
import train
from common import ROOT, WORK, digest, dump, load
from ownership import Owned

OUT = robustness.DIAGNOSTICS
LABEL_SELECTION = WORK / 'label-comparison-20260927/selection.json'
AUDIT = OUT / 'global-jamo-error-audit.json'
VERSION = 'child-jamo-context-pilot-v1'


def status(**kwargs):
    label_pilot.status(**kwargs)


def save_checkpoint(path, payload):
    temporary = Path(str(path) + '.tmp')
    torch.save(payload, temporary)
    for attempt in range(100):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def prepare(root, count):
    review = load(LABEL_SELECTION)
    label_pilot.check_answers(review)
    contract = integrity.contract({'train': robustness.TRAIN_SPLIT, 'valid': 'valid', 'test': 'test'})
    assert digest(robustness.PARENT) == robustness.PARENT_SHA
    bindings = {'labels_selection': digest(LABEL_SELECTION), 'human_audit': digest(AUDIT)}
    if (root / 'selection.json').exists():
        result = load(root / 'selection.json')
        assert result['bindings'] == bindings and result['contract'] == contract and result['count'] == count
        return result
    rows = data.rows(robustness.TRAIN_SPLIT)
    reviewed.require_repaired(rows)
    diagnostics = review['diagnostic_entries']
    excluded_wavs = {r['wav'] for group in ('train_reviewed', 'valid_reviewed') for r in review[group]}
    excluded_speakers = {r['speaker'] for group in ('train_reviewed', 'valid_reviewed') for r in review[group]}
    excluded_wavs.update(r['row']['wav'] for r in diagnostics)
    excluded_speakers.update(r['row'].get('speaker') for r in diagnostics)
    # Prevent learning either the judged recording or its speaker in this adaptation.
    eligible = [r for r in rows if r['speaker'] not in excluded_speakers]
    chosen, singles = robustness.select_training(eligible, count, 92801, excluded_wavs)
    valid, test = data.rows('valid'), data.rows('test')
    assert not ({r['speaker'] for r in chosen} & {r['speaker'] for r in valid + test})
    assert not ({r['pcm_sha256'] for r in chosen} & {r['pcm_sha256'] for r in valid + test})
    # A fixed validation stress subset covers all 40 jamo; no predictions used.
    order = sorted(valid, key=lambda r: robustness.rank(r, 'context-stress-v1'))
    stress, seen, missing = [], set(), set(range(2, len(labels.TOKENS)))
    for row in order:
        if missing & set(row['label_ids']):
            stress.append(row['wav'])
            seen.add(row['wav'])
            missing -= set(row['label_ids'])
    assert not missing
    for row in order:
        if len(stress) >= 256:
            break
        if row['wav'] not in seen:
            stress.append(row['wav'])
            seen.add(row['wav'])
    result = {'version': VERSION, 'count': count, 'contract': contract, 'bindings': bindings,
              'parent_sha256': robustness.PARENT_SHA, 'train_rows': chosen,
              'training_speakers': len({r['speaker'] for r in chosen}),
              'training_hours': sum(r['seconds'] for r in chosen) / 3600,
              'single_word_allocation': singles, 'excluded_wavs': sorted(excluded_wavs),
              'excluded_speakers': sorted(s for s in excluded_speakers if s),
              'stress_wavs': stress,
              'fit_wavs': [r['wav'] for r in sorted(chosen, key=lambda r: robustness.rank(r, 'context-fit'))[:32]],
              'selection_uses_predictions': False, 'human_examples_used_for_training': 0,
              'test_scored': False}
    dump(root / 'selection.json', result)
    return result


def make_plan(root, selection, epochs):
    plan = {'version': VERSION, 'created_at': datetime.now().astimezone().isoformat(),
            'selection_sha256': digest(root / 'selection.json'), 'epochs': epochs,
            'count': selection['count'], 'seeds': [92801, 92802],
            'arms': ['control', 'combined'], 'last_layers': 8,
            'encoder_lr': 1e-5, 'head_lr': 5e-5, 'warmup_updates': 32,
            'batch_budget': 320000, 'max_batch': 4, 'accumulate': 2,
            'stride_ms': 20, 'augmentation': aug.VERSION,
            'training_transform': '50% original; remaining examples split among zero padding, gain, both. Each pad independently 0-1.2s; requested gain -15 to +6dB, positive gain capped without clipping.',
            'padding_mask': 'Added silence is real input with mask=1; batch fill only is mask=0.',
            'loss': 'Same transcript-derived targets and length-normalized CTC in both arms.',
            'dropout': 'Disabled in both arms to isolate the input intervention; SpecAugment and layerdrop also disabled.',
            'evaluation': 'All 1500 clean validation clips; fixed 256 covering 40 jamo x 7 conditions; 27 existing human diagnoses x 7; 32 training clips x 7 for fit only.',
            'fixed_final_epoch': True, 'no_personal_checkpoint_selection': True,
            'gate': {'stress_relative_error_reduction_vs_parent_and_control': .10,
                     'clean_jer_max_absolute_increase_vs_parent': .001,
                     'personal_raw_exact_min_gain_vs_parent_and_control': 1,
                     'preserve_all_parent_correct_human_raw': True,
                     'do_not_rewrite_human_ssaek_to_chaek': True},
            'interpretation': 'A gain only on stress conditions is robustness improvement, not resolution of original errors. An inconclusive single seed requires replication or a diagnosed follow-up, not a success claim.',
            'deployment': False}
    path = root / 'plan.json'
    if path.exists():
        old = load(path)
        plan['created_at'] = old['created_at']
        assert old == plan
    else:
        dump(path, plan)
    return plan


def scores(ids, target):
    return {**robustness.describe_ids(ids), **label_pilot.scored(ids, target)}


def summarize_variants(entries):
    output = {}
    for condition in aug.EVAL:
        output[condition] = label_pilot.summarize([r['variants'][condition] for r in entries])
        output[condition]['changed_from_original'] = sum(
            robustness.compact(r['variants'][condition]['ids']) != robustness.compact(r['variants']['original']['ids'])
            for r in entries)
        output[condition]['original_correct_lost'] = sum(
            r['variants']['original']['exact'] and not r['variants'][condition]['exact'] for r in entries)
    transformed = [r['variants'][condition] for r in entries for condition in aug.EVAL if condition != 'original']
    output['transformed_total'] = label_pilot.summarize(transformed)
    output['transformed_total']['changed_from_original'] = sum(output[c]['changed_from_original'] for c in aug.EVAL if c != 'original')
    return output


def variant_records(model, head, inputs):
    result = []
    for item, wave, target in inputs:
        row = {**item, 'variants': {}}
        for name, (left, right, db) in aug.EVAL.items():
            changed, meta = aug.apply(wave, left, right, db)
            ids = robustness.prediction(model, head, changed)
            row['variants'][name] = {**scores(ids, target), 'transform': meta}
        result.append(row)
    return result


def human_inputs():
    selection = load(LABEL_SELECTION)
    label_pilot.check_answers(selection)
    source = {r['id']: r for group in ('train_reviewed', 'valid_reviewed') for r in selection[group]}
    source.update({r['id']: r for r in selection['diagnostic_entries'] if r['human_target']})
    sys.path.insert(0, str(ROOT / 'python/tools/child_jamo_web'))
    import engine
    inputs = []
    for judgment in load(AUDIT)['human_judged']['rows']:
        r = source[judgment['id']]
        if 'row' in r:
            wave = engine.read_row(r['row'])
            assert engine.pcm_sha256(wave) == r['variants']['baseline']['pcm_sha256']
        else:
            wave = label_pilot.reviewed_wave(r)
        item = {'id': judgment['id'], 'cohort': judgment['cohort'],
                'human_target': judgment['surface'], 'speaker': judgment.get('speaker'),
                'baseline_ids': judgment['ids']}
        target = [labels.VOCAB[j] for j in labels.jamo(judgment['surface'])]
        inputs.append((item, wave, target))
    return inputs


def evaluate(model, head, selection, directory, label):
    path = directory / (label + '-evaluation.json')
    if path.exists() and load(path).get('complete'):
        return load(path)
    started = time.monotonic()
    model.eval()
    head.eval()
    status(status='context_variation_evaluating', current_training_running=False,
           next_step_status=f'Evaluating {label}: clean, human judgments, and fixed silence/gain variants.')
    valid = data.rows('valid')
    # Batched clean evaluation has exactly the same layout for every model.
    clean = train.evaluate(model, head, valid, data.plan(valid, 640000, 16, 927, 'robustness-valid'))
    result = {'complete': False, 'name': label, 'clean_valid': clean, 'test_scored': False}
    dump(path, result)
    print('CLEAN', label, clean, flush=True)
    human = variant_records(model, head, human_inputs())
    if label == 'baseline':
        assert all(r['variants']['original']['ids'] == r['baseline_ids'] for r in human), 'Parent human prediction drift'
    result['human'] = human
    result['human_summary'] = summarize_variants(human)
    result['human_cohorts'] = {cohort: summarize_variants([r for r in human if r['cohort'] == cohort])
                               for cohort in sorted({r['cohort'] for r in human})}
    dump(path, result)
    print('HUMAN', label, {c: r['original']['exact'] for c, r in result['human_cohorts'].items()}, flush=True)
    for group, rows in [('stress', [r for r in valid if r['wav'] in set(selection['stress_wavs'])]),
                        ('fit', [r for r in selection['train_rows'] if r['wav'] in set(selection['fit_wavs'])])]:
        entries = []
        for start in range(0, len(rows), 32):
            batch = [(dict(id=r['wav'], speaker=r['speaker'], surface=r['surface']), robustness.verify_read(r), r['label_ids'])
                     for r in rows[start:start + 32]]
            entries.extend(variant_records(model, head, batch))
            print('VARIANTS', label, group, len(entries), '/', len(rows), flush=True)
        result[group] = entries
        result[group + '_summary'] = summarize_variants(entries)
        dump(path, result)
    result.update(complete=True, elapsed_seconds=round(time.monotonic() - started, 2))
    dump(path, result)
    return result


def build_model(plan, seed):
    model, head = train.build(argparse.Namespace(seed=seed, attention='sdpa', mask_time_prob=0., mask_feature_prob=0.))
    checkpoint = torch.load(robustness.PARENT, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    head.load_state_dict(checkpoint['head'], strict=True)
    del checkpoint
    assert list(model.config.conv_stride) == [5, 2, 2, 2, 2, 2, 2]
    model.config.apply_spec_augment = False
    for module in list(model.modules()) + list(head.modules()):
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
        if isinstance(getattr(module, 'dropout', None), (float, int)):
            module.dropout = 0.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.encoder.layers[-plan['last_layers']:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    return model, head


def run_arm(root, selection, plan, seed, arm):
    directory = root / f'seed-{seed}' / arm
    directory.mkdir(parents=True, exist_ok=True)
    batches_by_epoch = [data.groups(data.plan(
        [dict(r, samples=r['samples'] + 2 * aug.MAX_SIDE_SAMPLES) for r in selection['train_rows']],
        plan['batch_budget'], plan['max_batch'], seed, f'context:{epoch}'), plan['accumulate']) for epoch in range(plan['epochs'])]
    total = sum(map(len, batches_by_epoch))
    config = {'plan_sha256': digest(root / 'plan.json'), 'seed': seed, 'arm': arm,
              'total_updates': total, 'recording_exposures': plan['epochs'] * len(selection['train_rows']),
              'source_code_sha256': {name: digest(Path(__file__).parent / name) for name in
                                     ('context_pilot.py', 'context_augmentation.py', 'train.py', 'robustness.py', 'data.py', 'labels.py')}}
    if (directory / 'config.json').exists():
        assert load(directory / 'config.json') == config, 'Code/config drift'
    else:
        dump(directory / 'config.json', config)
    if (directory / 'result.json').exists():
        result = load(directory / 'result.json')
        assert result['complete'] and digest(directory / 'model.pt') == result['checkpoint_sha256']
        return result
    status(status='context_variation_loading', current_training_running=False,
           context_variation_seed=seed, context_variation_arm=arm)
    model, head = build_model(plan, seed)
    encoder_parameters = [p for p in model.parameters() if p.requires_grad]
    parameters = encoder_parameters + list(head.parameters())
    optimizer = torch.optim.AdamW([{'params': encoder_parameters, 'lr': plan['encoder_lr']},
                                  {'params': head.parameters(), 'lr': plan['head_lr']}], weight_decay=.01, foreach=False)
    state = {'update': 0, 'epoch': 0, 'next_group': 0, 'training_seconds': 0., 'counts': {},
             'exposures': 0, 'gain_capped': 0, 'epoch_loss': []}
    if (directory / 'last.pt').exists():
        saved = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['config'] == config
        model.load_state_dict(saved['model'], strict=True)
        head.load_state_dict(saved['head'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        state = saved['state']
        train.set_rng_state(saved['rng'])
        del saved
        print('RESUME', seed, arm, state['update'], '/', total, flush=True)
    else:
        evaluate(model, head, selection, root, 'baseline')
        train.seed_everything(seed)
    cache, counts = {}, Counter(state['counts'])
    start, before = time.monotonic(), state['training_seconds']
    trace = directory / 'trace.jsonl'
    status(status='context_variation_training', current_training_running=True,
           next_step_status=f'Training seed {seed} {arm}: {plan["epochs"]} epochs on {len(selection["train_rows"])} clips; 20ms stride, last 8 layers.')

    def persist(stage):
        state.update(training_seconds=before + time.monotonic() - start, counts=dict(counts), status=stage)
        save_checkpoint(directory / 'last.pt', {'model': model.state_dict(), 'head': head.state_dict(),
                        'optimizer': optimizer.state_dict(), 'state': state, 'rng': train.rng_state(), 'config': config})
        dump(directory / 'progress.json', state)

    model.train()
    head.train()
    for epoch in range(state['epoch'], plan['epochs']):
        groups = batches_by_epoch[epoch]
        for index in range(state['next_group'], len(groups)):
            group = groups[index]
            denominator = sum(map(len, group))
            optimizer.zero_grad(set_to_none=True)
            losses, exposures = [], []
            for indices in group:
                rows = [selection['train_rows'][i] for i in indices]
                waves, metas = [], []
                for row in rows:
                    if row['wav'] not in cache:
                        cache[row['wav']] = robustness.verify_read(row)
                    left, right, db, mode = aug.parameters(row['wav'], seed, epoch, arm)
                    wave, meta = aug.apply(cache[row['wav']], left, right, db)
                    waves.append(wave)
                    counts[mode] += 1
                    state['gain_capped'] += int(meta['gain_capped'])
                    exposures.append(row['wav'])
                    metas.append(meta)
                batch = robustness.pack(waves, rows)
                logits, lengths = train.forward(model, head, batch, grad=True, autocast=True)
                loss = train.ctc_sum(logits, lengths, batch)
                assert torch.isfinite(loss)
                (loss / denominator).backward()
                losses.append(float(loss.detach()))
            step = state['update'] + 1
            factor = min(1., step / plan['warmup_updates']) * (.1 + .9 * (1 + math.cos(math.pi * step / total)) / 2)
            optimizer.param_groups[0]['lr'] = plan['encoder_lr'] * factor
            optimizer.param_groups[1]['lr'] = plan['head_lr'] * factor
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state.update(update=step, epoch=epoch, next_group=index + 1,
                         exposures=state['exposures'] + denominator, mean_loss=sum(losses) / denominator)
            with trace.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'update': step, 'epoch': epoch, 'wavs': exposures,
                                         'loss': state['mean_loss'], 'grad_norm': float(norm)}) + '\n')
            if step % 32 == 0 or step == 1:
                state.update(training_seconds=before + time.monotonic() - start, counts=dict(counts),
                             status='training', total_updates=total, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                dump(directory / 'progress.json', state)
                print('TRAIN', seed, arm, step, '/', total, 'loss', round(state['mean_loss'], 4),
                      'seconds', round(state['training_seconds'], 1), flush=True)
            if step % 256 == 0:
                persist('training')
        state.update(epoch=epoch + 1, next_group=0)
        persist('training' if epoch + 1 < plan['epochs'] else 'evaluating')
    assert state['exposures'] == config['recording_exposures']
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    evaluation = evaluate(model, head, selection, directory, 'final')
    save_checkpoint(directory / 'model.pt', {'model': model.state_dict(), 'head': head.state_dict(),
                                            'config': config, 'valid': evaluation['clean_valid']})
    state['status'] = 'completed'
    dump(directory / 'progress.json', state)
    result = {'complete': True, 'config': config, 'state': state, 'evaluation': evaluation,
              'checkpoint_sha256': digest(directory / 'model.pt'), 'deployed': False}
    dump(directory / 'result.json', result)
    print('ARM_COMPLETE', seed, arm, flush=True)
    del model, head, parameters, encoder_parameters, cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(args):
    root = WORK / args.name
    with Owned(root, WORK):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        selection = prepare(root, args.count)
        plan = make_plan(root, selection, args.epochs)
        assert args.seed in plan['seeds']
        status(status='context_variation_prepared', automatic_diagnostics_running=True,
               current_training_running=False, active_experiment=str(Path(__file__).relative_to(ROOT)),
               context_variation_directory=str(root), original_problem_solved=False)
        arms = {}
        for arm in ('control', 'combined'):
            arms[arm] = run_arm(root, selection, plan, args.seed, arm)
        label_pilot.check_answers(load(LABEL_SELECTION))
        # Compare exact input order and update counts, not just nominal epochs.
        traces = []
        for arm in arms:
            rows = [json.loads(line) for line in (root / f'seed-{args.seed}' / arm / 'trace.jsonl').read_text().splitlines()]
            latest = {r['update']: r for r in rows}  # interrupted uncheckpointed updates may be replayed
            assert set(latest) == set(range(1, arms[arm]['config']['total_updates'] + 1))
            traces.append([(r['update'], r['epoch'], r['wavs']) for _, r in sorted(latest.items())])
        assert traces[0] == traces[1], 'Unpaired input schedule'
        summary = {'complete': True, 'seed': args.seed, 'paired_input_schedule_verified': True,
                   'parent': load(root / 'baseline-evaluation.json'),
                   'arms': {arm: {'evaluation': result['evaluation'], 'state': result['state']} for arm, result in arms.items()}}
        dump(root / f'comparison-{args.seed}.json', summary)
        status(status='context_variation_seed_complete', automatic_diagnostics_running=False,
               current_training_running=False, active_experiment=None,
               latest_context_comparison=str(root / f'comparison-{args.seed}.json'),
               next_step_status='Paired context-variation seed complete; interpret robustness, actual pronunciation, and clean regression before next action.')
        print('SEED_COMPLETE', args.seed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='context-variation-20260927')
    parser.add_argument('--seed', type=int, default=92801)
    parser.add_argument('--count', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        status(status='context_variation_interrupted', automatic_diagnostics_running=False,
               current_training_running=False, active_experiment=None, latest_diagnostic_error=repr(exc))
        raise
