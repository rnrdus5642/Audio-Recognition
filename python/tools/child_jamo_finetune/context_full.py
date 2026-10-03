"""Follow-up: allow the complete contextual encoder to learn silence/gain.

The first screen froze the lower forty layers and positional convolution.
This experiment updates every encoder layer, feature projection, positional
convolution, and CTC head. The waveform convolution remains frozen, as in
the original main fine-tuning. Identical paired epochs isolate augmentation
within this scope; comparisons against the small-batch screen are descriptive.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path
import gc
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_augmentation as aug
import context_pilot as base
import data
import integrity
import label_pilot
import robustness
import train
from common import ROOT, WORK, digest, dump, load
from optim import PartialCPUAdamW
from ownership import Owned

NAME = 'context-full-20260927'
SCREEN = WORK / 'context-variation-20260927'


def prepare(root):
    source = load(SCREEN / 'selection.json')
    assert source['contract'] == integrity.contract({'train': robustness.TRAIN_SPLIT, 'valid': 'valid', 'test': 'test'})
    label_pilot.check_answers(load(base.LABEL_SELECTION))
    assert digest(robustness.PARENT) == robustness.PARENT_SHA
    path = root / 'selection.json'
    if path.exists():
        assert load(path) == source
    else:
        dump(path, source)
    initial = load(SCREEN / 'baseline-evaluation.json')
    assert initial['complete']
    if not (root / 'baseline-evaluation.json').exists():
        dump(root / 'baseline-evaluation.json', initial)
    else:
        assert load(root / 'baseline-evaluation.json') == initial
    plan = {**load(SCREEN / 'plan.json'), 'version': 'child-jamo-context-full-v1',
            'created_at': datetime.now().astimezone().isoformat(),
            'seeds': [92801], 'last_layers': 48, 'accumulate': 16,
            'warmup_updates': 16, 'offload_layers': 48,
            'trainable': 'All encoder parameters except waveform convolution; includes positional convolution and feature projection.',
            'selection_sha256': digest(path), 'baseline_source_sha256': digest(SCREEN / 'baseline-evaluation.json'),
            'reason': 'The first screen left the implicated early context computation frozen. Test the same augmentation with that restriction removed.',
            'comparison_scope': 'Paired control versus combined within full-encoder training. Batch accumulation differs from the 8-layer screen; do not attribute cross-screen differences solely to trainable scope.',
            'checkpoint_selection': 'Fixed final two complete epochs; no best-personal-case checkpoint search.',
            'learning_scope_check': 'Verify positional convolution, layer 0, layer 47, and head receive finite nonzero gradients and change from the initial checkpoint.'}
    if (root / 'plan.json').exists():
        previous = load(root / 'plan.json')
        plan['created_at'] = previous['created_at']
        assert previous == plan
    else:
        dump(root / 'plan.json', plan)
    return source, plan


def fingerprint(parameter):
    return hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run(root, selection, plan, arm):
    seed = plan['seeds'][0]
    directory = root / f'seed-{seed}' / arm
    directory.mkdir(parents=True, exist_ok=True)
    planning = [dict(r, samples=r['samples'] + 2 * aug.MAX_SIDE_SAMPLES) for r in selection['train_rows']]
    schedule = [data.groups(data.plan(planning, plan['batch_budget'], plan['max_batch'], seed, f'context:{epoch}'),
                            plan['accumulate']) for epoch in range(plan['epochs'])]
    total = sum(map(len, schedule))
    config = {'plan_sha256': digest(root / 'plan.json'), 'seed': seed, 'arm': arm,
              'total_updates': total, 'recording_exposures': plan['epochs'] * len(planning),
              'source_code_sha256': {name: digest(Path(__file__).parent / name) for name in
                                     ('context_full.py', 'context_pilot.py', 'context_augmentation.py', 'train.py', 'optim.py', 'robustness.py')}}
    if (directory / 'config.json').exists():
        assert load(directory / 'config.json') == config, 'Refuse code/config drift'
    else:
        dump(directory / 'config.json', config)
    if (directory / 'result.json').exists():
        result = load(directory / 'result.json')
        assert result['complete'] and digest(directory / 'model.pt') == result['checkpoint_sha256']
        return result
    base.status(status='context_full_loading', current_training_running=False, context_variation_arm=arm,
                next_step_status=f'Loading full contextual encoder for {arm}; convolutional waveform frontend remains frozen.')
    model, head = base.build_model(plan, seed)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.feature_extractor._freeze_parameters()
    assert all(p.requires_grad == (not name.startswith('feature_extractor.')) for name, p in model.named_parameters())
    parameters = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    named = dict(model.named_parameters())
    witness_names = [max((n for n in named if n.startswith('encoder.pos_conv_embed') and 'weight' in n), key=lambda n: named[n].numel()),
                     'encoder.layers.0.attention.q_proj.weight', 'encoder.layers.47.attention.q_proj.weight']
    witnesses = {name: named[name] for name in witness_names}
    witnesses['head.linear.weight'] = head.linear.weight
    assert all(p.requires_grad for p in witnesses.values())
    initial_hashes = {name: fingerprint(p) for name, p in witnesses.items()}
    optimizer = PartialCPUAdamW(model, head, plan['offload_layers'], plan['encoder_lr'], plan['head_lr'], foreach=True)
    state = {'update': 0, 'epoch': 0, 'next_group': 0, 'training_seconds': 0., 'exposures': 0,
             'counts': {}, 'gain_capped': 0, 'scope_initial_hashes': initial_hashes,
             'first_gradient_norms': {}, 'trainable_parameters': sum(p.numel() for p in parameters)}
    if (directory / 'last.pt').exists():
        saved = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['config'] == config
        model.load_state_dict(saved['model'], strict=True)
        head.load_state_dict(saved['head'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        state = saved['state']
        train.set_rng_state(saved['rng'])
        del saved
        print('FULL_RESUME', arm, state['update'], '/', total, flush=True)
    else:
        train.seed_everything(seed)
    start, before = time.monotonic(), state['training_seconds']
    cache, counts = {}, Counter(state['counts'])
    trace = directory / 'trace.jsonl'
    base.status(status='context_full_training', current_training_running=True,
                next_step_status=f'Full contextual encoder {arm}: two paired epochs, {total} updates, 8192 exposures.')

    def persist(stage):
        state.update(status=stage, training_seconds=before + time.monotonic() - start, counts=dict(counts))
        base.save_checkpoint(directory / 'last.pt', {'model': model.state_dict(), 'head': head.state_dict(),
                             'optimizer': optimizer.state_dict(), 'state': state, 'rng': train.rng_state(), 'config': config})
        dump(directory / 'progress.json', state)

    model.train()
    head.train()
    for epoch in range(state['epoch'], plan['epochs']):
        for index in range(state['next_group'], len(schedule[epoch])):
            group = schedule[epoch][index]
            denominator = sum(map(len, group))
            optimizer.zero_grad()
            losses, exposed = [], []
            for indices in group:
                rows = [selection['train_rows'][i] for i in indices]
                waves = []
                for row in rows:
                    if row['wav'] not in cache:
                        cache[row['wav']] = robustness.verify_read(row)
                    left, right, db, mode = aug.parameters(row['wav'], seed, epoch, arm)
                    wave, meta = aug.apply(cache[row['wav']], left, right, db)
                    waves.append(wave)
                    counts[mode] += 1
                    state['gain_capped'] += int(meta['gain_capped'])
                    exposed.append(row['wav'])
                batch = robustness.pack(waves, rows)
                logits, lengths = train.forward(model, head, batch, grad=True, autocast=True)
                loss = train.ctc_sum(logits, lengths, batch)
                assert torch.isfinite(loss)
                (loss / denominator).backward()
                losses.append(float(loss.detach()))
            if not state['first_gradient_norms']:
                gradient_norms = {name: float(p.grad.norm()) if p.grad is not None else None for name, p in witnesses.items()}
                assert all(value is not None and math.isfinite(value) and value > 0 for value in gradient_norms.values())
                state['first_gradient_norms'] = gradient_norms
                print('FULL_GRADIENT_SCOPE', arm, gradient_norms, flush=True)
            step = state['update'] + 1
            factor = min(1., step / plan['warmup_updates']) * (.1 + .9 * (1 + math.cos(math.pi * step / total)) / 2)
            optimizer.set_lr(plan['encoder_lr'] * factor, plan['head_lr'] * factor)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()
            state.update(update=step, epoch=epoch, next_group=index + 1,
                         exposures=state['exposures'] + denominator, mean_loss=sum(losses) / denominator)
            with trace.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'update': step, 'epoch': epoch, 'wavs': exposed,
                                         'loss': state['mean_loss'], 'grad_norm': float(norm)}) + '\n')
            if step == 1 or step % 4 == 0:
                state.update(status='training', training_seconds=before + time.monotonic() - start,
                             counts=dict(counts), total_updates=total, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                dump(directory / 'progress.json', state)
                print('FULL_TRAIN', arm, step, '/', total, 'loss', round(state['mean_loss'], 4),
                      'seconds', round(state['training_seconds'], 1), flush=True)
            if step % 32 == 0:
                persist('training')
        state.update(epoch=epoch + 1, next_group=0)
        persist('training' if epoch + 1 < plan['epochs'] else 'evaluating')
    assert state['exposures'] == config['recording_exposures']
    state['scope_final_hashes'] = {name: fingerprint(p) for name, p in witnesses.items()}
    assert all(state['scope_initial_hashes'][name] != value for name, value in state['scope_final_hashes'].items())
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    evaluation = base.evaluate(model, head, selection, directory, 'final')
    base.save_checkpoint(directory / 'model.pt', {'model': model.state_dict(), 'head': head.state_dict(),
                                                'config': config, 'valid': evaluation['clean_valid']})
    state['status'] = 'completed'
    dump(directory / 'progress.json', state)
    result = {'complete': True, 'config': config, 'state': state, 'evaluation': evaluation,
              'checkpoint_sha256': digest(directory / 'model.pt'), 'deployed': False}
    dump(directory / 'result.json', result)
    print('FULL_ARM_COMPLETE', arm, flush=True)
    del model, head, named, witnesses, parameters, cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    root = WORK / NAME
    with Owned(root, WORK):
        assert all(load(SCREEN / f'comparison-{s}.json')['complete'] for s in (92801, 92802))
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        selection, plan = prepare(root)
        base.status(status='context_full_prepared', automatic_diagnostics_running=True, current_training_running=False,
                    active_experiment=str(Path(__file__).relative_to(ROOT)), context_variation_directory=str(root))
        arms = {arm: run(root, selection, plan, arm) for arm in ('control', 'combined')}
        schedules = []
        for arm in arms:
            rows = [json.loads(line) for line in (root / 'seed-92801' / arm / 'trace.jsonl').read_text().splitlines()]
            by_update = {r['update']: r for r in rows}
            assert set(by_update) == set(range(1, arms[arm]['config']['total_updates'] + 1))
            schedules.append([(r['update'], r['epoch'], r['wavs']) for _, r in sorted(by_update.items())])
        assert schedules[0] == schedules[1]
        label_pilot.check_answers(load(base.LABEL_SELECTION))
        dump(root / 'comparison-92801.json', {'complete': True, 'seed': 92801, 'paired_input_schedule_verified': True,
             'parent': load(root / 'baseline-evaluation.json'),
             'arms': {arm: {'state': r['state'], 'evaluation': r['evaluation']} for arm, r in arms.items()}})
        base.status(status='context_full_complete', automatic_diagnostics_running=False, current_training_running=False,
                    active_experiment=None, latest_context_comparison=str(root / 'comparison-92801.json'))
        print('FULL_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        base.status(status='context_full_interrupted', automatic_diagnostics_running=False,
                    current_training_running=False, active_experiment=None, latest_diagnostic_error=repr(exc))
        raise
