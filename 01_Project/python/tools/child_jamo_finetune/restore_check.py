"""Does a resume restore exactly what was saved?

This is deliberately separate from "does the next update come out the
same". Those are different questions and only one of them has a clean
answer.

Restoration is exact or it is not: load the checkpoint into a freshly
built model, head and optimizer, read the state straight back out, and
compare. No kernel runs, so any difference is a real bug in save/load.

Whether the *next update* then reproduces is a different matter. Adam
divides by the gradient's own magnitude, so a parameter whose gradient is
near zero takes a full learning-rate step in whichever direction the last
bits happened to fall. Two processes can pick different cuBLAS kernels
and land a whole step apart on such a parameter while both are perfectly
correct. That is measured in `smoke_cli.py` and bounded against the
learning rate, not asserted to be bit-identical.

    python python/tools/child_jamo_finetune/restore_check.py <checkpoint>
"""
from pathlib import Path
import argparse
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train
from common import dump
from optim import PartialCPUAdamW


def differences(a, b):
    """Max absolute gap per key. A checkpoint is on the CPU and a live
    model is on the GPU, so both sides are brought to the CPU first."""
    worst, where = 0.0, None
    for key in sorted(a):
        d = (a[key].detach().cpu().float()
             - b[key].detach().cpu().float()).abs().max().item()
        if d > worst:
            worst, where = d, key
    return {'max_abs_difference': worst, 'worst_tensor': where,
            'tensors': len(a)}


def check(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = saved['config']
    args = train.parser().parse_args([
        '--name', config['name'], '--seed', str(config['seed']),
        '--attention', config['attention'],
        '--offload-layers', str(config['offload_layers']),
        '--mask-time-prob', str(config['mask_time_prob']),
        '--mask-feature-prob', str(config['mask_feature_prob'])])
    train.seed_everything(args.seed)
    model, head = train.build(args)
    opt = PartialCPUAdamW(model, head, config['offload_layers'],
                          config['encoder_lr'], config['head_lr'])
    model.load_state_dict(saved['model'])
    head.load_state_dict(saved['head'])
    opt.load_state_dict(saved['optimizer'])
    train.set_rng_state(saved['rng'])

    again = train.checkpoint_payload(model, head, opt, saved['state'], config)
    report = {
        'checkpoint': str(path),
        'model': differences(saved['model'], again['model']),
        'head': differences(saved['head'], again['head']),
        'rng_restored': {
            'python': train.rng_state()['python'] == saved['rng']['python'],
            'numpy': all(
                (x == y).all() if hasattr(x, 'shape') else x == y
                for x, y in zip(train.rng_state()['numpy'], saved['rng']['numpy'])),
            'torch': torch.equal(train.rng_state()['torch'], saved['rng']['torch']),
            'cuda': all(torch.equal(x, y) for x, y in zip(
                train.rng_state()['cuda'], saved['rng']['cuda'])),
        },
    }
    for leg in ['gpu', 'cpu']:
        before, after = saved['optimizer'][leg], again['optimizer'][leg]
        if before is None:
            report[f'optimizer_{leg}'] = None
            continue
        worst, tensors = 0.0, 0
        for index in sorted(before['state']):
            for key in sorted(before['state'][index]):
                u, v = before['state'][index][key], after['state'][index][key]
                if not hasattr(u, 'float'):
                    assert u == v, (leg, index, key)
                    continue
                tensors += 1
                worst = max(worst, (u.detach().cpu().float()
                                    - v.detach().cpu().float())
                            .abs().max().item())
        report[f'optimizer_{leg}'] = {'max_abs_difference': worst,
                                      'tensors': tensors}

    # The CPU master copies are what the offloaded layers are updated
    # from, so a resume that rebuilt them wrongly would drift silently.
    master = max((gpu.detach().cpu().float() - cpu.detach().float())
                 .abs().max().item() for gpu, cpu in opt.pairs)
    report['cpu_masters_match_gpu_weights'] = {'max_abs_difference': master}

    parts = ['model', 'head', 'optimizer_gpu', 'optimizer_cpu',
             'cpu_masters_match_gpu_weights']
    report['exact'] = all(report[p]['max_abs_difference'] == 0.0
                          for p in parts if report[p])
    report['rng_exact'] = all(report['rng_restored'].values())
    report['pass'] = bool(report['exact'] and report['rng_exact'])
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--out')
    args = p.parse_args()
    result = check(Path(args.checkpoint))
    if args.out:
        dump(Path(args.out), result)
    print('restore exact:', result['exact'], '| rng exact:', result['rng_exact'])
    raise SystemExit(0 if result['pass'] else 1)
