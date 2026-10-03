"""Is a resumed run the same run, or a different one?

The earlier evidence compared two resumes from one checkpoint. That shows
a restore is repeatable; it does not show the resumed run matches the run
that was never interrupted, which is the thing that actually matters.

So this produces three checkpoints at the same encoder update with the
real 962M encoder:

    R   paused at update 2, resumed in a fresh process to update 3
    U1  one uninterrupted process straight to update 3
    U2  another uninterrupted process straight to update 3

U1 against U2 is the baseline: how far two identical uninterrupted runs
drift on this machine. R against U1 is the question. If the two are the
same size, interruption is not doing anything that running normally does
not already do.

`torch.nn.functional.ctc_loss` documents nondeterministic CUDA gradients,
which is why a baseline is measured rather than assumed - the point is to
compare R against that measured baseline, not to name a cause.

    python python/tools/child_jamo_finetune/control_experiment.py
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK, dump, log

PYTHON = ROOT / '.venv-gpu/Scripts/python.exe'
TRAIN = ROOT / 'python/tools/child_jamo_finetune/train.py'
NAME = 'control'
OUT = WORK / NAME
TARGET_UPDATE = 3
BASE = ['--name', NAME, '--train-split', 'smoketrain',
        '--valid-split', 'smokevalid', '--test-split', 'smoketest',
        '--accumulate', '4', '--max-epochs', '2',
        '--head-warmup-updates', '2', '--warmup-updates', '2',
        '--eval-every', str(TARGET_UPDATE), '--log-every', '1',
        # One write per run: the stop itself persists.
        '--checkpoint-every', '10000', '--skip-test']

#: Reported alongside every difference so the numbers have a scale.
CURSOR = ['phase', 'epoch', 'group', 'update', 'audio_seconds', 'phase_audio']


def run(*extra):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
               HF_HUB_DISABLE_PROGRESS_BARS='1')
    t0 = time.monotonic()
    p = subprocess.run([str(PYTHON), str(TRAIN), *BASE, *extra], cwd=ROOT,
                       env=env, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    log('control.log', f'CONTROL {" ".join(extra)} -> {p.returncode} '
                       f'({time.monotonic() - t0:.0f}s)')
    if p.returncode != 0:
        raise SystemExit(f'{extra} failed:\n{p.stdout}\n{p.stderr}')
    return p


def keep(tag):
    target = OUT / f'{tag}.pt'
    shutil.copy2(OUT / 'last.pt', target)
    return target


def summarise(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    out = {'cursor': {k: saved['state'][k] for k in CURSOR},
           'valid_history': [{k: h[k] for k in ['update', 'jer', 'at']}
                             for h in saved['state']['history']]}
    del saved
    return out


def difference(a, b, watch='encoder.layers.5.attention.k_proj.bias'):
    """Per-component maxima, a distribution, and the tensor under review."""
    x = torch.load(a, map_location='cpu', weights_only=False)
    y = torch.load(b, map_location='cpu', weights_only=False)
    out = {}
    for part in ['model', 'head']:
        worst, where, exceed, total = 0.0, None, 0, 0
        for key in sorted(x[part]):
            d = (x[part][key].float() - y[part][key].float()).abs().max().item()
            total += 1
            exceed += d > 1e-6
            if d > worst:
                worst, where = d, key
        out[part] = {'max_abs_difference': worst, 'worst_tensor': where,
                     'tensors': total, 'tensors_over_1e-6': exceed}
    for leg in ['gpu', 'cpu']:
        p, q = x['optimizer'][leg], y['optimizer'][leg]
        if p is None:
            out[f'optimizer_{leg}'] = None
            continue
        worst, tensors = 0.0, 0
        for index in sorted(p['state']):
            for key in sorted(p['state'][index]):
                u, v = p['state'][index][key], q['state'][index][key]
                if not hasattr(u, 'float'):
                    continue
                tensors += 1
                worst = max(worst, (u.float() - v.float()).abs().max().item())
        out[f'optimizer_{leg}'] = {'max_abs_difference': worst,
                                   'tensors': tensors}
    if watch in x['model']:
        u, v = x['model'][watch].float(), y['model'][watch].float()
        out['watched_tensor'] = {
            'name': watch,
            'max_abs_difference': (u - v).abs().max().item(),
            'parameter_magnitude_max': u.abs().max().item(),
            'parameter_magnitude_rms': u.pow(2).mean().sqrt().item(),
            'elements': u.numel(),
            'elements_differing_over_1e-6': int(((u - v).abs() > 1e-6).sum()),
        }
    out['cursor_identical'] = all(x['state'][k] == y['state'][k] for k in CURSOR)
    out['cursor'] = {k: x['state'][k] for k in CURSOR}
    out['worst_component'] = max(
        v['max_abs_difference'] for v in out.values()
        if isinstance(v, dict) and 'max_abs_difference' in v)
    del x, y
    return out


def main():
    root = Path(WORK).resolve()
    target = OUT.resolve()
    if target.parent != root or target.name != NAME:
        raise SystemExit(f'refusing to touch {target}')
    if target.exists():
        shutil.rmtree(target)

    log('control.log', 'CONTROL_START')
    run('--overwrite', '--stop-after-updates', '2')
    run('--resume', '--stop-after-updates', str(TARGET_UPDATE))
    resumed = keep('resumed')

    run('--overwrite', '--stop-after-updates', str(TARGET_UPDATE))
    first = keep('uninterrupted-1')

    run('--overwrite', '--stop-after-updates', str(TARGET_UPDATE))
    second = OUT / 'last.pt'

    baseline = difference(first, second)
    observed = difference(first, resumed)
    # The resumed run may differ from an uninterrupted one by no more
    # than two uninterrupted runs already differ from each other, with a
    # small absolute floor so a near-zero baseline is not a trap.
    allowance = max(4 * baseline['worst_component'], 1e-7)
    report = {
        'target_update': TARGET_UPDATE,
        'encoder': 'real pretrained XLarge (962M)',
        'runs': {'resumed': summarise(resumed),
                 'uninterrupted_1': summarise(first),
                 'uninterrupted_2': summarise(second)},
        'baseline_uninterrupted_vs_uninterrupted': baseline,
        'observed_resumed_vs_uninterrupted': observed,
        'tolerance': allowance,
        'tolerance_rule': '4 x the measured uninterrupted-vs-uninterrupted '
                          'baseline, floored at 1e-7',
        'ratio_observed_to_baseline':
            observed['worst_component'] / max(baseline['worst_component'], 1e-12),
        'context': 'torch.nn.functional.ctc_loss documents nondeterministic '
                   'CUDA gradients. That motivates measuring a baseline; it '
                   'is not offered as proof of the cause of any particular '
                   'difference.',
    }
    report['cursor_matches'] = (observed['cursor_identical']
                                and baseline['cursor_identical'])
    histories = {k: v['valid_history'] for k, v in report['runs'].items()}
    report['validation'] = {
        'resumed_matches_uninterrupted_1':
            histories['resumed'] == histories['uninterrupted_1'],
        'the_two_uninterrupted_runs_match':
            histories['uninterrupted_1'] == histories['uninterrupted_2'],
        'all_three_identical': len({json.dumps(h) for h in histories.values()}) == 1,
        'note': 'The validations taken before any encoder update agree '
                'exactly across all three runs. After encoder updates the '
                'two uninterrupted runs differ from each other, so a shared '
                'score between the resumed run and one control is not '
                'evidence that interruption contributes nothing.',
        'histories': histories,
    }
    report['within_baseline'] = observed['worst_component'] <= allowance
    report['pass'] = bool(report['cursor_matches'] and report['within_baseline'])
    dump(WORK / 'claude-control-experiment.json', report)

    for tag in ['resumed', 'uninterrupted-1']:
        (OUT / f'{tag}.pt').unlink(missing_ok=True)
    for leftover in ['last.pt', 'best.pt']:
        (OUT / leftover).unlink(missing_ok=True)

    print(json.dumps({k: report[k] for k in
                      ['tolerance', 'ratio_observed_to_baseline',
                       'cursor_matches', 'within_baseline', 'pass']}, indent=1))
    print('validation', json.dumps(
        {k: v for k, v in report['validation'].items()
         if k not in ('histories', 'note')}))
    print('baseline worst', baseline['worst_component'],
          '| observed worst', observed['worst_component'])
    log('control.log', f'CONTROL_COMPLETE pass={report["pass"]}')
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
