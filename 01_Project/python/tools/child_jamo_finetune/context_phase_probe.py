"""Follow up on the fixed primary evaluation using phase-aligned silence.

Training zero padding was a multiple of the 20-ms convolution stride.
The primary evaluation deliberately used unseen 10-ms offsets. This probe
checks whether failure is explained by that mismatch. It never changes
the primary gates or selects a checkpoint.
"""
from pathlib import Path
import argparse
import gc
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_augmentation as aug
import context_pilot as pilot
import robustness
from common import ROOT, WORK, digest, dump, load
from ownership import Owned

CONDITIONS = {
    'original': (0, 0, 0.),
    'left_400ms': (6400, 0, 0.),
    'right_400ms': (0, 6400, 0.),
    'both_400ms': (6400, 6400, 0.),
    'both_500ms': (8000, 8000, 0.),
    'both_1200ms': (19200, 19200, 0.),
    'gain_minus6': (0, 0, -6.),
}


def main(name):
    assert name in ('context-variation-20260927', 'context-full-20260927')
    root = WORK / name
    with Owned(root, WORK):
        training_plan = load(root / 'plan.json')
        seeds = training_plan['seeds']
        assert all(load(root / f'comparison-{seed}.json')['complete'] for seed in seeds)
        assert all(left % 320 == 0 and right % 320 == 0 for left, right, _ in CONDITIONS.values())
        plan = {'type': 'exploratory follow-up; primary gates unchanged', 'conditions': CONDITIONS,
                'reason': 'Check generalization on the same convolution phase as training padding.',
                'models': ['parent'] + [f'seed-{seed}/{arm}' for seed in seeds for arm in ('control', 'combined')],
                'code_sha256': digest(__file__), 'extra_training_updates': 0,
                'inputs': 'The same 27 existing human-judged clips; no newly created labels.'}
        path = root / 'phase-probe-plan.json'
        if path.exists():
            assert load(path) == load_json_compatible(plan)
        else:
            dump(path, plan)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        aug.EVAL = CONDITIONS  # This process only; no source file or service mutation.
        report = {'complete': False, 'plan': plan, 'models': {}}
        pilot.status(status='context_phase_probe_running', current_training_running=False,
                     automatic_diagnostics_running=True, active_experiment=str(Path(__file__).relative_to(ROOT)),
                next_step_status='Checking trained models with 20-ms-aligned silence; primary comparison is frozen.')
        started = time.monotonic()
        for model_name in plan['models']:
            model, head = pilot.build_model(training_plan, seeds[0])
            if model_name == 'parent':
                checkpoint_hash = digest(robustness.PARENT)
                expected = load(root / 'baseline-evaluation.json')['human']
            else:
                checkpoint = root / model_name / 'model.pt'
                checkpoint_hash = digest(checkpoint)
                assert checkpoint_hash == load(root / model_name / 'result.json')['checkpoint_sha256']
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                model.load_state_dict(saved['model'], strict=True)
                head.load_state_dict(saved['head'], strict=True)
                del saved
                expected = load(root / model_name / 'final-evaluation.json')['human']
            model.eval()
            head.eval()
            rows = pilot.variant_records(model, head, pilot.human_inputs())
            previous = {r['id']: r['variants']['original']['ids'] for r in expected}
            assert all(r['variants']['original']['ids'] == previous[r['id']] for r in rows), 'Reload parity failed'
            summary = {cohort: pilot.summarize_variants([r for r in rows if r['cohort'] == cohort])
                       for cohort in sorted({r['cohort'] for r in rows})}
            report['models'][model_name] = {'checkpoint_sha256': checkpoint_hash, 'reload_parity': True,
                                      'rows': rows, 'cohorts': summary}
            dump(root / 'phase-probe.json', report)
            print('PHASE', model_name, summary['personal_diagnostic'], flush=True)
            del model, head
            gc.collect()
            torch.cuda.empty_cache()
        report.update(complete=True, elapsed_seconds=round(time.monotonic() - started, 2),
                      parent_unchanged=digest(robustness.PARENT) == robustness.PARENT_SHA)
        dump(root / 'phase-probe.json', report)
        output_name = 'child-jamo-context-full-phase-probe-20260927.json' if name == 'context-full-20260927' else 'child-jamo-context-phase-probe-20260927.json'
        dump(ROOT / 'docs/data' / output_name, report)
        pilot.status(status='context_phase_probe_complete', current_training_running=False,
                     automatic_diagnostics_running=False, active_experiment=None,
                     latest_context_phase_probe='docs/data/' + output_name)


def load_json_compatible(value):
    import json
    return json.loads(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='context-variation-20260927')
    try:
        main(parser.parse_args().name)
    except BaseException as exc:
        pilot.status(status='context_phase_probe_failed', current_training_running=False,
                     automatic_diagnostics_running=False, active_experiment=None, latest_diagnostic_error=repr(exc))
        raise
