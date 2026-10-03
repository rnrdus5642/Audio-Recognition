"""Paired follow-up varying existing exact-zero margins without speech cuts.

The control runs are reused only after verifying the same initialization,
selection, schedule, hyperparameters, decoder, and evaluation transformation.
The immutable runner receives an explicit augmentation-module dependency;
its hash is bound through this experiment's plan and every checkpoint.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import io
import json
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio
import context_pilot as base
import context_span_augmentation as policy
import label_pilot
import robustness
from common import ROOT, WORK, digest, dump, load
from ownership import Owned

NAME = 'context-span-20260927'
SCREEN = WORK / 'context-variation-20260927'


def prepare(root):
    selection = base.prepare(root, 4096)
    assert selection == load(SCREEN / 'selection.json')
    original = load(SCREEN / 'plan.json')
    reused = {}
    for seed in original['seeds']:
        result_path = SCREEN / f'seed-{seed}/control/result.json'
        result = load(result_path)
        assert result['complete']
        assert digest(SCREEN / f'seed-{seed}/control/model.pt') == result['checkpoint_sha256']
        reused[str(seed)] = {'result_sha256': digest(result_path), 'checkpoint_sha256': result['checkpoint_sha256']}
    plan = {**original, 'version': 'child-jamo-context-span-v1',
            'created_at': datetime.now().astimezone().isoformat(),
            'selection_sha256': digest(root / 'selection.json'), 'augmentation': policy.VERSION,
            'policy_code_sha256': {n: digest(Path(__file__).parent / n) for n in ('context_span.py', 'context_span_augmentation.py')},
            'reason': 'The previous policy only lengthened already-padded recordings. Existing literal-zero margins conceal hundreds of short utterances.',
            'training_transform': '50% unchanged. Remaining half first shortens only literal-zero outer margins, retaining 80-100ms when available and the original 20-ms convolution phase. Gain-only, silence-only, and both remain equally likely. Added padding uses no/short 0-120ms/wide 0-1200ms bands with equal probability. Requested gain remains -15 to +6dB.',
            'ctc_length_guard': 'If a transformed input has fewer frames than the unchanged CTC target requires, retain the original margin before padding/gain; never shorten the target.',
            'reused_controls': reused,
            'reused_control_verification': 'All optimizer/data-order hyperparameters match; compare the full ordered update/exposure trace after training.',
            'extra_evaluation': 'After both fixed seeds, compare safe existing-zero shortening on the same fixed 256 validation clips and the 27 human clips; secondary diagnostic, not a replacement for primary gates.'}
    path = root / 'plan.json'
    if path.exists():
        previous = load(path)
        plan['created_at'] = previous['created_at']
        assert previous == plan
    else:
        dump(path, plan)
    for key in ('epochs', 'count', 'last_layers', 'encoder_lr', 'head_lr', 'warmup_updates',
                'batch_budget', 'max_batch', 'accumulate', 'stride_ms', 'dropout'):
        assert plan[key] == original[key]
    baseline = load(SCREEN / 'baseline-evaluation.json')
    assert baseline['complete']
    if (root / 'baseline-evaluation.json').exists():
        assert load(root / 'baseline-evaluation.json') == baseline
    else:
        dump(root / 'baseline-evaluation.json', baseline)
    policy.configure(selection['train_rows'])
    return selection, plan


def audit_policy(root, selection, plan, seed):
    path = root / f'policy-audit-{seed}.json'
    if path.exists():
        report = load(path)
        assert report['plan_sha256'] == digest(root / 'plan.json') and report['complete']
        return report
    counts, channels = Counter(), Counter()
    short_before = short_after = trimmed = removed_nonzero = fallbacks = capped = 0
    source_equal = 0
    records = []
    for row in selection['train_rows']:
        x = robustness.verify_read(row)
        blob = audio.payload(row['archive'], row['offset'], row['size'])
        info = sf.info(io.BytesIO(blob))
        channels[str(info.channels)] += 1
        raw = audio.decode(blob)
        lo_s, hi_s, _ = audio.span(row['speech_start'], row['speech_end'], len(raw) / 16000)
        direct = raw[max(0, round(lo_s * 16000)):min(len(raw), round(hi_s * 16000))]
        assert np.array_equal(direct, x)
        source_equal += 1
        entry = {'wav': row['wav'], 'source_channels': info.channels, 'source_samples': len(raw),
                 'training_samples': len(x), 'variants': []}
        for epoch in range(plan['epochs']):
            left, right, db, mode = policy.parameters(row['wav'], seed, epoch, 'combined')
            y, meta = policy.apply(x, left, right, db)
            counts[mode] += 1
            short_before += len(x) < 24000
            short_after += len(y) < 24000
            trimmed += bool(meta.get('removed_left_zero_samples', 0) + meta.get('removed_right_zero_samples', 0))
            removed_nonzero += meta.get('nonzero_samples_removed', 0)
            fallbacks += meta.get('ctc_length_fallback', False)
            capped += meta['gain_capped']
            entry['variants'].append({'epoch': epoch, 'mode': mode, **meta})
        records.append(entry)
    assert removed_nonzero == 0
    report = {'complete': True, 'plan_sha256': digest(root / 'plan.json'), 'seed': seed,
              'recordings': len(records), 'exposures': sum(counts.values()), 'transform_counts': dict(counts),
              'original_short_exposures': short_before, 'transformed_short_exposures': short_after,
              'shortened_zero_margin_exposures': trimmed, 'removed_nonzero_samples': removed_nonzero,
              'ctc_length_fallbacks': fallbacks, 'gain_capped': capped,
              'source_channels': dict(channels), 'crop_equals_direct_source_slice': source_equal,
              'original_audio_files_modified': False, 'rows': records}
    dump(path, report)
    print('SPAN_AUDIT', json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False), flush=True)
    return report


def main(seed):
    root = WORK / NAME
    with Owned(root, WORK):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        selection, plan = prepare(root)
        assert seed in plan['seeds']
        base.status(status='context_span_preparing', automatic_diagnostics_running=True, current_training_running=False,
                    active_experiment=str(Path(__file__).relative_to(ROOT)), context_variation_directory=str(root),
                    next_step_status='Auditing source zeros and exact signal preservation before varying existing margins.')
        audited = audit_policy(root, selection, plan, seed)
        # Explicit dependency injection. Numeric evaluation requests delegate
        # byte-for-byte to the original augmentation module (covered by tests).
        base.aug = policy
        combined = base.run_arm(root, selection, plan, seed, 'combined')
        control = load(SCREEN / f'seed-{seed}/control/result.json')
        assert digest(SCREEN / f'seed-{seed}/control/result.json') == plan['reused_controls'][str(seed)]['result_sha256']
        assert combined['state']['counts'] == audited['transform_counts']
        assert combined['state']['gain_capped'] == audited['gain_capped']
        schedules = []
        for path in (SCREEN / f'seed-{seed}/control/trace.jsonl', root / f'seed-{seed}/combined/trace.jsonl'):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            latest = {r['update']: r for r in rows}
            assert set(latest) == set(range(1, combined['config']['total_updates'] + 1))
            schedules.append([(r['update'], r['epoch'], r['wavs']) for _, r in sorted(latest.items())])
        assert schedules[0] == schedules[1]
        label_pilot.check_answers(load(base.LABEL_SELECTION))
        comparison = {'complete': True, 'seed': seed, 'paired_input_schedule_verified': True,
                      'parent': load(root / 'baseline-evaluation.json'),
                      'arms': {name: {'state': r['state'], 'evaluation': r['evaluation']} for name, r in
                               [('control', control), ('combined', combined)]},
                      'control_reuse': plan['reused_controls'][str(seed)],
                      'policy_audit_sha256': digest(root / f'policy-audit-{seed}.json')}
        dump(root / f'comparison-{seed}.json', comparison)
        base.status(status='context_span_seed_complete', current_training_running=False,
                    automatic_diagnostics_running=False, active_experiment=None,
                    latest_context_comparison=str(root / f'comparison-{seed}.json'))
        print('SPAN_SEED_COMPLETE', seed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=92801)
    try:
        main(parser.parse_args().seed)
    except BaseException as exc:
        base.status(status='context_span_interrupted', current_training_running=False,
                    automatic_diagnostics_running=False, active_experiment=None, latest_diagnostic_error=repr(exc))
        raise
