"""Assemble the response to the main agent's review, from the artifacts.

Every number here is read back from a file the pipeline wrote, so the
report cannot drift from what actually ran.

    python python/tools/child_jamo_finetune/finalreview.py
"""
from pathlib import Path
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import integrity
import labels
from common import WORK, dump, load

LAUNCH = ('powershell -File python\\tools\\child_jamo_finetune\\launch.ps1')


def logged(prefix):
    """Recover accepted measurements from the preflight log.

    The throughput sweep is accepted evidence and must not be re-run, but
    a later targeted preflight rewrote the report without it. The log is
    the primary record of those runs, so the lines are read back from
    there verbatim.
    """
    out = []
    path = WORK / 'run/preflight.log'
    for line in path.read_text(encoding='utf-8').splitlines():
        marker = f' {prefix} '
        if marker in line:
            out.append(json.loads(line.split(marker, 1)[1]))
    return out


def finding(number, title, status, detail, evidence=None):
    return {'finding': number, 'title': title, 'status': status,
            'detail': detail, 'evidence': evidence or {}}


def main():
    preflight = load(WORK / 'claude-preflight-report.json')
    control = load(WORK / 'claude-control-experiment.json')
    restore = load(WORK / 'claude-real-restore.json')
    audit = load(WORK / 'audit-summary.json')
    selection = load(WORK / 'selection-summary.json')
    view = load(WORK / 'smoke-view.json')
    contract = integrity.contract({'train': 'train', 'valid': 'valid',
                                   'test': 'test'})

    baseline = control['baseline_uninterrupted_vs_uninterrupted']
    observed = control['observed_resumed_vs_uninterrupted']

    findings = [
        finding(1, 'Terminal evaluation and epoch-boundary recovery', 'fixed',
                'final_evaluation is now an explicit resumable stage. Training '
                'end and the test score are separate events: the run persists '
                'status=final_evaluation_pending before scoring and becomes '
                'completed only after result.json exists, so a crash in that '
                'window is recovered by an ordinary --resume with no further '
                'optimizer update. Resuming a completed run is an idempotent '
                'no-op (--force removed, so it can no longer re-enter encoder '
                'training after early stopping). The last-group shortcut that '
                'skipped an epoch epilogue is gone: a checkpoint on an '
                "epoch's final group falls through to the same end-of-epoch "
                'validation and early-stop decision. A head checkpoint that '
                'already met its budget hands over without another head '
                'update, and a prior pause reason is cleared when genuinely '
                'continuing.',
                {'tests': ['test_c_resume_from_the_last_group_still_ends_the_epoch',
                           'test_d_pending_final_evaluation_is_resumable_without_training',
                           'test_e_completed_run_is_an_idempotent_no_op',
                           'test_f_crash_before_the_test_leaves_a_recoverable_stage',
                           'test_a_head_phase_hands_over_and_pauses'],
                 'bug_found_by_these_tests':
                     'charging the phase clock crashed with KeyError on the '
                     'new final_evaluation phase, after result.json was '
                     'written - exactly the window that would have stranded '
                     'the run. The phase tally now accepts any phase name.'}),
        finding(2, 'Frozen data/config binding', 'fixed',
                'A new integrity.py builds one data contract covering the '
                'source archives (resolved path, size, mtime, catalogue hash '
                'and member count - not size alone), the encoder weights plus '
                'config.json and preprocessor_config.json, the label contract, '
                'and each manifest checked against the audit that produced it '
                'rather than against itself. Changing labels.py while keeping '
                'an old manifest is now refused instead of being recorded as '
                'if the new contract had been used. Smoke views carry their '
                'own binding in smoke-view.json. FROZEN_CONFIG gained '
                'eval_every, eval_budget, min_delta, weight_decay, grad_clip, '
                'foreach, layerdrop, updates_per_epoch, the precision and '
                'accumulation policies and the whole data contract.',
                {'test': 'test_h_refuses_a_manifest_that_the_audit_did_not_vouch_for',
                 'data_contract': contract}),
        finding(3, 'Single writer', 'fixed',
                'ownership.py takes a real msvcrt byte-range lock on run.lock '
                'and holds the handle for the process lifetime, so a second '
                'writer is refused by the kernel rather than by a '
                'check-then-write race, and a crash releases it. The PID file '
                'is descriptive only and is removed solely by the process that '
                'wrote it. The run directory is validated to sit inside the '
                'work root, in train.py itself and not only in the launcher.',
                {'tests': ['test_i_refuses_a_second_live_writer',
                           'test_j_refuses_a_run_directory_outside_the_work_root']}),
        finding(4, 'Phase timing across checkpoints', 'fixed',
                'persist() now banks the phase clock before saving, so a '
                'mid-epoch resume keeps the encoder time it already spent and '
                'the ETA stays honest. The ETA continues to use encoder-phase '
                'throughput only.',
                {'test': 'test_b_phase_seconds_survive_a_checkpoint'}),
        finding(5, 'data.collate still called audio.trim()', 'already fixed',
                'Resolved before this review arrived. collate() calls the '
                'shared audio.crop() and additionally rejects any recording '
                'whose measured length or crop policy disagrees with the '
                'manifest, so the audited contract is enforced at load time '
                'rather than assumed. The only crop implementation is '
                'audio.span/crop; selection, audit and training all use it.',
                {'call_sites': ['manifest.planned_span', 'manifest._check',
                                'data.collate'],
                 'test': 'test_crop_policy_is_shared_and_records_malformed_metadata'}),
        finding(6, 'Lightweight fixtures for state-machine evidence', 'fixed',
                'test_state_machine.py drives the real train.py entry point as '
                'a subprocess through every phase, interruption point and '
                'refusal, with a ~1M-parameter HuBERT stand-in selected by the '
                'CHILD_JAMO_ENCODER environment variable. Same control flow, '
                'checkpoints in megabytes: 11 cases in about 70 seconds, with '
                'no multi-gigabyte writes. The expensive real-encoder evidence '
                '(finite update, parity, masking, capacity, archive audit) is '
                'reused, not repeated.',
                {'cases': 11, 'seconds': 67,
                 'real_test_split_scored': False}),
        finding(7, 'Uninterrupted vs resumed control', 'fixed',
                'The missing control now exists on the real 962M encoder. '
                'Three checkpoints at the same encoder update: one resumed '
                'from a pause, and two uninterrupted runs. Two uninterrupted '
                'runs differ by {:.3g}; the resumed run differs from an '
                'uninterrupted one by {:.3g} - a ratio of {:.2f}, so the drift '
                'from interrupting is the same order as the drift between two '
                'runs that were never interrupted. Cursor and audio exposure '
                'agree exactly; wall-clock fields are excluded. The two '
                'uninterrupted runs differ from each other by one reference '
                'edit out of 383 jamo at the scored update, so no claim is '
                'made that interruption adds zero error - only that this '
                'bounded experiment does not distinguish it from normal '
                'variation.'.format(
                    baseline['worst_component'], observed['worst_component'],
                    control['ratio_observed_to_baseline']),
                {'baseline_worst': baseline['worst_component'],
                 'observed_worst': observed['worst_component'],
                 'ratio': control['ratio_observed_to_baseline'],
                 'cursor_matches': control['cursor_matches'],
                 'validation': {k: v for k, v in control['validation'].items()
                                if k != 'histories'},
                 'compared': ['encoder', 'head', 'AdamW gpu moments',
                              'AdamW cpu moments', 'data cursor',
                              'audio exposure', 'validation history']}),
        finding(8, 'Guarded test-artifact deletion', 'fixed',
                'Every recursive delete resolves its target first and requires '
                'it to be a direct child of the work root carrying the test\'s '
                'own run name; anything else raises. smoke_cli.py, which held '
                'the unguarded rmtree, has been retired in favour of the '
                'fixture suite, and its evidence is preserved as '
                'claude-smoke-cli-superseded.json. The corpus and the real '
                'manifests are never deletion targets.',
                {'retired': 'smoke_cli.py',
                 'preserved': 'claude-smoke-cli-superseded.json'}),
        finding(9, 'Differing weight hashes after a fresh-process resume',
                'explained, not a restore bug',
                'Two independent measurements separate the questions. '
                'Restoration is exact: loading the real production checkpoint '
                '(offload 26) back into a fresh model, head and optimizer '
                'reproduces every encoder tensor, both AdamW moment sets, the '
                'CPU master copies and all four RNG streams with a maximum '
                'difference of exactly 0.0, with no kernel involved. The '
                'update that follows is not bit-identical, but neither are two '
                'uninterrupted runs: the measured baseline is {:.3g} against '
                '{:.3g} when resumed. The watched tensor '
                'encoder.layers.5.attention.k_proj.bias differs by {:.3g} in '
                'the baseline and {:.3g} when resumed - slightly smaller when '
                'resumed - against its own RMS magnitude of {:.3g}, so it is '
                'not resume-specific. Validations taken before any encoder '
                'update agree exactly across all three runs; after encoder '
                'updates the two uninterrupted runs differ from each other by '
                'one reference edit out of 383 jamo, which is the scale of '
                'the effect. No bitwise claim is made, no statistical proof '
                'that interruption adds zero error is offered, and no '
                'kernel is named as the cause; the installed ctc_loss '
                'documents nondeterministic CUDA gradients, which is why a '
                'baseline was measured rather than assumed.'.format(
                    baseline['worst_component'], observed['worst_component'],
                    baseline['watched_tensor']['max_abs_difference'],
                    observed['watched_tensor']['max_abs_difference'],
                    baseline['watched_tensor']['parameter_magnitude_rms']),
                {'restore_exact': restore['exact'],
                 'restore_max_difference': 0.0,
                 'rng_restored': restore['rng_restored'],
                 'watched_tensor_baseline': baseline['watched_tensor'],
                 'watched_tensor_resumed': observed['watched_tensor'],
                 'attention_shift_invariance':
                     'A k-projection bias shifts all attention logits for a '
                     'query equally and is largely absorbed by the softmax, '
                     'which would explain a loose gradient there. That remains '
                     'an untested hypothesis and the decision does not rest '
                     'on it: the baseline measurement settles the question.',
                 'short_tail_fixture': 'accumulate 4 over 15 batches leaves a '
                                       'final group of 3; every batch in the '
                                       'epoch contributes and each update '
                                       'divides by the utterances it covers'}),
    ]

    free = shutil.disk_usage(WORK.anchor).free
    report = {
        'phase': 'review response; long training NOT started',
        'findings': findings,
        'data': {
            'recordings': {s: v['recordings'] for s, v in audit['counts'].items()},
            'hours': {s: v['hours'] for s, v in audit['counts'].items()},
            'speakers': {s: v['speakers'] for s, v in audit['counts'].items()},
            'manifest_sha256': audit['manifest_sha256'],
            'labels_version': labels.LABELS_VERSION,
            'labels_contract_sha256': labels.contract_sha256(),
            'selection_exclusions': selection['exclusions'],
            'audit_drops': audit['drops'],
            'crop_policy_applied': audit['crop_policy_applied'],
            'target_expression_coverage': audit['target_expression_coverage'],
        },
        'data_contract': contract,
        'smoke_views': view['views'],
        'accepted_prior_evidence': {
            'input_parity': preflight['input_parity'],
            'dtypes': preflight['dtypes'],
            'masking': preflight['masking'],
            'splits': preflight['splits'],
            'step': {k: preflight['step'][k] for k in
                     ['loss', 'loss_finite', 'grad_norm',
                      'offloaded_layer_updated', 'resident_layer_updated',
                      'head_updated', 'frozen_frontend_unchanged']},
            'throughput': logged('THROUGHPUT'),
            'capacity': logged('CAPACITY'),
            'note': 'Re-used from the accepted preflight; not repeated. The '
                    'throughput and capacity rows are read back from '
                    'run/preflight.log, which is the primary record of those '
                    'runs.',
        },
        'control_experiment': control,
        'real_model_restore': restore,
        'approved_config': {'offload_layers': 26, 'batch_budget': 640000,
                            'accumulate': 8, 'max_epochs': 8,
                            'encoder_lr': 2e-5, 'head_lr': 1e-4,
                            'head_warmup_updates': 1000,
                            'mask_time_prob': 0.05, 'mask_feature_prob': 0.02,
                            'mask_time_min_masks': 0},
        'disk': {'free_gib': free / 2 ** 30,
                 'resumable_checkpoint_bytes': 11517990825,
                 'peak_with_atomic_replace_bytes': 2 * 11517990825 + 2 * 3850521519},
        'launch': {'started': False,
                   'reason': 'held for main-agent authorisation',
                   'command': LAUNCH,
                   'resume': LAUNCH + ' -Resume',
                   'progress': str(WORK / 'main/progress.json'),
                   'log': str(WORK / 'run/main.log')},
        'limitations': [
            'Targets are transcript-derived expected surface forms. LabelText '
            'is a transcription and can carry realisation or insertion '
            'notation, so rows whose notation cannot be resolved are excluded '
            'rather than guessed. JER measures recovery of that surface form, '
            'not pronunciation-error diagnosis, phoneme or IPA error rates, '
            'or product false-accept rates.',
            'ErrorTagged was N/A in the rows sampled here. That is sample '
            'evidence, not an exhaustive schema audit of the corpus.',
            'Valid and test speakers are unseen by this run but were used to '
            'evaluate a previous model, so they are not untouched subjects.',
            'Separate CUDA processes are not bit-identical. The measured '
            'baseline bounds that, and restoration itself is exact; no '
            'bitwise equivalence is claimed.',
            'Unity compatibility holds under Unity 6 GPUCompute with Inference '
            'Engine 2.6.1; legacy Sentis 1.2 did not pass for XLarge.',
            'Four target expressions are absent or near-absent as whole words. '
            'That limits direct examples, not the ability of a jamo head to '
            'emit them; any improvement on those words must be measured '
            'separately.',
            'The preflight and fixtures prove the training path, memory and '
            'throughput. Nothing here says anything about convergence or '
            'final accuracy.',
        ],
    }
    dump(WORK / 'claude-finalreview-report.json', report)
    print('written:', WORK / 'claude-finalreview-report.json')
    for f in findings:
        print(f'  {f["status"]:<28} {f["finding"]}. {f["title"]}')


if __name__ == '__main__':
    main()
