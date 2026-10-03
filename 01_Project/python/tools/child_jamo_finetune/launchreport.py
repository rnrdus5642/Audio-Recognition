"""Record what was launched, and what it is actually doing.

Read back from the run's own files and the live process, so nothing here
is a claim about a job that is not running.

    python python/tools/child_jamo_finetune/launchreport.py
"""
from pathlib import Path
import ctypes
import json
import re
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import integrity
from common import WORK, dump, load

NAME = 'main'
OUT = WORK / NAME
LOG = WORK / f'run/{NAME}.log'

#: Overhead the pure-encoder figure does not include.
EVAL_SECONDS = 130          # measured range for a 1,500-recording fp32 pass
CHECKPOINT_SECONDS = 45     # an 11.5 GB atomic write


def alive(pid):
    handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def encoder_rate(log):
    """Measured encoder-phase seconds per update, from the log itself."""
    stamps = []
    for line in log.splitlines():
        m = re.search(r'^(\S+ \S+) TRAIN phase=encoder .*update=(\d+) ', line)
        if m:
            stamps.append((time.mktime(time.strptime(m.group(1),
                                                     '%Y-%m-%d %H:%M:%S')),
                           int(m.group(2))))
    if len(stamps) < 2:
        return None
    (t0, u0), (t1, u1) = stamps[0], stamps[-1]
    return (t1 - t0) / max(1, u1 - u0)


def main():
    config = load(OUT / 'config.json')
    progress = load(OUT / 'progress.json')
    owner = load(OUT / 'run.owner.json')
    launch = json.loads((OUT / 'launch.json').read_text(
        encoding='utf-8-sig'))
    log = LOG.read_text(encoding='utf-8')

    per_update = encoder_rate(log)
    updates_per_epoch = config['updates_per_epoch']
    epochs = config['max_epochs']
    pure = (per_update or 0) * updates_per_epoch * epochs
    # eval_every plus one end-of-epoch validation per epoch
    evaluations = epochs * (updates_per_epoch // config['eval_every_updates'] + 1)
    checkpoints = epochs * (updates_per_epoch // 500 + 1)
    overhead = evaluations * EVAL_SECONDS + checkpoints * CHECKPOINT_SECONDS

    checkpoint_files = {p.name: p.stat().st_size
                        for p in sorted(OUT.glob('*.pt'))}
    report = {
        'status': 'RUNNING - long training is in progress, NOT complete',
        'run': NAME,
        'directory': str(OUT),
        'started': launch['started'],
        'report_written': time.strftime('%Y-%m-%d %H:%M:%S'),
        'observed_at': progress['updated'],
        'owned_pids': {
            'trainer': {'pid': owner['pid'], 'alive': alive(owner['pid']),
                        'source': 'run.owner.json, written by the process that '
                                  'holds the OS lock'},
            'launcher_shim': {'pid': launch['pid'],
                              'alive': alive(launch['pid']),
                              'source': 'launch.json; parent of the trainer, '
                                        'not the training process itself'},
        },
        'focused_fixes': {
            'audit_provenance': {
                'change': 'audit-summary.json now freezes the label contract, '
                          'the selection summary hash and inputs, and the full '
                          'archive bindings. integrity.py reads all of them '
                          'from that one snapshot instead of taking the label '
                          'contract from the mutable selection-summary.json, '
                          'so a re-run of select without a re-run of audit can '
                          'no longer pass both checks while describing '
                          'different generations.',
                'rebuild': 'bounded audit only; no catalog rescan, no '
                           'reselection',
                'manifest_hashes_unchanged': True,
                'refusal_tests': [
                    'test_reselecting_without_reauditing_is_refused',
                    'test_a_changed_archive_binding_is_refused',
                    'test_an_audit_without_provenance_is_refused',
                    'test_a_changed_label_contract_is_refused',
                    'test_a_manifest_the_audit_does_not_know_is_refused'],
            },
            'lock_metadata': 'ownership.release() now unlinks its own owner '
                             'file while still holding the OS lock, then '
                             'unlocks and closes, closing the window where a '
                             'new owner could write metadata between the read '
                             'and the unlink.',
            'control_report_correction':
                'The earlier claim that all three control runs had identical '
                'validation histories was wrong and has been corrected in '
                'claude-control-experiment.json, the docs report and the '
                'review report. Measured: the two validations taken before '
                'any encoder update agree exactly across all three runs; '
                'after encoder updates the two uninterrupted runs differ from '
                'each other by one reference edit in 383 jamo '
                '(2.856396866840731 vs 2.85378590078329). No statistical '
                'proof that interruption adds zero error is claimed, and no '
                'GPU kernel is named as a confirmed cause.',
        },
        'contract': config['data_contract'],
        'encoder_weights_match_required_sha256':
            config['data_contract']['encoder']['model.safetensors']
            == '14977266907588c28da9403bc4c04478a29b49968dfe206fb9cb8e71e4d2e31d',
        'fixture_encoder_env_var_set_for_trainer': False,
        'splits': {'train_recordings': config['train_recordings'],
                   'train_hours': config['train_hours'],
                   'valid_recordings': config['valid_recordings']},
        'hyperparameters': {k: config[k] for k in [
            'offload_layers', 'batch_budget_samples', 'accumulate',
            'max_epochs', 'head_warmup_updates', 'encoder_lr', 'head_lr',
            'head_warmup_lr', 'warmup_updates', 'mask_time_prob',
            'mask_feature_prob', 'weight_decay', 'grad_clip',
            'eval_every_updates', 'min_delta_jer', 'precision', 'input',
            'frozen', 'updates_per_epoch', 'encoder_phase_total_updates']},
        'observed': {
            'phase': progress['phase'], 'status': progress['status'],
            'epoch': progress['epoch'], 'update': progress['update'],
            'audio_hours_processed': progress['audio_hours_processed'],
            'audio_hours_by_phase': progress['audio_hours_by_phase'],
            'head_warmup_completed_updates': 1000,
            'validation_history': [
                {k: h[k] for k in ['phase', 'update', 'at', 'jer', 'exact']}
                for h in progress['history']],
            'recent_training_lines': [l for l in log.splitlines()
                                      if 'phase=encoder' in l and 'TRAIN' in l][-4:],
            'loss_finite': all(
                0.0 < float(m) < 1e4 for m in
                re.findall(r'TRAIN .* loss=([0-9.]+)', log)),
            'latest_training_line': [l for l in log.splitlines()
                                     if 'TRAIN phase=encoder' in l][-1],
        },
        'throughput': {
            'encoder_seconds_per_update_measured': per_update,
            'encoder_audio_seconds_per_wall_second':
                progress['encoder_audio_seconds_per_wall_second'],
            'note': 'Measured in the encoder phase only. The frozen-head '
                    'phase ran several times faster and is not used for any '
                    'estimate here.',
        },
        'memory': {
            'cuda_peak_allocated_bytes': progress['cuda_peak_allocated_bytes'],
            'cuda_peak_allocated_gib':
                progress['cuda_peak_allocated_bytes'] / 2 ** 30,
            'cuda_peak_reserved_bytes': progress['cuda_peak_reserved_bytes'],
            'cuda_peak_reserved_gib':
                progress['cuda_peak_reserved_bytes'] / 2 ** 30,
            'device_total_bytes': 16311 * 2 ** 20,
            'device_total_gib': 16311 / 1024,
            'note': 'Measurements, not guarantees. Peak allocated is what the '
                    'run has actually held; peak reserved is the caching '
                    'allocator pool, which the fp32 validation pass is what '
                    'sets. Reserved is close to device capacity, and nothing '
                    'here promises the run completes without an out-of-memory '
                    'failure. If one occurs the run stops and must be resumed '
                    'explicitly with the command below; there is no watchdog.',
            'units': 'GiB values are bytes / 2**30',
        },
        'checkpoints': {
            'files_observed_now': checkpoint_files,
            'observed_note': 'These sizes are what is on disk at the timestamp '
                             'above. last.pt is already a full resumable '
                             'checkpoint: model and head weights, the head '
                             'optimizer moments, the RNG streams, the data '
                             'cursor and the frozen config. The currently '
                             'saved one is at encoder update 0, written at the '
                             'handover out of head warmup.',
            'estimated_next_last_pt_bytes': 11_500_000_000,
            'estimated_note': 'The encoder AdamW moments are allocated on the '
                              'first encoder optimizer step, so they enter the '
                              'next regular last.pt - normally at encoder '
                              'update 500. That file is estimated at about '
                              '11.5 GB (decimal); it has not been written yet '
                              'and the figure is not an observation.',
            'free_disk_bytes': shutil.disk_usage(WORK.anchor).free,
            'free_disk_gib': shutil.disk_usage(WORK.anchor).free / 2 ** 30,
        },
        'duration_estimate': {
            'scope': 'Estimated runtime for the maximum configured 8 encoder '
                     'epochs, at the throughput observed so far. It is not a '
                     'hard limit on wall-clock time: throughput can change, '
                     'and an interruption pauses the run until it is resumed '
                     'explicitly.',
            'pure_encoder_training_hours': pure / 3600,
            'validation_and_checkpoint_overhead_hours': overhead / 3600,
            'estimated_hours_for_8_epochs': (pure + overhead) / 3600,
            'basis': f'{per_update:.2f} s per encoder update measured live, '
                     f'{updates_per_epoch} updates per epoch, {epochs} epochs, '
                     f'plus {evaluations} validations and {checkpoints} '
                     f'checkpoint writes',
            'earliest_possible_finish':
                'end of encoder epoch 2 if early stopping fires, roughly a '
                'quarter of the 8-epoch estimate',
            'caveat': 'Early stopping can end the run sooner. Nothing here '
                      'predicts accuracy.',
        },
        'resume_command':
            'powershell -File python\\tools\\child_jamo_finetune\\launch.ps1 -Resume',
        'monitor': {'progress': str(OUT / 'progress.json'), 'log': str(LOG)},
        'not_done': [
            'Training is running; no final result exists yet.',
            'The test split has not been scored and will be scored once, '
            'automatically, on the best checkpoint when training ends.',
        ],
    }
    dump(WORK / 'claude-launch-report.json', report)
    print(json.dumps({
        'status': report['status'],
        'trainer_pid': report['owned_pids']['trainer'],
        'encoder_sha_ok': report['encoder_weights_match_required_sha256'],
        'phase': report['observed']['phase'],
        'update': report['observed']['update'],
        'loss_finite': report['observed']['loss_finite'],
        'seconds_per_update': per_update,
        'estimated_hours_for_8_epochs':
            report['duration_estimate']['estimated_hours_for_8_epochs'],
    }, indent=1))


if __name__ == '__main__':
    main()
