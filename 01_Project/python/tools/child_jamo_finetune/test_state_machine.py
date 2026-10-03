"""The run's state machine, driven through the real entry point.

Every case here runs `train.py` as a subprocess, with the production
phases, cursor, checkpointing, integrity checks and refusal paths. Only
the encoder is swapped: a ~1M-parameter HuBERT with the same architecture
and the same 42-output head, so a checkpoint is megabytes and a full
run is seconds. That keeps the expensive real-encoder evidence for the
things that actually need a 962M model - a finite update, throughput,
memory - and lets the control flow be tested exhaustively.

The tiny encoder is built once into WORK and pointed at by
`CHILD_JAMO_ENCODER`. The data is the smoke-only view from `smoke.py`,
drawn entirely from the train pool; the real test manifest is never read.

    .venv-gpu/Scripts/python.exe -m pytest \
        python/tools/child_jamo_finetune/test_state_machine.py -q
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK, load

PYTHON = ROOT / '.venv-gpu/Scripts/python.exe'
TRAIN = ROOT / 'python/tools/child_jamo_finetune/train.py'
TINY = WORK / 'tiny-encoder'
NAME = 'statemachine'
OUT = WORK / NAME

BASE = ['--name', NAME, '--train-split', 'smoketrain',
        '--valid-split', 'smokevalid', '--test-split', 'smoketest',
        '--offload-layers', '1', '--batch-budget', '160000',
        '--accumulate', '4', '--max-epochs', '2',
        '--head-warmup-updates', '2', '--warmup-updates', '2',
        '--eval-every', '1000', '--log-every', '1', '--checkpoint-every', '1']


def build_tiny():
    """A real HubertModel, small enough to checkpoint in a heartbeat."""
    if (TINY / 'model.safetensors').exists():
        return TINY
    from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor
    torch.manual_seed(0)
    config = HubertConfig(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
        intermediate_size=128, conv_dim=[32] * 7,
        conv_kernel=[10, 3, 3, 3, 3, 2, 2], conv_stride=[5, 2, 2, 2, 2, 2, 2],
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
        do_stable_layer_norm=True, feat_extract_norm='layer', conv_bias=True,
        mask_time_prob=0.05, mask_feature_prob=0.02, layerdrop=0.0)
    TINY.mkdir(parents=True, exist_ok=True)
    HubertModel(config).save_pretrained(TINY)
    # The same input contract as the real checkpoint: raw audio, no
    # normalisation, so the tests exercise the path the run will take.
    Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=False, return_attention_mask=False).save_pretrained(TINY)
    return TINY


def train(*extra, expect_ok=True):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
               HF_HUB_DISABLE_PROGRESS_BARS='1',
               CHILD_JAMO_ENCODER=str(build_tiny()))
    p = subprocess.run([str(PYTHON), str(TRAIN), *BASE, *extra], cwd=ROOT,
                       env=env, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if expect_ok and p.returncode != 0:
        pytest.fail(f'expected success from {extra}\n{p.stdout}\n{p.stderr}')
    if not expect_ok and p.returncode == 0:
        pytest.fail(f'expected refusal from {extra}\n{p.stdout}')
    return p


def state():
    saved = torch.load(OUT / 'last.pt', map_location='cpu', weights_only=False)
    out = dict(saved['state'])
    del saved
    return out


@pytest.fixture(scope='module', autouse=True)
def fresh():
    assert (WORK / 'manifest-smoketrain.jsonl').exists(), 'run smoke.py first'
    if OUT.resolve().parent == WORK.resolve() and OUT.exists():
        shutil.rmtree(OUT)
    yield
    for leftover in ['last.pt', 'best.pt', 'last.pt.tmp']:
        (OUT / leftover).unlink(missing_ok=True)


def test_a_head_phase_hands_over_and_pauses():
    train('--overwrite', '--stop-after-updates', '1')
    s = state()
    assert s['phase'] == 'encoder' and s['status'] == 'paused'
    assert s['update'] == 1
    # The head phase ran its budget and its audio was charged separately.
    assert s['phase_audio']['head'] > 0 and s['phase_audio']['encoder'] > 0
    assert s['phase_seconds']['head'] > 0


def test_b_phase_seconds_survive_a_checkpoint():
    """A resumed epoch must not forget the time it already spent."""
    before = state()['phase_seconds']['encoder']
    train('--resume', '--stop-after-updates', '2')
    assert state()['phase_seconds']['encoder'] > before


def test_c_resume_from_the_last_group_still_ends_the_epoch():
    """The epoch epilogue has to survive an interruption at the last group."""
    groups = updates_per_epoch()
    train('--resume', '--stop-after-updates', str(groups))
    paused = state()
    assert paused['group'] == groups - 1, 'expected a last-group checkpoint'
    assert paused['epoch'] == 0 and not paused['epoch_best'][1:]
    train('--resume', '--stop-after-updates', str(groups + 1))
    after = state()
    assert after['epoch'] >= 1
    # Epoch 0's end-of-epoch validation happened on the resumed process.
    assert len(after['epoch_best']) >= 2
    assert any(h['at'] == 'epoch_end' and h['epoch'] == 1
               for h in after['history'])


def test_d_pending_final_evaluation_is_resumable_without_training():
    train('--resume')
    done = state()
    assert done['status'] == 'completed'
    assert (OUT / 'result.json').exists()
    result = load(OUT / 'result.json')
    assert result['split'] == 'smoketest'
    assert result['selected_checkpoint']['jer'] == done['best']['jer']
    assert 'stop-after-updates' not in (done['stopped'] or '')


def test_e_completed_run_is_an_idempotent_no_op():
    before = load(OUT / 'result.json')
    updates = state()['update']
    p = train('--resume')
    assert 'already complete' in p.stdout
    assert state()['update'] == updates
    assert load(OUT / 'result.json') == before


def test_f_crash_before_the_test_leaves_a_recoverable_stage():
    """Reproduce the window between the last update and the test score."""
    saved = torch.load(OUT / 'last.pt', map_location='cpu', weights_only=False)
    saved['state']['status'] = 'final_evaluation_pending'
    saved['state']['phase'] = 'final_evaluation'
    torch.save(saved, OUT / 'last.pt')
    updates = saved['state']['update']
    del saved
    (OUT / 'result.json').unlink()
    train('--resume')
    assert (OUT / 'result.json').exists(), 'resume must finish the evaluation'
    assert state()['update'] == updates, 'no further optimizer updates'
    assert state()['status'] == 'completed'


def test_f2_restoration_itself_is_exact():
    """Weights, both moment sets, CPU masters and all four RNGs, bit for bit.

    Separate from "does the next update reproduce": no kernel runs here,
    so any difference would be a real save/load bug rather than CUDA
    nondeterminism.
    """
    os.environ['CHILD_JAMO_ENCODER'] = str(build_tiny())
    import importlib

    import common
    importlib.reload(common)
    import restore_check
    importlib.reload(restore_check)
    result = restore_check.check(OUT / 'last.pt')
    assert result['exact'], result
    assert result['rng_exact'], result['rng_restored']
    assert result['cpu_masters_match_gpu_weights']['max_abs_difference'] == 0.0


def test_g_refuses_a_changed_frozen_setting():
    p = train('--resume', '--encoder-lr', '9e-5', expect_ok=False)
    assert 'encoder_lr' in p.stderr


def test_h_refuses_a_manifest_that_the_audit_did_not_vouch_for():
    """Editing a manifest after the audit must stop the run, not be trained on."""
    path = WORK / 'manifest-smoketrain.jsonl'
    original = path.read_bytes()
    rows = original.decode('utf-8').splitlines()
    path.write_bytes('\n'.join(rows[:-1]).encode('utf-8') + b'\n')
    try:
        p = train('--overwrite', '--stop-after-updates', '1', expect_ok=False)
        assert 'not produced by the recorded' in p.stderr
    finally:
        path.write_bytes(original)


def test_i_refuses_a_second_live_writer():
    """The lock is an OS lock, so the refusal is not a check-then-write race."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ownership import Owned
    held = Owned(OUT, WORK).acquire()
    try:
        p = train('--resume', expect_ok=False)
        assert 'second writer' in p.stderr
    finally:
        held.release()


def test_j_refuses_a_run_directory_outside_the_work_root():
    from ownership import Owned
    with pytest.raises(SystemExit):
        Owned(WORK.parent / 'not-a-run', WORK).acquire()


def updates_per_epoch():
    import data
    rows = data.rows('smoketrain')
    config = json.loads((OUT / 'config.json').read_text(encoding='utf-8'))
    batches = data.plan(rows, config['batch_budget_samples'],
                        config['max_batch'], config['seed'], 'encoder:0')
    return len(data.groups(batches, config['accumulate']))
