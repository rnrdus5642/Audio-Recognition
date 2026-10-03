"""Fine-tune the verified Korean HuBERT encoder on child speech, resumably.

Two phases. A fresh CTC head first learns to read the frozen encoder's
features, which keeps the first real gradients from being dominated by a
randomly initialised output layer. Then the transformer stack and the
head train together; only the convolutional waveform frontend stays
frozen throughout.

The run is expected to take days, so every piece of state that decides
the next update - weights, both AdamW moment sets, the data cursor and
all four random number generators - is checkpointed together, and
`--resume` continues from exactly there. A resume re-derives the
configuration from the current arguments and data and refuses to start
if anything frozen disagrees with the checkpoint, because silently
adopting the saved config while running with different arguments would
produce a run that no record describes.

Precision is explicit rather than incidental. Training runs under BF16
autocast on FP32 master weights; the CTC head and the loss always see
FP32, and validation and the final test run in full FP32 with no
autocast, matching the encoder comparison.

Input is raw, unnormalised waveform. That is not an oversight: this
checkpoint's own `preprocessor_config.json` sets `do_normalize: false`,
and `preflight.py` checks parity against that feature extractor.

What the validation score means: the targets are the corpus transcript
put through the project's Korean phonological rules, so the jamo error
rate measures how well the model recovers that expected surface form. It
is not a measurement of pronunciation-error diagnosis, and nothing here
supports a claim about phoneme accuracy or false accepts.

    python python/tools/child_jamo_finetune/train.py --help
"""
from pathlib import Path
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from transformers import HubertModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data
import integrity
import labels
from common import ENCODER, WORK, dump, log
import reviewed
from optim import PartialCPUAdamW
from ownership import Owned

MIN_DELTA = 0.002        # JER improvement that counts as meaningful
MAX_EPOCHS = 8

#: Settings a resume may not quietly change. Anything here that differs
#: between the checkpoint and the current invocation stops the run.
#: Everything that changes the arithmetic, the data, or what "best" and
#: "stop" mean belongs here - including the evaluation settings, since a
#: resume that quietly widened the validation batch or the early-stopping
#: threshold would make the second half of a run incomparable to the first.
FROZEN_CONFIG = (
    'seed', 'attention', 'offload_layers', 'foreach',
    'batch_budget_samples', 'max_batch', 'accumulate',
    'eval_budget_samples', 'eval_every_updates',
    'encoder_lr', 'head_lr', 'head_warmup_lr', 'head_warmup_updates',
    'warmup_updates', 'weight_decay', 'grad_clip', 'max_epochs',
    'min_delta_jer', 'mask_time_prob', 'mask_feature_prob', 'layerdrop',
    'encoder_phase_total_updates', 'updates_per_epoch',
    'train_split', 'valid_split', 'test_split',
    'frozen', 'input', 'precision', 'accumulation',
    'data_contract',
)

#: A run moves through these and stops. `final_evaluation_pending` exists
#: so that a crash between the last update and the test score is
#: recoverable by an ordinary resume, without another optimizer step.
RUNNING, PENDING_TEST, COMPLETED = 'running', 'final_evaluation_pending', 'completed'


class Head(torch.nn.Module):
    """The fresh jamo CTC head. Nothing is carried over from the pilot."""

    def __init__(self, hidden, vocab, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(hidden, vocab)

    def forward(self, x):
        return self.linear(self.dropout(x))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build(args):
    model = HubertModel.from_pretrained(
        ENCODER, local_files_only=True, attn_implementation=args.attention,
        dtype=torch.float32)
    model.config.layerdrop = 0.0
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = args.mask_time_prob
    model.config.mask_feature_prob = args.mask_feature_prob
    # Utterances as short as half a second are only 24 frames. A minimum
    # mask count would blanket them; letting the count fall out of the
    # sequence's own length leaves short utterances alone.
    model.config.mask_time_min_masks = 0
    model.config.mask_feature_min_masks = 0
    model.feature_extractor._freeze_parameters()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    seed_everything(args.seed)
    head = Head(model.config.hidden_size, len(labels.TOKENS))
    return model.cuda(), head.cuda()


def forward(model, head, batch, *, grad, autocast):
    """Encoder then head, with the precision policy stated at the call site.

    The head and everything after it always run in FP32. Which dtype the
    encoder's final layer norm returns under autocast is an
    implementation detail of the transformers version in use, so the cast
    is explicit instead of inherited.
    """
    context = torch.enable_grad() if grad else torch.no_grad()
    with context, torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
        hidden = model(input_values=batch['input_values'],
                       attention_mask=batch['attention_mask']).last_hidden_state
    logits = head(hidden.float())
    lengths = model._get_feat_extract_output_lengths(
        batch['attention_mask'].sum(-1)).to(torch.long)
    return logits, lengths


def ctc_sum(logits, lengths, batch):
    """Summed length-normalised CTC loss over the utterances in one batch.

    Returned as a sum, not a mean, so an optimizer update can divide once
    by the number of utterances it actually covers. Dividing each batch
    by a fixed accumulation count instead would weight a short tail group
    - and every batch that holds fewer recordings than the others - wrong.
    """
    logp = logits.float().log_softmax(-1).transpose(0, 1)
    per = torch.nn.functional.ctc_loss(
        logp, batch['targets'].to(logits.device), lengths,
        batch['target_lengths'], blank=labels.BLANK, reduction='none',
        zero_infinity=False)
    return (per / batch['target_lengths'].to(per.device)).sum()


def collapse(ids):
    """CTC greedy decode: merge repeats, then drop blanks."""
    out = []
    for i in ids:
        if not out or out[-1] != i:
            out.append(i)
    return [i for i in out if i != labels.BLANK]


@torch.inference_mode()
def evaluate(model, head, rows, batches, device='cuda'):
    """Jamo error rate over true lengths only; word boundaries ignored.

    FP32, no autocast: validation picks the checkpoint and the final test
    is reported, so neither should move with BF16 kernel choice.
    """
    model.eval()
    head.eval()
    errors = reference = exact = 0
    for i, batch in data.Prefetch(rows, batches, device):
        logits, lengths = forward(model, head, batch, grad=False, autocast=False)
        assert torch.isfinite(logits).all()
        argmax = logits.argmax(-1).cpu().numpy()
        for b, index in enumerate(batches[i]):
            width = int(lengths[b])
            hypothesis = [t for t in collapse(argmax[b, :width].tolist())
                          if t != labels.BOUNDARY]
            target = [t for t in rows[index]['label_ids'] if t != labels.BOUNDARY]
            err = Levenshtein.distance(target, hypothesis)
            errors += err
            reference += len(target)
            exact += err == 0
    return {'jer': errors / reference, 'errors': errors,
            'reference_jamo': reference, 'exact': exact,
            'recordings': sum(len(b) for b in batches), 'precision': 'float32'}


def learning_rate(base, update, total, warmup):
    if update <= warmup:
        return base * update / max(1, warmup)
    return base * max(0.0, (total - update) / max(1, total - warmup))


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def save(path, payload):
    """Atomic: a crash during a multi-gigabyte write cannot destroy the
    checkpoint that was already there."""
    temp = Path(str(path) + '.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def checkpoint_payload(model, head, opt, state, config):
    return {'model': model.state_dict(), 'head': head.state_dict(),
            'optimizer': opt.state_dict(), 'state': state,
            'rng': rng_state(), 'config': config}


def describe(args, splits, train_rows, valid_rows):
    """The configuration, derived fresh from the arguments and the data."""
    return {
        'name': args.name, 'seed': args.seed, 'encoder': str(ENCODER),
        'data_contract': integrity.contract(
            {'train': splits[0], 'valid': splits[1], 'test': splits[2]}),
        'attention': args.attention, 'offload_layers': args.offload_layers,
        'foreach': bool(args.foreach),
        'batch_budget_samples': args.batch_budget, 'max_batch': args.max_batch,
        'eval_budget_samples': args.eval_budget,
        'accumulate': args.accumulate, 'encoder_lr': args.encoder_lr,
        'head_lr': args.head_lr, 'head_warmup_lr': args.head_warmup_lr,
        'head_warmup_updates': args.head_warmup_updates,
        'warmup_updates': args.warmup_updates, 'weight_decay': 0.01,
        'grad_clip': 1.0, 'mask_time_prob': args.mask_time_prob,
        'mask_feature_prob': args.mask_feature_prob, 'layerdrop': 0.0,
        'max_epochs': args.max_epochs, 'min_delta_jer': MIN_DELTA,
        'eval_every_updates': args.eval_every,
        'frozen': 'convolutional waveform frontend only',
        'input': 'raw waveform; this checkpoint sets do_normalize=false',
        'precision': 'bf16 autocast encoder, fp32 head and CTC loss, '
                     'fp32 master weights, fp32 validation and test',
        'accumulation': 'loss summed as length-normalised per utterance and '
                        'divided by the utterances in the optimizer update',
        'train_split': splits[0], 'valid_split': splits[1],
        'test_split': splits[2],
        'train_recordings': len(train_rows),
        'train_hours': sum(r['seconds'] for r in train_rows) / 3600,
        'valid_recordings': len(valid_rows),
        'limitations': [
            'Targets are transcript-derived expected surface forms, not '
            'annotations of what the child actually pronounced.',
            'Validation JER does not measure pronunciation-error diagnosis, '
            'phoneme or IPA error rates, or product false-accept rates.'],
    }


def reconcile(saved, fresh):
    """Differences in frozen settings between a checkpoint and this run."""
    out = {}
    for key in FROZEN_CONFIG:
        a, b = saved.get(key), fresh.get(key)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            out.update({f'{key}.{k}': v
                        for k, v in integrity.differences(a, b).items()})
        else:
            out[key] = {'checkpoint': a, 'now': b}
    return out


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(args.cpu_threads)
    out = WORK / args.name
    # An OS-level lock held for the life of this process, and a check that
    # the run cannot write outside the work root.
    Owned(out, WORK).acquire()

    splits = (args.train_split, args.valid_split, args.test_split)
    train_rows = data.rows(splits[0])
    if not args.resume:
        reviewed.require_repaired(train_rows)
    valid_rows = data.rows(splits[1])
    valid_batches = data.plan(valid_rows, args.eval_budget, args.max_batch,
                              args.seed, 'valid')
    seed_everything(args.seed)
    model, head = build(args)

    config = describe(args, splits, train_rows, valid_rows)
    # The linear decay needs a horizon before the first update, so it comes
    # from the epoch plan rather than a guess. Early stopping can end the
    # run before it; the schedule is the same either way.
    epoch_groups = data.groups(
        data.plan(train_rows, args.batch_budget, args.max_batch, args.seed,
                  'encoder:0'), args.accumulate)
    config['encoder_phase_total_updates'] = (
        args.total_updates or max(1, args.max_epochs * len(epoch_groups)))
    config['updates_per_epoch'] = len(epoch_groups)

    state = {'phase': 'head', 'epoch': 0, 'group': -1, 'update': 0,
             'status': RUNNING, 'audio_seconds': 0.0, 'best': None,
             'history': [], 'epoch_best': [], 'stopped': None,
             'phase_audio': {'head': 0.0, 'encoder': 0.0},
             'phase_seconds': {'head': 0.0, 'encoder': 0.0}}
    opt = PartialCPUAdamW(model, head, args.offload_layers, args.encoder_lr,
                          args.head_lr, foreach=args.foreach)
    config['offloaded_parameters'] = opt.offloaded_parameters
    config['resident_parameters'] = opt.resident_parameters
    config['trainable_parameters'] = sum(
        p.numel() for p in model.parameters() if p.requires_grad) + \
        sum(p.numel() for p in head.parameters())

    last = out / 'last.pt'
    if args.resume:
        if not last.exists():
            raise SystemExit(f'nothing to resume: {last} is missing')
        saved = torch.load(last, map_location='cpu', weights_only=False)
        drift = reconcile(saved['config'], config)
        if drift:
            raise SystemExit(
                'refusing to resume: these frozen settings differ from the '
                'checkpoint\n  ' + '\n  '.join(
                    f'{k}: checkpoint={v["checkpoint"]!r} now={v["now"]!r}'
                    for k, v in drift.items()))
        if saved['state'].get('status') == COMPLETED:
            # Idempotent: resuming a finished run neither retrains nor
            # re-scores it. A launcher may safely retry.
            log(f'{args.name}.log', f'ALREADY_COMPLETE {saved["state"].get("stopped")}')
            print(f'run "{args.name}" is already complete '
                  f'({saved["state"].get("stopped")}); nothing to do')
            return
        model.load_state_dict(saved['model'])
        head.load_state_dict(saved['head'])
        opt.load_state_dict(saved['optimizer'])
        set_rng_state(saved['rng'])
        state = saved['state']
        config = saved['config']
        log(f'{args.name}.log', f'RESUME phase={state["phase"]} '
            f'epoch={state["epoch"]} update={state["update"]} '
            f'group={state["group"]} status={state["status"]}')
    else:
        if last.exists() and not args.overwrite:
            raise SystemExit(f'{last} exists; pass --resume or --overwrite')
        dump(out / 'config.json', config)

    pending_test = state['status'] == PENDING_TEST
    state['status'] = RUNNING
    if not pending_test:
        # The reason a previous invocation stopped describes that
        # invocation, not this one. Leaving it in place made a completed
        # run still report the pause that had interrupted it days earlier.
        state['stopped'] = None
    total_updates = config['encoder_phase_total_updates']
    started = time.monotonic()
    elapsed_before = state.get('elapsed_seconds', 0.0)
    phase_clock = {'mark': time.monotonic(), 'phase': state['phase']}

    def charge(phase=None):
        """Bank the time spent in the current phase; optionally switch.

        Phases are added over time - the final evaluation became one -
        so the tally accepts any name rather than assuming the two the
        run started with.
        """
        now = time.monotonic()
        current = phase_clock['phase']
        state['phase_seconds'][current] = (
            state['phase_seconds'].get(current, 0.0) + now - phase_clock['mark'])
        phase_clock.update(mark=now, phase=phase or current)

    def progress(extra=None):
        wall = elapsed_before + time.monotonic() - started
        # The ETA uses encoder-phase throughput only. The frozen-head phase
        # runs several times faster because it has no encoder backward, and
        # averaging the two would make a multi-day estimate look short.
        seconds = state['phase_seconds']['encoder'] + (
            time.monotonic() - phase_clock['mark']
            if phase_clock['phase'] == 'encoder' else 0.0)
        done = state['phase_audio']['encoder']
        rate = done / seconds if seconds > 1 else 0.0
        target = config['train_hours'] * 3600 * config['max_epochs']
        dump(out / 'progress.json', {
            'phase': state['phase'], 'status': state['status'],
            'epoch': state['epoch'], 'update': state['update'],
            'group': state['group'],
            'audio_hours_processed': state['audio_seconds'] / 3600,
            'audio_hours_by_phase': {k: v / 3600
                                     for k, v in state['phase_audio'].items()},
            'seconds_by_phase': dict(state['phase_seconds']),
            'audio_hours_per_epoch': config['train_hours'],
            'elapsed_seconds': wall,
            'encoder_audio_seconds_per_wall_second': rate,
            'projected_encoder_total_seconds': target / rate if rate else None,
            'projected_remaining_seconds': (target - done) / rate if rate else None,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'best': state['best'], 'epoch_best': state['epoch_best'],
            'stopped': state['stopped'], 'history': state['history'][-40:],
            'updated': time.strftime('%Y-%m-%d %H:%M:%S'), **(extra or {}),
        })

    def validate(tag='interval'):
        score = evaluate(model, head, valid_rows, valid_batches)
        record = {'phase': state['phase'], 'epoch': state['epoch'],
                  'update': state['update'], 'at': tag, **score}
        state['history'].append(record)
        if state['best'] is None or score['jer'] < state['best']['jer']:
            state['best'] = record
            save(out / 'best.pt', {'model': model.state_dict(),
                                   'head': head.state_dict(),
                                   'config': config, 'valid': record})
        log(f'{args.name}.log', f'VALID {tag} phase={state["phase"]} '
            f'epoch={state["epoch"]} update={state["update"]} '
            f'jer={score["jer"]:.4f} exact={score["exact"]}')
        progress()
        return score

    def persist():
        # Bank the phase clock first. Saving phase_seconds without the
        # time since the last mark lost every minute of a resumed epoch
        # and made the ETA read far faster than the run actually was.
        charge()
        state['elapsed_seconds'] = elapsed_before + time.monotonic() - started
        save(last, checkpoint_payload(model, head, opt, state, config))

    def train_phase(phase, epochs, encoder_lr, head_lr, warmup, total):
        """One phase; returns True if the whole run should stop."""
        frozen = phase == 'head'
        charge(phase)
        for p in model.parameters():
            p.requires_grad_(not frozen)
        if not frozen:
            model.feature_extractor._freeze_parameters()
        while state['epoch'] < epochs:
            batches = data.plan(train_rows, args.batch_budget, args.max_batch,
                                args.seed, f'{phase}:{state["epoch"]}')
            groups = data.groups(batches, args.accumulate)
            start = state['group'] + 1
            # A checkpoint taken at the last group of an epoch leaves
            # nothing to train here, but the epoch still has to end
            # properly. Falling through to the epilogue rather than
            # skipping ahead is what makes that epoch's validation and
            # early-stopping decision survive an interruption.
            model.train()
            head.train()
            if frozen:
                model.eval()
            losses = []
            # Groups are contiguous slices of the batch plan, so one
            # prefetching pass starting at the resumed group feeds them all.
            stream = iter(data.Prefetch(train_rows, batches, 'cuda',
                                        start * args.accumulate))
            for g in range(start, len(groups)):
                group = groups[g]
                # Divide by the utterances this update actually covers, so
                # the final short group of an epoch is weighted correctly.
                utterances = sum(len(b) for b in group)
                opt.zero_grad()
                for _ in group:
                    _i, batch = next(stream)
                    logits, lengths = forward(model, head, batch,
                                              grad=not frozen, autocast=True)
                    total_loss = ctc_sum(logits, lengths, batch)
                    if not torch.isfinite(total_loss):
                        raise RuntimeError(f'non-finite loss in group {g}')
                    (total_loss / utterances).backward()
                    losses.append(float(total_loss.detach())
                                  / len(batch['samples']))
                    seconds = sum(batch['samples']) / 16000
                    state['audio_seconds'] += seconds
                    state['phase_audio'][phase] = (
                        state['phase_audio'].get(phase, 0.0) + seconds)
                state['group'] = g
                state['update'] += 1
                lr_head = learning_rate(head_lr, state['update'], total, warmup)
                lr_enc = learning_rate(encoder_lr, state['update'], total, warmup)
                opt.set_lr(lr_enc, lr_head)
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad]
                    + list(head.parameters()), 1.0, error_if_nonfinite=True)
                opt.step()
                opt.zero_grad()

                if state['update'] % args.log_every == 0:
                    log(f'{args.name}.log',
                        f'TRAIN phase={phase} epoch={state["epoch"]} '
                        f'update={state["update"]} loss={np.mean(losses):.4f} '
                        f'grad_norm={float(norm):.2f} lr={lr_enc:.2e}/{lr_head:.2e} '
                        f'audio_h={state["audio_seconds"] / 3600:.1f} '
                        f'vram={torch.cuda.max_memory_allocated() / 2**30:.1f}G')
                    losses = []
                    progress()
                if state['update'] % args.eval_every == 0:
                    validate()
                    model.train()
                    head.train()
                    if frozen:
                        model.eval()
                if state['update'] % args.checkpoint_every == 0:
                    persist()
                if args.stop_after_updates and phase == 'encoder' \
                        and state['update'] >= args.stop_after_updates:
                    state['stopped'] = f'--stop-after-updates {args.stop_after_updates}'
                    state['status'] = 'paused'
                    persist()
                    return True
                if phase == 'head' and state['update'] >= total:
                    return False
            if phase == 'head':
                # The head phase has no epoch structure; it ends on its
                # update budget. Reaching the end of the data first just
                # means another pass.
                state['epoch'] += 1
                state['group'] = -1
                continue
            state['epoch'] += 1
            state['group'] = -1
            if not frozen:
                # Early stopping compares epochs, so the comparison point
                # has to be a validation actually taken at the epoch end.
                validate(tag='epoch_end')
                model.train()
                head.train()
                state['epoch_best'].append(state['best']['jer'])
                persist()
                marks = state['epoch_best']
                # Two completed encoder epochs without a meaningful gain
                # against where we stood two epochs ago.
                if len(marks) >= 3 and marks[-1] > marks[-3] - MIN_DELTA:
                    state['stopped'] = (f'no JER gain of {MIN_DELTA} over two '
                                        f'epochs ({marks[-3]:.4f} -> {marks[-1]:.4f})')
                    return True
        state['stopped'] = (f'{phase} phase reached its budget of '
                            f'{epochs} epochs')
        return True

    if not pending_test:
        if state['phase'] == 'head':
            if state['update'] >= args.head_warmup_updates:
                # This checkpoint already finished the head budget. Hand
                # over without spending another head update on it.
                log(f'{args.name}.log', 'HEAD_PHASE already complete; '
                                        'handing over to the encoder')
            else:
                if state['update'] == 0:
                    validate(tag='initial')
                train_phase('head', math.inf, 0.0, args.head_warmup_lr,
                            max(1, args.head_warmup_updates // 10),
                            args.head_warmup_updates)
            state.update(phase='encoder', epoch=0, group=-1, update=0)
            state['epoch_best'] = [validate(tag='encoder_start')['jer']]
            persist()

        if state['phase'] == 'encoder':
            train_phase('encoder', config['max_epochs'], args.encoder_lr,
                        args.head_lr, args.warmup_updates, total_updates)

        charge()
        if state['status'] == 'paused':
            persist()
            progress()
            log(f'{args.name}.log',
                f'PAUSED {state["stopped"]}; resume with --resume')
            return

        # Training is over but the run is not finished. Recording that
        # here means a crash during the final evaluation is recoverable
        # by an ordinary resume, with no further optimizer updates.
        state['phase'] = 'final_evaluation'
        state['status'] = PENDING_TEST
        persist()
        progress()
        log(f'{args.name}.log', f'TRAINING_DONE best={state["best"]} '
            f'stopped={state["stopped"]}; final evaluation pending')
    else:
        log(f'{args.name}.log', 'RESUME_FINAL_EVALUATION; no further training')

    final_test(args, out, model, head, config, state, splits[2])

    # Only now: the result is on disk, so "completed" is true.
    state['status'] = COMPLETED
    charge()
    persist()
    progress()
    log(f'{args.name}.log', f'COMPLETE best={state["best"]} '
        f'stopped={state["stopped"]}')


def final_test(args, out, model, head, config, state, split):
    """Score the frozen test manifest once, with the best weights.

    Not the last weights, and not more than once: the test split exists
    to be spent a single time on the checkpoint that validation chose.
    """
    if args.skip_test:
        log(f'{args.name}.log', 'TEST skipped by --skip-test')
        return
    result = out / 'result.json'
    if result.exists():
        # Spent once, and once only. A resume that reaches here again
        # must not re-score the test split.
        log(f'{args.name}.log', f'TEST already recorded in {result}')
        return
    best = out / 'best.pt'
    if not best.exists():
        log(f'{args.name}.log', 'TEST skipped: no best checkpoint')
        return
    saved = torch.load(best, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    head.load_state_dict(saved['head'])
    rows = data.rows(split)
    batches = data.plan(rows, args.eval_budget, args.max_batch, args.seed, 'test')
    score = evaluate(model, head, rows, batches)
    dump(result, {
        'test': score, 'split': split,
        'data_contract': config['data_contract'],
        'selected_checkpoint': saved['valid'],
        'selection': 'best validation JER; the test split was not used to '
                     'select anything',
        'stopped': state['stopped'], 'epoch_best': state['epoch_best'],
        'config': config,
        'limitations': config['limitations'],
    })
    log(f'{args.name}.log', f'TEST jer={score["jer"]:.4f} '
        f'exact={score["exact"]}/{score["recordings"]} -> {result}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='main')
    p.add_argument('--seed', type=int, default=917)
    p.add_argument('--attention', default='sdpa', choices=['sdpa', 'eager'])
    p.add_argument('--offload-layers', type=int, default=26)
    p.add_argument('--batch-budget', type=int, default=640000,
                   help='padded samples per batch (count x longest)')
    p.add_argument('--eval-budget', type=int, default=1280000)
    p.add_argument('--max-batch', type=int, default=32)
    p.add_argument('--accumulate', type=int, default=8)
    p.add_argument('--mask-time-prob', type=float, default=0.05)
    p.add_argument('--mask-feature-prob', type=float, default=0.02)
    p.add_argument('--encoder-lr', type=float, default=2e-5)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--head-warmup-lr', type=float, default=1e-3)
    p.add_argument('--head-warmup-updates', type=int, default=1000)
    p.add_argument('--warmup-updates', type=int, default=500)
    p.add_argument('--total-updates', type=int, default=0,
                   help='encoder-phase LR horizon; 0 derives it from the data')
    p.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--stop-after-updates', type=int, default=0,
                   help='bound the encoder phase; 0 runs the full schedule')
    p.add_argument('--train-split', default='train')
    p.add_argument('--valid-split', default='valid')
    p.add_argument('--test-split', default='test')
    p.add_argument('--skip-test', action='store_true')
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--no-foreach', dest='foreach', action='store_false')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
