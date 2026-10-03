"""Prove the training path before committing days of GPU time.

Each check is here because it can fail silently and cost the whole run:

  splits      the three manifests share no speaker and no audio, and the
              archives still match the catalogue their offsets came from
  parity      what we feed the encoder matches this checkpoint's own
              feature extractor, which specifies raw unnormalised audio
  dtypes      the observed precision at each stage, not the assumed one
  masking     padding is masked correctly, judged against a measured
              noise floor and a control that withholds the mask
  padding     the same comparison across bf16 and fp32, for the record
  decode      mixed-length greedy decoding respects each true length
  step        real audio gives a finite loss and actually moves weights
  resume      a checkpoint reproduces the next update bit for bit
  capacity    measured VRAM and throughput at several batch and offload
              settings, on the longest utterances in the manifest
  throughput  the same on real planned batches, with accumulation

Checks that can fail do fail: `masking`, `dtypes`, `splits` and `step`
assert their verdicts rather than printing a number for someone else to
judge.

Nothing here trains to convergence and nothing here produces a score
worth reporting as accuracy. The bounded end-to-end exercise of the
production entry point lives in `smoke.py` plus the CLI runs recorded in
the review report, and uses a smoke-only data view so the real test
split is never spent.

    python python/tools/child_jamo_finetune/preflight.py
"""
from pathlib import Path
import gc
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog
import data
import labels
import manifest
import train
from common import ENCODER, WORK, dump, load, log
from optim import PartialCPUAdamW

REPORT = WORK / 'claude-preflight-report.json'


def gpu_free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def check_splits():
    rows = {s: data.rows(s) for s in ['train', 'valid', 'test']}
    speakers = {s: {r['speaker'] for r in rows[s]} for s in rows}
    pcm = {s: {r['pcm_sha256'] for r in rows[s]} for s in rows}
    dev = manifest.old_dev_speakers()
    overlaps = {f'{a}|{b}': len(speakers[a] & speakers[b])
                for a in rows for b in rows if a < b}
    audio_overlaps = {f'{a}|{b}': len(pcm[a] & pcm[b])
                      for a in rows for b in rows if a < b}
    result = {
        'recordings': {s: len(rows[s]) for s in rows},
        'speakers': {s: len(speakers[s]) for s in rows},
        'speaker_overlap': overlaps,
        'identical_audio_overlap': audio_overlaps,
        'previous_dev_speakers_reserved': len(dev),
        'previous_dev_speakers_in_train': len(dev & speakers['train']),
        'label_ids_within_vocabulary': all(
            0 <= t < len(labels.TOKENS) and t != labels.BLANK
            for s in rows for r in rows[s] for t in r['label_ids']),
        'ctc_feasible': all(
            data.audio.frames(r['samples']) >= labels.ctc_width(r['label_ids'])
            for s in rows for r in rows[s]),
    }
    assert not any(overlaps.values()) and not any(audio_overlaps.values())
    assert result['previous_dev_speakers_in_train'] == 0
    assert result['label_ids_within_vocabulary'] and result['ctc_feasible']
    return result


def check_input_parity(rows):
    """What this package feeds the encoder must be what the checkpoint expects.

    This checkpoint's own `preprocessor_config.json` sets
    `do_normalize: false`, so the correct input is the raw waveform and
    applying wav2vec2-style zero-mean unit-variance normalisation would
    be wrong. Rather than trust that reading, the local feature extractor
    is run and its output compared with what `data.collate` produces.

    The same config sets `return_attention_mask: false`, which is only
    safe for a single unpadded input. Batched training needs a mask, so
    this package builds one; here it is checked against the mask the
    feature extractor produces when asked for it.
    """
    from transformers import AutoFeatureExtractor
    extractor = AutoFeatureExtractor.from_pretrained(ENCODER,
                                                     local_files_only=True)
    order = sorted(range(len(rows)), key=lambda i: rows[i]['samples'])
    picks = [order[0], order[len(order) // 2], order[-1]]
    waves = []
    for i in picks:
        r = rows[i]
        x, _policy = data.audio.crop(
            data.audio.decode(data.audio.payload(r['archive'], r['offset'],
                                                 r['size'])),
            r['speech_start'], r['speech_end'])
        waves.append(x)
    reference = extractor(waves, sampling_rate=16000, return_tensors='pt',
                          padding=True, return_attention_mask=True)
    ours = data.collate(rows, picks, 'cpu')
    values = ours['input_values'].numpy()
    return {
        'do_normalize': bool(extractor.do_normalize),
        'config_return_attention_mask': bool(extractor.return_attention_mask),
        'policy': 'raw unnormalised waveform, as this checkpoint specifies',
        'max_abs_difference_vs_feature_extractor': float(
            np.abs(reference['input_values'].numpy() - values).max()),
        'attention_mask_matches_feature_extractor': bool(
            (reference['attention_mask'].numpy()
             == ours['attention_mask'].numpy()).all()),
        'raw_waveform_passed_through_unchanged': bool(np.allclose(
            values[0, :len(waves[0])], waves[0], atol=0, rtol=0)),
        'would_normalisation_change_input': float(
            np.abs((waves[0] - waves[0].mean()) / (waves[0].std() + 1e-7)
                   - waves[0]).max()),
    }


def check_dtypes(model, head, rows):
    """Record the precision actually observed, rather than assuming it."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]['samples'])
    batch = data.collate(rows, order[len(order) // 2: len(order) // 2 + 2], 'cuda')
    model.eval()
    head.eval()
    seen = {}
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = model(input_values=batch['input_values'],
                           attention_mask=batch['attention_mask']).last_hidden_state
        seen['encoder_output_under_autocast'] = str(hidden.dtype)
        seen['head_input_after_explicit_cast'] = str(hidden.float().dtype)
        seen['logits_training_path'] = str(head(hidden.float()).dtype)
        logits, _ = train.forward(model, head, batch, grad=False, autocast=False)
        seen['logits_evaluation_path'] = str(logits.dtype)
    seen['head_parameters'] = str(head.linear.weight.dtype)
    seen['encoder_master_weights'] = str(
        next(model.encoder.parameters()).dtype)
    seen['policy'] = ('the head and CTC loss are cast to float32 explicitly, '
                      'so the policy does not depend on which op autocast '
                      'happens to leave last')
    assert seen['logits_training_path'] == 'torch.float32'
    assert seen['logits_evaluation_path'] == 'torch.float32'
    assert seen['encoder_master_weights'] == 'torch.float32'
    return seen


def check_masking(model, head, rows, args):
    """Decide whether padding is masked correctly, with a stated tolerance.

    A single measurement proves nothing on its own, because a batched and
    an unbatched forward differ anyway - different shapes pick different
    kernels and reduction orders. So this measures three things:

      floor     a batch of identical copies, which carries no padding at
                all, giving the pure batch-shape noise level
      masked    a short utterance batched with a long one, masked properly
      unmasked  the same batch with the mask withheld

    Correct masking means `masked` sits at the noise floor while
    `unmasked` is far above it. The last part matters most: without it
    the test cannot distinguish a working mask from a test too blunt to
    notice a broken one.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i]['samples'])
    short, long_ = order[0], order[-1]
    model.eval()
    head.eval()

    def logits_for(indices, mask=True, width_from=None):
        batch = data.collate(rows, indices, 'cuda')
        if not mask:
            batch = dict(batch, attention_mask=torch.ones_like(
                batch['attention_mask']))
        with torch.inference_mode():
            out, lengths = train.forward(model, head, batch, grad=False,
                                         autocast=False)
        return out, lengths

    alone, alone_len = logits_for([short])
    width = int(alone_len[0])
    reference = alone[0, :width].float()

    copies, _ = logits_for([short] * 3)
    floor = float((copies[0, :width].float() - reference).abs().max())

    masked, masked_len = logits_for([short, long_])
    assert int(masked_len[0]) == width
    observed = float((masked[0, :width].float() - reference).abs().max())

    unmasked, _ = logits_for([short, long_], mask=False)
    leaked = float((unmasked[0, :width].float() - reference).abs().max())

    scale = float(reference.std())
    # Padding may cost no more than the noise floor plus a small absolute
    # allowance; the mask must also demonstrably matter.
    allowance = max(4 * floor, 0.02 * scale)
    verdict = {
        'short_samples': rows[short]['samples'],
        'long_samples': rows[long_]['samples'],
        'frames_compared': width,
        'logit_standard_deviation': scale,
        'same_length_batch_noise_floor': floor,
        'masked_padded_difference': observed,
        'unmasked_padded_difference': leaked,
        'tolerance': allowance,
        'tolerance_rule': 'max(4 x same-length noise floor, 2% of logit sd)',
        'masking_within_tolerance': observed <= allowance,
        'withholding_mask_is_detected': leaked > max(10 * observed, allowance),
        'leak_to_signal_ratio': leaked / max(observed, 1e-9),
        'precision': 'float32, no autocast',
    }
    verdict['pass'] = bool(verdict['masking_within_tolerance']
                           and verdict['withholding_mask_is_detected'])
    assert verdict['pass'], verdict
    return verdict


def check_padding(model, head, rows):
    """A padded batch must not change what the model says about each row.

    Picks the extremes on purpose: the shortest and longest utterances in
    one batch is the worst case for mask handling. The publisher's
    preprocessor config sets `return_attention_mask: false`, which is only
    safe for a single unpadded input, so the mask this package builds
    itself is exactly what this check is testing.

    Run twice. Under BF16 autocast a batched and an unbatched forward use
    different kernels and reduction orders, so they differ a little even
    when the mask is perfect. The fp32 pass has no such excuse: if
    padding were leaking, the difference would survive there too.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i]['samples'])
    picks = [order[0], order[len(order) // 2], order[-1]]
    model.eval()
    head.eval()
    out = {'samples': [rows[i]['samples'] for i in picks],
           'padding_ratio': 1 - sum(rows[i]['samples'] for i in picks)
                            / (len(picks) * rows[picks[-1]]['samples'])}
    for precision in ['bfloat16', 'float32']:
        autocast = precision == 'bfloat16'
        with torch.inference_mode():
            batch = data.collate(rows, picks, 'cuda')
            grouped, lengths = _forward(model, head, batch, autocast)
            scale = float(grouped.float().std())
            differences, disagreement = [], []
            for b, i in enumerate(picks):
                one = data.collate(rows, [i], 'cuda')
                alone, length = _forward(model, head, one, autocast)
                assert int(length[0]) == int(lengths[b])
                width = int(length[0])
                a = grouped[b, :width].float()
                c = alone[0, :width].float()
                differences.append(float((a - c).abs().max()))
                disagreement.append(
                    float((a.argmax(-1) != c.argmax(-1)).float().mean()))
        out['frames'] = [int(x) for x in lengths]
        out[precision] = {
            'max_abs_logit_difference_vs_unbatched': differences,
            'logit_standard_deviation': scale,
            'relative_difference': [d / scale for d in differences],
            'argmax_disagreement_fraction': disagreement,
        }
        gpu_free()
    return out


def _forward(model, head, batch, autocast):
    if autocast:
        return train.forward(model, head, batch, grad=False, autocast=autocast)
    with torch.no_grad():
        hidden = model(input_values=batch['input_values'],
                       attention_mask=batch['attention_mask']).last_hidden_state
    lengths = model._get_feat_extract_output_lengths(
        batch['attention_mask'].sum(-1)).to(torch.long)
    return head(hidden), lengths


def check_decode(model, head, rows):
    """Greedy decoding over true lengths, on a deliberately mixed batch."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]['samples'])
    picks = [order[0], order[len(order) // 3], order[-1]]
    model.eval()
    head.eval()
    with torch.inference_mode():
        batch = data.collate(rows, picks, 'cuda')
        logits, lengths = train.forward(model, head, batch, grad=False,
                                        autocast=False)
        argmax = logits.argmax(-1).cpu().numpy()
    out = []
    for b, i in enumerate(picks):
        width = int(lengths[b])
        hyp = train.collapse(argmax[b, :width].tolist())
        padded = train.collapse(argmax[b].tolist())
        out.append({
            'wav': rows[i]['wav'], 'text': rows[i]['text'],
            'surface': rows[i]['surface'], 'frames': width,
            'reference_jamo': ''.join(labels.TOKENS[t] for t in rows[i]['label_ids']),
            'untrained_hypothesis': ''.join(labels.TOKENS[t] for t in hyp),
            'hypothesis_tokens': len(hyp),
            'padded_region_adds_tokens': len(padded) - len(hyp),
            'all_tokens_in_vocabulary': all(0 <= t < len(labels.TOKENS) for t in hyp),
        })
    return out


def check_step(model, head, rows, args):
    """Finite CTC loss on real audio, and weights that actually move."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]['samples'])
    picks = order[len(order) // 2: len(order) // 2 + 4]
    opt = PartialCPUAdamW(model, head, args.offload_layers, 2e-5, 1e-4)
    watched = {'head.linear.weight': head.linear.weight,
               'encoder.layers.0.final_layer_norm.weight':
                   dict(model.named_parameters())['encoder.layers.0.final_layer_norm.weight'],
               'encoder.layers.47.final_layer_norm.weight':
                   dict(model.named_parameters())['encoder.layers.47.final_layer_norm.weight']}
    before = {k: v.detach().clone() for k, v in watched.items()}
    frontend = next(model.feature_extractor.parameters()).detach().clone()
    model.train()
    head.train()
    opt.zero_grad()
    batch = data.collate(rows, picks, 'cuda')
    logits, lengths = train.forward(model, head, batch, grad=True, autocast=True)
    loss = train.ctc_sum(logits, lengths, batch) / len(batch['samples'])
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad]
        + list(head.parameters()), 1.0, error_if_nonfinite=True)
    opt.step()
    moved = {k: float((watched[k].detach() - before[k]).abs().max())
             for k in watched}
    result = {
        'loss': float(loss), 'loss_finite': bool(torch.isfinite(loss)),
        'grad_norm': float(norm),
        'random_head_loss_vs_uniform_reference': {
            'observed': float(loss),
            'note': 'a fresh head sits near -log(1/42) per frame; this is '
                    'initialisation, not a score'},
        'max_abs_parameter_change': moved,
        'offloaded_layer_updated': moved['encoder.layers.0.final_layer_norm.weight'] > 0,
        'resident_layer_updated': moved['encoder.layers.47.final_layer_norm.weight'] > 0,
        'head_updated': moved['head.linear.weight'] > 0,
        'frozen_frontend_unchanged': bool(torch.equal(
            frontend, next(model.feature_extractor.parameters()))),
        'offloaded_parameters': opt.offloaded_parameters,
        'resident_parameters': opt.resident_parameters,
    }
    assert result['loss_finite'] and result['head_updated']
    assert result['offloaded_layer_updated'] and result['resident_layer_updated']
    assert result['frozen_frontend_unchanged']
    del opt
    gpu_free()
    return result


def check_resume(model, head, rows, args, path):
    """Save, take one update, restore, retake it - the two must agree.

    Dropout and SpecAugment make this a real test: if the RNG state were
    not part of the checkpoint, the second attempt would diverge.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i]['samples'])
    picks = order[len(order) // 2: len(order) // 2 + 4]
    batch = data.collate(rows, picks, 'cuda')

    def one_update(opt):
        model.train()
        head.train()
        opt.zero_grad()
        logits, lengths = train.forward(model, head, batch, grad=True,
                                        autocast=True)
        loss = train.ctc_sum(logits, lengths, batch) / len(batch['samples'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad]
            + list(head.parameters()), 1.0)
        opt.step()
        return float(loss), head.linear.weight.detach().clone()

    opt = PartialCPUAdamW(model, head, args.offload_layers, 2e-5, 1e-4)
    one_update(opt)                                   # give AdamW real state
    state = {'phase': 'encoder', 'epoch': 0, 'batch_index': 7, 'update': 1}
    train.save(path, train.checkpoint_payload(model, head, opt, state, {}))
    first_loss, first_weight = one_update(opt)

    saved = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    head.load_state_dict(saved['head'])
    opt.load_state_dict(saved['optimizer'])
    train.set_rng_state(saved['rng'])
    second_loss, second_weight = one_update(opt)

    checkpoint_bytes = path.stat().st_size
    path.unlink()
    del opt
    gpu_free()
    return {
        'loss_before_checkpoint_restore': first_loss,
        'loss_after_checkpoint_restore': second_loss,
        'loss_identical': first_loss == second_loss,
        'max_abs_head_weight_difference': float(
            (first_weight - second_weight).abs().max()),
        'restored_state': saved['state'],
        'checkpoint_bytes': checkpoint_bytes,
        'covers': ['model', 'head', 'both AdamW moment sets', 'data cursor',
                   'python/numpy/torch/cuda RNG'],
    }


def capacity(model, head, rows, args):
    """Peak VRAM and throughput at the settings the long run might use."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]['samples'])
    longest = order[-64:]
    trials = []
    for offload in args.sweep_offload:
        for budget in args.sweep_budget:
            gpu_free()
            opt = PartialCPUAdamW(model, head, offload, 2e-5, 1e-4)
            # Worst case on purpose: a batch of the longest utterances,
            # filled to the budget exactly as `plan` would fill it.
            width = rows[longest[-1]]['samples']
            count = max(1, budget // width)
            picks = longest[-count:]
            record = {'offload_layers': offload, 'batch_budget': budget,
                      'recordings_in_batch': len(picks),
                      'longest_samples': width,
                      'batch_audio_seconds': sum(rows[i]['samples'] for i in picks) / 16000}
            try:
                batch = data.collate(rows, picks, 'cuda')
                times = []
                for it in range(args.sweep_iters):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model.train()
                    head.train()
                    opt.zero_grad()
                    logits, lengths = train.forward(model, head, batch,
                                                    grad=True, autocast=True)
                    loss = train.ctc_sum(logits, lengths, batch) / len(batch['samples'])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad]
                        + list(head.parameters()), 1.0)
                    step0 = time.perf_counter()
                    opt.step()
                    torch.cuda.synchronize()
                    now = time.perf_counter()
                    if it:
                        times.append((now - t0, now - step0))
                record['seconds_per_update'] = float(np.median([t for t, _ in times]))
                record['seconds_in_optimizer_step'] = float(np.median([s for _, s in times]))
                record['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
                record['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2 ** 30
                record['audio_seconds_per_wall_second'] = (
                    record['batch_audio_seconds'] / record['seconds_per_update'])
                record['ok'] = True
            except torch.OutOfMemoryError as exc:
                record['ok'] = False
                record['error'] = str(exc).split('\n')[0]
            trials.append(record)
            log('preflight.log', 'CAPACITY ' + json.dumps(
                {k: (round(v, 3) if isinstance(v, float) else v)
                 for k, v in record.items() if k != 'error'}))
            del opt
            gpu_free()
    return trials


def throughput(model, head, rows, args):
    """Time real accumulated updates on the batches `plan` actually builds.

    The capacity sweep pays for an optimizer step on every batch, which
    is not how the run works: with gradient accumulation one CPU-side
    step covers `--accumulate` batches. This measures the thing that
    decides the ETA, data loading included.
    """
    out = []
    for offload in args.sweep_offload:
        for budget in args.sweep_budget:
            gpu_free()
            batches = data.plan(rows, budget, args.max_batch, args.seed, 'bench')
            widths = [max(rows[i]['samples'] for i in b) for b in batches]
            padded = sum(w * len(b) for w, b in zip(widths, batches))
            real = sum(rows[i]['samples'] for b in batches for i in b)
            need = args.accumulate * (args.bench_updates + 1)
            start = len(batches) // 2
            window = batches[start:start + need]
            record = {'offload_layers': offload, 'batch_budget': budget,
                      'accumulate': args.accumulate,
                      'batches_per_epoch': len(batches),
                      'mean_recordings_per_batch': len(rows) / len(batches),
                      'padding_fraction_of_epoch': 1 - real / padded,
                      'updates_per_epoch': len(batches) // args.accumulate}
            opt = PartialCPUAdamW(model, head, offload, 2e-5, 1e-4)
            try:
                model.train()
                head.train()
                opt.zero_grad()
                times, audio, seen, t0 = [], 0.0, 0, None
                for n, (_i, batch) in enumerate(
                        data.Prefetch(rows, window, 'cuda'), 1):
                    logits, lengths = train.forward(
                        model, head, batch, grad=True, autocast=True)
                    loss = train.ctc_sum(logits, lengths, batch) / len(batch['samples'])
                    (loss / args.accumulate).backward()
                    audio += sum(batch['samples']) / 16000
                    if n % args.accumulate:
                        continue
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad]
                        + list(head.parameters()), 1.0)
                    opt.step()
                    opt.zero_grad()
                    torch.cuda.synchronize()
                    now = time.perf_counter()
                    if t0 is not None:                 # skip the warm-up update
                        times.append(now - t0)
                        seen += audio
                    t0, audio = now, 0.0
                record['seconds_per_update'] = float(np.median(times))
                record['audio_seconds_per_update'] = seen / len(times)
                rate = seen / sum(times)
                epoch = sum(r['seconds'] for r in rows) / rate
                record.update({
                    'audio_seconds_per_wall_second': rate,
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
                    'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30,
                    'projected_hours_per_epoch': epoch / 3600,
                    'projected_hours_8_epochs': 8 * epoch / 3600,
                    'ok': True})
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                record['ok'] = False
                record['error'] = str(exc).split('\n')[0][:200]
            out.append(record)
            log('preflight.log', 'THROUGHPUT ' + json.dumps(
                {k: (round(v, 3) if isinstance(v, float) else v)
                 for k, v in record.items()}))
            del opt
            gpu_free()
    return out


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(args.cpu_threads)
    report = load(REPORT) if REPORT.exists() and args.keep else {}
    report['started'] = time.strftime('%Y-%m-%d %H:%M:%S')
    report['gpu'] = torch.cuda.get_device_name(0)
    report['torch'] = torch.__version__
    report['total_vram_gib'] = torch.cuda.get_device_properties(0).total_memory / 2 ** 30

    if 'splits' in args.only:
        log('preflight.log', 'CHECK splits')
        report['splits'] = check_splits()
        report['source_archives'] = catalog.verify()
        dump(REPORT, report)

    rows = data.rows('train')
    if 'parity' in args.only:
        log('preflight.log', 'CHECK parity')
        report['input_parity'] = check_input_parity(rows)
        dump(REPORT, report)
    train.seed_everything(args.seed)
    model, head = train.build(args)
    report['model'] = {
        'attention_implementation': model.config._attn_implementation,
        'encoder_layers': model.config.num_hidden_layers,
        'hidden_size': model.config.hidden_size,
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
        'frozen_frontend_parameters': sum(
            p.numel() for p in model.feature_extractor.parameters()),
        'gradient_checkpointing': model.is_gradient_checkpointing,
        'mask_time_prob': model.config.mask_time_prob,
        'mask_feature_prob': model.config.mask_feature_prob,
        'layerdrop': model.config.layerdrop,
        'head_outputs': head.linear.out_features,
    }
    # Order matters: `padding` and `decode` need the pretrained weights
    # untouched, and `step` moves them.
    for name, fn in [('dtypes', check_dtypes),
                     ('masking', lambda m, h, r: check_masking(m, h, r, args)),
                     ('padding', check_padding), ('decode', check_decode),
                     ('step', lambda m, h, r: check_step(m, h, r, args)),
                     ('resume', lambda m, h, r: check_resume(
                         m, h, r, args, WORK / 'preflight-checkpoint.pt')),
                     ('capacity', lambda m, h, r: capacity(m, h, r, args)),
                     ('throughput', lambda m, h, r: throughput(m, h, r, args))]:
        if name not in args.only:
            continue
        log('preflight.log', f'CHECK {name}')
        report[name] = fn(model, head, rows)
        dump(REPORT, report)
        gpu_free()

    report['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    dump(REPORT, report)
    log('preflight.log', 'PREFLIGHT_COMPLETE')


if __name__ == '__main__':
    p = train.parser()
    p.add_argument('--sweep-offload', type=int, nargs='+', default=[40, 32, 26])
    p.add_argument('--sweep-budget', type=int, nargs='+', default=[480000, 640000])
    p.add_argument('--sweep-iters', type=int, default=4)
    p.add_argument('--bench-updates', type=int, default=3)
    p.add_argument('--only', nargs='+',
                   default=['splits', 'parity', 'dtypes', 'masking', 'padding',
                            'decode', 'step', 'resume', 'capacity',
                            'throughput'])
    p.add_argument('--keep', action='store_true')
    main(p.parse_args())
