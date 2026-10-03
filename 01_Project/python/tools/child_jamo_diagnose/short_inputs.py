"""Offline cause isolation for short child utterances, without training.

Reads the existing checkpoint, original clips and historical audit. Writes
only beneath --output. It does not start the app, touch review answers, alter
audio, or change inference defaults. The 83 clips are development examples;
transcript agreement here is not a measure of actual pronunciation accuracy.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / 'python/tools/child_jamo_web'
TRAIN = ROOT / 'python/tools/child_jamo_finetune'
sys.path[:0] = [str(WEB), str(TRAIN)]

import numpy as np
import torch
import transformers

import compose
import engine as engine_module
import labels
import personal
from common import digest
from ctc import log_softmax, prefix_beam, self_check, sequence_logp

SR = 16000
FOCUS = ['personal:child1_09', 'personal:child1_17',
         'personal:child1_18', 'personal:child1_22',
         'personal:child2_09', 'personal:child2_16']
HISTORICAL = ROOT / '.venv/child-jamo-web-20260916/consonant-review/baseline.jsonl'


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def announce(stage, **values):
    print(json.dumps({'stage': stage, **values}, ensure_ascii=False), flush=True)


def environment():
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'numpy': np.__version__,
            'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'at_utc': datetime.now(timezone.utc).isoformat()}


@contextmanager
def capture_stages(model):
    captured, handles = {}, []

    def save(name, convolution=False):
        def hook(module, args, output):
            x = output[0] if isinstance(output, tuple) else output
            if convolution:
                x = x.transpose(1, 2)
            captured[name] = x.detach().float().cpu().clone()
        return hook

    for name, module, convolution in [
        ('conv', model.feature_extractor, True),
        ('projected', model.feature_projection, False),
        ('position', model.encoder.pos_conv_embed, False),
        *[(f'layer_{i + 1}', model.encoder.layers[i], False)
          for i in (0, 3, 11, 23, 47)],
    ]:
        handles.append(module.register_forward_hook(save(name, convolution)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def forward(engine, waveform, *, true_samples=None, capture=False):
    values = torch.tensor(np.asarray(waveform)[None, :], dtype=torch.float32,
                          device=engine.device)
    mask = torch.ones(values.shape, dtype=torch.long, device=engine.device)
    if true_samples is not None:
        mask[:, true_samples:] = 0
    length = int(engine.model._get_feat_extract_output_lengths(mask.sum(-1))[0])

    def compute():
        hidden = engine.model(values, attention_mask=mask).last_hidden_state
        return engine.head(hidden.float())[0, :length].cpu().numpy()

    if capture:
        with capture_stages(engine.model) as stages:
            logits = compute()
    else:
        logits, stages = compute(), {}
    assert np.isfinite(logits).all()
    return logits, stages


def describe(logits, row, beam=False):
    lp = log_softmax(logits)
    ids = compose.collapse(logits.argmax(-1).tolist())
    rendered = compose.compose(ids)
    reference = row['label_ids']
    result = {'text': rendered['text'], 'ids': ids, 'jamo': rendered['jamo'],
              'frames': len(logits), 'metrics': compose.compare(reference, ids),
              'greedy_logp': sequence_logp(lp, ids, labels.BLANK),
              'reference_logp': sequence_logp(lp, reference, labels.BLANK)}
    result['reference_to_greedy_ratio'] = float(np.exp(np.clip(
        result['reference_logp'] - result['greedy_logp'], -700, 700)))
    if beam:
        candidates = prefix_beam(lp, blank=labels.BLANK)
        result['beam'] = [
            {'ids': seq, 'text': compose.compose(seq)['text'],
             'search_logp_lower_bound': score,
             'exact_logp': sequence_logp(lp, seq, labels.BLANK),
             'metrics': compose.compare(reference, seq)}
            for seq, score in candidates[:5]]
    if row['id'] in FOCUS:
        probability = np.exp(lp)
        evidence = {}
        for token in ('ㅊ', 'ㅅ', 'ㄱ', 'ㅂ', 'ㄹ', 'ㄸ'):
            index = labels.VOCAB[token]
            peak = int(probability[:, index].argmax())
            top = np.argsort(-probability[peak])[:4]
            evidence[token] = {
                'peak_frame': peak, 'peak_probability': float(probability[peak, index]),
                'wins_any_frame': bool((logits.argmax(-1) == index).any()),
                'competitors': [{'token': labels.TOKENS[i],
                                 'probability': float(probability[peak, i])} for i in top]}
        result['token_evidence'] = evidence
    return result


def difference(a, b):
    delta = (a.double() - b.double()).numpy()
    scale = float(np.sqrt(np.mean(np.square(a.double().numpy()))))
    return {'max_abs': float(np.max(np.abs(delta))),
            'rms': float(np.sqrt(np.mean(delta ** 2))),
            'relative_rms': float(np.sqrt(np.mean(delta ** 2)) / max(scale, 1e-15))}


@torch.inference_mode()
def encoder_interventions(engine, row, baseline, padded, offset, frames):
    """Cross position-convolution context with attention context.

    Both interventions are diagnostics only. They feed internal features;
    neither is presented as a valid serving policy or an acoustic label.
    """
    model = engine.model
    projected = padded['projected'].to(engine.device)
    position = padded['position'].to(engine.device)
    span = slice(offset, offset + frames)
    original_position = baseline['position'].to(engine.device)
    all_mask = torch.ones(projected.shape[:2], dtype=torch.bool, device=engine.device)
    real_mask = torch.zeros_like(all_mask)
    real_mask[:, span] = True
    results = {}
    for name in ('all_context', 'masked_context_control',
                 'position_context_only', 'attention_context_only'):
        mask = all_mask if name in ('all_context', 'attention_context_only') else real_mask
        override = None
        if name == 'position_context_only':
            override = position.clone()
        elif name == 'attention_context_only':
            override = position.clone()
            override[:, span] = original_position
        handle = None
        if override is not None:
            handle = model.encoder.pos_conv_embed.register_forward_hook(
                lambda module, args, output: override)
        try:
            hidden = model.encoder(projected.clone(), attention_mask=mask).last_hidden_state
            logits = engine.head(hidden.float())[0, span].cpu().numpy()
            results[name] = describe(logits, row)
        finally:
            if handle is not None:
                handle.remove()
    return results


def aggregate(entries, variant):
    errors = reference = exact = better = worse = same = broken = repaired = 0
    changes = []
    for item in entries:
        prediction = item['variants'][variant]
        before = item['variants']['baseline']['metrics']
        after = prediction['metrics']
        errors += after['errors']
        reference += after['reference_jamo']
        exact += int(after['exact'])
        better += int(after['errors'] < before['errors'])
        worse += int(after['errors'] > before['errors'])
        same += int(after['errors'] == before['errors'])
        broken += int(before['exact'] and not after['exact'])
        repaired += int(not before['exact'] and after['exact'])
        if prediction['ids'] != item['variants']['baseline']['ids']:
            changes.append({'id': item['id'], 'reference': item['surface'],
                            'before': item['variants']['baseline']['text'],
                            'after': prediction['text'],
                            'before_errors': before['errors'], 'after_errors': after['errors']})
    return {'recordings': len(entries), 'errors': errors, 'reference_jamo': reference,
            'jer': errors / reference, 'exact': exact, 'improved': better,
            'worsened': worse, 'same_error_count': same,
            'correct_to_incorrect': broken, 'incorrect_to_correct': repaired,
            'changed': changes}


def run(output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    checks = self_check()
    write(output / 'preflight.json', {'environment': environment(), 'ctc_checks': checks})
    announce('loading_checkpoint')
    engine = engine_module.Engine()
    engine.load()
    write(output / 'identity.json', {'environment': environment(),
                                    'engine': engine.identity, 'ctc_checks': checks,
                                    'source_sha256': {str(p.relative_to(ROOT)): digest(p)
                                      for p in (Path(__file__), Path(__file__).with_name('ctc.py'),
                                                WEB / 'engine.py', WEB / 'compose.py',
                                                TRAIN / 'labels.py', HISTORICAL)}})
    old = {r['id']: r for r in map(json.loads, HISTORICAL.read_text(encoding='utf-8').splitlines())}
    rows = personal.clip_rows()  # validates original WAV file hashes; writes nothing
    assert len(rows) == 83
    records, artifacts = [], {}
    focus_stages, focus_logits = {}, {}
    variants = {'baseline': (0, 0), 'both_200ms': (3200, 3200),
                'both_500ms': (8000, 8000), 'left_500ms': (8000, 0),
                'right_500ms': (0, 8000)}
    for number, row in enumerate(rows, 1):
        historical = old[row['id']]
        assert row['pcm_sha256'] == historical['audio_pcm_sha256'], row['id']
        assert row['label_ids'] == historical['reference_ids'], row['id']
        assert engine.identity['checkpoint_sha256'] == historical['checkpoint_sha256']
        wave = engine_module.read_row(row)
        item = {'id': row['id'], 'transcript': row['text'], 'surface': row['surface'],
                'pcm_sha256': row['pcm_sha256'], 'samples': len(wave), 'variants': {}}
        focus = row['id'] in FOCUS
        for name, (left, right) in variants.items():
            x = np.pad(wave, (left, right))
            assert np.array_equal(x[left:left + len(wave)], wave)
            logits, stages = forward(engine, x, capture=focus and name in ('baseline', 'both_500ms'))
            prediction = describe(logits, row, beam=name == 'baseline')
            item['variants'][name] = prediction
            if name == 'baseline':
                assert prediction['ids'] == historical['hypothesis_ids'], (
                    'baseline mismatch', row['id'], prediction['text'], historical['predicted_text'])
                served = engine.transcribe(wave, use_cache=False) if focus else None
                if served:
                    assert served['token_ids'] == prediction['ids'], ('app parity', row['id'])
                item['variants']['beam25'] = prediction['beam'][0]
                artifacts[row['id'].replace(':', '_')] = logits
            if focus:
                focus_logits[row['id'], name] = logits
                if stages:
                    focus_stages[row['id'], name] = stages
        records.append(item)
        if number % 10 == 0 or focus or number == len(rows):
            announce('personal_controls', done=number, total=len(rows),
                     id=row['id'], baseline=item['variants']['baseline']['text'],
                     padded=item['variants']['both_500ms']['text'])
            write(output / 'progress.json', {'stage': 'personal_controls', 'done': number,
                                             'elapsed_seconds': time.monotonic() - started})
    write(output / 'personal-results.json', records)
    summary = {name: aggregate(records, name) for name in (*variants, 'beam25')}
    write(output / 'personal-summary.json', summary)
    np.savez_compressed(output / 'baseline-logits.npz', **artifacts)
    announce('baseline_reproduced', recordings=len(records), baseline=summary['baseline']['jer'],
             beam=summary['beam25']['jer'], padding=summary['both_500ms']['jer'])

    focused = []
    row_map = {r['id']: r for r in rows}
    for sample_id in FOCUS:
        row = row_map[sample_id]
        wave = engine_module.read_row(row)
        base_logits = focus_logits[sample_id, 'baseline']
        base_stages = focus_stages[sample_id, 'baseline']
        pad_stages = focus_stages[sample_id, 'both_500ms']
        frames = len(base_logits)
        item = {'id': sample_id, 'surface': row['surface'], 'sweep': {},
                'stage_difference_aligned_500ms': {
                    name: difference(value, pad_stages[name][:, 25:25 + frames])
                    for name, value in base_stages.items()}}
        repeated, _ = forward(engine, wave)
        item['repeat_max_abs_logit_difference'] = float(np.max(np.abs(repeated - base_logits)))
        assert np.array_equal(repeated.argmax(-1), base_logits.argmax(-1))
        masked, _ = forward(engine, np.pad(wave, (0, 16000)), true_samples=len(wave))
        item['masked_batch_padding_control'] = describe(masked, row)
        item['masked_batch_padding_max_abs_logit_difference'] = float(np.max(np.abs(masked - base_logits)))
        for milliseconds in (40, 100, 200, 500, 1000):
            samples = milliseconds * 16
            logits, _ = forward(engine, np.pad(wave, (samples, samples)))
            item['sweep'][f'both_{milliseconds}ms'] = describe(logits, row)
            artifacts[sample_id.replace(':', '_') + f'_both_{milliseconds}ms'] = logits
        # Constant total padding (20 ms), varied placement probes convolution grid phase.
        for left in (0, 80, 160, 240, 320):
            logits, _ = forward(engine, np.pad(wave, (left, 320 - left)))
            item['sweep'][f'phase_left_{left}_total_320'] = describe(logits, row)
        item['encoder_interventions'] = encoder_interventions(
            engine, row, base_stages, pad_stages, 25, frames)
        # Same CPU/GPU weights, attention computed without the fused SDPA kernel.
        engine.model.set_attn_implementation('eager')
        try:
            eager, _ = forward(engine, wave)
        finally:
            engine.model.set_attn_implementation('sdpa')
        item['eager_attention_control'] = describe(eager, row)
        item['eager_max_abs_logit_difference'] = float(np.max(np.abs(eager - base_logits)))
        lp = log_softmax(base_logits)
        item['beam_width_check'] = {
            str(width): [{'text': compose.compose(seq)['text'], 'ids': seq,
                          'exact_logp': sequence_logp(lp, seq, labels.BLANK)}
                         for seq, _ in prefix_beam(lp, width=width, blank=labels.BLANK)[:3]]
            for width in (5, 25, 100)}
        focused.append(item)
        write(output / 'focused-results.json', focused)
        announce('focused_done', id=sample_id,
                 intervention_text={k: v['text'] for k, v in item['encoder_interventions'].items()},
                 conv_difference=item['stage_difference_aligned_500ms']['conv']['max_abs'])
    np.savez_compressed(output / 'diagnostic-logits.npz', **artifacts)
    write(output / 'progress.json', {'status': 'completed', 'stage': 'cause_isolation',
                                     'personal_recordings': len(records),
                                     'focused_recordings': len(focused),
                                     'elapsed_seconds': time.monotonic() - started})
    announce('completed', seconds=round(time.monotonic() - started, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=ROOT / '.venv/child-jamo-cause-20260925')
    args = parser.parse_args()
    # Diagnostics must stay within the workspace; this program never writes elsewhere.
    if not args.output.resolve().is_relative_to(ROOT.resolve()):
        parser.error('--output must be inside the repository')
    if args.preflight:
        result = {'environment': environment(), 'ctc_checks': self_check(),
                  'personal_audio_available': personal.available(),
                  'checkpoint_exists': engine_module.CHECKPOINT.exists()}
        write(args.output / 'preflight.json', result)
        announce('preflight', **result)
    else:
        try:
            run(args.output)
        except Exception as exc:
            write(args.output / 'failure.json', {'error': type(exc).__name__, 'message': str(exc)})
            raise


if __name__ == '__main__':
    main()
