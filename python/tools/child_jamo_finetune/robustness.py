"""Isolated, resumable comparison of extra training with/without acoustic variation.

Both arms start from the same trained checkpoint, see the same ordered
training examples, and use identical optimizer settings. No personal clip
is used for optimization. This bounded experiment trains the last eight
transformer layers and the head; it does not overwrite or deploy the parent.
"""
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
import argparse
import gc
import hashlib
import io
import json
import math
import sqlite3
import sys
import time

import numpy as np
import soundfile as sf
import torch
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio
import augmentation
import data
import integrity
import labels
import reviewed
import train
from common import ROOT, WORK, digest, dump, load
from ownership import Owned

DIAGNOSTICS = ROOT / '.venv/child-jamo-web-20260916/cause-analysis-20260927'
PARENT = WORK / 'main/best.pt'
PARENT_SHA = '5a330f1bea3a17d1d063fdfd71a0cc25394695b0b797de01b293f6f688257aa7'
TRAIN_SPLIT = 'train-repaired-20260927'
VERSION = 'child-jamo-robustness-pilot-2026-09-27'


def rank(row, salt):
    return hashlib.sha256((salt + ':' + row['wav']).encode()).hexdigest()


def select_training(rows, count, seed, excluded):
    eligible = [r for r in rows if r['wav'] not in excluded]
    words = defaultdict(list)
    for row in eligible:
        if len(row['surface'].split()) == 1:
            words[row['surface']].append(row)
    # Round-robin among transcript types prevents a common word from
    # consuming the single-word allocation. No diagnosis word is privileged.
    buckets = {word: sorted(group, key=lambda r: rank(r, f'{seed}:word'))
               for word, group in words.items()}
    order = sorted(buckets, key=lambda word: hashlib.sha256(f'{seed}:{word}'.encode()).hexdigest())
    picked, used = [], set()
    depth = 0
    while len(picked) < count // 2:
        candidates = [buckets[word][depth] for word in order if len(buckets[word]) > depth]
        if not candidates:
            break
        for row in candidates:
            if len(picked) >= count // 2:
                break
            picked.append(row)
            used.add(row['wav'])
        depth += 1
    word_count = len(picked)
    for row in sorted(eligible, key=lambda r: rank(r, f'{seed}:general')):
        if len(picked) >= count:
            break
        if row['wav'] not in used:
            picked.append(row)
            used.add(row['wav'])
    assert len(picked) == count
    return picked, word_count


def verify_read(row):
    blob = audio.payload(row['archive'], row['offset'], row['size'])
    if hashlib.sha256(blob).hexdigest() != row['source_sha256']:
        raise RuntimeError('Source changed: ' + row['wav'])
    wave, policy = audio.crop(audio.decode(blob), row['speech_start'], row['speech_end'])
    if (policy != row['span_policy'] or len(wave) != row['samples']
            or hashlib.sha256(wave.tobytes()).hexdigest() != row['pcm_sha256']):
        raise RuntimeError('Audited crop changed: ' + row['wav'])
    return wave


def pack(waves, rows, device='cuda'):
    values = np.zeros((len(waves), max(map(len, waves))), dtype=np.float32)
    mask = np.zeros(values.shape, dtype=np.int64)
    for i, wave in enumerate(waves):
        values[i, :len(wave)] = wave
        mask[i, :len(wave)] = 1
    return {'input_values': torch.from_numpy(values).to(device),
            'attention_mask': torch.from_numpy(mask).to(device),
            'targets': torch.tensor([t for row in rows for t in row['label_ids']], dtype=torch.long),
            'target_lengths': torch.tensor([len(row['label_ids']) for row in rows], dtype=torch.long),
            'samples': list(map(len, waves))}


def compact(ids):
    return [i for i in ids if i != labels.BOUNDARY]


@torch.inference_mode()
def prediction(model, head, wave):
    batch = pack([wave], [{'label_ids': []}])
    logits, lengths = train.forward(model, head, batch, grad=False, autocast=False)
    assert torch.isfinite(logits).all()
    return train.collapse(logits[0, :int(lengths[0])].argmax(-1).cpu().tolist())


def describe_ids(ids):
    # Formatting is isolated from prediction and receives no reference.
    sys.path.insert(0, str(ROOT / 'python/tools/child_jamo_web'))
    import compose
    return {'ids': ids, 'text': compose.compose(ids)['text'],
            'jamo': ''.join(labels.TOKENS[i] for i in ids)}


def check_review():
    report = load(DIAGNOSTICS / 'context-review-results.json')
    if not report['complete'] or len(report['findings']) != 6:
        raise RuntimeError('The pronunciation review is incomplete')
    folder = DIAGNOSTICS / 'context-listening'
    with closing(sqlite3.connect((folder / 'listening-answers.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        latest = {r['item_id']: dict(r) for r in db.execute('SELECT * FROM answers ORDER BY revision')}
    for item in report['findings']:
        if latest[str(item['number'])] != item['answer']:
            raise RuntimeError('Listening answer changed; reclassify before training')
        if not item['pronunciation_preserved_by_human']:
            raise RuntimeError('This policy requires the six completed preservation checks')
        for clip in item['identity']['clips']:
            if digest(folder / 'listen-audio' / clip['audio']) != clip['audio_sha256']:
                raise RuntimeError('Listening audio changed')
    return digest(DIAGNOSTICS / 'context-review-results.json')


def prepare(out, args):
    manifest = out / 'selection.json'
    current = integrity.contract({'train': TRAIN_SPLIT, 'valid': 'valid'})
    review_sha = check_review()
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError('Parent checkpoint changed')
    if manifest.exists():
        selection = load(manifest)
        assert selection['data_contract'] == current
        assert selection['review_sha256'] == review_sha
        assert selection['seed'] == args.seed and selection['count'] == args.count
        return selection
    rows = data.rows(TRAIN_SPLIT)
    reviewed.require_repaired(rows)
    diagnostic = load(DIAGNOSTICS / 'context-generalization.json')
    excluded = {e['row']['wav'] for e in diagnostic['entries'] if e['cohort'] == 'human_verified_training'}
    chosen, word_count = select_training(rows, args.count, args.seed, excluded)
    valid = data.rows('valid')
    test = data.rows('test')  # Identity/separation only; never scored here.
    train_speakers = {r['speaker'] for r in chosen}
    holdout_speakers = {r['speaker'] for r in valid + test}
    assert not (train_speakers & holdout_speakers)
    assert not ({r['pcm_sha256'] for r in chosen} & {r['pcm_sha256'] for r in valid + test})
    assert len({r['wav'] for r in chosen}) == len(chosen)
    selection = {'version': VERSION, 'seed': args.seed, 'count': args.count,
                 'parent_checkpoint_sha256': PARENT_SHA, 'data_contract': current,
                 'review_sha256': review_sha, 'excluded_human_training_examples': sorted(excluded),
                 'balanced_single_word_allocation': word_count,
                 'training_speakers': len(train_speakers), 'training_hours': sum(r['seconds'] for r in chosen) / 3600,
                 'train_rows': chosen,
                 'stress_valid_wavs': [r['wav'] for r in sorted(valid, key=lambda r: rank(r, f'{args.seed}:stress'))[:96]],
                 'selection_uses_model_predictions': False,
                 'personal_examples_in_training': 0, 'test_scored': False}
    dump(manifest, selection)
    return selection


def fixed_config(args, selection):
    import librosa
    import scipy
    import transformers
    return {'version': VERSION, 'seed': args.seed, 'arm': args.arm,
            'updates': args.updates, 'last_layers': args.last_layers,
            'encoder_lr': args.encoder_lr, 'head_lr': args.head_lr,
            'batch_budget': args.batch_budget, 'max_batch': args.max_batch,
            'accumulate': args.accumulate, 'warmup_updates': 8,
            'weight_decay': .01, 'grad_clip': 1.0,
            'initialization_sha256': PARENT_SHA,
            'selection_sha256': digest(WORK / args.name / 'selection.json'),
            'data_contract': selection['data_contract'],
            'source_code_sha256': {name: digest(Path(__file__).parent / name)
                                   for name in ('robustness.py', 'augmentation.py', 'train.py', 'audio.py', 'labels.py')},
            'libraries': {'torch': torch.__version__, 'numpy': np.__version__,
                          'librosa': librosa.__version__, 'scipy': scipy.__version__,
                          'transformers': transformers.__version__},
            'augmentation': augmentation.VERSION if args.arm == 'augmented' else None,
            'precision': 'BF16 training encoder; FP32 head/loss and complete validation',
            'training_scope': 'last transformer layers and CTC head; frozen rest of encoder',
            'validation_rule': 'fixed final update, no checkpoint chosen from personal diagnostics',
            'test_scored': False, 'deployed': False}


def evaluate_all(model, head, selection, out, name, args):
    model.eval()
    head.eval()
    started = time.monotonic()
    valid = data.rows('valid')
    clean = train.evaluate(model, head, valid, data.plan(valid, 640000, 16, args.seed, 'robustness-valid'))
    result = {'name': name, 'clean_valid': clean, 'stress_valid': {},
              'personal_diagnostics': [], 'listening_pairs': [], 'test_scored': False}
    stress = [r for r in valid if r['wav'] in set(selection['stress_valid_wavs'])]
    for mode in ('identity', 'rate_down_2', 'rate_up_2', 'pitch_down_2', 'pitch_up_2', 'tempo_0.65'):
        errors = reference = exact = 0
        for row in stress:
            wave, _meta = augmentation.apply(verify_read(row), mode, target_ids=row['label_ids'])
            ids = prediction(model, head, wave)
            target = compact(row['label_ids'])
            err = Levenshtein.distance(target, compact(ids))
            errors += err
            reference += len(target)
            exact += err == 0
        result['stress_valid'][mode] = {'recordings': len(stress), 'errors': errors,
                                       'reference_jamo': reference, 'jer': errors / reference, 'exact': exact}
    sys.path.insert(0, str(ROOT / 'python/tools/child_jamo_web'))
    import engine
    for item in load(DIAGNOSTICS / 'context-generalization.json')['entries']:
        if item['cohort'] != 'personal_diagnostic':
            continue
        wave = engine.read_row(item['row'])
        assert engine.pcm_sha256(wave) == item['variants']['baseline']['pcm_sha256']
        ids = prediction(model, head, wave)
        target = [labels.VOCAB[t] for t in labels.jamo(item['human_target'])] if item['human_target'] else None
        result['personal_diagnostics'].append({'id': item['id'], 'human_target': item['human_target'],
                                               **describe_ids(ids),
                                               'human_exact': compact(ids) == compact(target) if target else None})
    for item in load(DIAGNOSTICS / 'context-review-results.json')['findings']:
        pair = {'number': item['number'], 'reference': item['identity']['reference'], 'variants': {}}
        target = [labels.VOCAB[t] for t in labels.jamo(pair['reference'])]
        for clip in item['identity']['clips']:
            payload = (DIAGNOSTICS / 'context-listening/listen-audio' / clip['audio']).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == clip['audio_sha256']
            wave, sr = sf.read(io.BytesIO(payload), dtype='float32')
            assert sr == 16000
            ids = prediction(model, head, wave)
            pair['variants'][clip['key']] = {**describe_ids(ids), 'human_exact': compact(ids) == compact(target)}
        result['listening_pairs'].append(pair)
    result['elapsed_seconds'] = round(time.monotonic() - started, 1)
    result['complete'] = True
    dump(out / f'{name}-evaluation.json', result)
    print('EVALUATION', name, 'clean_jer', clean['jer'], 'personal',
          sum(r['human_exact'] is True for r in result['personal_diagnostics']),
          '/', sum(r['human_exact'] is not None for r in result['personal_diagnostics']), flush=True)
    return result


def main(args):
    if not 1 <= args.last_layers <= 48 or args.updates < 1:
        raise ValueError('invalid training budget')
    out = WORK / args.name
    with Owned(out, WORK):
        selection = prepare(out, args)
        config = fixed_config(args, selection)
        run = out / args.arm
        run.mkdir(parents=True, exist_ok=True)
        if (run / 'result.json').exists():
            if load(run / 'config.json') != config:
                raise RuntimeError('Completed run has different configuration')
            print('ALREADY_COMPLETE', run, flush=True)
            return
        if (run / 'config.json').exists():
            if load(run / 'config.json') != config:
                raise RuntimeError('Refuse resume after config/code/provenance drift')
        else:
            dump(run / 'config.json', config)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        train.seed_everything(args.seed)
        build_args = argparse.Namespace(seed=args.seed, attention='sdpa', mask_time_prob=.05, mask_feature_prob=.02)
        model, head = train.build(build_args)
        parent = torch.load(PARENT, map_location='cpu', weights_only=False)
        model.load_state_dict(parent['model'], strict=True)
        head.load_state_dict(parent['head'], strict=True)
        del parent
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for layer in model.encoder.layers[-args.last_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        parameters = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
        optimizer = torch.optim.AdamW([{'params': [p for p in model.parameters() if p.requires_grad], 'lr': args.encoder_lr},
                                       {'params': head.parameters(), 'lr': args.head_lr}], weight_decay=.01, foreach=False)
        state = {'update': 0, 'epoch': 0, 'next_group': 0, 'augmentation_counts': {}, 'training_seconds': 0.0}
        if (run / 'last.pt').exists():
            saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
            if saved['config'] != config:
                raise RuntimeError('Resume checkpoint configuration differs')
            model.load_state_dict(saved['model'], strict=True)
            head.load_state_dict(saved['head'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            state = saved['state']
            train.set_rng_state(saved['rng'])
            del saved
            print('RESUME', args.arm, state['update'], flush=True)
        else:
            baseline_path = out / 'baseline-evaluation.json'
            if not baseline_path.exists():
                evaluate_all(model, head, selection, out, 'baseline', args)
            train.seed_everything(args.seed)
        rows = selection['train_rows']
        # The same conservative batch plan is used by BOTH arms; the augmented
        # arm cannot accidentally receive a smaller number of training rows.
        planning = [dict(r, samples=math.ceil(r['samples'] * augmentation.MAX_LENGTH_FACTOR) + 2) for r in rows]
        cache = {}
        counts = Counter(state['augmentation_counts'])
        clock = time.monotonic()
        elapsed_before = state['training_seconds']
        losses = []
        def persist(status):
            state['training_seconds'] = elapsed_before + time.monotonic() - clock
            state['augmentation_counts'] = dict(counts)
            state['status'] = status
            train.save(run / 'last.pt', {'model': model.state_dict(), 'head': head.state_dict(),
                       'optimizer': optimizer.state_dict(), 'state': state, 'rng': train.rng_state(), 'config': config})
            dump(run / 'progress.json', state)
        try:
            while state['update'] < args.updates:
                batches = data.plan(planning, args.batch_budget, args.max_batch, args.seed, f'paired:{state["epoch"]}')
                groups = data.groups(batches, args.accumulate)
                model.train()
                head.train()
                for group_index in range(state['next_group'], len(groups)):
                    group = groups[group_index]
                    denominator = sum(map(len, group))
                    optimizer.zero_grad(set_to_none=True)
                    for indices in group:
                        waves, batch_rows = [], [rows[i] for i in indices]
                        for row in batch_rows:
                            if row['wav'] not in cache:
                                cache[row['wav']] = verify_read(row)
                            mode = (augmentation.mode_for(row['wav'], args.seed, state['epoch'])
                                    if args.arm == 'augmented' else 'identity')
                            wave, meta = augmentation.apply(cache[row['wav']], mode, target_ids=row['label_ids'])
                            counts[meta['applied']] += 1
                            waves.append(wave)
                        batch = pack(waves, batch_rows)
                        logits, lengths = train.forward(model, head, batch, grad=True, autocast=True)
                        loss = train.ctc_sum(logits, lengths, batch)
                        if not torch.isfinite(loss):
                            raise RuntimeError('non-finite training loss')
                        (loss / denominator).backward()
                        losses.append(float(loss.detach()) / len(batch_rows))
                    update = state['update'] + 1
                    factor = min(1.0, update / 8) * (.1 + .9 * (1 + math.cos(math.pi * update / args.updates)) / 2)
                    optimizer.param_groups[0]['lr'] = args.encoder_lr * factor
                    optimizer.param_groups[1]['lr'] = args.head_lr * factor
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    state.update(update=update, next_group=group_index + 1)
                    if update % 8 == 0 or update == 1:
                        state.update(training_seconds=elapsed_before + time.monotonic() - clock,
                                     augmentation_counts=dict(counts), mean_loss=float(np.mean(losses)),
                                     grad_norm=float(norm), status='training',
                                     cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                        dump(run / 'progress.json', state)
                        print('TRAIN', args.arm, 'update', update, '/', args.updates,
                              'loss', round(state['mean_loss'], 4), 'seconds', round(state['training_seconds'], 1), flush=True)
                        losses = []
                    if update % 32 == 0:
                        persist('training')
                    if update >= args.updates:
                        break
                if state['next_group'] >= len(groups):
                    state.update(epoch=state['epoch'] + 1, next_group=0)
            persist('evaluating')
        except BaseException:
            # Persist only completed optimizer updates, never partial gradients.
            # An exception inside a group requires reloading the last periodic
            # checkpoint; do not label partially modified state as resumable.
            dump(run / 'failure.json', {'update': state['update'], 'resume_from': 'last.pt if present',
                                       'at': time.strftime('%Y-%m-%d %H:%M:%S')})
            raise
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        score = evaluate_all(model, head, selection, run, 'final', args)
        train.save(run / 'model.pt', {'model': model.state_dict(), 'head': head.state_dict(),
                                     'config': config, 'valid': score['clean_valid']})
        state['status'] = 'completed'
        dump(run / 'progress.json', state)
        dump(run / 'result.json', {'complete': True, 'config': config, 'state': state,
                                   'evaluation': score, 'checkpoint_sha256': digest(run / 'model.pt'),
                                   'deployed': False})
        print('COMPLETE', args.arm, run, flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='robustness-pilot-20260927')
    p.add_argument('--arm', choices=['control', 'augmented'], required=True)
    p.add_argument('--seed', type=int, default=927)
    p.add_argument('--count', type=int, default=2048)
    p.add_argument('--updates', type=int, default=128)
    p.add_argument('--last-layers', type=int, default=8)
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=5e-5)
    p.add_argument('--batch-budget', type=int, default=320000)
    p.add_argument('--max-batch', type=int, default=8)
    p.add_argument('--accumulate', type=int, default=2)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
