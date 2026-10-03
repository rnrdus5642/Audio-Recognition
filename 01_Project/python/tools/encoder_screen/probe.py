"""Frozen-encoder Korean CTC probe and measured CUDA forward screening.

Same data, jamo targets, linear CTC head, seeds, and optimizer search for every
encoder. These probe scores do not claim to predict fully fine-tuned accuracy.
"""
from pathlib import Path
from itertools import groupby
import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoFeatureExtractor
from common import ROOT, WORK, MODELS, dump, load, digest


def encoder(key, device='cuda'):
    directory = WORK / 'models' / key
    verified = load(directory / 'verified.json')
    if MODELS[key]['family'] == 'omni':
        from convert_omni import converted_encoder
        model, feature, info = converted_encoder(key)
    elif MODELS[key]['family'] == 'xeus':
        from xeus import xeus_encoder
        model, feature, info = xeus_encoder()
    else:
        feature = AutoFeatureExtractor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        model, info = AutoModel.from_pretrained(directory, local_files_only=True,
                trust_remote_code=False, output_loading_info=True)
        assert not info.get('missing_keys'), info
        assert not info.get('mismatched_keys'), info
        assert not info.get('error_msgs'), info
        # Pretraining quantizer / prediction-only parameters are not encoder inputs.
    info={k:sorted(v) if isinstance(v,set) else v for k,v in info.items()}
    model.eval().to(device)
    return model, feature, info, verified


def input_values(feature, audio, device):
    values = feature(audio, sampling_rate=16000, return_tensors='pt')
    return {k: v.to(device) for k, v in values.items() if k in {'input_values', 'input_features', 'attention_mask'}}


def extract(key):
    dest = WORK / 'features' / key
    dest.mkdir(parents=True, exist_ok=True)
    manifest = load(WORK / 'manifest.json')
    manifest_sha = digest(WORK / 'manifest.json')
    if (dest / 'metadata.json').exists():
        rec = load(dest / 'metadata.json')
        assert rec['manifest_sha256'] == manifest_sha
        print('FEATURES_REUSED', key, flush=True)
        return rec
    model, feature, info, verified = encoder(key)
    parameters = sum(p.numel() for p in model.parameters())
    dummy = np.random.default_rng(917).normal(0, .1, 40000).astype(np.float32)
    values = input_values(feature, dummy, 'cuda')
    times = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(24):
            torch.cuda.synchronize()
            t = time.perf_counter()
            y = model(**values).last_hidden_state
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t) * 1000
            if i >= 4: times.append(dt)
        assert torch.isfinite(y).all()
    rec = {'key': key, 'download': verified, 'loading_info': info,
           'manifest_sha256': manifest_sha, 'parameters': parameters,
           'cuda_forward_fp32_2_5s_ms': {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95)), 'samples': times},
           'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
           'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
           'output_shape_2_5s': list(y.shape), 'input_shapes': {k: list(v.shape) for k,v in values.items()},
           'feature_dtype': 'float16 cache from float32 encoder output; encoder weights float32',
           'representation': 'last_hidden_state, pretrained encoder frozen; one utterance at a time'}
    dump(dest / 'preflight.json', rec)
    t0 = time.monotonic()
    for i, row in enumerate(manifest['rows'], 1):
        target = dest / (row['id'] + '.npy')
        if target.exists(): continue
        audio, rate = sf.read(WORK / 'audio' / row['wav'], dtype='float32')
        assert rate == 16000 and hashlib_audio(audio) == row['pcm_sha256']
        values = input_values(feature, audio, 'cuda')
        with torch.inference_mode(): output = model(**values).last_hidden_state[0]
        assert torch.isfinite(output).all()
        minimum = len(row['label_ids']) + sum(a==b for a,b in zip(row['label_ids'], row['label_ids'][1:]))
        assert output.shape[0] >= minimum, (row['id'], output.shape, minimum)
        np.save(target, output.cpu().numpy().astype(np.float16))
        if i % 300 == 0: print('FEATURES', key, i, '/', len(manifest['rows']), 'seconds', round(time.monotonic()-t0, 1), flush=True)
    rec['feature_seconds'] = time.monotonic()-t0
    rec['feature_files'] = len(manifest['rows'])
    rec['cached_shapes'] = {s: [int(x) for x in np.load(dest/(manifest['rows'][0]['id']+'.npy'), mmap_mode='r').shape] for s in ['first']}
    dump(dest / 'metadata.json', rec)
    del model, values, output
    gc.collect(); torch.cuda.empty_cache()
    print('FEATURES_COMPLETE', key, flush=True)
    return rec


def hashlib_audio(audio):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def edit_distance(a, b):
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a, b)


def decode(ids):
    return [k for k, _ in groupby(ids) if k != 0]


def tensor_batch(rows, features, indices, device='cuda'):
    lengths = [len(features[i]) for i in indices]
    width = max(lengths)
    dim = features[indices[0]].shape[-1]
    x = np.zeros((len(indices), width, dim), dtype=np.float32)
    targets, target_lengths = [], []
    for b,i in enumerate(indices):
        x[b,:lengths[b]] = features[i]
        targets.extend(rows[i]['label_ids'])
        target_lengths.append(len(rows[i]['label_ids']))
    return torch.from_numpy(x).to(device), torch.tensor(lengths), torch.tensor(targets,device=device), torch.tensor(target_lengths)


@torch.inference_mode()
def evaluate(head, rows, features, indices):
    head.eval()
    errors = total = exact = 0
    predictions = []
    for lo in range(0, len(indices), 32):
        group = indices[lo:lo+32]
        x,lengths,_,_ = tensor_batch(rows,features,group)
        logits = head(x)
        assert torch.isfinite(logits).all()
        ids = logits.argmax(-1).cpu().numpy()
        for j,i in enumerate(group):
            hyp = decode(ids[j,:lengths[j]].tolist())
            # Boundaries are excluded from the principal jamo error rate.
            ref = [x for x in rows[i]['label_ids'] if x!=1]
            scored = [x for x in hyp if x!=1]
            err = edit_distance(ref, scored)
            errors += err; total += len(ref); exact += (err == 0)
            predictions.append({'id':rows[i]['id'], 'hyp_ids':hyp, 'errors':err, 'reference_jamo':len(ref)})
    return {'jer': errors/total, 'errors':errors,'reference_jamo':total,'exact':exact,'recordings':len(indices)}, predictions


def probe(key, steps=1600):
    path = WORK / 'probes' / key
    path.mkdir(parents=True, exist_ok=True)
    if (path / 'result.json').exists():
        print('PROBE_REUSED',key,flush=True)
        return load(path/'result.json')
    manifest = load(WORK/'manifest.json')
    meta = load(WORK/'features'/key/'metadata.json')
    assert meta['manifest_sha256'] == digest(WORK/'manifest.json')
    rows = manifest['rows']
    indices = {s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['train','valid','test']}
    print('PROBE_LOAD',key,flush=True)
    features = [np.load(WORK/'features'/key/(r['id']+'.npy')) for r in rows]
    dim = features[0].shape[-1]
    summaries = []
    best = None
    started = time.monotonic()
    # Same two learning rates for all encoders, chosen only on the valid split.
    # A second seed is reserved for the selected rate after this search.
    for lr in [.001, .003]:
        torch.manual_seed(917); np.random.seed(917)
        head = torch.nn.Linear(dim, len(manifest['vocab'])).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.01)
        rng = random.Random(917)
        history=[]; run_best = None
        for step in range(1, steps+1):
            head.train()
            group = rng.sample(indices['train'], 32)
            x,lengths,target,tlen=tensor_batch(rows,features,group)
            logp=head(x).float().log_softmax(-1).transpose(0,1)
            loss=torch.nn.functional.ctc_loss(logp,target,lengths,tlen,blank=0,reduction='mean',zero_infinity=False)
            assert torch.isfinite(loss), (key,step,loss.item())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
            if step % 200 == 0:
                score,_=evaluate(head,rows,features,indices['valid'])
                h={'step':step,'loss':loss.item(),**score}; history.append(h)
                if run_best is None or score['jer'] < run_best['valid']['jer']:
                    run_best={'step':step,'lr':lr,'seed':917,'valid':score}
                    torch.save(head.state_dict(),path/f'head-lr{lr}.pt')
                print('PROBE',key,'lr',lr,'step',step,'valid_JER',round(score['jer'],4),flush=True)
        summaries.append({**run_best,'history':history})
        if best is None or run_best['valid']['jer']<best['valid']['jer']:
            best=run_best
        del head,opt
    head=torch.nn.Linear(dim,len(manifest['vocab'])).cuda()
    head.load_state_dict(torch.load(path/f"head-lr{best['lr']}.pt",weights_only=True))
    test,predictions=evaluate(head,rows,features,indices['test'])
    torch.save(head.state_dict(),path/'head.pt')
    dump(path/'test-predictions.json',predictions)
    result={'key':key,'method':'frozen pretrained encoder + linear CTC, surface-form jamo proxy labels',
            'manifest_sha256':digest(WORK/'manifest.json'),'steps_per_lr':steps,
            'search':summaries,'selected':best,'test':test,'elapsed_seconds':time.monotonic()-started,
            'head_sha256':digest(path/'head.pt'),'encoder_parameters':meta['parameters'],
            'limitations':['Not full encoder fine-tuning','Derived transcript labels do not establish pronunciation-error diagnostic accuracy','One seed in initial screen']}
    dump(path/'result.json',result)
    print('PROBE_COMPLETE',key,'test_JER',test['jer'],flush=True)
    del features,head
    gc.collect();torch.cuda.empty_cache()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('key');p.add_argument('--mode',choices=['features','probe','all'],default='all');p.add_argument('--steps',type=int,default=1600)
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.mode in {'features','all'}:extract(args.key)
    if args.mode in {'probe','all'}:probe(args.key,args.steps)
