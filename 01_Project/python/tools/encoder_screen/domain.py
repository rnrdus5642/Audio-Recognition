"""Held-out owned short words; no threshold tuning or temporary demo aliases."""
from collections import defaultdict
import hashlib
import numpy as np
import soundfile as sf
import torch
from common import ROOT, WORK, load, dump, digest
from probe import input_values, decode, edit_distance
from prepare import labels, VOCAB


@torch.inference_mode()
def score(model, head, feature, out):
    model.eval(); head.eval()
    source_file = ROOT/'.venv/adult-model-comparison-0913/inputs.json'
    source = load(source_file)
    reverse = {v:k for k,v in VOCAB.items()}
    records = []
    for key, row in source['clips'].items():
        path = row['path']
        assert digest(path) == row['sha256'], path
        wave, sr = sf.read(path, dtype='float32')
        assert sr == 16000 and wave.ndim == 1
        surface, reference = labels(row['label'])
        ref = [i for i in reference if i != 1]
        blocks = wave[:len(wave)//320*320].reshape(-1,320)
        rms = np.sqrt(np.mean(blocks.astype(np.float64)**2,axis=1))
        active = blocks[rms > max(float(rms.max())*.1, 1e-7)]
        level = float(np.sqrt(np.mean(active.astype(np.float64)**2))) if active.size else 0.
        rng = np.random.default_rng(int(hashlib.sha256(('encoder-domain:'+key).encode()).hexdigest()[:16],16))
        noise = rng.normal(size=len(wave)); noise /= max(np.sqrt(np.mean(noise**2)),1e-12)
        variants = {'clean':wave, 'pad_1s':np.pad(wave,(16000,16000))}
        for snr in [20,10]:
            y = wave + noise*level*10**(-snr/20)
            variants[f'noise_{snr}db'] = (y/max(1.,float(np.abs(y).max())/.98)).astype(np.float32)
        for condition, audio in variants.items():
            logits = head(model(**input_values(feature, audio, 'cuda')).last_hidden_state)
            assert torch.isfinite(logits).all()
            hyp = decode(logits.argmax(-1)[0].cpu().tolist())
            scored = [i for i in hyp if i != 1]
            err = edit_distance(ref,scored)
            # Dialect sensitivity is a separate score, not a changed reference.
            merge = {VOCAB['ㅐ']:VOCAB['ㅔ'],VOCAB['ㅒ']:VOCAB['ㅖ']}
            merged_err = edit_distance([merge.get(i,i) for i in ref], [merge.get(i,i) for i in scored])
            records.append({'id':key,'set':row['set'],'speaker':row['speaker_id'],'label':row['label'],
                            'surface':surface,'condition':condition,'source_sha256':row['sha256'],
                            'reference_ids':reference,'hyp_ids':hyp,'hyp_jamo':''.join(reverse[i] for i in hyp),
                            'errors':err,'ae_merged_errors':merged_err,'reference_jamo':len(ref)})
    groups = defaultdict(list)
    for row in records: groups[(row['set'],row['condition'])].append(row)
    summary = []
    for (group,condition), rows in groups.items():
        n=sum(r['reference_jamo'] for r in rows)
        summary.append({'set':group,'condition':condition,'recordings':len(rows),
                        'exact':sum(r['errors']==0 for r in rows),
                        'jer':sum(r['errors'] for r in rows)/n,
                        'ae_merged_exact':sum(r['ae_merged_errors']==0 for r in rows)})
    result = {'source_manifest_sha256':digest(source_file),'unique_recordings':53,
              'notes':['7 adult user recordings; age unconfirmed for the other 46 recordings from two speakers',
                       'Whole utterance greedy CTC; no context/word forcing, matching thresholds or demo aliases',
                       'Noise is controlled white noise based on active 20 ms block RMS; not real room noise',
                       'Exact jamo sequence agreement is not product acceptance or a pronunciation-error diagnosis'],
              'summary':summary,'records':records}
    dump(out,result)
    return result
