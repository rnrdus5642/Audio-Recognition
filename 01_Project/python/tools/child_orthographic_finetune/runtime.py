"""Model, CTC update and resumable evaluation shared by training and preflight."""
from collections import defaultdict
import copy
import os
import random
import sys
import time
import unicodedata as ud

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from transformers import HubertModel
from transformers.utils import logging as hf_logging

from common import CODE, ENCODER, decode, read, dump, frames
import data
sys.path.append(str(CODE.parent/'child_jamo_finetune'))
from optim import PartialCPUAdamW


def seed(value=917):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def build(config,device='cuda'):
    os.environ['HF_HUB_OFFLINE']='1'
    hf_logging.disable_progress_bar()
    torch.set_num_threads(config.get('cpu_threads',8))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    assert read(ENCODER/'preprocessor_config.json')['do_normalize'] is False
    model=HubertModel.from_pretrained(ENCODER,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa')
    for name in ('hidden_dropout','attention_dropout','activation_dropout','feat_proj_dropout'):
        assert getattr(model.config,name)==.1
    model.config.layerdrop=0
    model.config.apply_spec_augment=True
    model.config.mask_time_prob=.05; model.config.mask_time_length=5; model.config.mask_time_min_masks=0
    model.config.mask_feature_prob=0; model.config.mask_feature_min_masks=0
    model.feature_extractor._freeze_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    seed(config.get('seed',917))
    head=torch.nn.Sequential(torch.nn.Dropout(.1),torch.nn.Linear(1280,69))
    model.to(device); head.to(device)
    opt=PartialCPUAdamW(model,head,config.get('offload_layers',40),2e-5,1e-4,foreach=True)
    return model,head,opt


def forward(model,head,batch,grad=True,amp=True):
    with torch.set_grad_enabled(grad), torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp and batch['input_values'].is_cuda):
        hidden=model(input_values=batch['input_values'],attention_mask=batch['attention_mask']).last_hidden_state
    logits=head(hidden.float())
    lengths=model._get_feat_extract_output_lengths(batch['attention_mask'].sum(-1)).long()
    return logits,lengths


def ctc_sum(logits,lengths,batch):
    # Native CUDA CTC backward is nondeterministic in the pinned PyTorch build.
    # Differentiable FP32 copies retain the same objective and return gradients
    # to the GPU encoder, while the small CTC dynamic program runs on the CPU.
    losses=torch.nn.functional.ctc_loss(logits.float().log_softmax(-1).transpose(0,1).cpu(),
        batch['targets'].cpu(),lengths.cpu(),batch['target_lengths'].cpu(),
        blank=0,reduction='none',zero_infinity=False)
    return (losses/batch['target_lengths'].to(losses.device)).sum()


class PauseRequested(Exception):
    pass


def update(model,head,opt,rows,group,head_only=False,encoder_lr=2e-5,head_lr=1e-4,device='cuda'):
    model.train(not head_only); head.train()
    opt.zero_grad(); opt.set_lr(0 if head_only else encoder_lr,head_lr)
    count=sum(map(len,group)); loss_sum=0.; samples=0; by_age=defaultdict(lambda:{'records':0,'seconds':0.})
    by_dataset=defaultdict(lambda:{'records':0,'seconds':0.})
    start=time.perf_counter()
    for indices in group:
        batch=data.collate(rows,indices,device)
        logits,lengths=forward(model,head,batch,grad=not head_only)
        total=ctc_sum(logits,lengths,batch)
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite CTC loss; original committed state is retained')
        (total/count).backward()
        loss_sum+=float(total.detach()); samples+=sum(batch['samples'])
        for i in indices:
            for key,registry in ((str(rows[i]['age']),by_age),(str(rows[i]['dataset_id']),by_dataset)):
                registry[key]['records']+=1; registry[key]['seconds']+=rows[i]['seconds']
        del batch,logits,lengths,total
    params=[p for p in model.parameters() if p.requires_grad]+list(head.parameters())
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    opt.step(); opt.zero_grad()
    if device=='cuda':
        torch.cuda.synchronize()
    return {'loss':loss_sum/count,'gradient_norm':float(norm),'records':count,'audio_seconds':samples/16000,
            'wall_seconds':time.perf_counter()-start,'by_age':dict(by_age),'by_dataset':dict(by_dataset)}


def empty_score():
    return {'records':0,'errors':0,'reference':0,'errors_with_space':0,'reference_with_space':0,
            'jamo_errors':0,'jamo_reference':0,'exact':0,'nonblank':0,'malformed':0,'seconds':0.}


def add_score(total,reference,hypothesis,seconds):
    ref=reference.replace(' ',''); hyp=hypothesis.replace(' ','')
    rj=ud.normalize('NFD',ref); hj=ud.normalize('NFD',hyp)
    total['records']+=1; total['seconds']+=seconds
    total['errors']+=Levenshtein.distance(ref,hyp); total['reference']+=len(ref)
    total['errors_with_space']+=Levenshtein.distance(reference,hypothesis); total['reference_with_space']+=len(reference)
    total['jamo_errors']+=Levenshtein.distance(rj,hj); total['jamo_reference']+=len(rj)
    total['exact']+=reference==hypothesis; total['nonblank']+=bool(hyp)
    total['malformed']+=any('\u1100'<=c<='\u11ff' for c in hypothesis)


def finish_score(state):
    result=copy.deepcopy(state)
    for registry in ('groups','speakers'):
        for value in result[registry].values():
            value['cer']=value['errors']/max(1,value['reference'])
            value['cer_with_space']=value['errors_with_space']/max(1,value['reference_with_space'])
            value['jamo_er']=value['jamo_errors']/max(1,value['jamo_reference'])
    ages=[result['groups'][f'age:{a}']['cer'] for a in (3,4,5,6) if f'age:{a}' in result['groups']]
    result['macro_cer']=sum(ages)/len(ages)
    all_=result['groups']['all']; result['nonblank_fraction']=all_['nonblank']/all_['records']
    return result


@torch.inference_mode()
def evaluate(model,head,rows,run,key,identity,budget=320000,device='cuda'):
    directory=run/'evaluations'; directory.mkdir(exist_ok=True)
    completed=directory/f'{key}.json'; partial=directory/f'{key}.partial.json'
    if completed.exists():
        result=read(completed)
        assert result['identity']==identity
        return result
    state=read(partial) if partial.exists() else {'identity':identity,'cursor':0,'groups':{},'speakers':{},'elapsed_seconds':0.}
    assert state['identity']==identity
    batches=data.plan(rows,budget,32,917,'evaluation')
    old_rng=rng_state()
    model.eval(); head.eval()
    start=time.perf_counter(); previous=state['elapsed_seconds']
    try:
        for b in range(state['cursor'],len(batches)):
            if (run/'pause.request').exists():
                state['elapsed_seconds']=previous+time.perf_counter()-start
                dump(partial,state)
                raise PauseRequested('Evaluation cursor saved; resume from the committed model')
            indices=batches[b]; batch=data.collate(rows,indices,device)
            logits,lengths=forward(model,head,batch,grad=False,amp=False)
            predictions=logits.argmax(-1).cpu().tolist(); widths=lengths.cpu().tolist()
            for i,ids,width in zip(indices,predictions,widths):
                row=rows[i]; text=decode(ids[:width])
                keys=['all',f'age:{row["age"]}',f'dataset:{row["dataset_id"]}',f'age:{row["age"]}/dataset:{row["dataset_id"]}']
                if row['seconds']<=2.5:
                    keys.append('duration:<=2.5')
                for group in keys:
                    add_score(state['groups'].setdefault(group,empty_score()),row['text'],text,row['seconds'])
                speaker=state['speakers'].setdefault(row['speaker_key'],{**empty_score(),'age':row['age']})
                add_score(speaker,row['text'],text,row['seconds'])
            state['cursor']=b+1
            del batch,logits,lengths,predictions
            if (b+1)%100==0:
                state['elapsed_seconds']=previous+time.perf_counter()-start
                dump(partial,state)
        state['elapsed_seconds']=previous+time.perf_counter()-start
        result=finish_score(state); result['status']='complete'
        dump(completed,result)
        return result
    finally:
        restore_rng(old_rng)


def comparison(natural,balanced):
    a=natural['groups']; b=balanced['groups']
    children=[s for s,v in natural['speakers'].items() if v['age']==3]
    improved=sum(balanced['speakers'][s]['cer']<natural['speakers'][s]['cer'] for s in children)
    checks={'age3_relative_improvement':a['age:3']['cer']>0 and b['age:3']['cer']<=a['age:3']['cer']*.95,
            'at_least_five_age3_speakers':improved>=5,
            'macro_nonregression':balanced['macro_cer']<=natural['macro_cer']}
    for age in (4,5,6):
        checks[f'age{age}_nonregression']=b[f'age:{age}']['cer']<=a[f'age:{age}']['cer']*1.02
    rng=np.random.default_rng(917); relative=[]
    for _ in range(10000):
        selected=[children[int(i)] for i in rng.integers(0,len(children),len(children))]
        ar=sum(natural['speakers'][s]['reference'] for s in selected)
        ac=sum(natural['speakers'][s]['errors'] for s in selected)/ar
        bc=sum(balanced['speakers'][s]['errors'] for s in selected)/ar
        relative.append((ac-bc)/max(ac,1e-12))
    return {'winner':'sqrt' if all(checks.values()) else 'natural','checks':checks,'age3_speakers_improved':improved,
            'age3_speaker_ids':len(children),'relative_improvement_bootstrap90_exploratory':np.percentile(relative,[5,95]).tolist()}
