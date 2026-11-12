"""WAV input and deterministic, resumable natural/age-weighted sampling."""
from collections import defaultdict
import copy
import random
import wave

import numpy as np
import torch

from common import safe_audio


def plan(rows, budget=320000, max_batch=32, seed=917, epoch=0):
    rng=random.Random(f'{seed}:{epoch}')
    order=list(range(len(rows)))
    rng.shuffle(order)
    order.sort(key=lambda i:rows[i]['samples'])
    batches=[]; current=[]; width=0
    for i in order:
        n=rows[i]['samples']
        if current and (max(width,n)*(len(current)+1)>budget or len(current)>=max_batch):
            batches.append(current); current=[]; width=0
        current.append(i); width=max(width,n)
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


class Sampler:
    def __init__(self, rows, budget=320000, accumulate=16, seed=917, policy='natural', state=None):
        self.rows,self.budget,self.accumulate,self.seed=rows,budget,accumulate,seed
        self.state=copy.deepcopy(state) if state else {'policy':policy,'epoch':0,'cursor':0,'draws':0,'cycles':{},'age_cursors':{}}
        self.cache={}
        self.rng=random.Random(f'{seed}:balanced')
        if self.state.get('rng'):
            self.rng.setstate(self.state['rng'])
        self.age_indices={age:[i for i,r in enumerate(rows) if r['age']==age] for age in (3,4,5,6)}
        self.age_weights=[len(self.age_indices[age])**.5 for age in (3,4,5,6)]

    def snapshot(self):
        result=copy.deepcopy(self.state)
        if self.state['policy']=='sqrt':
            result['rng']=self.rng.getstate()
        return result

    def age_order(self,age):
        cycle=self.state['cycles'].get(age,0)
        key=(age,cycle)
        if key not in self.cache:
            order=list(self.age_indices[age])
            random.Random(f'{self.seed}:age:{age}:cycle:{cycle}').shuffle(order)
            self.cache={k:v for k,v in self.cache.items() if not isinstance(k,tuple) or k[0]!=age}
            self.cache[key]=order
        return self.cache[key]

    def next_group(self):
        if self.state['policy']=='natural':
            epoch=self.state['epoch']
            if epoch not in self.cache:
                self.cache={epoch:plan(self.rows,self.budget,32,self.seed,epoch)}
            batches=self.cache[epoch]
            cursor=self.state['cursor']
            group=batches[cursor:cursor+self.accumulate]
            self.state['cursor']+=len(group)
            if self.state['cursor']>=len(batches):
                self.state['epoch']+=1; self.state['cursor']=0
            self.state['draws']+=sum(map(len,group))
            return group
        # Decide ages per utterance, then length-sort a bounded candidate pool.
        # The pool and its unconsumed batches are part of the checkpoint state.
        pending=self.state.setdefault('pending',[])
        while len(pending)<self.accumulate:
            chosen=[]
            for _ in range(512):
                age=self.rng.choices((3,4,5,6),weights=self.age_weights,k=1)[0]
                order=self.age_order(age)
                cursor=self.state['age_cursors'].get(age,0)
                chosen.append(order[cursor])
                cursor+=1
                if cursor==len(order):
                    cursor=0
                    self.state['cycles'][age]=self.state['cycles'].get(age,0)+1
                self.state['age_cursors'][age]=cursor
            local=[self.rows[i] for i in chosen]
            packed=plan(local,self.budget,32,self.seed,f'pool:{self.state.get("pools",0)}')
            pending.extend([[chosen[i] for i in b] for b in packed])
            self.state['pools']=self.state.get('pools',0)+1
        group=pending[:self.accumulate]
        del pending[:self.accumulate]
        self.state['draws']+=sum(map(len,group))
        return group


def collate(rows, indices, device='cuda'):
    width=max(rows[i]['samples'] for i in indices)
    values=np.zeros((len(indices),width),np.float32)
    mask=np.zeros((len(indices),width),np.int64)
    targets=[]; lengths=[]; samples=[]
    for b,i in enumerate(indices):
        row=rows[i]; path=safe_audio(row['audio_path']); stat=path.stat()
        if stat.st_size!=row['audio_bytes'] or stat.st_mtime_ns!=row['mtime_ns']:
            raise ValueError(f'Audio changed since verified manifest: {path}')
        with wave.open(str(path),'rb') as wav:
            if (wav.getnframes(),wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(row['samples'],16000,1,2):
                raise ValueError(f'WAV header changed: {path}')
            pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2')
        assert len(pcm)==row['samples']
        values[b,:len(pcm)]=pcm.astype(np.float32)/32768
        mask[b,:len(pcm)]=1
        targets.extend(row['label_ids']); lengths.append(len(row['label_ids'])); samples.append(len(pcm))
    return {'input_values':torch.from_numpy(values).to(device), 'attention_mask':torch.from_numpy(mask).to(device),
            'targets':torch.tensor(targets,dtype=torch.long,device=device),
            'target_lengths':torch.tensor(lengths,dtype=torch.long), 'samples':samples}
