"""Training-only feature standardization before identical linear CTC probes.

Native final-feature scales differ by >20x. Preserve the raw-scale experiment,
and test one scalar scale correction without changing relative channel weights.
No validation/test feature statistics enter this normalization.
"""
import argparse
import gc
import random
import time
import numpy as np
import torch
from filelock import FileLock
from common import WORK,load,dump,digest
from probe import tensor_batch,evaluate


class StandardHead(torch.nn.Module):
    def __init__(self,mean,scale,classes):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32).clone())
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32).clone())
        self.linear=torch.nn.Linear(len(mean),classes)
    def forward(self,x):return self.linear((x-self.mean)/self.scale)


def statistics(key,rows,features):
    path=WORK/'probes-normalized'/key/'statistics.json'
    if path.exists():
        r=load(path);assert r['manifest_sha256']==digest(WORK/'manifest.json')
        assert r['kind']=='single_scalar_no_centering';return r
    d=features[0].shape[1];total=0;sums=np.zeros(d,np.float64);squares=np.zeros(d,np.float64)
    for row,feature in zip(rows,features):
        if row['split']!='train':continue
        x=feature.astype(np.float64);total+=len(x);sums+=x.sum(0);squares+=(x*x).sum(0)
    mean=sums/total;std=np.sqrt(np.maximum(squares/total-mean*mean,0))
    scalar=max(float(std.mean()),.001)
    scale=np.full(d,scalar)
    r={'key':key,'kind':'single_scalar_no_centering','split':'train only','frames':total,
       'mean':np.zeros(d).tolist(),'scale':scale.tolist(),'native_channel_mean':mean.tolist(),
       'native_mean_channel_std':float(std.mean()),'native_min_channel_std':float(std.min()),
       'floor':.001,'floored_channels':int((std<.001).sum()),'manifest_sha256':digest(WORK/'manifest.json')}
    dump(path,r);return r


def _run(key,steps=1600):
    path=WORK/'probes-normalized'/key;path.mkdir(parents=True,exist_ok=True)
    if (path/'result.json').exists():
        prior=load(path/'result.json')
        assert prior['steps_per_lr']==steps, 'Existing head uses a different update budget'
        assert prior['manifest_sha256']==digest(WORK/'manifest.json'), 'Dataset changed'
        assert load(path/'statistics.json')['kind']=='single_scalar_no_centering'
        return prior
    torch.set_num_threads(4)
    manifest=load(WORK/'manifest.json');rows=manifest['rows']
    features=[np.load(WORK/'features'/key/(r['id']+'.npy')) for r in rows]
    stats=statistics(key,rows,features)
    indices={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['train','valid','test']}
    summaries=[];best=None;start=time.monotonic()
    print('NORMALIZED_PROBE_START',key,'native_mean_std',stats['native_mean_channel_std'],flush=True)
    for lr in [.001,.003]:
        torch.manual_seed(917);np.random.seed(917)
        head=StandardHead(stats['mean'],stats['scale'],len(manifest['vocab'])).cuda()
        opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.01)
        rng=random.Random(917);history=[];run_best=None
        for step in range(1,steps+1):
            head.train();group=rng.sample(indices['train'],32)
            x,lengths,target,tlen=tensor_batch(rows,features,group)
            logp=head(x).float().log_softmax(-1).transpose(0,1)
            loss=torch.nn.functional.ctc_loss(logp,target,lengths,tlen,blank=0,reduction='mean',zero_infinity=False)
            assert torch.isfinite(loss),(key,step)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
            if step%200==0:
                score,_=evaluate(head,rows,features,indices['valid'])
                history.append({'step':step,'valid':score})
                if run_best is None or score['jer']<run_best['valid']['jer']:
                    run_best={'step':step,'lr':lr,'seed':917,'valid':score}
                    torch.save(head.state_dict(),path/f'head-lr{lr}.pt')
                print('NORMALIZED_PROBE',key,lr,step,round(score['jer'],4),flush=True)
        summaries.append({**run_best,'history':history})
        if best is None or run_best['valid']['jer']<best['valid']['jer']:
            best=run_best
            torch.save(torch.load(path/f'head-lr{lr}.pt',map_location='cpu',weights_only=True),path/'head.pt')
        del head,opt;gc.collect();torch.cuda.empty_cache()
    head=StandardHead(stats['mean'],stats['scale'],len(manifest['vocab'])).cuda()
    head.load_state_dict(torch.load(path/'head.pt',map_location='cpu',weights_only=True))
    test,predictions=evaluate(head,rows,features,indices['test'])
    dump(path/'test-predictions.json',predictions)
    result={'key':key,'method':'train-only scalar amplitude normalization + frozen encoder linear CTC',
            'manifest_sha256':digest(WORK/'manifest.json'),'statistics_sha256':digest(path/'statistics.json'),
            'steps_per_lr':steps,'search':summaries,'selected':best,'test':test,
            'head_sha256':digest(path/'head.pt'),'elapsed_seconds':time.monotonic()-start,
            'native_mean_channel_std':stats['native_mean_channel_std'],
            'limitations':['Fixed last-layer representation; not convergence ranking','One seed for the frozen probe',
                           'Transcript-derived jamo reference, not audited pronunciation-error labels']}
    dump(path/'result.json',result)
    print('NORMALIZED_PROBE_COMPLETE',key,test,flush=True)
    del features,head;gc.collect();torch.cuda.empty_cache()
    return result


def run(key,steps=1600):
    path=WORK/'probes-normalized'/key;path.mkdir(parents=True,exist_ok=True)
    # A head-only job may overlap encoder training, never standalone timings.
    with FileLock(str(path/'training.lock'),timeout=7200):
        return _run(key,steps)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('keys',nargs='+');args=p.parse_args()
    for key in args.keys:run(key)
