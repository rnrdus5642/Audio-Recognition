"""Bounded real-model verification, including a fresh-model optimizer/RNG replay."""
import gc
import gzip
import hashlib
import json
import random
import time
import unittest
import uuid

import torch

import common
import data
import runtime
import train
from checkpoints import Checkpoints
import test_core


def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(str((v.dtype,tuple(v.shape))).encode())
            x=v.detach().cpu().contiguous()
            h.update(x.numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):
                h.update(repr(k).encode()); visit(v[k])
        elif isinstance(v,(list,tuple)):
            for item in v:visit(item)
        else:h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def verify():
    start=time.monotonic()
    config=train.configuration(); contract=train.contract(config)
    run=common.WORK/'verification'/('gpu-'+uuid.uuid4().hex[:10])
    manager=Checkpoints(run)
    result={'status':'running','contract_id':train.fingerprint(contract),'contract':contract,
            'run':str(run.relative_to(common.ROOT)),'checks':{}}
    def record(event,**values):
        common.log(run,event,**values)
        common.dump(common.WORK/'preflight-progress.json',{'status':'running','event':event,'run':str(run),'elapsed_seconds':time.monotonic()-start,**values})
    record('core_tests')
    outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_core))
    assert outcome.wasSuccessful()
    result['checks']['core_tests']=outcome.testsRun
    report=common.read(common.MANIFESTS/'report.json')
    dataset=[]; rng=random.Random(917)
    with gzip.open(common.MANIFESTS/'train.jsonl.gz','rt',encoding='utf8') as stream:
        for i,line in enumerate(stream):
            r=json.loads(line)
            if len(dataset)<256:dataset.append(r)
            else:
                k=rng.randrange(i+1)
                if k<256:dataset[k]=r
    dataset.append(report['longest']['train'])
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(config['cuda_allocator_fraction'])
    model,head,opt=runtime.build(config)
    cnn_before=digest(model.feature_extractor.state_dict())
    encoder_before=digest(model.state_dict())
    head_before=digest(head.state_dict())
    assert all(not p.requires_grad for p in model.feature_extractor.parameters())
    # Confirm padding and true frame lengths with unequal real utterances.
    short=min(range(len(dataset)),key=lambda i:dataset[i]['samples'])
    long=max(range(len(dataset)),key=lambda i:dataset[i]['samples'])
    b=data.collate(dataset,[short,long]); alone=data.collate(dataset,[short])
    assert torch.equal(b['input_values'][0,:dataset[short]['samples']],alone['input_values'][0])
    assert b['attention_mask'].sum(1).tolist()==[dataset[i]['samples'] for i in (short,long)]
    assert not b['input_values'][0,dataset[short]['samples']:].count_nonzero()
    model.eval(); head.eval()
    with torch.inference_mode():
        logits,lengths=runtime.forward(model,head,b,grad=False,amp=False)
        only,only_lengths=runtime.forward(model,head,alone,grad=False,amp=False)
    assert lengths.tolist()==[common.frames(dataset[i]['samples']) for i in (short,long)]
    difference=float((logits[0,:only_lengths[0]]-only[0,:only_lengths[0]]).abs().max())
    assert difference<.002, difference
    result['checks']['padding_max_abs_logit_difference']=difference
    del b,alone,logits,lengths,only,only_lengths
    sampler=data.Sampler(dataset,budget=config['batch_budget'],accumulate=config['accumulate'],seed=917)
    record('head_update')
    measured=runtime.update(model,head,opt,dataset,sampler.next_group(),head_only=True,head_lr=.001)
    assert digest(model.state_dict())==encoder_before
    assert digest(head.state_dict())!=head_before
    result['checks']['head_update']=measured
    record('longest_training_backward',seconds=dataset[long]['seconds'])
    measured=runtime.update(model,head,opt,dataset,[[long]])
    assert digest(model.feature_extractor.state_dict())==cnn_before
    assert digest(model.state_dict())!=encoder_before
    result['checks']['longest_training_update']=measured
    # Exercise ordinary accumulated joint training and full FP32 long evaluation
    # while the actual Adam moment tensors are already allocated.
    record('accumulated_joint_update')
    result['checks']['joint_update']=runtime.update(model,head,opt,dataset,sampler.next_group())
    model.eval(); head.eval()
    for split in ('dev','test'):
        record('longest_'+split+'_capacity_only',seconds=report['longest'][split]['seconds'])
        with torch.inference_mode():
            b=data.collate([report['longest'][split]],[0])
            logits,lengths=runtime.forward(model,head,b,grad=False,amp=False)
            assert bool(torch.isfinite(logits).all())
            assert lengths.item()==common.frames(report['longest'][split]['samples'])
        result['checks']['longest_'+split+'_capacity_seconds']=report['longest'][split]['seconds']
        del b,logits,lengths
    record('save_full_optimizer_checkpoint')
    begin=time.monotonic()
    receipt=manager.save({'model':model.state_dict(),'head':head.state_dict(),'optimizer':opt.state_dict(),
        'state':{'total_updates':3,'stage':'joint','weights_id':'preflight'},'sampler':sampler.snapshot(),
        'rng':runtime.rng_state(),'contract':contract})
    result['checks']['checkpoint_bytes']=receipt['bytes']
    result['checks']['checkpoint_save_seconds']=time.monotonic()-begin
    before_reload=digest({'model':model.state_dict(),'head':head.state_dict(),'optimizer':opt.state_dict()})
    before_rng=digest(runtime.rng_state())
    next_group=sampler.next_group()
    expected=runtime.update(model,head,opt,dataset,next_group)
    expected_state=digest({'model':model.state_dict(),'head':head.state_dict(),'optimizer':opt.state_dict()})
    expected_rng=digest(runtime.rng_state()); expected_sampler=sampler.snapshot()
    expected_path=run/'expected-next-update.pt'
    torch.save({'model':model.state_dict(),'head':head.state_dict(),'optimizer':opt.state_dict()},expected_path)
    result['checks']['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    result['checks']['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    del model,head,opt,sampler
    gc.collect(); torch.cuda.empty_cache()
    record('rebuild_and_reload_full_checkpoint')
    model,head,opt=runtime.build(config)
    begin=time.monotonic()
    saved,actual,failures=manager.load()
    assert not failures and saved['contract']==contract and actual==receipt
    model.load_state_dict(saved['model']); head.load_state_dict(saved['head']); opt.load_state_dict(saved['optimizer'])
    sampler=data.Sampler(dataset,budget=config['batch_budget'],accumulate=config['accumulate'],seed=917,state=saved['sampler'])
    runtime.restore_rng(saved['rng']); del saved; gc.collect()
    assert before_reload==digest({'model':model.state_dict(),'head':head.state_dict(),'optimizer':opt.state_dict()}), 'Saved state did not restore bitwise'
    assert before_rng==digest(runtime.rng_state())
    result['checks']['checkpoint_load_seconds']=time.monotonic()-begin
    replay_group=sampler.next_group(); assert next_group==replay_group
    record('replay_next_update')
    replay=runtime.update(model,head,opt,dataset,replay_group)
    actual_state=digest({'model':model.state_dict(),'head':head.state_dict(),'optimizer':opt.state_dict()})
    assert expected['loss']==replay['loss'], (expected['loss'],replay['loss'])
    reference=torch.load(expected_path,map_location='cpu',weights_only=False,mmap=True)
    differences=[]
    def compare(a,b,path=''):
        if torch.is_tensor(a):
            b=b.detach().cpu()
            assert a.dtype==b.dtype and a.shape==b.shape, path
            if not torch.equal(a,b):
                delta=a.double()-b.double()
                differences.append({'path':path,'max_abs':float(delta.abs().max()),
                    'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a.double()).clamp_min(1e-30))})
                assert torch.allclose(a,b,atol=1e-7,rtol=1e-5), (path,differences[-1])
        elif isinstance(a,dict):
            assert set(a)==set(b), path
            for k in a:compare(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b), path
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        else:assert a==b, (path,a,b)
    compare(reference,{'model':model.state_dict(),'head':head.state_dict(),'optimizer':opt.state_dict()})
    assert expected_state==actual_state, 'Deterministic next-update weights and Adam states must match bitwise'
    del reference; gc.collect()
    result['checks']['restored_state_bitwise_equal']=True
    result['checks']['fresh_model_resume_next_update_bitwise_equal']=expected_state==actual_state
    result['checks']['fresh_model_resume_next_update_within_tolerance']=True
    result['checks']['resume_tolerance']={'atol':1e-7,'rtol':1e-5}
    result['checks']['resume_changed_tensors']=len(differences)
    result['checks']['resume_max_abs_difference']=max((v['max_abs'] for v in differences),default=0.)
    common.dump(run/'numerical-replay-differences.json',differences)
    assert expected_sampler==sampler.snapshot()
    assert expected_rng==digest(runtime.rng_state())
    assert cnn_before==digest(model.feature_extractor.state_dict())
    result['checks']['resume_replayed_loss']=replay['loss']
    result['status']='passed'; result['elapsed_seconds']=time.monotonic()-start
    common.dump(run/'result.json',result)
    common.dump(common.WORK/'preflight.json',result)
    record('all_preflight_checks_passed',elapsed_seconds=result['elapsed_seconds'])
    common.dump(common.WORK/'preflight-progress.json',{'status':'passed','elapsed_seconds':result['elapsed_seconds'],'checks':result['checks']})
    manager.close()
    return result


if __name__=='__main__':
    try:
        output=verify()
        print(json.dumps({'status':output['status'],'checks':output['checks']},ensure_ascii=False),flush=True)
    except BaseException as exc:
        common.dump(common.WORK/'preflight-progress.json',{'status':'failed','error':str(exc)})
        raise
