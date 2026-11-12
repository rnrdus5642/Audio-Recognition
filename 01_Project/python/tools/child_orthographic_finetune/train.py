"""Resumable head adaptation, joint CTC training, age comparison and evaluation."""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from common import ROOT, CODE, PREP, WORK, MANIFESTS, PLAN, ENCODER, read, dump, sha, rows, log
from checkpoints import Checkpoints, durable_json
from data import Sampler
import runtime


def configuration():
    return {'version':'orthographic-jamo-run-20261027-v1','seed':917,'cpu_threads':8,
            'offload_layers':40,'batch_budget':320000,'accumulate':16,'eval_budget':320000,
            'head_updates':1000,'head_warmup':100,'joint_warmup':500,'maximum_equivalents':3.,
            'monitor_every':1000,'checkpoint_every_updates':100,'checkpoint_every_seconds':600,
            'encoder_lr':2e-5,'head_lr':1e-4,'head_adaptation_lr':.001,
            'branch_audio_seconds':200*3600,'allow_tf32':False,'cuda_allocator_fraction':.80,
            'ctc_device':'cpu','deterministic_algorithms':True,'cublas_workspace_config':':4096:8',
            'long_training_authorized':'User explicitly requested start and interruption-safe checkpoint/resume.'}


def contract(config):
    report=read(MANIFESTS/'report.json')
    assert report['status']=='passed'
    for name,digest in report['manifest_hashes'].items():
        assert sha(MANIFESTS/name)==digest
    for name,digest in report['frozen_input_hashes'].items():
        assert sha(PREP/name)==digest
    model_sha=sha(ENCODER/'model.safetensors')
    assert model_sha==read(PLAN)['model']['model_sha256']
    import transformers, numpy
    return {'configuration':config,'manifest_hashes':report['manifest_hashes'],
            'frozen_inputs':report['frozen_input_hashes'],'model_sha256':model_sha,
            'model_config_sha256':sha(ENCODER/'config.json'),'preprocessor_sha256':sha(ENCODER/'preprocessor_config.json'),
            'code':{name:sha(CODE/name) for name in ('common.py','data.py','runtime.py','checkpoints.py','train.py')},
            'optimizer_source':sha(CODE.parent/'child_jamo_finetune/optim.py'),
            'software':{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__}}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def initial_state():
    return {'stage':'head','status':'running','total_updates':0,'head_updates':0,'joint_updates':0,
            'joint_audio_seconds':0.,'arm_audio_seconds':0.,'lineage':'natural','weights_id':'initial',
            'branch_done':False,'spent_audio':{},'wall_seconds_by_stage':{},'exposure_by_age':{},'exposure_by_dataset':{},
            'decay_start':None,'significant_reference':None,'plateau':0,'regression':0,
            'best_monitor_macro':None,'best_monitor_age3':None,'best_late_macro':None,
            'last_monitor_joint_update':0,'pending_monitor':False,'pending_pins':[],'elapsed_seconds':0.}


class Trainer:
    def __init__(self,run,resume=False):
        self.run=Path(run); self.manager=Checkpoints(run); self.config=configuration()
        self.contract=contract(self.config); self.contract_id=fingerprint(self.contract)
        proof=read(WORK/'preflight.json')
        if proof.get('status')!='passed' or proof.get('contract_id')!=self.contract_id:
            raise RuntimeError('Run the GPU preflight for this exact code, data and configuration before training')
        self.train=rows('train'); self.monitor=rows('monitor'); self.dev=None; self.test=None
        self.train_seconds=sum(r['seconds'] for r in self.train)
        self.model,self.head,self.opt=runtime.build(self.config)
        self.state=initial_state(); self.sampler=Sampler(self.train,seed=917)
        self.last_save=time.monotonic(); self.started=time.monotonic(); self.elapsed_before=0.
        if resume:
            payload,receipt,failures=self.manager.load()
            self.restore(payload)
            del payload; gc.collect()
            self.elapsed_before=self.state.get('elapsed_seconds',0.)
            for label in self.state.pop('pending_pins',[]):
                self.manager.pin(label,receipt)
            log(self.run,'resumed',checkpoint=receipt,failed_newer_generations=failures)
        else:
            if self.manager.committed():
                raise RuntimeError('Run already has checkpoints; use --resume')
            durable_json(self.run/'contract.json',self.contract)
            durable_json(self.run/'recipe.json',read(PLAN))
            self.persist()
        self.progress()

    def restore(self,payload,preserve=None):
        if payload['contract']!=self.contract:
            raise RuntimeError('Code, data, software or run settings differ from checkpoint; resume refused')
        self.model.load_state_dict(payload['model']); self.head.load_state_dict(payload['head'])
        self.opt.load_state_dict(payload['optimizer']); self.opt.zero_grad()
        self.state=copy.deepcopy(payload['state'])
        if preserve:
            self.state.update(copy.deepcopy(preserve))
        self.sampler=Sampler(self.train,self.config['batch_budget'],self.config['accumulate'],917,state=payload['sampler'])
        runtime.restore_rng(payload['rng'])

    def payload(self):
        return {'model':self.model.state_dict(),'head':self.head.state_dict(),'optimizer':self.opt.state_dict(),
                'state':copy.deepcopy(self.state),'sampler':self.sampler.snapshot(),'rng':runtime.rng_state(),'contract':self.contract}

    def persist(self,pins=()):
        self.state['pending_pins']=list(pins)
        self.state['elapsed_seconds']=self.elapsed_before+time.monotonic()-self.started
        begin=time.monotonic(); receipt=self.manager.save(self.payload())
        for label in pins:
            self.manager.pin(label,receipt)
        self.state['pending_pins']=[]
        self.last_save=time.monotonic()
        log(self.run,'checkpoint_committed',file=receipt['file'],update=receipt['update'],
            size_bytes=receipt['bytes'],seconds=time.monotonic()-begin,pins=list(pins))
        self.progress()
        return receipt

    def progress(self,**extra):
        state=copy.deepcopy(self.state)
        state.update(updated=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),
                     train_records=len(self.train),train_hours=self.train_seconds/3600,
                     completed_equivalents=self.state['joint_audio_seconds']/self.train_seconds,
                     elapsed_seconds=self.elapsed_before+time.monotonic()-self.started,
                     latest_committed=self.manager.committed()[:1],**extra)
        if torch.cuda.is_available():
            state.update(cuda_allocated_gib=torch.cuda.memory_allocated()/2**30,
                         cuda_reserved_gib=torch.cuda.memory_reserved()/2**30)
        dump(self.run/'progress.json',state)

    def identity(self,split):
        return {'weights_id':self.state['weights_id'],'manifest_sha256':self.contract['manifest_hashes'][split+'.jsonl.gz'],
                'metric':'NFC_no_space_CER_equal_age_v1','contract':self.contract_id}

    def evaluate(self,split,key):
        if split=='dev' and self.dev is None:
            self.dev=rows('dev')
        if split=='test' and self.test is None:
            self.test=rows('test')
        dataset={'monitor':self.monitor,'dev':self.dev,'test':self.test}[split]
        self.progress(activity='evaluating_'+split)
        result=runtime.evaluate(self.model,self.head,dataset,self.run,key,self.identity(split),self.config['eval_budget'])
        log(self.run,'evaluation',split=split,key=key,macro_cer=result['macro_cer'],
            nonblank_fraction=result['nonblank_fraction'],seconds=result['elapsed_seconds'])
        return result

    def monitor_now(self):
        score=self.evaluate('monitor','monitor-'+self.state['weights_id'][:20])
        if self.state['stage']=='head_eval':
            if score['nonblank_fraction']<.5 or score['macro_cer']>=1:
                self.state.update(stage='head_gate_failed',status='needs_diagnosis',head_gate_score=score)
                self.persist(); return
            self.state['stage']='joint'
            self.sampler=Sampler(self.train,self.config['batch_budget'],self.config['accumulate'],917)
            self.state['pending_monitor']=False
            self.persist(pins=('head_adapted',)); return
        self.state['pending_monitor']=False
        self.state['last_monitor_joint_update']=self.state['joint_updates']
        pins=[]; macro=score['macro_cer']; age3=score['groups']['age:3']['cer']
        if self.state['stage']=='joint':
            if self.state['best_monitor_macro'] is None or macro<self.state['best_monitor_macro']:
                self.state['best_monitor_macro']=macro; pins.append('candidate_macro')
            if self.state['best_monitor_age3'] is None or age3<self.state['best_monitor_age3']:
                self.state['best_monitor_age3']=age3; pins.append('candidate_age3')
            if self.state['decay_start'] is not None and (self.state['best_late_macro'] is None or macro<self.state['best_late_macro']):
                self.state['best_late_macro']=macro; pins.append('candidate_late')
            reference=self.state['significant_reference']
            if reference is None or macro<=reference*.98:
                self.state['significant_reference']=macro; self.state['plateau']=0
            else:
                self.state['plateau']+=1
            if macro>self.state['best_monitor_macro']*1.03:
                self.state['regression']+=1
            else:
                self.state['regression']=0
            if self.state['regression']>=3:
                self.state.update(stage='diagnosis_required',status='needs_diagnosis',diagnosis='Three monitor regressions >=3% relative; inspect before continuing.')
            elif self.state['decay_start'] is None and self.equivalents()>=1.5 and self.state['plateau']>=3:
                self.state['decay_start']=self.state['joint_audio_seconds']
        self.persist(pins)

    def equivalents(self):
        return self.state['joint_audio_seconds']/self.train_seconds

    def train_update(self):
        head_only=self.state['stage']=='head'
        if head_only:
            lr=.001*min(1.,(self.state['head_updates']+1)/100)
            enc=0
        else:
            scale=min(1.,(self.state['joint_updates']+1)/500)
            if self.state['decay_start'] is not None:
                progress=(self.state['joint_audio_seconds']-self.state['decay_start'])/(.6*self.train_seconds)
                scale*=max(.05,1-.95*progress)
            enc=2e-5*scale; lr=1e-4*scale
        group=self.sampler.next_group()
        measured=runtime.update(self.model,self.head,self.opt,self.train,group,head_only,enc,lr)
        self.state['total_updates']+=1
        key='head_updates' if head_only else 'joint_updates'; self.state[key]+=1
        stage=self.state['stage']; seconds=measured['audio_seconds']
        self.state['spent_audio'][stage]=self.state['spent_audio'].get(stage,0.)+seconds
        self.state['wall_seconds_by_stage'][stage]=self.state['wall_seconds_by_stage'].get(stage,0.)+measured['wall_seconds']
        for field,source in (('exposure_by_age','by_age'),('exposure_by_dataset','by_dataset')):
            for group_name,counts in measured[source].items():
                target=self.state[field].setdefault(group_name,{'records':0,'seconds':0.})
                target['records']+=counts['records']; target['seconds']+=counts['seconds']
        if not head_only:
            self.state['joint_audio_seconds']+=seconds
        if stage.startswith('branch_'):
            self.state['arm_audio_seconds']+=seconds
        self.state['weights_id']=fingerprint([self.contract_id,self.state['lineage'],self.state['head_updates'],
                                              self.state['joint_updates'],self.state['joint_audio_seconds']])
        if self.state['total_updates']%10==0:
            log(self.run,'train_update',stage=stage,update=self.state['total_updates'],head_updates=self.state['head_updates'],
                joint_updates=self.state['joint_updates'],equivalents=self.equivalents(),encoder_lr=enc,head_lr=lr,**measured)
            self.progress(last_update=measured)

    def preserve_accounting(self):
        return {k:copy.deepcopy(self.state[k]) for k in ('total_updates','spent_audio','wall_seconds_by_stage','exposure_by_age','exposure_by_dataset')}

    def load_pin(self,label,preserve=True):
        payload,receipt,failures=self.manager.load(self.manager.pins()[label])
        self.restore(payload,self.preserve_accounting() if preserve else None)
        del payload; gc.collect(); torch.cuda.empty_cache()
        return receipt

    def branch_evaluation(self):
        natural=self.state['stage']=='branch_natural_eval'
        label='natural_arm' if natural else 'sqrt_arm'
        score=self.evaluate('dev',label+'-full-dev')
        if natural:
            self.load_pin('fork')
            self.state.update(stage='branch_sqrt',arm_audio_seconds=0.,lineage='sqrt',pending_monitor=False)
            self.sampler=Sampler(self.train,self.config['batch_budget'],self.config['accumulate'],917,policy='sqrt')
            self.persist()
        else:
            natural_score=read(self.run/'evaluations/natural_arm-full-dev.json')
            decision=runtime.comparison(natural_score,score)
            dump(self.run/'age-comparison.json',decision)
            if decision['winner']=='natural':
                self.load_pin('natural_arm')
            self.state.update(stage='joint',branch_done=True,branch_result=decision,pending_monitor=False,
                              significant_reference=None,plateau=0,regression=0,last_monitor_joint_update=self.state['joint_updates'])
            self.persist(pins=('selected_arm',))
            log(self.run,'age_comparison_decided',**decision)

    def selection(self):
        pins=self.manager.pins()
        candidates=[]; seen=set()
        labels=['candidate_macro','candidate_age3','candidate_late','natural_arm']
        if self.state.get('branch_result',{}).get('winner')=='sqrt':
            labels.append('sqrt_arm')
        evaluated=[]
        control=copy.deepcopy(self.state)
        for label in labels:
            if label not in pins or pins[label]['weights_id'] in seen:
                continue
            seen.add(pins[label]['weights_id']); candidates.append((label,pins[label]))
        for label,receipt in candidates:
            payload,_,_=self.manager.load(receipt)
            self.model.load_state_dict(payload['model']); self.head.load_state_dict(payload['head'])
            self.state['weights_id']=receipt['weights_id']
            del payload; gc.collect()
            key=(label+'-full-dev') if label in ('natural_arm','sqrt_arm') else 'candidate-'+receipt['weights_id'][:20]
            score=self.evaluate('dev',key)
            evaluated.append({'label':label,'receipt':receipt,'macro_cer':score['macro_cer'],
                              'age3_cer':score['groups']['age:3']['cer'],'evaluation_key':key})
        chosen=min(evaluated,key=lambda r:(r['macro_cer'],r['age3_cer'],r['receipt']['update']))
        dump(self.run/'selection.json',{'candidates':evaluated,'selected':chosen})
        payload,receipt,_=self.manager.load(chosen['receipt'])
        self.restore(payload,{k:control[k] for k in ('total_updates','spent_audio','wall_seconds_by_stage','exposure_by_age','exposure_by_dataset')})
        del payload; gc.collect()
        self.state.update(stage='test',status='running',selection=chosen,pending_monitor=False)
        self.persist(pins=('final_selected',))

    def final_test(self):
        score=self.evaluate('test','final-test')
        temp=self.run/'selected-model.pt.tmp'
        with temp.open('wb') as stream:
            torch.save({'model':self.model.state_dict(),'head':self.head.state_dict(),'contract':self.contract,
                        'selection':self.state['selection'],'test':score},stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp,self.run/'selected-model.pt')
        dump(self.run/'result.json',{'status':'complete','selection':self.state['selection'],'test':score,
                                    'model_sha256':sha(self.run/'selected-model.pt'),'product_word_acceptance_validated':False})
        self.state.update(stage='complete',status='complete')
        self.persist(pins=('completed',))

    def run_loop(self,stop_after=None):
        while self.state['stage'] not in ('complete','head_gate_failed','diagnosis_required'):
            if (self.run/'pause.request').exists():
                self.state['status']='paused'; self.persist(); return
            self.state['status']='running'
            stage=self.state['stage']
            if self.state['pending_monitor'] or stage=='head_eval':
                self.monitor_now(); continue
            if stage in ('branch_natural_eval','branch_sqrt_eval'):
                self.branch_evaluation(); continue
            if stage=='selection':
                self.selection(); continue
            if stage=='test':
                self.final_test(); continue
            if stage=='joint' and self.state['decay_start'] is None and self.equivalents()>=2.4:
                self.state['decay_start']=self.state['joint_audio_seconds']
            self.train_update()
            if stage=='head' and self.state['head_updates']>=1000:
                self.state.update(stage='head_eval',pending_monitor=True); self.persist(); continue
            if stage=='joint' and not self.state['branch_done'] and self.equivalents()>=1:
                self.state.update(stage='branch_natural',arm_audio_seconds=0.,pending_monitor=False)
                self.persist(pins=('fork',)); continue
            if stage.startswith('branch_') and self.state['arm_audio_seconds']>=200*3600:
                self.state['stage']=stage+'_eval'
                self.persist(pins=('natural_arm' if stage=='branch_natural' else 'sqrt_arm',)); continue
            if stage=='joint':
                end=3*self.train_seconds
                if self.state['decay_start'] is not None:
                    end=min(end,self.state['decay_start']+.6*self.train_seconds)
                if self.state['joint_audio_seconds']>=end:
                    self.state['pending_monitor']=True
                    self.monitor_now()
                    if self.state['stage']!='diagnosis_required':
                        if 'candidate_late' not in self.manager.pins():
                            self.persist(pins=('candidate_late',))
                        self.state['stage']='selection'; self.persist()
                    continue
            if stage!='head' and self.state['joint_updates']-self.state['last_monitor_joint_update']>=1000:
                self.state['pending_monitor']=True; self.persist(); continue
            if self.state['total_updates']%100==0 or time.monotonic()-self.last_save>=600:
                self.persist()
            if stop_after and self.state['total_updates']>=stop_after:
                self.state['status']='paused_for_bounded_test'; self.persist(); return
        self.progress()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--name',default='main'); parser.add_argument('--stop-after-updates',type=int)
    args=parser.parse_args()
    if Path(args.name).name!=args.name:
        raise ValueError('Run name must be a simple directory name')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(.80)
    trainer=None; run=WORK/args.name
    try:
        trainer=Trainer(run,args.resume)
        trainer.run_loop(args.stop_after_updates)
    except runtime.PauseRequested:
        # Selection temporarily loads candidate weights without their optimizer.
        # Keep the prior consistent checkpoint; the evaluation cursor is durable.
        if trainer:
            trainer.progress(status='paused',activity='evaluation_cursor_saved')
    except BaseException as exc:
        run.mkdir(parents=True,exist_ok=True)
        dump(run/'failure.json',{'at':time.strftime('%Y-%m-%d %H:%M:%S'),'error':str(exc),'traceback':traceback.format_exc(),
                                'recovery':'Run the same launcher with -Resume; latest verified committed checkpoint is used. Partial updates are not committed.'})
        if trainer:
            trainer.progress(status='interrupted_or_failed',error=str(exc))
        raise
    finally:
        if trainer:
            trainer.manager.close()


if __name__=='__main__':
    main()
