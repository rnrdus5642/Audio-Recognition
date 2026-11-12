"""Durable generations, atomic commit pointer, SHA verification and fallback."""
import atexit
import hashlib
import json
import msvcrt
import os
import pickle
from pathlib import Path
import time
import uuid

import torch

from common import WORK, read, sha


def durable_json(path,value):
    path=Path(path); temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)


class Checkpoints:
    def __init__(self, run):
        self.run=Path(run).resolve()
        if not self.run.is_relative_to(WORK.resolve()) or self.run==WORK.resolve():
            raise ValueError('Checkpoint directory must be below the new experiment root')
        self.run.mkdir(parents=True,exist_ok=True)
        self.directory=self.run/'checkpoints'; self.directory.mkdir(exist_ok=True)
        self.index=self.directory/'committed.json'
        self.pins_file=self.directory/'pins.json'
        self.last_verified=None
        self.lock=(self.run/'run.lock').open('a+b')
        self.lock.seek(0)
        try:
            msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            self.lock.close()
            raise RuntimeError('Another process owns this training run')
        durable_json(self.run/'owner.json',{'pid':os.getpid(),'started':time.strftime('%Y-%m-%d %H:%M:%S')})
        atexit.register(self.close)

    def close(self):
        if self.lock.closed:
            return
        owner=self.run/'owner.json'
        if owner.exists() and read(owner).get('pid')==os.getpid():
            owner.unlink()
        self.lock.seek(0)
        msvcrt.locking(self.lock.fileno(),msvcrt.LK_UNLCK,1)
        self.lock.close()

    def safe(self,name):
        if Path(name).name!=name or not name.startswith('ckpt-') or not name.endswith('.pt'):
            raise ValueError('Invalid checkpoint generation name')
        result=(self.directory/name).resolve()
        if result.parent!=self.directory.resolve():
            raise ValueError('Checkpoint escapes owned directory')
        return result

    def committed(self):
        try:
            values=read(self.index)
            if not isinstance(values,list) or not values:
                raise ValueError('Invalid checkpoint index')
            for receipt in values:
                if not isinstance(receipt,dict) or not all(k in receipt for k in ('file','bytes','sha256','update')):
                    raise ValueError('Invalid generation receipt in checkpoint index')
                self.safe(receipt['file'])
            return values
        except (OSError,ValueError):
            # A damaged index can be reconstructed from complete generation receipts.
            recovered=[]
            for path in self.directory.glob('ckpt-*.pt.json'):
                try:
                    receipt=read(path)
                    self.safe(receipt['file'])
                    recovered.append((path.stat().st_mtime_ns,receipt))
                except (OSError,ValueError,KeyError):
                    continue
            recovered.sort(key=lambda item:(item[1]['update'],item[0]),reverse=True)
            return [item[1] for item in recovered[:2]]

    def pins(self):
        return read(self.pins_file) if self.pins_file.exists() else {}

    def save(self,payload,pin=None):
        # Never overwrite a completed generation. A killed writer can only leave an uncommitted temporary.
        name=f'ckpt-{payload["state"]["total_updates"]:09d}-{uuid.uuid4().hex[:12]}.pt'
        final=self.safe(name); temp=final.with_suffix('.pt.tmp')
        with temp.open('wb') as stream:
            torch.save(payload,stream)
            stream.flush(); os.fsync(stream.fileno())
        digest=sha(temp); size=temp.stat().st_size
        os.replace(temp,final)
        receipt={'file':name,'sha256':digest,'bytes':size,'update':payload['state']['total_updates'],
                 'stage':payload['state']['stage'],'weights_id':payload['state'].get('weights_id'),
                 'committed_at':time.strftime('%Y-%m-%d %H:%M:%S')}
        # Receipt first, then a single atomic pointer commit. Retention runs only after this commit.
        durable_json(self.directory/(name+'.json'),receipt)
        previous=[self.last_verified] if self.last_verified else self.committed()[:1]
        durable_json(self.index,[receipt]+previous)
        self.last_verified=receipt
        if pin:
            self.pin(pin,receipt)
        self.prune()
        return receipt

    def pin(self,label,receipt):
        pins=self.pins(); pins[label]=receipt
        durable_json(self.pins_file,pins)

    def unpin(self,label):
        pins=self.pins(); pins.pop(label,None)
        durable_json(self.pins_file,pins)
        self.prune()

    def prune(self):
        keep={r['file'] for r in self.committed()}|{r['file'] for r in self.pins().values()}
        for path in self.directory.glob('ckpt-*.pt'):
            if path.name not in keep:
                path=self.safe(path.name)
                path.unlink()
                receipt=self.directory/(path.name+'.json')
                if receipt.exists():
                    receipt.unlink()

    def load(self,receipt=None):
        attempts=[receipt] if receipt else self.committed()
        failures=[]
        for item in attempts:
            try:
                path=self.safe(item['file'])
                if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
                    raise ValueError('Checkpoint size/SHA256 mismatch')
                payload=torch.load(path,map_location='cpu',weights_only=False)
                if not all(k in payload for k in ('model','head','optimizer','state','sampler','rng','contract')):
                    raise ValueError('Incomplete checkpoint payload')
                # A resumed run must retain the verified fallback, not a corrupt
                # newer generation still named by the old commit pointer.
                if receipt is None:
                    self.last_verified=item
                return payload,item,failures
            except (OSError,ValueError,RuntimeError,EOFError,pickle.UnpicklingError) as exc:
                failures.append({'file':item['file'],'error':str(exc)})
        raise RuntimeError(f'No valid committed checkpoint: {failures}')
