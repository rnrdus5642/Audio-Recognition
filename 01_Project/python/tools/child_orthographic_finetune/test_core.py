"""Behavioral checks for decoding, accumulation, sampling and crash recovery."""
import copy
import json
import os
import pickle
from pathlib import Path
import random
import subprocess
import sys
import time
import unittest
import uuid

import torch

import common
import data
import runtime
from checkpoints import Checkpoints


def directory():
    path=common.WORK/'verification'/('test-'+uuid.uuid4().hex[:10])
    path.mkdir(parents=True)
    assert path.resolve().is_relative_to(common.WORK.resolve())
    return path


def payload(update):
    return {'model':{'w':torch.tensor([float(update)])},'head':{},'optimizer':{'moments':torch.tensor([.5])},
            'state':{'total_updates':update,'stage':'head','weights_id':str(update)},
            'sampler':{'cursor':update},'rng':{'python':random.getstate()},'contract':{'test':True}}


class CoreTests(unittest.TestCase):
    def test_all_hangul_roundtrip_and_repeats(self):
        for code in range(0xac00,0xd7a4):
            text=chr(code)
            ids=common.encode(text)
            sequence=[v for i in ids for v in (i,i,0)]
            self.assertEqual(common.decode(sequence),text)
        self.assertEqual(common.decode([2,2,0,2]),'ᄀᄀ')
        self.assertEqual(common.decode([0,2,21,0,2,21]),'가가')
        self.assertEqual(common.normalize('  책!  놔요. '),'책 놔요')
        with self.assertRaises(ValueError):
            common.normalize('(NO:) 책')

    def test_unequal_microbatches_preserve_loss_and_gradients(self):
        torch.manual_seed(5)
        original=torch.randn(12,4,69,dtype=torch.float64)
        targets=[[2,21,2],[3,22],[3,22,2],[4,23]]; lengths=torch.tensor([12,11,10,9])
        def result(groups):
            logits=original.clone().requires_grad_()
            loss=0
            for indices in groups:
                batch={'targets':torch.tensor([t for i in indices for t in targets[i]]),
                       'target_lengths':torch.tensor([len(targets[i]) for i in indices])}
                loss=loss+runtime.ctc_sum(logits[:,indices].transpose(0,1),lengths[indices],batch)/4
            loss.backward()
            return loss.detach(),logits.grad
        a,ga=result([[0,1,2,3]]); b,gb=result([[0],[1,2,3]])
        self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=1e-6))
        self.assertTrue(torch.allclose(ga,gb,atol=1e-7,rtol=1e-6))

    def test_sampler_resume_and_natural_coverage(self):
        rows=[{'samples':16000+(i%7)*1000,'age':3+i%4} for i in range(103)]
        sample=data.Sampler(rows,budget=40000,accumulate=3)
        visited=[]
        while sample.state['epoch']==0:
            visited.extend(i for batch in sample.next_group() for i in batch)
        self.assertEqual(sorted(visited),list(range(len(rows))))
        for policy in ('natural','sqrt'):
            a=data.Sampler(rows,budget=40000,accumulate=3,policy=policy)
            for _ in range(5): a.next_group()
            state=pickle.loads(pickle.dumps(a.snapshot()))
            b=data.Sampler(rows,budget=40000,accumulate=3,state=state)
            self.assertEqual([a.next_group() for _ in range(7)],[b.next_group() for _ in range(7)])

    def test_checkpoint_corruption_falls_back_and_index_recovers(self):
        manager=Checkpoints(directory())
        try:
            first=manager.save(payload(1)); second=manager.save(payload(2))
            manager.safe(second['file']).write_bytes(b'forced corrupted generation')
            restored,receipt,failures=manager.load()
            self.assertEqual(receipt['file'],first['file'])
            self.assertEqual(restored['sampler']['cursor'],1)
            self.assertEqual(len(failures),1)
            manager.index.write_text('{interrupted',encoding='utf-8')
            restored,receipt,failures=manager.load()
            self.assertEqual(receipt['file'],first['file'])
            third=manager.save(payload(3))
            self.assertEqual([v['file'] for v in manager.committed()],[third['file'],first['file']])
            manager.index.unlink()
            restored,receipt,failures=manager.load()
            self.assertEqual(receipt['file'],third['file'])
        finally:
            manager.close()

    def test_duplicate_writer_is_rejected(self):
        path=directory(); manager=Checkpoints(path)
        try:
            with self.assertRaises(RuntimeError):
                Checkpoints(path)
        finally:
            manager.close()

    def test_forced_termination_during_write_retains_last_commit(self):
        path=directory(); manager=Checkpoints(path)
        receipt=manager.save(payload(7)); manager.close()
        script="""
import os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
from checkpoints import Checkpoints
run=Path(sys.argv[2]); manager=Checkpoints(run)
def torn_save(payload,stream):
    stream.write(b'incomplete checkpoint'); stream.flush(); os.fsync(stream.fileno())
    (run/'writing.marker').write_text(str(os.getpid()))
    time.sleep(45)
torch.save=torn_save
manager.save({'state':{'total_updates':8,'stage':'head'}})
"""
        proc=subprocess.Popen([sys.executable,'-B','-c',script,str(common.CODE),str(path)],creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline=time.monotonic()+30
            while not (path/'writing.marker').exists() and proc.poll() is None and time.monotonic()<deadline:
                time.sleep(.1)
            self.assertTrue((path/'writing.marker').exists())
            # On Windows a venv executable may be a launcher with a real child.
            # Terminate this test's entire spawned tree, including the lock owner.
            actual_pid=int((path/'writing.marker').read_text())
            killed=subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],
                                  capture_output=True,text=True,check=True)
            proc.wait(timeout=10)
            self.assertNotEqual(actual_pid,os.getpid())
        finally:
            if proc.poll() is None:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
                proc.wait(timeout=10)
        manager=Checkpoints(path)
        try:
            restored,actual,failures=manager.load()
            self.assertEqual(actual['file'],receipt['file'])
            self.assertEqual(restored['state']['total_updates'],7)
            self.assertTrue(list(manager.directory.glob('*.tmp')))
        finally:
            manager.close()

    def test_age4_regression_blocks_balanced_adoption(self):
        def score(age3,age4):
            ages={3:age3,4:age4,5:.2,6:.2}
            return {'macro_cer':sum(ages.values())/4,'groups':{f'age:{a}':{'cer':v} for a,v in ages.items()},
                    'speakers':{str(i):{'age':3,'cer':age3,'errors':int(age3*1000),'reference':1000} for i in range(7)}}
        self.assertEqual(runtime.comparison(score(.2,.2),score(.18,.2))['winner'],'sqrt')
        self.assertEqual(runtime.comparison(score(.2,.2),score(.18,.21))['winner'],'natural')


if __name__=='__main__':
    unittest.main(verbosity=2)
