"""Phase boundaries must commit before evaluations or sampler forks."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

import common
import runtime
from train import Trainer, initial_state, configuration


class ControlTests(unittest.TestCase):
    def fake(self,stage):
        trainer=Trainer.__new__(Trainer)
        trainer.run=common.WORK/'verification'/('control-'+uuid.uuid4().hex[:10])
        trainer.run.mkdir(parents=True)
        trainer.config=configuration(); trainer.state=initial_state()
        trainer.state['stage']=stage; trainer.train_seconds=1000000.
        trainer.saved=[]
        def persist(pins=()):
            trainer.saved.append((copy.deepcopy(trainer.state),pins))
            (trainer.run/'pause.request').write_text('test phase boundary',encoding='utf8')
        trainer.persist=persist
        def update():
            trainer.state['total_updates']+=1
            if trainer.state['stage']=='head':trainer.state['head_updates']+=1
            else:
                trainer.state['joint_updates']+=1
                trainer.state['joint_audio_seconds']+=2
                trainer.state['arm_audio_seconds']+=2
        trainer.train_update=update
        return trainer

    def test_head_gate_is_committed_before_evaluation(self):
        trainer=self.fake('head'); trainer.state['head_updates']=999
        trainer.run_loop()
        state,pins=trainer.saved[0]
        self.assertEqual(state['stage'],'head_eval')
        self.assertTrue(state['pending_monitor'])
        self.assertEqual(state['head_updates'],1000)

    def test_first_pass_commits_fork_before_branch_training(self):
        trainer=self.fake('joint'); trainer.state['joint_audio_seconds']=trainer.train_seconds-1
        trainer.run_loop()
        state,pins=trainer.saved[0]
        self.assertEqual(state['stage'],'branch_natural')
        self.assertEqual(pins,('fork',))
        self.assertEqual(state['arm_audio_seconds'],0)
        self.assertFalse(state['branch_done'])

    def test_each_branch_commits_endpoint_before_full_dev(self):
        for stage,label in (('branch_natural','natural_arm'),('branch_sqrt','sqrt_arm')):
            trainer=self.fake(stage); trainer.state['arm_audio_seconds']=720000-1
            trainer.run_loop()
            state,pins=trainer.saved[0]
            self.assertEqual(state['stage'],stage+'_eval')
            self.assertEqual(pins,(label,))

    def test_evaluation_pause_saves_cursor_without_using_model(self):
        run=common.WORK/'verification'/('eval-pause-'+uuid.uuid4().hex[:10])
        run.mkdir(parents=True); (run/'pause.request').write_text('pause')
        class ForbiddenModel:
            def eval(self):pass
            def __call__(self,*args,**kwargs):raise AssertionError('Paused evaluation should not invoke model')
        with patch.object(runtime,'rng_state',return_value={}),patch.object(runtime,'restore_rng'):
            with self.assertRaises(runtime.PauseRequested):
                runtime.evaluate(ForbiddenModel(),ForbiddenModel(),[{'samples':16000}],run,'paused',{'test':True},device='cpu')
        progress=common.read(run/'evaluations/paused.partial.json')
        self.assertEqual(progress['cursor'],0)
        self.assertEqual(progress['identity'],{'test':True})


if __name__=='__main__':
    unittest.main(verbosity=2)
