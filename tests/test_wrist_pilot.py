"""Protect the camera intervention and simulator action conventions."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from vla_eval.contracts import Observation

spec=importlib.util.spec_from_file_location('wrist_runner',Path(__file__).parents[1]/'scripts/run_gr00t_wrist_pilot.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class WristPilotTests(unittest.TestCase):
    def test_batched_cache_fill_then_only_changed_camera_recomputes(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional Torch required for vision cache test')
        from vla_eval.gr00t16_pilot import Gr00t16Pilot
        pilot=object.__new__(Gr00t16Pilot)
        pilot.torch=torch;pilot.synchronize=lambda:None
        pilot.cache_enabled=True;pilot.cache_cold_batch=True;pilot.cache={};pilot.stage={}
        pilot.camera_keys=['image','wrist_image'];pilot.image_keys=['a','b']
        calls=[]
        def encode(pixels,flags=None):
            calls.append(len(pixels))
            return torch.stack([p.sum().repeat(2) for p in pixels])
        pilot.original_extract=encode
        pixels=[torch.ones(3,2,2),torch.ones(3,2,2)*2]
        first=pilot._extract(pixels)
        torch.testing.assert_close(first,pilot._extract(pixels))
        self.assertEqual(calls,[2])
        pilot.image_keys=['a','c'];pixels[1]=torch.ones(3,2,2)*3
        changed=pilot._extract(pixels)
        self.assertEqual(calls,[2,1])
        torch.testing.assert_close(changed[0],first[0])
        torch.testing.assert_close(changed[1],torch.tensor([36.,36.]))
        pilot.clear_cache();pilot._extract(pixels)
        self.assertEqual(calls,[2,1,2])

    def test_camera_intervention_keeps_fresh_state_and_instruction(self):
        fresh=Observation(np.ones(8),{'image':np.ones((2,2,3)), 'wrist_image':np.ones((2,2,3))*2},'fresh')
        stale=Observation(np.zeros(8),{'image':np.zeros((2,2,3)), 'wrist_image':np.ones((2,2,3))*3},'old')
        for mode,external,wrist in [('both_fresh',1,2),('external_stale',0,2),
                                     ('wrist_stale',1,3),('both_stale',0,3)]:
            got=runner.camera_obs(fresh,stale,mode)
            np.testing.assert_array_equal(got.state,fresh.state)
            self.assertEqual(got.instruction,'fresh')
            self.assertTrue(np.all(got.images['image']==external))
            self.assertTrue(np.all(got.images['wrist_image']==wrist))
        self.assertTrue(np.all(fresh.images['image']==1))

    def test_gripper_convention_and_no_input_mutation(self):
        actions=np.arange(21,dtype=np.float32).reshape(3,7)/20
        actions[:,-1]=[0,1,.75]
        before=actions.copy()
        result=runner.command_actions(actions)
        np.testing.assert_array_equal(result[:,-1],[1,-1,-1])
        np.testing.assert_array_equal(result[:,:6],actions[:,:6])
        np.testing.assert_array_equal(actions,before)


if __name__=='__main__':unittest.main()
