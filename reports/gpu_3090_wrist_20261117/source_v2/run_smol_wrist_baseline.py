"""Same recorded observations and LIBERO initial states, full-step Small baseline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from run_gr00t_wrist_pilot import ROOT, cases, closed_run, save, append, rows
from vla_eval.smolvla import SmolPilot


class SmolBridge:
    def __init__(self):
        self.pilot=SmolPilot(ROOT/'checkpoints/smolvla_libero',ROOT/'checkpoints/smolvlm_config',device='cuda')
        self.torch=self.pilot.torch
        self.native_horizon=self.output_horizon=50
        self.last={}

    def clear_cache(self):pass

    def predict(self,obs,horizon=16,steps=10,seed=0,cache=False):
        if cache:raise ValueError('Small baseline always uses fresh observations')
        self.pilot.synchronize();start=time.perf_counter()
        with self.torch.autocast('cuda',dtype=self.torch.bfloat16):
            condition=self.pilot.condition(obs)
            actions,_,ae_ms=self.pilot.predict_condition(condition,horizon,steps,seed)
        self.pilot.synchronize()
        self.last=dict(H=horizon,N=steps,total_ms=(time.perf_counter()-start)*1000,
                       ae_ms=ae_ms,backbone_ms=condition.backbone_ms,
                       preprocess_ms=condition.preparation_ms,vision_ms=None,vlm_ms=None)
        # Shared live runner expects the GR00T [0,1]-open convention at its boundary.
        actions[:,-1]=(1-actions[:,-1])/2
        return actions


def main():
    p=argparse.ArgumentParser();p.add_argument('--minutes',type=float,default=8)
    p.add_argument('--output',default='results/wrist_gr00t_20261117/small')
    p.add_argument('--phase',choices=['all','offline','closed'],default='all')
    p.add_argument('--steps',type=int,default=10)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'replays').mkdir(exist_ok=True)
    save(out/'small_status.json',{'status':'running','phase':a.phase})
    config=json.loads((ROOT/'configs/gr00t16_wrist_pilot.json').read_text())
    config['model_label']='SmolVLA'
    config['closed_cells']=[['both_fresh',16,k,a.steps] for k in [4,16,8]]
    config['flow_steps']=a.steps
    config['initial_states']=[0,1]
    save(out/'config.json',config)
    deadline=time.monotonic()+a.minutes*60
    samples,std=cases(config);pilot=SmolBridge()
    with pilot.torch.autocast('cuda',dtype=pilot.torch.bfloat16):
        sanity=pilot.pilot.sanity(samples[0]['fresh'],config['seed'])
    save(out/'manifest.json',dict(model=pilot.pilot.metadata,sanity=sanity,
                                 compute_precision='BF16 autocast, FP32 stored weights'))
    for _ in range(2):pilot.predict(samples[0]['fresh'],steps=a.steps)
    if a.phase in ('all','offline'):
        completed={r['trial_id'] for r in rows(out/'small_offline.jsonl')}
        from run_gr00t_wrist_pilot import command_actions
        for case in samples:
            for h in [16,50,8,4]:
                if time.monotonic()>deadline:break
                trial_id=f"{case['id']}:{h}:{a.steps}"
                if trial_id in completed:continue
                actions=command_actions(pilot.predict(case['fresh'],h,a.steps,config['seed']))
                for k in [4,8,16]:
                    if k>h:continue
                    append(out/'small_offline.jsonl',dict(trial_id=trial_id,case=case['id'],episode=case['episode'],
                        H=h,K=k,N=a.steps,expert_nmse=float(np.mean(((actions[:k]-case['target'][:k])/std)**2)),
                        actions=actions[:k].tolist(),timing=pilot.last))
            print(json.dumps({'event':'small_offline_case','case':case['id']}),flush=True)
    if a.phase in ('all','closed') and time.monotonic()<deadline:
        closed_run(pilot,config,out,deadline)
    save(out/'small_status.json',{'status':'budget_finished' if time.monotonic()>deadline else 'completed'})


if __name__=='__main__':main()
