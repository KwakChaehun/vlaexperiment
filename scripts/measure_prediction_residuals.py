"""Untimed dense-reference sparsity diagnostics on saved predicted/actual pairs.

These expensive diagnostics NEVER select tokens for the timed controller. Both
streams are fully recomputed to inspect how concentrated prediction error is.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import run_gr00t_wrist_pilot as base
import numpy as np
from vla_eval.contracts import Observation
from vla_eval.gr00t16_arrival import Gr00t16Arrival
from vla_eval.speculative_features import PredictedMLPCache


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);args=p.parse_args();out=Path(args.input)
    if (out/'residuals.jsonl').exists():raise ValueError('Diagnostics already exist')
    pilot=Gr00t16Arrival(str(base.ROOT/'checkpoints/gr00t_n16_libero'));cache=PredictedMLPCache(pilot)
    pairs=[r for r in base.rows(out/'calls.jsonl') if 'observation_pair' in r]
    for r in pairs:
        data=np.load(out/r['observation_pair'])
        def obs(prefix):
            return Observation(data[prefix+'_state'],dict(image=data[prefix+'_external'],wrist_image=data[prefix+'_wrist']),str(data['instruction']))
        pred=cache.capture(obs('predicted'),int(data['seed']))
        actual=cache.capture(obs('actual'),int(data['seed']))
        groups=defaultdict(list)
        for name,(pred_input,_) in pred['mlps'].items():
            dense=actual['mlps'][name][0]
            score=(dense.float()-pred_input.float()).square().mean(-1).flatten()
            ordered=score.sort(descending=True).values
            total=float(ordered.sum())
            n=ordered.numel()
            row=dict(layer=name,n=n,energy=total)
            if total>0:
                cumulative=ordered.cumsum(0)
                row['top25_energy_fraction']=float(ordered[:max(1,int(np.ceil(.25*n)))].sum()/total)
                row['top50_energy_fraction']=float(ordered[:max(1,int(np.ceil(.50*n)))].sum()/total)
                row['fraction_for_90pct_energy']=(int(pilot.torch.searchsorted(cumulative,.9*total))+1)/n
                row['fraction_for_99pct_energy']=(int(pilot.torch.searchsorted(cumulative,.99*total))+1)/n
            else:
                row.update(top25_energy_fraction=1.,top50_energy_fraction=1.,fraction_for_90pct_energy=0.,fraction_for_99pct_energy=0.)
            groups['vision' if name.startswith('vision_model.') else 'vlm'].append(row)
        summary={g:{k:statistics.mean(x[k] for x in rr) for k in ['energy','top25_energy_fraction','top50_energy_fraction','fraction_for_90pct_energy','fraction_for_99pct_energy']} for g,rr in groups.items()}
        result={k:r[k] for k in ['trial','case','method','step','prediction_age_steps','prediction_position_mm','prediction_pixel_mae','observation_pair']}
        result.update(groups=summary,layers=dict(groups),scope='offline dense-reference MLP input residual; never part of timed token selector')
        teacher=base.command_actions(actual['actions'])[:4]
        agreement={}
        for ratio in [0.,.25,.50,1.]:
            actions,_,_=cache.correct(obs('actual'),pred,int(data['seed']),ratio)
            delta=base.command_actions(actions)[:4]-teacher
            agreement[str(ratio)]=dict(action_rmse=float(np.sqrt(np.mean(delta**2))),
                translation_action_rmse=float(np.sqrt(np.mean(delta[:,:3]**2))),
                gripper_agreement=float(np.mean(delta[:,-1]==0)))
        action1,_=pilot.finish(obs('actual'),int(data['seed']),steps=1)
        delta=base.command_actions(action1)[:4]-teacher
        agreement['fresh1']=dict(action_rmse=float(np.sqrt(np.mean(delta**2))),
            translation_action_rmse=float(np.sqrt(np.mean(delta[:,:3]**2))),
            gripper_agreement=float(np.mean(delta[:,-1]==0)))
        result['action_agreement']=agreement
        base.append(out/'residuals.jsonl',result)
        del pred,actual
    rows=base.rows(out/'residuals.jsonl');summary=[]
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method and r['prediction_age_steps']>0]
        for group in ['vision','vlm']:
            summary.append(dict(method=method,group=group,frames=len(rr),**{
                k:statistics.mean(r['groups'][group][k] for r in rr) for k in ['top25_energy_fraction','top50_energy_fraction','fraction_for_90pct_energy','fraction_for_99pct_energy']}))
    base.save(out/'residual_summary.json',summary)
    print(json.dumps(dict(event='residual_diagnostics_done',pairs=len(rows),summary=summary)),flush=True)
    cache.close()


if __name__=='__main__':main()
