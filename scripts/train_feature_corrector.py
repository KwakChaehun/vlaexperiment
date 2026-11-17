"""Fit an arrival-only low-resolution image/state residual correction ablation."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time
import run_gr00t_wrist_pilot as base
import numpy as np
import torch
from vla_eval.contracts import Observation
from vla_eval.direct_feature_prediction import FeaturePredictor, context, correction_context, load_predictor


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--predictor',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();data=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'corrector.pt').exists():raise ValueError('Use a new output')
    cfg=json.loads((data/'config.json').read_text());torch.manual_seed(cfg['seed']+1);torch.set_num_threads(4)
    forecast,fc=load_predictor(args.predictor);basis=forecast.basis;groups={};start=time.perf_counter()
    for split in ['train','validation']:
        grouped=defaultdict(list)
        with torch.no_grad():
            for row in base.rows(data/'episodes.jsonl'):
                if row['split']!=split:continue
                ep=torch.load(data/row['file'],weights_only=False,map_location='cpu');mask=ep['masks']['image_mask'].cuda()
                for pair in ep['pairs']:
                    frames=ep['frames'];a=frames[pair['origin']];b=frames[pair['target']];prev=frames[pair['previous']]
                    current=a['feature'].cuda();previous=prev['feature'].cuda();target=b['feature'].cuda()
                    ctx=context(a['state'],prev['state'],pair['planned_actions'],pair['horizon'])
                    pred=forecast(current,previous,torch.tensor(ctx,device='cuda')[None],mask)
                    pred=(current.float()+fc['alpha']*(pred.float()-current.float())).to(current.dtype)
                    cc=correction_context(ctx,Observation(a['state'],a['images']),Observation(b['state'],b['images']))
                    full=(target.float()-pred.float())[0];w=.2+.8*mask[0].float()
                    grouped[current.shape[1]].append(dict(x=(pred.float()@basis)[0],p=(current.float()@basis)[0],
                        y=((target.float()-pred.float())@basis)[0],ctx=torch.tensor(cc,device='cuda'),mask=mask[0].float(),
                        energy=(full.square().sum(-1)*w).sum()))
        groups[split]={n:{k:torch.stack([x[k] for x in items]) for k in items[0]} for n,items in grouped.items()}
    allctx=torch.cat([g['ctx'] for g in groups['train'].values()])
    model=FeaturePredictor(basis.clone(),allctx.mean(0),allctx.std(0).clamp_min(.05)).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    def validate():
        numerator=0.;denominator=0.
        with torch.no_grad():
            for g in groups['validation'].values():
                pred=model.coefficients(g['x'],g['p'],g['ctx'],g['mask']);w=.2+.8*g['mask']
                numerator+=float(((pred-g['y']).square().sum(-1)*w).sum());denominator+=float(w.sum())
        return numerator/denominator
    best=validate();best_epoch=0;best_state=copy.deepcopy(model.state_dict());rng=np.random.default_rng(cfg['seed']+1)
    base.append(out/'training.jsonl',dict(epoch=0,val_projected_mse=best))
    for epoch in range(1,cfg['epochs']+1):
        batches=[]
        for length,g in groups['train'].items():
            idx=torch.randperm(len(g['x']),device='cuda');batches += [(length,idx[i:i+16]) for i in range(0,len(idx),16)]
        rng.shuffle(batches)
        for length,idx in batches:
            g=groups['train'][length];pred=model.coefficients(g['x'][idx],g['p'][idx],g['ctx'][idx],g['mask'][idx]);w=.2+.8*g['mask'][idx]
            loss=((pred-g['y'][idx]).square().sum(-1)*w).sum()/w.sum()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if epoch%5==0:
            val=validate()
            if val<best:best=val;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
            base.append(out/'training.jsonl',dict(epoch=epoch,val_projected_mse=val,best_epoch=best_epoch))
    model.load_state_dict(best_state);blends=[0.,.25,.5,1.];errors=np.zeros(4);uncorrected=0.
    with torch.no_grad():
        for g in groups['validation'].values():
            pred=model.coefficients(g['x'],g['p'],g['ctx'],g['mask']);w=.2+.8*g['mask'];energy=g['energy'].sum();uncorrected+=float(energy)
            for i,a in enumerate(blends):errors[i]+=float(energy+(((a*pred-g['y']).square()-g['y'].square()).sum(-1)*w).sum())
    alpha=blends[int(np.argmin(errors))]
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},alpha=alpha,best_epoch=best_epoch),out/'corrector.pt')
    summary=dict(parameters=sum(p.numel() for p in model.parameters()),alpha=alpha,best_epoch=best_epoch,
        validation_energy_relative_to_uncorrected={str(a):float(v/uncorrected) for a,v in zip(blends,errors)},
        train_pairs=sum(len(g['x']) for g in groups['train'].values()),validation_pairs=sum(len(g['x']) for g in groups['validation'].values()),
        test_initial_states_never_loaded=cfg['splits']['test'],wall_s=time.perf_counter()-start,
        predictor_sha256=hashlib.sha256(Path(args.predictor).read_bytes()).hexdigest(),
        corrector_sha256=hashlib.sha256((out/'corrector.pt').read_bytes()).hexdigest(),
        source_hashes={s:hashlib.sha256((base.ROOT/s).read_bytes()).hexdigest() for s in ['scripts/train_feature_corrector.py','vla_eval/direct_feature_prediction.py']})
    base.save(out/'training_summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
