"""Train a compact residual predictor; split by complete initial-state episodes."""
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
from vla_eval.direct_feature_prediction import FeaturePredictor, context


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args(); data=Path(args.data); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'predictor.pt').exists():raise ValueError('Use a new output')
    cfg=json.loads((data/'config.json').read_text()); torch.manual_seed(cfg['seed']); torch.set_num_threads(4)
    rows=base.rows(data/'episodes.jsonl')
    # Test episodes are never opened during fit, hyperparameter selection, or PCA.
    episodes={split:[torch.load(data/r['file'],weights_only=False,map_location='cpu') for r in rows if r['split']==split]
              for split in ['train','validation']}
    rng=np.random.default_rng(cfg['seed']); sample=[]; contexts=[]
    for ep in episodes['train']:
        for pair in ep['pairs']:
            f=ep['frames']; a=f[pair['origin']]; b=f[pair['target']]; prev=f[pair['previous']]
            eligible=torch.where(ep['masks']['image_mask'][0].bool())[0].numpy()
            idx=rng.choice(eligible,size=min(32,len(eligible)),replace=False)
            sample.append((b['feature'][0,idx].float()-a['feature'][0,idx].float()))
            contexts.append(context(a['state'],prev['state'],pair['planned_actions'],pair['horizon']))
    samples=torch.cat(sample).cuda(); start=time.perf_counter()
    _,_,basis=torch.pca_lowrank(samples,q=cfg['rank'],center=False,niter=3)
    basis=basis.contiguous(); basis_fraction=float((samples@basis).square().sum()/samples.square().sum())
    ctx=torch.tensor(np.stack(contexts),device='cuda')
    model=FeaturePredictor(basis,ctx.mean(0),ctx.std(0).clamp_min(.05)).cuda()
    del samples, sample
    groups={}
    for split,eps in episodes.items():
        grouped=defaultdict(list)
        with torch.no_grad():
            for ep in eps:
                projected={step:frame['feature'].cuda().float()@basis for step,frame in ep['frames'].items()}
                for pair in ep['pairs']:
                    f=ep['frames']; a=f[pair['origin']]; b=f[pair['target']]; prev=f[pair['previous']]
                    cur=projected[pair['origin']][0]; old=projected[pair['previous']][0]
                    target=projected[pair['target']][0]-cur
                    mask=ep['masks']['image_mask'][0].cuda().float()
                    full=(b['feature'].float()-a['feature'].float()).cuda()[0]
                    history=(a['feature'].float()-prev['feature'].float()).cuda()[0]
                    weight=.2+.8*mask
                    grouped[cur.shape[0]].append(dict(x=cur,p=old,y=target,mask=mask,
                        ctx=torch.tensor(context(a['state'],prev['state'],pair['planned_actions'],pair['horizon']),device='cuda'),
                        energy=(full.square().sum(-1)*weight).sum(),
                        temporal_errors=torch.stack([((full-alpha*pair['horizon']/16*history).square().sum(-1)*weight).sum()
                                                     for alpha in [0.,.25,.5,1.]])))
        groups[split]={length:{key:torch.stack([r[key] for r in items]) for key in items[0]}
                       for length,items in grouped.items()}
    del episodes
    def validation():
        total=0.; denominator=0.
        with torch.no_grad():
            for g in groups['validation'].values():
                pred=model.coefficients(g['x'],g['p'],g['ctx'],g['mask'])
                w=.2+.8*g['mask']
                total+=float(((pred-g['y']).square().sum(-1)*w).sum())
                denominator+=float(w.sum())
        return total/denominator
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=validation(); best_epoch=0; best_state=copy.deepcopy(model.state_dict())
    base.append(out/'training.jsonl',dict(epoch=0,val_projected_mse=best))
    for epoch in range(1,cfg['epochs']+1):
        model.train(); losses=[]; batches=[]
        for length,g in groups['train'].items():
            indices=torch.randperm(len(g['x']),device='cuda')
            batches += [(length,indices[i:i+16]) for i in range(0,len(indices),16)]
        rng.shuffle(batches)
        for length,idx in batches:
            g=groups['train'][length]; pred=model.coefficients(g['x'][idx],g['p'][idx],g['ctx'][idx],g['mask'][idx])
            w=.2+.8*g['mask'][idx]
            loss=((pred-g['y'][idx]).square().sum(-1)*w).sum()/w.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            losses.append(float(loss.detach()))
        if epoch%5==0:
            model.eval(); val=validation()
            if val<best:best=val;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
            row=dict(epoch=epoch,train_projected_mse=float(np.mean(losses)),val_projected_mse=val,best_epoch=best_epoch)
            base.append(out/'training.jsonl',row);print(json.dumps(row),flush=True)
    model.load_state_dict(best_state);model.eval()
    blends=[0.,.25,.5,1.]; errors=np.zeros(4); temporal=np.zeros(4); stale_energy=0.
    with torch.no_grad():
        for g in groups['validation'].values():
            pred=model.coefficients(g['x'],g['p'],g['ctx'],g['mask']);w=.2+.8*g['mask']
            base_energy=g['energy'].sum(); stale_energy+=float(base_energy)
            for i,alpha in enumerate(blends):
                # Full-dimensional residual norm, including energy outside PCA span.
                errors[i]+=float(base_energy+(((alpha*pred-g['y']).square()-g['y'].square()).sum(-1)*w).sum())
            temporal+=g['temporal_errors'].sum(0).cpu().numpy()
    chosen=float(blends[int(np.argmin(errors))]); temporal_alpha=float(blends[int(np.argmin(temporal))])
    checkpoint=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    alpha=chosen,temporal_alpha=temporal_alpha,best_epoch=best_epoch)
    torch.save(checkpoint,out/'predictor.pt')
    summary=dict(parameters=sum(p.numel() for p in model.parameters()),rank=cfg['rank'],
        train_pairs=sum(len(g['x']) for g in groups['train'].values()),validation_pairs=sum(len(g['x']) for g in groups['validation'].values()),
        training_initial_states=cfg['splits']['train'],validation_initial_states=cfg['splits']['validation'],
        test_initial_states_never_loaded=cfg['splits']['test'],pca_training_image_residual_energy_fraction=basis_fraction,
        best_epoch=best_epoch,blend_alpha=chosen,temporal_alpha=temporal_alpha,
        validation_residual_energy_relative_to_stale={str(a):float(e/stale_energy) for a,e in zip(blends,errors)},
        validation_temporal_energy_relative_to_stale={str(a):float(e/stale_energy) for a,e in zip(blends,temporal)},
        training_wall_s=time.perf_counter()-start,
        checkpoint_sha256=hashlib.sha256((out/'predictor.pt').read_bytes()).hexdigest(),
        source_hashes={s:hashlib.sha256((base.ROOT/s).read_bytes()).hexdigest() for s in ['scripts/train_direct_features.py','vla_eval/direct_feature_prediction.py']})
    base.save(out/'training_summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
