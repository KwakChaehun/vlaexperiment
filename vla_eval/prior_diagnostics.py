"""Controlled diagnosis of stale visual/state conditions and heuristic re-noising.

Arrival RGB/state in a precomputed prior are privileged diagnostics. They must
never be described as a deployable forecast or as a learned feature predictor.
"""
from __future__ import annotations
from collections import defaultdict
import gc,json,time
import numpy as np
from .contracts import Observation


def run(model,out,cases,config,deadline,progress):
    from scripts.research_pilot import append,scores
    directory=out/'prior_diagnostics';directory.mkdir(exist_ok=True)
    done_path=directory/'completed_cases.jsonl'
    done={json.loads(s)['case_id'] for s in done_path.read_text().splitlines()} if done_path.exists() else set()
    grouped=defaultdict(list)
    for case in cases:
        if case['split']==config['split']:grouped[case['episode_id']].append(case)
    selected=[grouped[e][i] for i in range(max(map(len,grouped.values()),default=0))
              for e in sorted(grouped) if i<len(grouped[e])][:config['max_cases']]
    h,n=config['horizon'],config['total_steps']
    refinements=config['refine_steps'];seed=config['old_noise_seed']
    if not (1<=h<=model.native_horizon and n>0 and refinements and all(0<r<n for r in refinements)):
        raise ValueError('Invalid prior diagnostic integration budget')
    noise=model.noise(h,seed);independent=model.noise(h,config['independent_renoise_seed'])
    plan={'cases':[c['id'] for c in selected],'horizon':h,'total_steps':n,'refine_steps':refinements,
          'old_noise_seed':seed,'independent_renoise_seed':config['independent_renoise_seed'],
          'score_prefix':4,'diagnostic_only':True,
          'privileged_input':'arrival images/states in oracle prior conditions',
          'meaning':'teacher agreement and expert action error; not closed-loop success or grasp precision'}
    (directory/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    for case in selected:
        if case['id'] in done:continue
        if time.monotonic()>deadline:return False
        progress(case['id'],'fresh_baselines',len(done),len(selected))
        current=model.condition(case['actual'])
        teacher,full,_=model.predict_condition(current,h,n,seed)
        records=[]
        def emit(method,condition,actions,post_steps,**extra):
            k=4
            metric=scores(actions[:k],case['target'][:k],case['mask'][:k],model.action_std)
            records.append({'case_id':case['id'],'episode_id':case['episode_id'],'phase':case['phase'],
                'method':method,'prior_condition':condition,'H':h,'K_scored':k,'N_reference':n,
                'post_velocity_evaluations':post_steps,'diagnostic_only':True,
                'expert_nmse':metric['normalized_action_mse'],
                'teacher_nmse':float(np.mean(((actions[:k]-teacher[:k])/model.action_std)**2)),
                'gripper_accuracy':metric['gripper_accuracy'],'first_actions':actions[:k].tolist(),**extra})
        emit('fresh_full','none',teacher,n)
        for r in refinements:
            a,_,ms=model.predict_condition(current,h,r,seed)
            emit('fresh_few','none',a,r,ae_ms=ms,backbone_ms=current.backbone_ms)
        # The actual intermediate latent should recover the full solution under
        # unchanged conditioning. Re-noising the completed action need not do so.
        r=refinements[0]
        partial,_=model.integrate(current,h,n,seed,stop_step=n-r)
        resumed,ms=model.integrate(current,h,n,seed,latent=partial,start_step=n-r)
        parity=float((resumed-full).abs().max())
        if parity>1e-5:raise ValueError(f'Fresh split-trajectory parity failed: {parity}')
        emit('fresh_split_identity','arrival_rgb_arrival_state',model.decode(resumed),r,
             latent_max_abs_difference=parity,ae_ms=ms)
        variants=[('stale_rgb_predicted_state',case['forecast'],False),
                  ('stale_rgb_arrival_state',case['oracle_proprio'],True),
                  ('arrival_rgb_predicted_state',Observation(case['forecast'].state,case['actual'].images,case['actual'].instruction),True),
                  ('arrival_rgb_arrival_state',case['actual'],True),
                  ('stale_rgb_stale_state',case['stale'],False)]
        for name,obs,privileged in variants:
            if time.monotonic()>deadline:return False
            progress(case['id'],name,len(done),len(selected))
            cond=current if name=='arrival_rgb_arrival_state' else model.condition(obs)
            cuts=sorted({0,n,*[n-r for r in refinements]})
            latent=None;snapshots={};prior_ms=0.
            for lo,hi in zip(cuts,cuts[1:]):
                latent,ms=model.integrate(cond,h,n,seed,latent=latent,start_step=lo,stop_step=hi)
                snapshots[hi]=latent;prior_ms+=ms
            prior_full=latent
            emit('prior_unmodified',name,model.decode(prior_full),0,privileged_prior=privileged,
                 pre_ae_ms=prior_ms,pre_backbone_ms=cond.backbone_ms)
            for r in refinements:
                if time.monotonic()>deadline:return False
                s=r/n
                for label,epsilon in [('renoise_original_noise',noise),('renoise_independent_noise',independent)]:
                    initial=(1-s)*prior_full+s*epsilon
                    refined,ms=model.integrate(current,h,n,seed,latent=initial,start_step=n-r)
                    emit(label,name,model.decode(refined),r,privileged_prior=privileged,
                         ae_ms=ms,backbone_ms=current.backbone_ms,
                         pre_ae_ms=prior_ms,restart_flow_time=s)
                continued,ms=model.integrate(current,h,n,seed,latent=snapshots[n-r],start_step=n-r)
                emit('continue_intermediate',name,model.decode(continued),r,privileged_prior=privileged,
                     ae_ms=ms,backbone_ms=current.backbone_ms,restart_flow_time=s)
            if cond is not current:del cond
            del snapshots,prior_full,latent
        # Commit a whole case so live summaries always compare balanced conditions.
        for record in records:append(directory/'trials.jsonl',record)
        append(done_path,{'case_id':case['id']});done.add(case['id'])
        progress(case['id'],'case_completed',len(done),len(selected))
        del current,full,teacher,resumed,partial;gc.collect()
    return True
