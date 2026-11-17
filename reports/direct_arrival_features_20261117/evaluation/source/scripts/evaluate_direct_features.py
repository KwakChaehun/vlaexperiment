"""Held-out feature/action agreement and matched synchronous contact rollouts.

Forecast arms refresh full VLM every OTHER K16 chunk. They predict one arrival,
not a whole bank. All preparation/arrival calls are charged; no hidden overlap.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import run_gr00t_wrist_pilot as base
import numpy as np
import torch
from vla_eval.contracts import Observation
from vla_eval.direct_feature_prediction import DirectFeaturePilot,context,correction_context,load_predictor


def observation(frame, instruction):
    return Observation(frame['state'],frame['images'],instruction)


class Engine:
    def __init__(self,pilot,predictor,corrector):
        self.pilot=pilot;self.model,self.cfg=load_predictor(predictor)
        self.corrector,self.ccfg=load_predictor(corrector)

    def forecast(self, method, feature, previous, origin, old_state, actions, horizon, masks):
        p=self.pilot;p.synchronize();begin=time.perf_counter()
        ctx=context(origin.state,old_state,actions,horizon)
        if method=='stale':pred=feature.clone()
        elif method=='temporal':
            pred=(feature.float()+self.cfg['temporal_alpha']*horizon/16*(feature.float()-previous.float())).to(feature.dtype)
        else:
            raw=self.model(feature,previous,torch.tensor(ctx,device=feature.device)[None],masks['image_mask'])
            pred=(feature.float()+self.cfg['alpha']*(raw.float()-feature.float())).to(feature.dtype)
        p.synchronize();elapsed=(time.perf_counter()-begin)*1000
        return dict(feature=pred,origin_feature=feature,origin=origin,context=ctx,masks=masks,prepare_ms=elapsed)

    def arrive(self, method, forecast, actual, seed):
        p=self.pilot;p.synchronize();begin=time.perf_counter();feature=forecast['feature'];correction_ms=0.
        if method=='corrected':
            c=correction_context(forecast['context'],forecast['origin'],actual)
            raw=self.corrector(feature,forecast['origin_feature'],torch.tensor(c,device=feature.device)[None],forecast['masks']['image_mask'])
            feature=(feature.float()+self.ccfg['alpha']*(raw.float()-feature.float())).to(feature.dtype)
            p.synchronize();correction_ms=(time.perf_counter()-begin)*1000
        cond=p.condition_from_features(actual,feature,forecast['masks'])
        actions=p.decode(cond,p.integrate(cond,50,4,seed));p.synchronize()
        return actions,(time.perf_counter()-begin)*1000,feature,correction_ms


def offline(pilot,engine,data,out):
    completed=0
    for ep_row in base.rows(data/'episodes.jsonl'):
        if ep_row['split']!='test':continue
        ep=torch.load(data/ep_row['file'],weights_only=False,map_location='cpu')
        masks={k:v.cuda() for k,v in ep['masks'].items()}
        for pair in ep['pairs']:
            frames=ep['frames'];a=frames[pair['origin']];prev=frames[pair['previous']]
            current=a['feature'].cuda();previous=prev['feature'].cuda();origin=observation(a,ep['instruction'])
            # Construct all forecasts BEFORE accessing arrival input in this loop.
            predictions={m:engine.forecast(m,current,previous,origin,prev['state'],pair['planned_actions'],pair['horizon'],masks)
                         for m in ['stale','temporal','direct']}
            b=frames[pair['target']];actual=observation(b,ep['instruction']);truth=b['feature'].cuda();seed=ep['seed']+pair['target']
            teacher,fresh_ms=pilot.finish(actual,seed,steps=4)
            # Evaluation-only ceiling for the fixed low-rank representation.
            # This deliberately uses true future features; NEVER a controller arm.
            basis=engine.model.basis
            lowrank=(current.float()+((truth.float()-current.float())@basis)@basis.T).to(current.dtype)
            predictions['oracle_rank32']=dict(predictions['stale'],feature=lowrank,prepare_ms=0.)
            variants={}
            for m in ['stale','temporal','direct','corrected','oracle_rank32']:
                pred=predictions['direct' if m=='corrected' else m]
                actions,post,feature,corr=engine.arrive(m,pred,actual,seed)
                diff=feature.float()-truth.float();image=masks['image_mask'].bool()
                variants[m]=dict(actions=actions,feature_sse=float(diff.square().sum()),
                    image_feature_sse=float(diff[image].square().sum()),feature_elements=diff.numel(),image_elements=diff[image].numel(),
                    prepare_ms=pred['prepare_ms'],arrival_ms=post,correction_ms=corr,charged_ms=post+pred['prepare_ms'])
            for n in [1,2]:
                actions,ms=pilot.finish(actual,seed,steps=n)
                variants['fresh'+str(n)]=dict(actions=actions,prepare_ms=0.,arrival_ms=ms,charged_ms=ms)
            variants['fresh4']=dict(actions=teacher,prepare_ms=0.,arrival_ms=fresh_ms,charged_ms=fresh_ms)
            for method,v in variants.items():
                actions=v.pop('actions');diff=actions-teacher
                row=dict(episode=ep_row['file'],task=ep['task'],initial_state=ep['initial_state'],
                         origin=pair['origin'],target=pair['target'],horizon=pair['horizon'],method=method,
                         origin_before_precision=pair['origin']<ep['spec']['start'],
                         evaluation_only_oracle=method=='oracle_rank32',
                         action_mse=float(np.mean(diff**2)),translation_mse=float(np.mean(diff[:,:3]**2)),
                         gripper_agreement=float(np.mean((actions[:,-1]>.5)==(teacher[:,-1]>.5))),**v)
                base.append(out/'offline.jsonl',row)
            completed+=1
        print(json.dumps(dict(phase='offline',episode=ep_row['file'],completed_pairs=completed)),flush=True)
    rows=base.rows(out/'offline.jsonl');summary=[]
    for horizon in [4,8,16]:
        for method in sorted({r['method'] for r in rows}):
            rr=[r for r in rows if r['horizon']==horizon and r['method']==method]
            item=dict(horizon=horizon,method=method,pairs=len(rr),episodes=len({r['episode'] for r in rr}),
                      action_rmse=float(np.sqrt(np.mean([r['action_mse'] for r in rr]))),
                      translation_rmse=float(np.sqrt(np.mean([r['translation_mse'] for r in rr]))),
                      gripper_agreement=float(np.mean([r['gripper_agreement'] for r in rr])))
            for k in ['prepare_ms','arrival_ms','charged_ms']:
                item[k+'_p50']=float(np.median([r[k] for r in rr]));item[k+'_p95']=float(np.percentile([r[k] for r in rr],95))
            if 'feature_sse' in rr[0]:
                stale=[r for r in rows if r['horizon']==horizon and r['method']=='stale']
                item['feature_energy_vs_stale']=sum(r['feature_sse'] for r in rr)/sum(r['feature_sse'] for r in stale)
                item['image_feature_energy_vs_stale']=sum(r['image_feature_sse'] for r in rr)/sum(r['image_feature_sse'] for r in stale)
            summary.append(item)
    base.save(out/'offline_summary.json',summary)


def closed(pilot,engine,data,out,cfg,arrival_start=False):
    base.setup_libero()
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suites={s:benchmark.get_benchmark_dict()[s]() for s in {t['suite'] for t in cfg['tasks']}}
    methods=['fresh4','fresh2','fresh1','stale','direct','corrected']
    for ep_row in base.rows(data/'episodes.jsonl'):
        if ep_row['split']!='test':continue
        ep=torch.load(data/ep_row['file'],weights_only=False,map_location='cpu');spec=ep['spec']
        task=next(t for t in suites[spec['suite']].tasks if t.name==spec['name'])
        initial=torch.load(Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file,weights_only=False,map_location='cpu')[ep['initial_state']]
        order=methods.copy();np.random.default_rng(ep['seed']+ep['task_index']).shuffle(order)
        for method in order:
            env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                                  camera_heights=256,camera_widths=256,ignore_done=True)
            try:
                env.seed(0);env.reset();obs=env.set_init_state(initial)
                for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
                initial_pending=None;initial_prepare_ms=0.;prefix_previous=None;plan_error=None
                for prefix_step,action in enumerate(ep['applied_actions'][:spec['start']]):
                    if arrival_start and method in ['stale','direct','corrected']:
                        if prefix_step==spec['start']-32:
                            # Reconstruct an actually observed past cache from the
                            # shared prefix, never from a future frame.
                            prefix_view=base.sim_obs(obs,task.language)
                            prefix_previous=(pilot.condition(prefix_view)[2].backbone_features,prefix_view.state)
                        if prefix_step==spec['start']-16:
                            origin=base.sim_obs(obs,task.language);stored=ep['frames'][prefix_step]
                            assert all(np.array_equal(origin.images[k],stored['images'][k]) for k in origin.images)
                            assert np.array_equal(origin.state,stored['state'])
                            masks={k:v.cuda() for k,v in ep['masks'].items()}
                            # These commands were already committed by the common
                            # Fresh controller at this origin, before arrival.
                            planned=ep['applied_actions'][prefix_step:spec['start']]
                            origin_condition=pilot.condition_from_features(origin,stored['feature'].cuda(),masks)
                            reproduced=base.command_actions(pilot.decode(origin_condition,pilot.integrate(origin_condition,50,4,ep['seed']+prefix_step)))
                            plan_error=float(np.max(np.abs(reproduced[:16]-planned)))
                            assert plan_error<=1e-6, 'Initial forecast used commands not known at origin'
                            initial_pending=engine.forecast(method,stored['feature'].cuda(),prefix_previous[0],
                                                            origin,prefix_previous[1],planned,16,masks)
                            initial_prepare_ms=initial_pending['prepare_ms']
                    obs,_,_,_=env.step(action.tolist())
                view=base.sim_obs(obs,task.language);anchor=ep['frames'][spec['start']]
                state_error=float(np.max(np.abs(env.get_sim_state()-anchor['sim_state'])))
                rgb_equal=all(np.array_equal(view.images[k],anchor['images'][k]) for k in view.images)
                assert state_error<1e-7 and rgb_equal
                step=spec['start'];success=bool(env.check_success());pending=initial_pending
                # This is an observed past prefix feature, not an actual future feature.
                past=ep['frames'][spec['start']-16]
                previous_feature=past['feature'].cuda();previous_state=past['state'].copy()
                calls=[];trace=[];begin=time.perf_counter()
                while step<spec['end'] and not success:
                    view=base.sim_obs(obs,task.language);seed=ep['seed']+step
                    pilot.synchronize();call_start=time.perf_counter();prepare_ms=0.;corr=0.
                    if method.startswith('fresh'):
                        actions,arrival_ms=pilot.finish(view,seed,steps=int(method[-1]));path=method
                    elif pending is not None:
                        actions,arrival_ms,arrival_feature,corr=engine.arrive(method,pending,view,seed)
                        path='arrival';pending=None
                        previous_feature=arrival_feature;previous_state=view.state.copy()
                        prepare_ms=initial_prepare_ms;initial_prepare_ms=0.
                    else:
                        cond=pilot.condition(view)
                        actions=pilot.decode(cond,pilot.integrate(cond,50,4,seed));pilot.synchronize()
                        arrival_ms=(time.perf_counter()-call_start)*1000;path='refresh'
                        feature=cond[2].backbone_features
                        masks={k:getattr(cond[0],k) for k in ['image_mask','backbone_attention_mask']}
                        # Past arrival feature is predicted/corrected after the first
                        # shared prefix. No fresh future labels enter this controller.
                        if step+16<spec['end']:
                            pending=engine.forecast(method,feature,previous_feature,view,previous_state,
                                                    base.command_actions(actions),16,masks)
                            prepare_ms=pending['prepare_ms']
                    command=base.command_actions(actions)
                    calls.append(dict(step=step,path=path,arrival_ms=arrival_ms,prepare_ms=prepare_ms,correction_ms=corr,
                                      charged_ms=arrival_ms+prepare_ms))
                    for action in command[:min(16,spec['end']-step)]:
                        applied=action.copy();applied[:3]*=cfg['gain']
                        obs,_,ok,_=env.step(applied.tolist());step+=1;success|=bool(ok);trace.append(applied.tolist())
                goal_at_end=bool(env.check_success());stable=goal_at_end
                hold=[0.]*6+[trace[-1][-1] if trace else -1.]
                for _ in range(10):obs,_,_,_=env.step(hold);stable &= bool(env.check_success())
                row=dict(method=method,episode=ep_row['file'],task=task.name,initial_state=ep['initial_state'],
                    arrival_start=arrival_start,
                    initial_plan_reproduction_max_abs=plan_error,
                    success=success,goal_at_end=goal_at_end,stable_success=stable,steps=step-spec['start'],
                    paired_sim_max_abs=state_error,paired_images_equal=rgb_equal,
                    calls=calls,action_trace=trace,wall_s=time.perf_counter()-begin,
                    total_service_ms=sum(c['charged_ms'] for c in calls),arrival_ms_sum=sum(c['arrival_ms'] for c in calls),
                    preparation_ms_sum=sum(c['prepare_ms'] for c in calls),status='completed')
                base.append(out/'closed_loop.jsonl',row)
                print(json.dumps({k:row[k] for k in ['method','task','initial_state','success','stable_success','steps','total_service_ms']}),flush=True)
            finally:env.close()
    rows=base.rows(out/'closed_loop.jsonl');summary=[]
    for method in methods:
        rr=[r for r in rows if r['method']==method];cc=[c for r in rr for c in r['calls']]
        arrivals=[c['arrival_ms'] for c in cc if c['path']=='arrival']
        summary.append(dict(method=method,trials=len(rr),successes=sum(r['success'] for r in rr),
                            stable_successes=sum(r['stable_success'] for r in rr),
                            mean_service_ms=float(np.mean([r['total_service_ms'] for r in rr])),
                            charged_ms_per_executed_action=sum(r['total_service_ms'] for r in rr)/sum(r['steps'] for r in rr),
                            arrival_only_p50_ms=float(np.median(arrivals)) if arrivals else None,
                            mean_prepare_ms_per_call=float(np.mean([c['prepare_ms'] for c in cc])),
                            all_call_p50_ms=float(np.median([c['charged_ms'] for c in cc]))))
    base.save(out/'closed_summary.json',summary)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--predictor',required=True)
    ap.add_argument('--corrector',required=True);ap.add_argument('--output',required=True);ap.add_argument('--phase',choices=['offline','closed'],required=True)
    ap.add_argument('--arrival-start',action='store_true')
    args=ap.parse_args();data=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    target='offline.jsonl' if args.phase=='offline' else 'closed_loop.jsonl'
    if (out/target).exists():raise ValueError('Use a new output')
    cfg=json.loads((data/'config.json').read_text());pilot=DirectFeaturePilot(str(base.ROOT/cfg['checkpoint']))
    engine=Engine(pilot,args.predictor,args.corrector)
    sources=['scripts/evaluate_direct_features.py','vla_eval/direct_feature_prediction.py']
    base.save(out/(args.phase+'_manifest.json'),dict(started_utc=base.now(),source_hashes={s:hashlib.sha256((base.ROOT/s).read_bytes()).hexdigest() for s in sources},
        predictor_sha256=hashlib.sha256(Path(args.predictor).read_bytes()).hexdigest(),
        corrector_sha256=hashlib.sha256(Path(args.corrector).read_bytes()).hexdigest(),H=50,K=16,
        scope='Synchronous physics; prediction/arrival costs both charged. No asynchronous speedup claim. Every-other-chunk full VLM refresh in forecast arms.',
        arrival_start=args.arrival_start,test_initial_states=cfg['splits']['test']))
    with torch.inference_mode():
        example=next(r for r in base.rows(data/'episodes.jsonl') if r['split']=='train')
        ep=torch.load(data/example['file'],weights_only=False,map_location='cpu');frame=ep['frames'][ep['spec']['start']]
        for _ in range(4):pilot.finish(observation(frame,ep['instruction']),37)
        if args.phase=='offline':offline(pilot,engine,data,out)
        else:closed(pilot,engine,data,out,cfg,args.arrival_start)
    base.save(out/(args.phase+'_status.json'),dict(status='completed',updated_utc=base.now()))


if __name__=='__main__':main()
