"""Paired contact/containment H/K evaluation, with predeclared task and stress sets."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import run_gr00t_wrist_pilot as base
import numpy as np


def comparisons(data):
    """Exact paired tests; CIs resample initial states within each fixed task."""
    output=[]
    for scenario in sorted({r['scenario'] for r in data}):
        subset=[r for r in data if r['scenario']==scenario and r['status']=='completed']
        by_cell=defaultdict(dict)
        for r in subset:
            key=(r['suite'],r['task'],r['initial_state'])
            cell=(r['H'],r['K'])
            if key in by_cell[cell]:raise ValueError('Duplicate paired condition')
            by_cell[cell][key]=r
        for a,b in [((16,4),(50,16)),((16,4),(50,4)),((50,4),(50,16))]:
            keys=sorted(by_cell[a].keys() & by_cell[b].keys())
            if not keys:continue
            for metric in ['success','sustained_success']:
                delta=np.array([int(by_cell[a][k][metric])-int(by_cell[b][k][metric]) for k in keys])
                win=int((delta>0).sum());loss=int((delta<0).sum());discordant=win+loss
                pvalue=min(1.,2*sum(math.comb(discordant,i) for i in range(min(win,loss)+1))/2**discordant) if discordant else 1.
                clusters=defaultdict(list)
                for k,d in zip(keys,delta):clusters[k[:2]].append(d)
                rng=np.random.default_rng(719)
                boot=np.zeros(3000)
                for values in clusters.values():
                    v=np.array(values)
                    boot+=v[rng.integers(len(v),size=(3000,len(v)))].sum(axis=1)/len(keys)
                output.append(dict(scenario=scenario,comparison=f'{a[0]}/{a[1]} - {b[0]}/{b[1]}',metric=metric,
                    pairs=len(keys),wins=win,losses=loss,ties=len(keys)-win-loss,
                    delta_pp=float(delta.mean()*100),bootstrap_ci95_pp=(np.quantile(boot,[.025,.975])*100).tolist(),
                    exact_mcnemar_p=pvalue,ci_scope='initial states within these fixed tasks; not task-generalization CI',
                    caveat='exploratory, unadjusted multiple comparisons; all-zero bootstrap is not equivalence evidence'))
    return output


def summarize(out,status):
    data=base.rows(out/'trials.jsonl');groups=defaultdict(list)
    for r in data:
        if r['status']=='completed':groups[(r['scenario'],r['H'],r['K'])].append(r)
    summary=[];per_task=[]
    for (scenario,h,k),rr in sorted(groups.items()):
        times=[x for r in rr for x in r['call_ms']]
        summary.append(dict(scenario=scenario,H=h,K=k,n=len(rr),successes=sum(r['success'] for r in rr),
            sustained=sum(r['sustained_success'] for r in rr),ci95=base.wilson(sum(r['success'] for r in rr),len(rr)),
            p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),
            amortized_ms=sum(times)/sum(r['steps'] for r in rr),
            median_steps=float(np.median([r['steps'] for r in rr])),
            perturbations_applied=sum(r.get('perturbation') is not None for r in rr)))
        for task in sorted({r['task'] for r in rr}):
            tt=[r for r in rr if r['task']==task]
            per_task.append(dict(scenario=scenario,H=h,K=k,task=task,n=len(tt),successes=sum(r['success'] for r in tt)))
    base.save(out/'summary.json',dict(summary=summary,per_task=per_task,paired=comparisons(data)))
    base.save(out/'status.json',dict(status=status,updated_utc=base.now(),completed=sum(len(v) for v in groups.values()),
        errors=sum(r['status']=='error' for r in data),censored=sum(r['status']=='censored' for r in data)))


def displace(env,name,dx):
    obj=env.env.get_object(name)
    joint=obj.joints[-1]
    q=np.asarray(env.sim.data.get_joint_qpos(joint)).copy()
    if q.shape!=(7,):raise ValueError('Perturbation requires free body')
    before=q[:3].copy();q[0]+=dx
    env.sim.data.set_joint_qpos(joint,q)
    env.sim.data.set_joint_qvel(joint,np.zeros(6))
    obs=env.regenerate_obs_from_state(env.get_sim_state())
    return obs,dict(object=name,before_xyz=before.tolist(),after_xyz=q[:3].tolist(),dx_m=dx)


def run_trial(pilot,task,Env,get_path,spec,init,h,k,config,out,scenario,shift,trigger,deadline):
    trial=f'{scenario}:{spec["suite"]}:{task.name}:{init}:{h}:{k}'
    record=dict(trial_id=trial,scenario=scenario,suite=spec['suite'],task=task.name,initial_state=init,
                H=h,K=k,N=config['flow_steps'],status='running',success=False,sustained_success=False,
                steps=0,call_ms=[],perturbation=None,seed=config['seed']+init*10000,
                seed_rule='base + initial_state*10000 + control_step')
    env=None;started=time.monotonic();frames=[]
    try:
        env=Env(bddl_file_name=str(Path(get_path('bddl_files'))/task.problem_folder/task.bddl_file),
                camera_heights=256,camera_widths=256,ignore_done=True)
        env.seed(0);env.reset()
        states=pilot.torch.load(Path(get_path('init_states'))/task.problem_folder/task.init_states_file,
                               weights_only=False,map_location='cpu')
        obs=env.set_init_state(states[init])
        for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
        record['goal_predicates']=env.env.parsed_problem['goal_state']
        pilot.clear_cache()
        while record['steps']<spec['max_steps']:
            if time.monotonic()>deadline:record['status']='censored';break
            actions=base.command_actions(pilot.predict(base.sim_obs(obs,task.language),h,config['flow_steps'],
                                                      record['seed']+record['steps']))
            record['call_ms'].append(pilot.last['total_ms'])
            base.append(out/'calls.jsonl',dict(trial_id=trial,control_step=record['steps'],**pilot.last))
            for action in actions[:k]:
                # Trigger before the next physical action, even inside an existing chunk.
                # Control phases are balanced over K16; K4 also sees nonzero observation delay.
                if trigger is not None and record['steps']==trigger:
                    obs,record['perturbation']=displace(env,config['stress']['target_object'],shift)
                    record['perturbation']['control_step']=record['steps']
                obs,_,success,_=env.step(action.tolist());record['steps']+=1
                if record['steps']%8==0:frames.append(obs['agentview_image'][::-1,::-1].copy())
                if success:
                    record['success']=True
                    # Fixed post-success check; no extra inference and no commanded release.
                    held=[0.]*6+[float(action[-1])];checks=[]
                    for _ in range(10):
                        obs,_,ok,_=env.step(held);checks.append(bool(ok))
                    record['sustained_success']=all(checks)
                    record['post_success_checks']=checks
                    break
                if record['steps']>=spec['max_steps']:break
            if record['success']:break
        if record['status']=='running':record['status']='completed'
        # Keep reviewable movies for first two states and every failure.
        if frames and (init<2 or not record['success']):
            from PIL import Image
            imgs=[Image.fromarray(x).resize((192,192)) for x in frames]
            filename=hashlib.sha256(trial.encode()).hexdigest()[:16]+'.gif'
            imgs[0].save(out/'replays'/filename,save_all=True,append_images=imgs[1:],duration=400,loop=0)
            record['replay']='replays/'+filename
    except Exception:
        record['status']='error';record['error']=traceback.format_exc()
    finally:
        if env is not None:env.close()
    record['wall_s']=time.monotonic()-started
    base.append(out/'trials.jsonl',record);summarize(out,'running')
    print(json.dumps({x:y for x,y in record.items() if x not in ['call_ms','post_success_checks']},ensure_ascii=False),flush=True)
    if record['status']=='error':raise RuntimeError(record['error'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/gr00t16_contact_hk.json')
    p.add_argument('--output',default='results/contact_hk_20261117')
    p.add_argument('--minutes',type=float,default=40)
    p.add_argument('--phase',choices=['all','contact','stress'],default='all')
    args=p.parse_args();config=json.loads(Path(args.config).read_text());out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True);(out/'replays').mkdir(exist_ok=True)
    import fcntl
    lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise ValueError('Changed config requires a new output')
    base.save(out/'config.json',config)
    base.setup_libero()
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    tasks={s:{t.name:t for t in benchmark.get_benchmark_dict()[s]().tasks}
           for s in {x['suite'] for x in config['tasks']} | {config['stress']['suite']}}
    jobs=[]
    if args.phase in ('all','contact'):
        for init in config['initial_states']:
            for spec in config['tasks']:jobs.append((spec,init,'contact',None,None))
    if args.phase in ('all','stress'):
        spec=config['stress']
        for init,trigger in zip(spec['initial_states'],spec['trigger_steps'],strict=True):
            for shift in spec['shift_x_m']:jobs.append((spec,init,f'plate_shift_{shift:+.3f}m',shift,trigger))
    for spec,init,*_ in jobs:
        if spec['name'] not in tasks[spec['suite']]:raise ValueError(spec['name'])
    pilot=base.Gr00t16Pilot(str(base.ROOT/config['checkpoint']),config['device'])
    examples,_=base.cases(json.loads((base.ROOT/'configs/gr00t16_wrist_pilot.json').read_text()))
    parity=pilot.parity(examples[0]['fresh'])
    for h,k in config['cells']:pilot.predict(examples[0]['fresh'],h,config['flow_steps'],config['seed'])
    base.save(out/'manifest.json',dict(started_utc=base.now(),config=config,parity=parity,
        device=pilot.torch.cuda.get_device_name(),torch=pilot.torch.__version__,
        code_sha256={x:hashlib.sha256((base.ROOT/x).read_bytes()).hexdigest() for x in
            ['scripts/run_contact_hk.py','scripts/run_gr00t_wrist_pilot.py','vla_eval/gr00t16_pilot.py']},
        original_manifest=json.loads((base.ROOT/'results/wrist_gr00t_20261117/run/manifest.json').read_text()),
        intended_trials=len(jobs)*len(config['cells'])))
    deadline=time.monotonic()+args.minutes*60
    done={r['trial_id'] for r in base.rows(out/'trials.jsonl') if r['status']=='completed'}
    for i,(spec,init,scenario,shift,trigger) in enumerate(jobs):
        cells=list(config['cells']);np.random.default_rng(config['seed']+i).shuffle(cells)
        task=tasks[spec['suite']][spec['name']]
        for h,k in cells:
            trial=f'{scenario}:{spec["suite"]}:{task.name}:{init}:{h}:{k}'
            if trial in done:continue
            if time.monotonic()>deadline:summarize(out,'budget_finished');return
            run_trial(pilot,task,OffScreenRenderEnv,get_libero_path,spec,init,h,k,config,out,scenario,shift,trigger,deadline)
    summarize(out,'completed')


if __name__=='__main__':main()
