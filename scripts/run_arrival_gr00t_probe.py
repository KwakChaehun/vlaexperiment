"""Small paired GR00T arrival probe; privileged future inputs remain labelled."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import run_gr00t_wrist_pilot as base
import numpy as np
from vla_eval.contracts import Observation
from vla_eval.gr00t16_arrival import Gr00t16Arrival


def forecast(history,lookahead):
    """Only observations at or before the pre-arrival timestamp are accepted."""
    from scipy.spatial.transform import Rotation
    if len(history)<=lookahead:raise ValueError('Insufficient causal history')
    old,current=history[-1-lookahead],history[-1]
    state=current.state.copy()
    state[:3]+=current.state[:3]-old.state[:3]
    a,b=Rotation.from_rotvec(old.state[3:6]),Rotation.from_rotvec(current.state[3:6])
    q=((b*a.inv())*b).as_quat()
    # Same noncanonical axis-angle convention as the LIBERO policy input.
    w=np.clip(q[3],-1,1);den=np.sqrt(1-w*w)
    state[3:6]=np.zeros(3) if np.isclose(den,0) else q[:3]*2*np.arccos(w)/den
    return Observation(state,current.images,current.instruction)


def summarize(out,status):
    d=base.rows(out/'diagnostics.jsonl');r=base.rows(out/'rollouts.jsonl')
    groups=defaultdict(list);result=[]
    for row in d:groups[row['method']].append(row)
    for method,dd in groups.items():
        rr=[x for x in r if x['method']==method and x['status']=='completed']
        item=dict(method=method,cases=len(dd),completed=len(rr),successes=sum(x['success'] for x in rr),
                  arrival_already_success=sum(x['arrival_already_success'] for x in rr))
        for key in ['precompute_ms','post_ms','analytical_arrival_delay_ms','teacher_nmse_first16','teacher_nmse_first4']:
            item[key]=float(np.mean([x[key] for x in dd]))
        item['post_p50_ms']=float(np.median([x['post_ms'] for x in dd]))
        result.append(item)
    base.save(out/'summary.json',result)
    base.save(out/'status.json',dict(status=status,updated_utc=base.now(),diagnostics=len(d),rollouts=len(r),
        errors=sum(x['status']=='error' for x in r)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/gr00t16_arrival_probe.json')
    p.add_argument('--output',default='results/arrival_gr00t_20261117');p.add_argument('--minutes',type=float,default=30)
    args=p.parse_args();config=json.loads(Path(args.config).read_text());out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    import fcntl
    lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'rollouts.jsonl').exists():raise ValueError('Use a new output; probe does not resume partial cases')
    base.save(out/'config.json',config);base.setup_libero()
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suites={s:benchmark.get_benchmark_dict()[s]() for s in {x['suite'] for x in config['tasks']}}
    pilot=Gr00t16Arrival(str(base.ROOT/config['checkpoint']),config['device'])
    examples,std=base.cases(json.loads((base.ROOT/'configs/gr00t16_wrist_pilot.json').read_text()))
    parity=pilot.split_parity(examples[0]['fresh'])
    base.save(out/'manifest.json',dict(started_utc=base.now(),config=config,parity=parity,expert_std=std.tolist(),
        device=pilot.torch.cuda.get_device_name(),torch=pilot.torch.__version__,
        source_hashes={s:hashlib.sha256((base.ROOT/s).read_bytes()).hexdigest() for s in
            ['scripts/run_arrival_gr00t_probe.py','vla_eval/gr00t16_arrival.py','vla_eval/gr00t16_pilot.py']},
        baseline_manifest=json.loads((base.ROOT/'results/contact_hk_20261117/manifest.json').read_text())))
    deadline=time.monotonic()+args.minutes*60
    try:
        for init in config['initial_states']:
            for ti,spec in enumerate(config['tasks']):
                if time.monotonic()>deadline:summarize(out,'budget_finished');return
                task=next(t for t in suites[spec['suite']].tasks if t.name==spec['name'])
                case=f'{spec["suite"]}:{task.name}:{init}';seed=config['seed']+init*10000
                initial=pilot.torch.load(Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file,
                                        weights_only=False,map_location='cpu')[init]
                def make():
                    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                            camera_heights=256,camera_widths=256,ignore_done=True)
                    env.seed(0);env.reset();obs=env.set_init_state(initial)
                    for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
                    return env,obs
                env,obs=make();prefix=[];history=[base.sim_obs(obs,task.language)]
                while len(prefix)<spec['arrival_step']:
                    actions=base.command_actions(pilot.predict(base.sim_obs(obs,task.language),50,4,seed+len(prefix)))
                    for action in actions[:16]:
                        obs,_,_,_=env.step(action.tolist());prefix.append(action)
                        history.append(base.sim_obs(obs,task.language))
                        if len(prefix)==spec['arrival_step']:break
                arrival=base.sim_obs(obs,task.language);sim_state=env.get_sim_state().copy()
                arrival_already_success=bool(env.check_success());env.close()
                look=config['lookahead_steps'];before=history[-1-look]
                predicted=forecast(history[:-look],look)
                sample_seed=seed+spec['arrival_step']
                tag=hashlib.sha256(case.encode()).hexdigest()[:16]
                np.savez_compressed(out/'cases'/f'{tag}.npz',prefix=np.array(prefix),arrival_state=arrival.state,
                    before_state=before.state,forecast_state=predicted.state,sim_state=sim_state,
                    arrival_external=arrival.images['image'],arrival_wrist=arrival.images['wrist_image'],
                    before_external=before.images['image'],before_wrist=before.images['wrist_image'])
                conditions={'oracle':arrival,'stale':before,'forecast':predicted,
                    'oracle_state':Observation(arrival.state,before.images,arrival.instruction),
                    'oracle_rgb':Observation(predicted.state,arrival.images,arrival.instruction)}
                for mm in [5,20]:
                    state=arrival.state.copy();state[0]+=(-1 if ti%2 else 1)*mm/1000
                    conditions[f'bias{mm}mm']=Observation(state,arrival.images,arrival.instruction)
                priors={key:pilot.prepare_prior(value,sample_seed) for key,value in conditions.items()}
                teacher=base.command_actions(priors['oracle']['actions'])
                methods=list(config['methods']);np.random.default_rng(sample_seed).shuffle(methods)
                prepared={}
                for method in methods:
                    pre=0.
                    if method.startswith('fresh'):
                        actions,post=pilot.finish(arrival,sample_seed,steps=int(method[-1]))
                    elif method.endswith('_resume3_1'):
                        prior=priors[method.removesuffix('_resume3_1')];pre=prior['partial_ms']
                        actions,post=pilot.finish(arrival,sample_seed,latent=prior['partial'],start_step=3)
                    elif method=='forecast_full_refine1':
                        prior=priors['forecast'];pre=prior['full_ms']
                        with pilot.torch.inference_mode():
                            # Noise generation is post-arrival work and included in post timing.
                            pilot.synchronize();begin=time.perf_counter();cond=pilot.condition(arrival)
                            latent=.75*prior['full']+.25*pilot.noise(cond,50,sample_seed)
                            latent=pilot.integrate(cond,50,4,sample_seed,start=3,latent=latent)
                            actions=pilot.decode(cond,latent);pilot.synchronize();post=(time.perf_counter()-begin)*1000
                    else:
                        prior=priors[method.removesuffix('_replay')];pre=prior['full_ms']
                        begin=time.perf_counter();actions=prior['actions'].copy();post=(time.perf_counter()-begin)*1000
                    command=base.command_actions(actions);prepared[method]=command
                    row=dict(case=case,method=method,task=task.name,initial_state=init,
                        precompute_ms=pre,post_ms=post,analytical_arrival_delay_ms=max(0.,pre-look*50)+post,
                        teacher_nmse_first16=float(np.mean(((command-teacher)/std)**2)),
                        teacher_nmse_first4=float(np.mean(((command[:4]-teacher[:4])/std)**2)),
                        gripper_agreement=float(np.mean(command[:,-1]==teacher[:,-1])),
                        forecast_position_error_mm=float(np.linalg.norm(predicted.state[:3]-arrival.state[:3])*1000),
                        before_position_error_mm=float(np.linalg.norm(before.state[:3]-arrival.state[:3])*1000),
                        privileged=method.startswith(('oracle','bias')),actions=command.tolist(),
                        observation_file=f'cases/{tag}.npz',arrival_already_success=arrival_already_success)
                    base.append(out/'diagnostics.jsonl',row)
                print(json.dumps(dict(event='diagnostics_ready',case=case,forecast_error_mm=row['forecast_position_error_mm'],
                                     parity=parity)),flush=True)
                # All methods see exactly the same physical arrival; precomputed post-actions are
                # valid only if replay reproduces BOTH sensor images and full simulator state.
                for method in methods:
                    if time.monotonic()>deadline:summarize(out,'budget_finished');return
                    env=None;r=dict(case=case,method=method,task=task.name,initial_state=init,status='running',
                        success=False,steps=spec['arrival_step'],arrival_already_success=arrival_already_success)
                    started=time.monotonic()
                    try:
                        env,obs=make()
                        for action in prefix:obs,_,_,_=env.step(action.tolist())
                        current=base.sim_obs(obs,task.language)
                        r['arrival_sim_max_abs']=float(np.max(np.abs(env.get_sim_state()-sim_state)))
                        r['arrival_images_equal']=all(np.array_equal(current.images[k],arrival.images[k]) for k in arrival.images)
                        if r['arrival_sim_max_abs']>1e-7 or not r['arrival_images_equal']:
                            raise ValueError('Paired arrival mismatch')
                        for action in prepared[method][:config['intervention_steps']]:
                            obs,_,ok,_=env.step(action.tolist());r['steps']+=1;r['success']|=bool(ok)
                        r['endpoint_state']=base.sim_obs(obs,task.language).state.tolist()
                        r['goal_after_intervention']=bool(env.check_success())
                        while not r['success'] and r['steps']<spec['max_steps']:
                            action_chunk=base.command_actions(pilot.predict(base.sim_obs(obs,task.language),50,4,seed+r['steps']))
                            for action in action_chunk[:config['continuation_K']]:
                                obs,_,ok,_=env.step(action.tolist());r['steps']+=1;r['success']=bool(ok)
                                if ok or r['steps']>=spec['max_steps']:break
                        r['status']='completed'
                    except Exception:
                        r['status']='error';r['error']=traceback.format_exc()
                    finally:
                        if env is not None:env.close()
                    r['wall_s']=time.monotonic()-started
                    base.append(out/'rollouts.jsonl',r);summarize(out,'running')
                    print(json.dumps({k:v for k,v in r.items() if k!='endpoint_state'}),flush=True)
                    if r['status']=='error':raise RuntimeError(r['error'])
                del priors,prepared,history,conditions
        summarize(out,'completed')
    except Exception:
        summarize(out,'failed');raise


if __name__=='__main__':main()
