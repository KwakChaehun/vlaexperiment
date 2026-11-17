"""Measured 20Hz closed loop with one asynchronous GPU inference worker.

No predicted-future free overlap: whole banks are prepared while the simulator is
paused BEFORE the real-time phase. Preparation time is recorded and added to end
to end time. During the phase, simulator physics continues with zero-motion holds
when no new action is ready. Graphics and inference share the actual RTX3090.
"""
from __future__ import annotations
import argparse
from collections import defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import traceback

import run_gr00t_wrist_pilot as base
import numpy as np
from vla_eval.gr00t16_arrival import Gr00t16Arrival
from vla_eval.speculative_features import PredictedMLPCache


METHODS={
    'fresh4_k16':dict(K=16,N=4), 'fresh1_k16':dict(K=16,N=1),
    'fresh2_k16':dict(K=16,N=2),
    'fresh4_k4':dict(K=4,N=4), 'fresh1_k4':dict(K=4,N=1),
    'whole_s25_k16':dict(K=16,prediction=True),
    'whole_s25_k4':dict(K=4,prediction=True),
    'whole_gate_k16':dict(K=16,prediction=True,gate=True),
}


def gate_metrics(actual,predicted):
    from scipy.spatial.transform import Rotation
    angle=(Rotation.from_rotvec(actual.state[3:6])*Rotation.from_rotvec(predicted.state[3:6]).inv()).magnitude()
    return dict(position_m=float(np.linalg.norm(actual.state[:3]-predicted.state[:3])),
        angle_rad=float(angle),gripper_m=float(np.max(np.abs(actual.state[6:]-predicted.state[6:]))),
        external_mae=float(np.abs(actual.images['image'].astype(np.float32)-predicted.images['image']).mean()),
        wrist_mae=float(np.abs(actual.images['wrist_image'].astype(np.float32)-predicted.images['wrist_image']).mean()))


def summarize(out,status):
    rows=base.rows(out/'rollouts.jsonl');groups=defaultdict(list)
    for r in rows:groups[r['method']].append(r)
    result=[]
    for method,rr in groups.items():
        good=[r for r in rr if r['status']=='completed']
        if not good:continue
        calls=[c for r in good for c in r['calls']]
        result.append(dict(method=method,completed=len(good),successes=sum(r['success'] for r in good),
            stable_successes=sum(r['stable_success'] for r in good),
            mean_hold_steps=float(np.mean([r['hold_steps'] for r in good])),
            mean_phase_wall_s=float(np.mean([r['phase_wall_s'] for r in good])),
            mean_startup_s=float(np.mean([r['startup_s'] for r in good])),
            mean_inclusive_wall_s=float(np.mean([r['phase_wall_s']+r['startup_s'] for r in good])),
            response_compute_p50_ms=float(np.median([c['compute_ms'] for c in calls])),
            gate_accepts=sum(c['path']=='accepted_prediction' for c in calls),calls=len(calls),
            mean_worker_compute_ms_per_action=float(np.mean([sum(c['compute_ms'] for c in r['calls'])/max(r['action_steps'],1) for r in good]))
        ))
    base.save(out/'summary.json',result)
    base.save(out/'status.json',dict(status=status,completed=len(rows),errors=sum(r['status']=='error' for r in rows),updated_utc=base.now()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'rollouts.jsonl').exists():raise ValueError('Use new output')
    base.save(out/'config.json',config);base.setup_libero()
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suites={name:benchmark.get_benchmark_dict()[name]() for name in {t['suite'] for t in config['tasks']}}
    pilot=Gr00t16Arrival(str(base.ROOT/config['checkpoint']));cache=PredictedMLPCache(pilot)
    examples,_=base.cases(json.loads((base.ROOT/'configs/gr00t16_wrist_pilot.json').read_text()))
    parity=pilot.split_parity(examples[0]['fresh'])
    for n in (1,4):pilot.finish(examples[0]['fresh'],0,steps=n)
    base.save(out/'manifest.json',dict(scope=__doc__,parity=parity,gpu=pilot.torch.cuda.get_device_name(),
        control_hz=20,source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            ['scripts/run_realtime_feature_probe.py','vla_eval/speculative_features.py','vla_eval/gr00t16_arrival.py','vla_eval/gr00t16_pilot.py']},
        gate='Fixed untrained thresholds chosen before these real-time trials, not calibrated success probabilities',
        predictor='Privileged current simulator state via past command replay; nominal dynamics, no actual future observations'))
    try:
        for init in config['initial_states']:
            for spec in config['tasks']:
                task=next(t for t in suites[spec['suite']].tasks if t.name==spec['name']);seed=config['seed']+init*10000
                initial=pilot.torch.load(Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file,weights_only=False,map_location='cpu')[init]
                def make():
                    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                        camera_heights=256,camera_widths=256,ignore_done=True)
                    env.seed(0);env.reset();obs=env.set_init_state(initial)
                    for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
                    return env,obs
                ref,obs=make();prefix=[]
                while len(prefix)<spec['start']:
                    chunk=base.command_actions(pilot.predict(base.sim_obs(obs,task.language),50,4,seed+len(prefix)))
                    for a in chunk[:16]:
                        obs,_,_,_=ref.step(a.tolist());prefix.append(a.copy())
                        if len(prefix)>=spec['start']:break
                anchor=ref.get_sim_state().copy();anchor_obs=base.sim_obs(obs,task.language)
                if ref.check_success():raise ValueError('Phase already successful')
                ref.close();methods=config['methods'].copy();np.random.default_rng(seed).shuffle(methods)
                for method in methods:
                    arm=METHODS[method];K=arm['K'];bank={};env=None;pool=None
                    r=dict(task=task.name,initial_state=init,method=method,K=K,status='running',success=False,
                        stable_success=False,action_steps=0,hold_steps=0,physical_steps=0,startup_s=0.,
                        startup_model_ms=0.,calls=[],tick_lateness_ms=[],reaction_ms=[],trace=[])
                    pending=None;pending_record=None
                    try:
                        env,obs=make()
                        for a in prefix:obs,_,_,_=env.step(a.tolist())
                        fresh=base.sim_obs(obs,task.language)
                        r['paired_sim_max_abs']=float(np.max(np.abs(env.get_sim_state()-anchor)))
                        r['paired_images_equal']=all(np.array_equal(fresh.images[k],anchor_obs.images[k]) for k in fresh.images)
                        if r['paired_sim_max_abs']>1e-7 or not r['paired_images_equal']:raise ValueError('Phase mismatch')
                        if arm.get('prediction'):
                            prepare=time.perf_counter();future,fobs=make()
                            try:
                                for a in prefix:fobs,_,_,_=future.step(a.tolist())
                                clone=base.sim_obs(fobs,task.language)
                                if np.max(np.abs(future.get_sim_state()-anchor))>1e-7 or not all(np.array_equal(clone.images[k],fresh.images[k]) for k in fresh.images):raise ValueError('Forecast clone mismatch')
                                for offset in range(0,spec['end']-spec['start'],K):
                                    predicted=base.sim_obs(fobs,task.language)
                                    prior=cache.capture(predicted,seed+spec['start']+offset)
                                    prior['obs']=predicted;bank[offset]=prior;r['startup_model_ms']+=prior['pre_ms']
                                    for a in base.command_actions(prior['actions'])[:K]:fobs,_,_,_=future.step(a.tolist())
                            finally:
                                future.close();env.sim._render_context_offscreen.gl_ctx.make_current()
                            r['startup_s']=time.perf_counter()-prepare
                            r['bank_MiB']=sum(cache.bytes(v) for v in bank.values())/2**20
                        def work(observation,offset):
                            begin=time.perf_counter();metrics=None;path='fresh'
                            if arm.get('prediction'):
                                prior=bank[offset]
                                if arm.get('gate'):
                                    metrics=gate_metrics(observation,prior['obs'])
                                    accepted=all(metrics[key]<=value for key,value in config['gate_thresholds'].items())
                                else:accepted=False
                                if accepted:
                                    action=prior['actions'].copy();path='accepted_prediction'
                                else:
                                    action,_,_=cache.correct(observation,prior,seed+spec['start']+offset,.25)
                                    path='sparse_correction'
                            else:action,_=pilot.finish(observation,seed+spec['start']+offset,steps=arm['N'])
                            finish=time.perf_counter()
                            return base.command_actions(action),dict(offset=offset,path=path,gate_metrics=metrics,
                                compute_ms=(finish-begin)*1000,worker_started=begin,worker_finished=finish)
                        pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='vla')
                        # Warm the worker thread before the measured phase to avoid first-thread CUDA setup.
                        pool.submit(pilot.finish,fresh,seed+spec['start'],1).result()
                        queue=deque();last_gripper=float(prefix[-1][-1]);start=time.perf_counter();deadline=start+.05
                        observed_at=time.perf_counter();pending=pool.submit(work,fresh,0)
                        while r['physical_steps']<spec['end']-spec['start']:
                            pause=deadline-time.perf_counter()
                            if pause>0:time.sleep(pause)
                            tick=time.perf_counter();r['tick_lateness_ms'].append(max(0.,(tick-deadline)*1000))
                            if pending is not None and pending.done():
                                actions,detail=pending.result();pending=None
                                detail['observed_at']=observed_at;detail['applied_at']=tick
                                detail['reaction_ms']=(tick-observed_at)*1000
                                r['reaction_ms'].append(detail['reaction_ms']);r['calls'].append(detail)
                                queue.extend(actions[:K])
                            if queue:
                                a=queue.popleft().copy();a[:3]*=config['gain'];last_gripper=float(a[-1]);r['action_steps']+=1
                            else:
                                a=np.array([0.]*6+[last_gripper]);r['hold_steps']+=1
                            obs,_,ok,_=env.step(a.tolist());r['physical_steps']+=1;r['success']|=bool(ok)
                            r['trace'].append(dict(step=r['physical_steps'],wall_s=time.perf_counter()-start,
                                action=a.tolist(),goal=bool(ok)))
                            if r['success']:break
                            if not queue and pending is None and r['physical_steps']<spec['end']-spec['start']:
                                current=base.sim_obs(obs,task.language);observed_at=time.perf_counter()
                                pending=pool.submit(work,current,r['action_steps'])
                            deadline+=.05
                        r['phase_wall_s']=time.perf_counter()-start
                        r['success_at_phase_end']=bool(env.check_success())
                        if pending is not None:
                            _,detail=pending.result();detail['unused_at_end']=True;r['calls'].append(detail);pending=None
                        pool.shutdown(wait=True);pool=None
                        r['endpoint_state']=base.sim_obs(obs,task.language).state.tolist()
                        if r['success']:
                            # Separate native goal stability diagnostic, not part of completion time.
                            stable=True
                            for _ in range(10):
                                obs,_,ok,_=env.step([0.]*6+[last_gripper]);stable&=bool(ok)
                            r['stable_success']=stable
                        r['tick_lateness_p95_ms']=float(np.percentile(r['tick_lateness_ms'],95))
                        r['reaction_p50_ms']=float(np.median(r['reaction_ms'])) if r['reaction_ms'] else None
                        r['status']='completed'
                    except Exception:r['status']='error';r['error']=traceback.format_exc()
                    finally:
                        if pool is not None:pool.shutdown(wait=True)
                        bank.clear()
                        if env is not None:env.close()
                    base.append(out/'rollouts.jsonl',r);summarize(out,'running')
                    print(json.dumps({k:r.get(k) for k in ['task','initial_state','method','status','success','stable_success','hold_steps','phase_wall_s','startup_s','reaction_p50_ms']}),flush=True)
                    if r['status']=='error':raise RuntimeError(r['error'])
        summarize(out,'completed')
    except Exception:summarize(out,'failed');raise
    finally:cache.close()


if __name__=='__main__':main()
