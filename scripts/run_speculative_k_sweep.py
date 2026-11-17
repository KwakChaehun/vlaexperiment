"""Causal nominal-simulator forecast banks with prediction-relative sparse correction.

The predictor has privileged CURRENT full simulator state via action-history replay;
it never reads the real future. Real execution applies an unmodelled command gain.
This is an optimistic world-model diagnostic, not a deployable learned predictor.
All inference is sequential. Planning stalls are measured, not hidden as free overlap.
"""
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
from vla_eval.gr00t16_arrival import Gr00t16Arrival
from vla_eval.speculative_features import PredictedMLPCache


METHODS = {
    'fresh4_k16': dict(kind='fresh', n=4, K=16),
    'fresh1_k16': dict(kind='fresh', n=1, K=16),
    'fresh4_k4': dict(kind='fresh', n=4, K=4),
    'fresh1_k4': dict(kind='fresh', n=1, K=4),
    'whole_s25_k16': dict(kind='whole', ratio=.25, K=16),
    'whole_s25_k4': dict(kind='whole', ratio=.25, K=4),
}



def summarize(out, status):
    rows=base.rows(out/'rollouts.jsonl');calls=base.rows(out/'calls.jsonl')
    groups=defaultdict(list)
    for r in rows:groups[(r['gain'],r['method'])].append(r)
    summary=[]
    for (gain,method),rr in groups.items():
        good=[r for r in rr if r['status']=='completed']
        cc=[r for r in calls if r['gain']==gain and r['method']==method]
        pp=[r['post_ms'] for r in cc]
        summary.append(dict(gain=gain,method=method,completed=len(good),
            successes=sum(r['success'] for r in good),
            success_at_end=sum(r['goal_at_end'] for r in good),
            post_p50_ms=float(np.median(pp)) if pp else None,
            post_p95_ms=float(np.percentile(pp,95)) if pp else None,
            mean_plan_wall_ms=float(np.mean([r['plan_wall_ms'] for r in good])) if good else None,
            mean_service_ms=float(np.mean([r['plan_wall_ms']+r['post_sum_ms'] for r in good])) if good else None,
            mean_prediction_position_mm=float(np.mean([r['prediction_position_mm'] for r in cc if 'prediction_position_mm' in r]))
                if any('prediction_position_mm' in r for r in cc) else None))
    base.save(out/'summary.json',summary)
    base.save(out/'status.json',dict(status=status,completed=len(rows),updated_utc=base.now(),
        errors=sum(r['status']=='error' for r in rows)))
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    (out/'observation_pairs').mkdir(exist_ok=True)
    import fcntl
    lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'rollouts.jsonl').exists():raise ValueError('Use a new output directory')
    base.save(out/'config.json',config);base.setup_libero()
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suites={s:benchmark.get_benchmark_dict()[s]() for s in {x['suite'] for x in config['tasks']}}
    pilot=Gr00t16Arrival(str(base.ROOT/config['checkpoint']))
    cache=PredictedMLPCache(pilot)
    sources=['scripts/run_speculative_k_sweep.py','vla_eval/speculative_features.py',
             'vla_eval/gr00t16_arrival.py','vla_eval/gr00t16_pilot.py','scripts/run_gr00t_wrist_pilot.py']
    base.save(out/'manifest.json',dict(started_utc=base.now(),source_hashes={s:hashlib.sha256((base.ROOT/s).read_bytes()).hexdigest() for s in sources},
        gpu=pilot.torch.cuda.get_device_name(),torch=pilot.torch.__version__,
        predictor='Nominal LIBERO simulator + same VLA, initialized from replay of actual past commands; privileged current state, no future observations',
        correction='Per-layer MLP input residual top-k against predicted frame. Dense attention, embedding, state fusion; one remaining native flow step.',
        timing='Sequential measured planning+replay+render wall cost and post-arrival model cost. No asynchronous execution or PIM.',
        H=50,K='per-arm 4 or 16',N='fresh 1 or 4, correction 3+1',full_task_success_claim=False))
    # Warm up outside timing, then verify full-update and sampler equivalence.
    examples,_=base.cases(json.loads((base.ROOT/'configs/gr00t16_wrist_pilot.json').read_text()))
    parity=pilot.split_parity(examples[0]['fresh'])
    prediction=cache.capture(examples[0]['fresh'],37)
    native,_=pilot.finish(examples[0]['fresh'],37)
    corrected,_,_=cache.correct(examples[0]['fresh'],prediction,37,1.)
    error=float(np.max(np.abs(native-corrected)))
    if error>1e-6:raise ValueError('All-update sampler mismatch')
    base.save(out/'parity.json',dict(**parity,all_update_action_max_abs=error,
        cache_MiB=cache.bytes(prediction)/2**20));del prediction,examples
    try:
        for gain in config['gains']:
            for init in config['initial_states']:
                for spec in config['tasks']:
                    task=next(t for t in suites[spec['suite']].tasks if t.name==spec['name'])
                    case=f'{task.name}:{init}:gain{gain}'
                    seed=config['seed']+init*10000
                    initial=pilot.torch.load(Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file,
                        weights_only=False,map_location='cpu')[init]
                    def make():
                        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                            camera_heights=256,camera_widths=256,ignore_done=True)
                        env.seed(0);env.reset();obs=env.set_init_state(initial)
                        for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
                        return env,obs
                    # Fixed-time task prefix, shared across every arm. No outcome-based selection.
                    reference,obs=make();prefix=[]
                    while len(prefix)<spec['start']:
                        actions=base.command_actions(pilot.predict(base.sim_obs(obs,task.language),50,4,seed+len(prefix)))
                        for action in actions[:16]:
                            obs,_,_,_=reference.step(action.tolist());prefix.append(action.copy())
                            if len(prefix)>=spec['start']:break
                    anchor_state=reference.get_sim_state().copy();anchor_obs=base.sim_obs(obs,task.language)
                    already=bool(reference.check_success());reference.close()
                    if already:raise ValueError(f'Preset phase already successful: {case}')
                    methods=config['methods'].copy();np.random.default_rng(seed).shuffle(methods)
                    for method in methods:
                        arm=METHODS[method];K=arm['K'];env=None;bank={}
                        trial=hashlib.sha256(f'{case}:{method}'.encode()).hexdigest()[:16]
                        r=dict(trial=trial,case=case,task=task.name,initial_state=init,gain=gain,method=method,K=K,
                            status='running',success=False,goal_at_end=False,start=spec['start'],end=spec['end'],
                            steps=spec['start'],plan_wall_ms=0.,post_sum_ms=0.,plan_count=0,plans=[],
                            max_cache_MiB=0.,calls=0)
                        begin=time.perf_counter()
                        try:
                            env,obs=make();history=[a.copy() for a in prefix]
                            for action in prefix:obs,_,_,_=env.step(action.tolist())
                            sim_error=float(np.max(np.abs(env.get_sim_state()-anchor_state)))
                            real=base.sim_obs(obs,task.language)
                            equal=all(np.array_equal(real.images[k],anchor_obs.images[k]) for k in real.images)
                            if sim_error>1e-7 or not equal:raise ValueError('Paired initial state mismatch')
                            r['paired_sim_max_abs']=sim_error;r['paired_images_equal']=equal
                            while r['steps']<spec['end']:
                                step=r['steps'];fresh=base.sim_obs(obs,task.language)
                                needs_plan=arm['kind'] not in ('fresh',) and (
                                    step==spec['start'] or (arm['kind']=='rolling' and step not in bank))
                                if needs_plan:
                                    bank.clear();pilot.torch.cuda.empty_cache()
                                    pbegin=time.perf_counter();forecast_env,forecast_obs=make()
                                    try:
                                        # Replay only already executed commands, never real future states.
                                        for old_action in history:
                                            forecast_obs,_,_,_=forecast_env.step(old_action.tolist())
                                        clone_error=float(np.max(np.abs(forecast_env.get_sim_state()-env.get_sim_state())))
                                        clone_obs=base.sim_obs(forecast_obs,task.language)
                                        clone_images=all(np.array_equal(clone_obs.images[k],fresh.images[k]) for k in fresh.images)
                                        if clone_error>1e-7 or not clone_images:
                                            pixel_errors={k:int(np.abs(clone_obs.images[k].astype(np.int16)-fresh.images[k]).max()) for k in fresh.images}
                                            raise ValueError(f'Predictor start mismatch: sim={clone_error}, pixels={pixel_errors}, history={len(history)}, step={step}')
                                        horizon=arm.get('span',spec['end']-step)
                                        stop=min(step+horizon,spec['end'])
                                        model_ms=0.
                                        for future_step in range(step,stop,K):
                                            predicted_obs=base.sim_obs(forecast_obs,task.language)
                                            prior=cache.capture(predicted_obs,seed+future_step)
                                            prior['obs']=predicted_obs;prior['origin_step']=step
                                            bank[future_step]=prior;model_ms+=prior['pre_ms']
                                            actions=base.command_actions(prior['actions'])
                                            for a in actions[:min(K,stop-future_step)]:
                                                forecast_obs,_,_,_=forecast_env.step(a.tolist())
                                    finally:
                                        forecast_env.close()
                                        # This robosuite version does not make the GL context
                                        # current inside render(). Restore the real world after
                                        # using/destroying the forecast world's EGL context.
                                        env.sim._render_context_offscreen.gl_ctx.make_current()
                                    plan_ms=(time.perf_counter()-pbegin)*1000
                                    memory=sum(cache.bytes(v) for v in bank.values())/2**20
                                    r['plan_wall_ms']+=plan_ms;r['plan_count']+=1
                                    r['max_cache_MiB']=max(r['max_cache_MiB'],memory)
                                    r['plans'].append(dict(origin_step=step,stop=stop,wall_ms=plan_ms,
                                        model_ms=model_ms,cache_MiB=memory,clone_sim_max_abs=clone_error,clone_images_equal=clone_images))
                                call=dict(trial=trial,case=case,method=method,gain=gain,step=step)
                                pred=bank.get(step)
                                if arm['kind']=='fresh' or pred is None:
                                    actions,post=pilot.finish(fresh,seed+step,steps=arm.get('n',4))
                                    call['path']='fresh'+str(arm.get('n',4))
                                elif arm['kind']=='replay':
                                    copy_start=time.perf_counter();actions=pred['actions'].copy()
                                    post=(time.perf_counter()-copy_start)*1000;call['path']='replay_copy_only'
                                else:
                                    actions,post,groups=cache.correct(fresh,pred,seed+step,arm['ratio'])
                                    call.update(path='sparse_correction',groups=groups)
                                if pred is not None:
                                    call['prediction_age_steps']=step-pred['origin_step']
                                    call['prediction_position_mm']=float(np.linalg.norm(fresh.state[:3]-pred['obs'].state[:3])*1000)
                                    call['prediction_pixel_mae']={k:float(np.abs(fresh.images[k].astype(np.float32)-pred['obs'].images[k]).mean()) for k in fresh.images}
                                    # Diagnostics are excluded from timed post correction and never used by selection.
                                    call['state_max_abs']=float(np.max(np.abs(fresh.state-pred['obs'].state)))
                                    offset=step-spec['start']
                                    save_pair=(method.startswith('whole_s25') and offset in (0,16,32,64,128)) or (
                                        False and offset in (4,20,36,68,132))
                                    if save_pair:
                                        name=f'{trial}_{step}.npz'
                                        np.savez_compressed(out/'observation_pairs'/name,
                                            predicted_state=pred['obs'].state,actual_state=fresh.state,
                                            predicted_external=pred['obs'].images['image'],actual_external=fresh.images['image'],
                                            predicted_wrist=pred['obs'].images['wrist_image'],actual_wrist=fresh.images['wrist_image'],
                                            instruction=task.language,seed=seed+step)
                                        call['observation_pair']='observation_pairs/'+name
                                call['post_ms']=post;r['post_sum_ms']+=post;r['calls']+=1
                                command=base.command_actions(actions)
                                for a in command[:min(K,spec['end']-step)]:
                                    applied=a.copy();applied[:3]*=gain
                                    obs,_,ok,_=env.step(applied.tolist());history.append(applied.copy());r['steps']+=1
                                    r['success']|=bool(ok)
                                call['goal_after_chunk']=bool(env.check_success())
                                base.append(out/'calls.jsonl',call)
                                if step in bank:del bank[step]
                                del pred
                                # Stop at first successful control chunk, no shared rescue policy.
                                if r['success']:break
                            r['goal_at_end']=bool(env.check_success())
                            r['endpoint_state']=base.sim_obs(obs,task.language).state.tolist()
                            r['status']='completed'
                        except Exception:
                            r['status']='error';r['error']=traceback.format_exc()
                        finally:
                            bank.clear()
                            if env is not None:env.close()
                        r['wall_s']=time.perf_counter()-begin
                        r['executed_actions']=r['steps']-r['start']
                        r['post_ms_per_action']=r['post_sum_ms']/max(1,r['executed_actions'])
                        r['model_plus_post_ms_per_action']=(sum(p['model_ms'] for p in r['plans'])+r['post_sum_ms'])/max(1,r['executed_actions'])
                        base.append(out/'rollouts.jsonl',r);summary=summarize(out,'running')
                        print(json.dumps({k:r[k] for k in ['trial','task','initial_state','gain','method','success','steps','status','wall_s']}),flush=True)
                        if r['status']=='error':raise RuntimeError(r['error'])
        summarize(out,'completed')
    except Exception:
        summarize(out,'failed');raise
    finally:cache.close()


if __name__=='__main__':main()
