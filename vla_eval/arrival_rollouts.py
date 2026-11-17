"""Paired arrival-only interventions after a committed, open-gripper reaching prefix.

The target pose forecast uses only current proprioception and the known reaching
commands. Images stay stale; no future RGB or future simulator state enters the
prior. Ground-truth object position is used only to annotate the evaluation phase.
"""
from __future__ import annotations
from pathlib import Path
import json,time,traceback,gc
import numpy as np
from .contracts import Observation
from .small import RidgeSmall


def quat_rotvec(quat):
    q=np.asarray(quat);w=np.clip(q[3],-1.,1.);den=np.sqrt(1-w*w)
    return np.zeros(3) if np.isclose(den,0.) else q[:3]*2*np.arccos(w)/den


def observation(obs,instruction):
    state=np.r_[obs['robot0_eef_pos'],quat_rotvec(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']].astype(np.float32)
    return Observation(state,{'image':np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),
                              'wrist_image':np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])},instruction)


def append(path,row):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def run_arrival_pilot(model,out,state,deadline,update,forecast_mode='controller_target_sum',methods=None):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from scipy.spatial.transform import Rotation
    from PIL import Image
    suite=benchmark.get_benchmark_dict()['libero_spatial']()
    methods=methods or ['A_fresh','B_few_steps','C_flow_refine','D_partial','C_ridge','B_ridge']
    existing=[]
    if (out/'arrival_closed_loop.jsonl').exists():
        existing=[json.loads(s) for s in (out/'arrival_closed_loop.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    finished={(r['task_id'],r['method']) for r in existing if r.get('status')=='completed'}
    small=RidgeSmall(out/'correction_ridge.npz') if (out/'correction_ridge.npz').exists() else None
    for task_id in [0,1,2]:
        if time.monotonic()>deadline:return
        task=suite.get_task(task_id)
        def make_env():
            env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
            env.seed(0);env.reset();obs=env.set_init_state(suite.get_task_init_states(task_id)[0])
            for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
            return env,obs
        plan_file=out/f'arrival_plan_task{task_id}.npz'
        if plan_file.exists():
            with np.load(plan_file,allow_pickle=False) as d:
                committed=d['committed'].copy();expected=d['expected'].copy()
        else:
            state.update(phase='arrival_reaching',task_id=task_id);update(out,state)
            env,obs=make_env();committed=[];expected=None
            try:
                while len(committed)<120 and time.monotonic()<deadline:
                    current=observation(obs,task.language)
                    cond=model.condition(current);actions,_,_=model.predict_condition(cond,50,10,len(committed))
                    targets=[np.asarray(value) for key,value in obs.items()
                             if 'black_bowl' in key and key.endswith('_pos') and np.asarray(value).shape==(3,)]
                    near=bool(targets and min(np.linalg.norm(current.state[:3]-p) for p in targets)<.12)
                    predicted_close=np.flatnonzero(actions[:,6]>0)
                    close_soon=bool(len(predicted_close) and 4<=predicted_close[0]<=16)
                    open_prefix=bool(np.all(actions[:4,6]<=0) and np.max(np.abs(current.state[6:]))>.02)
                    if len(committed)>=8 and open_prefix and (near or close_soon):
                        # Controller-scaled target, not the future achieved simulator state.
                        controller=env.env.robots[0].controller
                        prefix=np.clip(actions[:4,:6],np.asarray(controller.input_min),np.asarray(controller.input_max))
                        scale=(np.asarray(controller.output_max)-np.asarray(controller.output_min))/(np.asarray(controller.input_max)-np.asarray(controller.input_min))
                        bias=(np.asarray(controller.output_max)+np.asarray(controller.output_min))/2
                        center=(np.asarray(controller.input_max)+np.asarray(controller.input_min))/2
                        deltas=(prefix-center)*scale+bias
                        expected=current.state.copy();expected[:3]+=deltas[:,:3].sum(axis=0)
                        rotation=Rotation.from_quat(obs['robot0_eef_quat'])
                        for delta in deltas[:,3:6]:rotation=Rotation.from_rotvec(delta)*rotation
                        expected[3:6]=quat_rotvec(rotation.as_quat())
                        committed.extend(actions[:4].tolist())
                        np.savez_compressed(plan_file,committed=np.asarray(committed),expected=expected)
                        break
                    for action in actions[:4]:
                        obs,_,done,_=env.step(action.tolist());committed.append(action.tolist())
                        if done:break
                    if done:break
                    del cond;gc.collect()
            finally:env.close()
            if expected is None:
                append(out/'arrival_closed_loop.jsonl',{'task_id':task_id,'method':'phase_selection',
                       'status':'phase_not_found','success':None,'reaching_steps':len(committed)})
                continue
            committed=np.asarray(committed)
        # Reconstruct the pre-arrival sensor snapshot, then precompute before executing the final 4 steps.
        env,obs=make_env()
        past_states=[observation(obs,task.language).state.copy()]
        for action in committed[:-4]:
            obs,_,_,_=env.step(action.tolist())
            past_states.append(observation(obs,task.language).state.copy())
        before=observation(obs,task.language)
        if forecast_mode=='past_calibrated_command':
            controller=env.env.robots[0].controller
            lower,upper=np.asarray(controller.input_min),np.asarray(controller.input_max)
            scale=(np.asarray(controller.output_max)-np.asarray(controller.output_min))/(upper-lower)
            bias=(np.asarray(controller.output_max)+np.asarray(controller.output_min))/2
            deltas=(np.clip(committed[:,:6],lower,upper)-(upper+lower)/2)*scale+bias
            past_states=np.asarray(past_states)
            old_commands=deltas[-8:-4,:3]
            old_moves=np.diff(past_states[-5:,:3],axis=0)
            gain=np.sum(old_commands*old_moves,axis=0)/(np.sum(old_commands**2,axis=0)+1e-8)
            expected=before.state.copy()
            expected[:3]+=(deltas[-4:,:3]*gain).sum(axis=0)
            # SO(3) velocity estimate, avoiding the axis-angle wrap at pi.
            rotations=Rotation.from_rotvec(past_states[-2:,3:6])
            angular_velocity=(rotations[1]*rotations[0].inv()).as_rotvec()
            expected[3:6]=quat_rotvec((Rotation.from_rotvec(4*angular_velocity)*rotations[1]).as_quat())
        elif forecast_mode!='controller_target_sum':
            raise ValueError(f'Unknown causal forecast: {forecast_mode}')
        forecast=Observation(expected,before.images,before.instruction)
        pc=model.condition(forecast)
        partial,t8=model.integrate(pc,50,10,777,stop_step=8)
        full,t2=model.integrate(pc,50,10,777,latent=partial,start_step=8)
        prior=model.decode(full)
        pre_ms=pc.preparation_ms+pc.backbone_ms+t8+t2
        for action in committed[-4:]:obs,_,_,_=env.step(action.tolist())
        reference_arrival=observation(obs,task.language)
        env.close()
        position_error=float(np.linalg.norm(reference_arrival.state[:3]-expected[:3])*1000)
        rotation_error=float((Rotation.from_rotvec(reference_arrival.state[3:6])*Rotation.from_rotvec(expected[3:6]).inv()).magnitude())
        for method in methods:
            if time.monotonic()>deadline:return
            if (task_id,method) in finished:continue
            if 'ridge' in method and small is None:continue
            state.update(phase='arrival_rollout',task_id=task_id,method=method);update(out,state)
            print('ARRIVAL',task_id,method,'reaching_steps',len(committed),'forecast_error_mm',position_error,flush=True)
            env=None;success=False;status='completed';frames=[];first_actions=None;trace=[];steps=len(committed)
            start=time.perf_counter();post_ms=0.;arrival_difference=None
            try:
                env,obs=make_env()
                for action in committed:obs,_,_,_=env.step(action.tolist())
                current=observation(obs,task.language)
                arrival_difference=float(np.max(np.abs(current.state-reference_arrival.state)))
                if arrival_difference>1e-5:raise ValueError(f'Paired arrival states differ: {arrival_difference}')
                t=time.perf_counter()
                if method in {'B_ridge','C_ridge'}:
                    norm_prior=(prior-model.action_mean)/model.action_std
                    predicted=(small.predict(current,50,10,0) if method=='B_ridge'
                               else small.correct(current,norm_prior,forecast))
                    actions=predicted*model.action_std+model.action_mean
                    post_ms=(time.perf_counter()-t)*1000
                else:
                    cond=model.condition(current)
                    if method=='A_fresh':actions,_,_=model.predict_condition(cond,50,10,777)
                    elif method=='B_few_steps':actions,_,_=model.predict_condition(cond,50,2,777)
                    elif method=='D_partial':
                        x,_=model.integrate(cond,50,10,777,latent=partial,start_step=8);actions=model.decode(x)
                    else:
                        x,_=model.integrate(cond,50,10,777,latent=.8*full+.2*model.noise(50,777),start_step=8);actions=model.decode(x)
                    post_ms=(time.perf_counter()-t)*1000
                first_actions=actions[:4].tolist()
                calls=0
                while steps<220 and not success:
                    if time.monotonic()>deadline:status='censored_budget';break
                    for action in actions[:4]:
                        obs,_,done,_=env.step(action.tolist());steps+=1;success=bool(done)
                        if steps%5==0:frames.append(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]))
                        if success or steps>=220:break
                    calls+=1
                    if success or steps>=220:break
                    current=observation(obs,task.language);cond=model.condition(current)
                    actions,_,ae=model.predict_condition(cond,50,10,1000+calls)
                    trace.append({'call':calls,'step':steps,'backbone_ms':cond.backbone_ms,'ae_ms':ae})
                    if calls%10==0:print('  arrival steps',steps,flush=True)
                    gc.collect()
            except Exception as error:
                status='error';append(out/'errors.jsonl',{'phase':'arrival','task_id':task_id,'method':method,'error':str(error),'traceback':traceback.format_exc()})
            finally:
                if env is not None:env.close()
            row={'task_id':task_id,'task':task.language,'initial_state':0,'method':method,'status':status,
                 'success':success if status=='completed' else None,'steps':steps,'reaching_steps':len(committed),
                 'forecast_position_error_mm':position_error,'forecast_rotation_error_rad':rotation_error,
                 'forecast_mode':forecast_mode,'arrival_state_max_difference':arrival_difference,
                 'precompute_cpu_ms':pre_ms if method.startswith(('C','D')) else 0.,'arrival_post_cpu_ms':post_ms,
                 'first_actions':first_actions,'intervention_scope':'first four actions, identical Heavy controller afterwards',
                 'forecast_source':'stale RGB plus '+forecast_mode+' pose; no future simulator state',
                 'phase_selection':'object-position annotation or planned gripper transition; evaluation oracle',
                 'latency_mode':'zero_inference_delay','wall_seconds':time.perf_counter()-start,'trace':trace}
            if method=='D_partial':row['precompute_cpu_ms']=pc.preparation_ms+pc.backbone_ms+t8
            row['overlap_window_sim_ms']=200.
            row['analytical_first_action_delay_cpu_ms']=max(0.,row['precompute_cpu_ms']-200.)+post_ms
            row['latency_accounting']='sequential CPU stage measurements with analytical overlap; no concurrency measurement'
            if frames:
                directory=out/'arrival_replays';directory.mkdir(exist_ok=True)
                images=[Image.fromarray(x) for x in frames];file=directory/f'task{task_id}_{method}.gif'
                images[0].save(file,save_all=True,append_images=images[1:],duration=250,loop=0)
                row['replay']=str(file.relative_to(out))
            append(out/'arrival_closed_loop.jsonl',row);update(out,state)
