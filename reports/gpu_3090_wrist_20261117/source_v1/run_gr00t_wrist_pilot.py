"""Incremental RTX 3090 camera/H/K screening; open-loop and live results stay separate."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'.runtime/Isaac-GR00T-n16'), str(ROOT/'.runtime/vendor/LIBERO-master')]
os.environ.setdefault('HF_HOME', str(ROOT/'.runtime/hf'))
os.environ.setdefault('HF_MODULES_CACHE', str(ROOT/'.runtime/hf/modules'))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT/'.runtime/libero_config'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('ALBUMENTATIONS_OFFLINE', '1')

import numpy as np
from vla_eval.contracts import Observation
from vla_eval.gr00t16_pilot import Gr00t16Pilot
from vla_eval.statistics import wilson


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x] if path.exists() else []


def command_actions(actions):
    # Match NVIDIA's LIBERO wrapper: [0,1] open gripper -> [-1,1] close gripper.
    x = actions.copy()
    x[:,-1] = -np.sign(2*x[:,-1]-1)
    return x


def camera_obs(fresh, stale, mode):
    images = dict(fresh.images)
    if mode in ('external_stale','both_stale'):
        images['image'] = stale.images['image']
    if mode in ('wrist_stale','both_stale'):
        images['wrist_image'] = stale.images['wrist_image']
    return Observation(fresh.state, images, fresh.instruction)


def cases(config):
    import pyarrow.parquet as pq
    from PIL import Image
    tasks = {r['task_index']:r['task'] for r in rows(ROOT/'data/wrist_pilot/tasks.jsonl')}
    result, all_actions = [], []
    for episode in config['episodes']:
        source = ROOT/f'data/wrist_pilot/episode_{episode:06d}.parquet'
        data = pq.read_table(source).to_pylist()
        targets = np.asarray([r['actions'] for r in data],np.float32)
        all_actions.append(targets)
        lower, upper = max(config['lags']),len(data)-16
        events = [int(i)-2 for i in np.flatnonzero(np.diff(np.sign(targets[:,-1])))+1
                  if lower <= i-2 <= upper]
        selection = events[:2]
        for i in np.linspace(lower,upper,12,dtype=int):
            if all(abs(i-j)>4 for j in selection): selection.append(int(i))
            if len(selection)>=config['samples_per_episode']: break
        def get(i):
            row = data[i]
            images = {}
            for k in ('image','wrist_image'):
                with Image.open(io.BytesIO(row[k]['bytes'])) as im:
                    images[k] = np.asarray(im.convert('RGB')).copy()
            return Observation(np.asarray(row['state'],np.float32),images,tasks[row['task_index']])
        for frame in selection:
            result.append(dict(id=f'{episode}:{frame}',episode=episode,frame=frame,
                               phase='gripper_transition' if frame in events else 'uniform',
                               fresh=get(frame),stale={lag:get(frame-lag) for lag in config['lags']},
                               target=targets[frame:frame+16]))
    std = np.maximum(np.concatenate(all_actions).std(axis=0),0.05)
    return result,std


def summarize(out, status):
    offline, closed = rows(out/'offline.jsonl'),rows(out/'closed_loop.jsonl')
    groups = defaultdict(list)
    for r in offline:
        groups[(r['mode'],r['H'],r['K'],r['N'],r['lag'])].append(r)
    summary = []
    for key,values in groups.items():
        item = dict(zip(['mode','H','K','N','lag'],key),samples=len(values),
                    episodes=len({x['episode'] for x in values}))
        for metric in ['expert_nmse','teacher_nmse','total_ms','vision_ms','vlm_ms','ae_ms']:
            measured = [x[metric] for x in values if x.get(metric) is not None]
            item[metric] = float(np.mean(measured)) if measured else None
        item['call_p50_ms'] = float(np.median([x['total_ms'] for x in values]))
        item['call_p95_ms'] = float(np.quantile([x['total_ms'] for x in values],.95))
        item['call_p99_ms'] = float(np.quantile([x['total_ms'] for x in values],.99))
        item['amortized_ms_per_action'] = item['total_ms']/item['K']
        summary.append(item)
    save(out/'offline_summary.json',summary)
    groups = defaultdict(list)
    for r in closed:
        groups[(r['mode'],r['H'],r['K'],r['N'])].append(r)
    live = []
    for key,values in groups.items():
        complete = [v for v in values if v['status']=='completed']
        wins = sum(v['success'] for v in complete)
        times = [t for v in complete for t in v['call_ms']]
        item = dict(zip(['mode','H','K','N'],key),completed=len(complete),successes=wins,
                    wilson95=wilson(wins,len(complete)),censored=len(values)-len(complete))
        item['p50_ms'] = float(np.median(times)) if times else None
        item['p95_ms'] = float(np.quantile(times,.95)) if times else None
        item['p99_ms'] = float(np.quantile(times,.99)) if times else None
        item['amortized_ms_per_action'] = (sum(sum(v['call_ms']) for v in complete)/
                                          sum(v['steps'] for v in complete)) if complete else None
        live.append(item)
    save(out/'closed_loop_summary.json',live)
    def table(data,columns):
        lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
        for row in data:
            lines.append('| '+' | '.join(f'{row[c]:.5g}' if isinstance(row.get(c),float)
                         else str(row.get(c,'')) for c in columns)+' |')
        return '\n'.join(lines)
    label=json.loads((out/'config.json').read_text()).get('model_label','GR00T N1.6')
    report = '# RTX 3090 '+label+' pilot\n\n'+now()+'\n\n'
    report += (('Small SmolVLA-LIBERO baseline; BF16 autocast, full N=10. H is generated latent length, K is executed length.\n\n'
                if label=='SmolVLA' else
                'Public community LIBERO checkpoint; BF16. H is generated latent length; K is executed length. '
               'Native H=50 is decoded to 16 by this checkpoint. Shorter H is an untrained inference change.\n\n'
                )+'Offline NMSE is demo-action error, not task success; public training overlap is possible. '
               'Teacher NMSE is agreement with fresh H16/N4. K rows sharing an inference are not independent samples. '
               'GPU calls include preprocessing, vision, VLM, action head and decode; rendering/control are excluded. '
               'Simulator pauses during inference, so live success is not a real-time deadline result.\n\n')
    report += '## Offline\n\n'+table(summary,['mode','H','K','N','lag','samples','expert_nmse','teacher_nmse','call_p50_ms','call_p95_ms','amortized_ms_per_action'])
    report += '\n\n## Closed loop\n\n'+table(live,['mode','H','K','N','completed','successes','wilson95','p50_ms','p95_ms','amortized_ms_per_action'])
    (out/'RESULTS.md').write_text(report)
    save(out/'status.json',dict(status,updated_utc=now(),offline_rows=len(offline),rollouts=len(closed)))


def offline_run(pilot, config, out, samples, std, deadline):
    done = {r['trial_id'] for r in rows(out/'offline.jsonl')}
    for case in samples:
        if time.monotonic()>deadline:return
        fresh = case['fresh']
        teacher = command_actions(pilot.predict(fresh,16,4,config['seed']))
        variants = [('both_fresh',h,4,0,False) for h in config['horizons']]
        for lag in config['lags']:
            variants += [(mode,16,4,lag,False) for mode in ['external_stale','wrist_stale','both_stale']]
            variants += [('external_stale_cached',16,n,lag,True) for n in [2,4,8]]
        # Rotate measurement order by case; warmups and cache fills are excluded but saved.
        rng = np.random.default_rng(config['seed']+case['frame'])
        rng.shuffle(variants)
        for mode,h,n,lag,cache in variants:
            if time.monotonic()>deadline:return
            eligible = [k for k in config['execute'] if k<=min(h,pilot.output_horizon)]
            ids = [f"{case['id']}:{mode}:{h}:{k}:{n}:{lag}" for k in eligible]
            if all(i in done for i in ids):continue
            obs = fresh if lag==0 else camera_obs(fresh,case['stale'][lag],mode.replace('_cached',''))
            if cache:
                pilot.clear_cache()
                pilot.predict(case['stale'][lag],h,n,config['seed'],cache=True)
                fill_ms = pilot.last['total_ms']
            else: fill_ms=0.0
            actions = command_actions(pilot.predict(obs,h,n,config['seed'],cache=cache))
            timing = {key:value for key,value in pilot.last.items() if key not in ('H','N')}
            for k,trial_id in zip(eligible,ids):
                if trial_id in done:continue
                row = dict(trial_id=trial_id,case=case['id'],episode=case['episode'],frame=case['frame'],
                           phase=case['phase'],mode=mode,H=h,K=k,N=n,lag=lag,age_ms=lag*50,
                           expert_nmse=float(np.mean(((actions[:k]-case['target'][:k])/std)**2)),
                           teacher_nmse=float(np.mean(((actions[:k]-teacher[:k])/std)**2)),
                           gripper_accuracy=float(np.mean(actions[:k,-1]==case['target'][:k,-1])),
                           actions=actions[:k].tolist(),cache_fill_ms=fill_ms,**timing)
                if lag:
                    row['pixel_mae_0_255']={camera:float(np.mean(np.abs(
                        fresh.images[camera].astype(float)-case['stale'][lag].images[camera].astype(float))))
                        for camera in ('image','wrist_image')}
                append(out/'offline.jsonl',row)
            summarize(out,{'status':'running','phase':'offline','last_case':case['id']})
        print(json.dumps({'event':'offline_case','case':case['id'],'time':now()}),flush=True)


def setup_libero():
    import yaml
    base = ROOT/'.runtime/vendor/LIBERO-master/libero/libero'
    config = ROOT/'.runtime/libero_config'
    config.mkdir(exist_ok=True)
    (config/'config.yaml').write_text(yaml.safe_dump(dict(benchmark_root=str(base),
        bddl_files=str(base/'bddl_files'),init_states=str(base/'init_files'),
        assets=str(base/'assets'),datasets=str(ROOT/'data'))))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    return benchmark.get_benchmark_dict()['libero_spatial'](),get_libero_path,OffScreenRenderEnv


def sim_obs(obs,instruction):
    quat=obs['robot0_eef_quat'].copy();w=np.clip(quat[3],-1,1);den=np.sqrt(1-w*w)
    rotation=np.zeros(3) if np.isclose(den,0) else quat[:3]*2*np.arccos(w)/den
    return Observation(np.concatenate([obs['robot0_eef_pos'],rotation,obs['robot0_gripper_qpos']]).astype(np.float32),
                       {'image':obs['agentview_image'][::-1,::-1].copy(),
                        'wrist_image':obs['robot0_eye_in_hand_image'][::-1,::-1].copy()},instruction)


def closed_run(pilot,config,out,deadline):
    suite,get_path,Env=setup_libero()
    done = {r['trial_id'] for r in rows(out/'closed_loop.jsonl') if r['status']=='completed'}
    cells=config.get('closed_cells') or [('both_fresh',50,8,4),('both_fresh',16,4,4),('both_fresh',16,16,4),
           ('both_fresh',16,8,4),('both_fresh',8,4,4),('both_fresh',4,4,4),
           ('external_stale_cached',16,4,4),('wrist_stale',16,4,4),('both_stale',16,4,4),
           ('external_stale',16,4,4),('external_stale_cached',16,4,8),('both_fresh',16,4,2)]
    # First cover every cell at the same initial state, then add paired repetitions.
    for init in config['initial_states']:
        for task_id in config['tasks']:
            for mode,h,k,n in cells:
                if time.monotonic()>deadline:return
                trial_id=f'{task_id}:{init}:{mode}:{h}:{k}:{n}'
                if trial_id in done:continue
                task=suite.get_task(task_id)
                env=None
                record=dict(trial_id=trial_id,task=task_id,initial_state=init,mode=mode,H=h,K=k,N=n,
                            seed=config['seed'],status='running',call_ms=[],steps=0,success=False)
                start=time.monotonic()
                try:
                    env=Env(bddl_file_name=str(Path(get_path('bddl_files'))/task.problem_folder/task.bddl_file),
                            camera_heights=256,camera_widths=256)
                    env.seed(0);env.reset()
                    # These are official public LIBERO simulator state tensors, not arbitrary checkpoints.
                    init_file=Path(get_path('init_states'))/task.problem_folder/task.init_states_file
                    initial=pilot.torch.load(init_file,weights_only=False,map_location='cpu')[init]
                    obs=env.set_init_state(initial)
                    for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
                    pilot.clear_cache();slow=None;slow_step=-1000;calls=0;age=[];frames=[]
                    while record['steps']<config['max_control_steps']:
                        if time.monotonic()>deadline:
                            record['status']='censored';break
                        fresh=sim_obs(obs,task.language)
                        if slow is None or record['steps']-slow_step>=config['external_refresh_steps']:
                            slow=fresh;slow_step=record['steps']
                        current=camera_obs(fresh,slow,mode.replace('_cached',''))
                        candidate_age=record['steps']-slow_step
                        external_age=candidate_age if mode in ('external_stale','external_stale_cached','both_stale') else 0
                        wrist_age=candidate_age if mode in ('wrist_stale','both_stale') else 0
                        age.append(max(external_age,wrist_age))
                        act=command_actions(pilot.predict(current,h,n,config['seed']+calls,
                                                          cache=mode.endswith('_cached')))
                        record['call_ms'].append(pilot.last['total_ms'])
                        append(out/'calls.jsonl',dict(trial_id=trial_id,call=calls,control_step=record['steps'],
                                                     camera_age_steps=age[-1],external_age_steps=external_age,
                                                     wrist_age_steps=wrist_age,**pilot.last))
                        calls+=1
                        for a in act[:k]:
                            obs,_,success,_=env.step(a.tolist());record['steps']+=1
                            if record['steps']%4==0: frames.append(obs['agentview_image'][::-1,::-1].copy())
                            if success:record['success']=True;break
                            if record['steps']>=config['max_control_steps']:break
                        if record['success']:break
                    if record['status']=='running':record['status']='completed'
                    record['mean_camera_age_steps']=float(np.mean(age)) if age else None
                    if frames:
                        from PIL import Image
                        images=[Image.fromarray(x).resize((192,192)) for x in frames]
                        name=trial_id.replace(':','_')+'.gif'
                        images[0].save(out/'replays'/name,save_all=True,append_images=images[1:],duration=200,loop=0)
                except Exception:
                    record['status']='error';record['error']=traceback.format_exc()
                finally:
                    if env is not None:env.close()
                record['wall_s']=time.monotonic()-start
                append(out/'closed_loop.jsonl',record)
                summarize(out,{'status':'running','phase':'closed_loop','last_trial':trial_id})
                print(json.dumps({k:v for k,v in record.items() if k!='call_ms'}),flush=True)
                if record['status']=='error':raise RuntimeError(record['error'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/gr00t16_wrist_pilot.json')
    parser.add_argument('--output',default='results/wrist_gr00t_20261117/run')
    parser.add_argument('--minutes',type=float,default=40)
    parser.add_argument('--phase',choices=['all','offline','closed','smoke'],default='all')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True);(out/'replays').mkdir(exist_ok=True)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise ValueError('Use a new output for changed configuration')
    save(out/'config.json',config)
    import fcntl
    lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+args.minutes*60
    try:
        samples,std=cases(config)
        pilot=Gr00t16Pilot(str(ROOT/config['checkpoint']),config['device'])
        parity=pilot.parity(samples[0]['fresh'])
        for _ in range(2):pilot.predict(samples[0]['fresh'],16,4,config['seed'])
        revision=subprocess.check_output(['git','-C',str(ROOT/'.runtime/Isaac-GR00T-n16'),'rev-parse','HEAD'],text=True).strip()
        manifest=dict(started_utc=now(),device=pilot.torch.cuda.get_device_name(),
            torch=pilot.torch.__version__,cuda=pilot.torch.version.cuda,parity=parity,
            upstream_revision=revision,config=config,expert_std=std.tolist(),
            data_repo='physical-intelligence/libero',data_revision='a4336d589d589045d1c56423ffdf3b88a0e19b1f',
            libero_revision='8f1084e3132a39270c3a13ebe37270a43ece2a01',
            source_split='train; checkpoint overlap possible',
            runtime_patch='Gr00tPolicy AutoModel.from_pretrained: torch_dtype=bfloat16, low_cpu_mem_usage=True (16GB host RAM)',
            peak_allocated_gib=pilot.torch.cuda.max_memory_allocated()/1024**3,
            code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                         ['vla_eval/gr00t16_pilot.py','scripts/run_gr00t_wrist_pilot.py']})
        save(out/'manifest.json',manifest);print(json.dumps(manifest),flush=True)
        if args.phase in ('all','offline'):offline_run(pilot,config,out,samples,std,deadline)
        if args.phase in ('all','closed'):closed_run(pilot,config,out,deadline)
        summarize(out,{'status':'budget_finished' if time.monotonic()>deadline else 'completed',
                       'phase':args.phase})
    except Exception:
        save(out/'status.json',{'status':'failed','error':traceback.format_exc(),'updated_utc':now()})
        raise


if __name__=='__main__':main()
