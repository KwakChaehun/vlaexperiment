"""Real-model local pilot with incremental artifacts; no synthetic accuracy results."""
from __future__ import annotations
import argparse, csv, gc, hashlib, io, json, os, sys, time, traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from vla_eval.contracts import Observation
from vla_eval.smolvla import SmolPilot
from vla_eval.statistics import wilson


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def append(path, row):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def scores(actions, target, mask, std):
    length = min(len(actions), len(target))
    valid = mask[:length]
    error = actions[:length][valid]-target[:length][valid]
    normalized = error/std
    return {'action_mse':float(np.mean(error**2)), 'normalized_action_mse':float(np.mean(normalized**2)),
            'translation_command_mse':float(np.mean(error[:,:3]**2)),
            'rotation_command_mse':float(np.mean(error[:,3:6]**2)),
            'gripper_accuracy':float(np.mean((actions[:length][valid,6]>0)==(target[:length][valid,6]>0))),
            'out_of_range_fraction':float(np.mean(np.abs(actions[:length])>1)),
            'first_action_nmse':float(np.mean(((actions[0]-target[0])/std)**2))}


def quat_to_training_rotvec(quat):
    """Match LIBERO's quat2axisangle convention, including negative scalar quaternions."""
    quat=np.asarray(quat,dtype=float)
    scalar=float(np.clip(quat[3],-1.,1.))
    denominator=np.sqrt(1.-scalar*scalar)
    if np.isclose(denominator,0.):return np.zeros(3)
    return quat[:3]*(2.*np.arccos(scalar))/denominator


def get_cases():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from PIL import Image
    metadata = json.loads((ROOT/'data/libero_pilot.source.json').read_text())
    cases = []
    train_episodes = {1261,1267,1280}
    for episode in metadata['episodes']:
        file = hf_hub_download('physical-intelligence/libero',
            f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet',
            repo_type='dataset', revision=metadata['source_revision'], cache_dir=str(ROOT/'data/hf_cache'),
            local_files_only=True, token=False)
        rows = pq.read_table(file).to_pylist()
        states = np.asarray([r['state'] for r in rows], dtype=np.float32)
        actions = np.asarray([r['actions'] for r in rows], dtype=np.float32)
        with np.load(ROOT/'data/libero_pilot.npz',allow_pickle=False) as data:
            matches=np.flatnonzero(data['episode_id']==f'libero_{episode:06d}')
            instruction=str(data['instruction'][matches[0]])
        events = []
        for i in np.flatnonzero(np.diff(np.sign(actions[:,6]))!=0)+1:
            p=int(i)-2
            if 8 <= p < len(rows)-4:
                events.append((p,'before_gripper_close' if actions[i,6]>0 else 'before_gripper_open'))
        chosen=events[:2]
        for p in np.linspace(8,len(rows)-5,8,dtype=int):
            if all(abs(int(p)-q)>4 for q,_ in chosen):
                chosen.append((int(p),'uniform_motion'))
            if len(chosen)>=4:
                break
        def obs_at(index, state=None):
            images={}
            for key, field in [('image','image'),('wrist_image','wrist_image')]:
                with Image.open(io.BytesIO(rows[index][field]['bytes'])) as im:
                    images[key]=np.asarray(im.convert('RGB')).copy()
            return Observation(states[index].copy() if state is None else state.copy(),images,instruction)
        for p,phase in chosen:
            lag=4; previous=p-lag
            predicted = states[previous] + lag*(states[previous]-states[previous-1])
            actual=obs_at(p)
            stale=obs_at(previous)
            forecast=obs_at(previous,predicted)
            exact_state=obs_at(previous,states[p])
            target=np.zeros((50,7),dtype=np.float32)
            valid=min(50,len(rows)-p)
            target[:valid]=actions[p:p+valid]
            mask=np.arange(50)<valid
            cases.append({'id':f'{episode}:{p}','episode_id':str(episode),'frame':p,'phase':phase,
                          'split':'train' if episode in train_episodes else 'test','instruction':instruction,
                          'actual':actual,'stale':stale,'forecast':forecast,'oracle_proprio':exact_state,
                          'target':target,'mask':mask,
                          'forecast_position_error_mm':float(np.linalg.norm(predicted[:3]-states[p,:3])*1000),
                          'forecast_rotation_vector_error_rad':float(np.linalg.norm(predicted[3:6]-states[p,3:6])),
                          'lag_frames':lag})
    return cases


def report(out, state):
    state['updated_utc']=now()
    rows=read_rows(out/'offline.jsonl'); closed=read_rows(out/'closed_loop.jsonl')
    groups=defaultdict(list)
    for r in rows:
        groups[(r['experiment'],r['method'],r.get('H'),r.get('K'),r.get('split'))].append(r)
    summary=[]
    for key,rr in groups.items():
        s=dict(zip(['experiment','method','H','K','split'],key));s['samples']=len(rr)
        s['episodes']=len({r['episode_id'] for r in rr})
        for metric in ['normalized_action_mse','teacher_nmse','first_action_nmse','gripper_accuracy',
                       'ae_ms','backbone_ms','post_cpu_ms','forecast_position_error_mm']:
            values=[r[metric] for r in rr if metric in r]
            if values:s[metric]=float(np.mean(values))
        summary.append(s)
    write_json(out/'offline_summary.json',summary)
    cl_groups=defaultdict(list)
    for r in closed:
        cl_groups[(r['H'],r['K'])].append(r)
    cls=[]
    for (h,k),rr in cl_groups.items():
        completed=[r for r in rr if r['status']=='completed']
        wins=sum(int(r['success']) for r in completed)
        cls.append({'H':h,'K':k,'completed':len(completed),'successes':wins,
                    'success_rate':wins/len(completed) if completed else None,
                    'wilson_ci95':wilson(wins,len(completed)),
                    'censored_or_errors':len(rr)-len(completed)})
    write_json(out/'closed_loop_summary.json',cls)
    pairs=[]
    prior_rows={(r['case_id'],r['method']):r for r in rows if r['experiment']=='prior' and r['split']=='test'}
    for a,b in [('C_ridge','B_ridge'),('C_flow_refine','B_few_steps'),('D_partial','A_fresh'),
                ('forecast_prior','A_fresh'),('C_ridge','C_shuffled_ridge')]:
        for metric in ['normalized_action_mse','teacher_nmse','gripper_accuracy']:
            units=defaultdict(list)
            for (case,method),rr in prior_rows.items():
                ref=prior_rows.get((case,b))
                if method==a and ref and metric in rr and metric in ref:
                    units[rr['episode_id']].append(rr[metric]-ref[metric])
            if units:
                x=np.asarray([np.mean(v) for v in units.values()]);rng=np.random.default_rng(442)
                boot=x[rng.integers(0,len(x),size=(2000,len(x)))].mean(axis=1)
                pairs.append({'comparison':a+' - '+b,'metric':metric,'episodes':len(x),
                              'mean_delta':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()})
    write_json(out/'paired_prior_comparisons.json',pairs)
    def table(records,columns):
        lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
        for r in records:
            lines.append('| '+' | '.join(f'{r[c]:.5g}' if isinstance(r.get(c),float) else str(r.get(c,'')) for c in columns)+' |')
        return '\n'.join(lines)
    text=f'''# 로컬 실제 모델 평가 결과

마지막 갱신(UTC): {now()}<br>
상태: **{state['status']} / {state.get('phase','')}**<br>
모델: **HuggingFaceVLA/smolvla_libero**, 고정 revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. CPU float32, native H=50/N=10. RTX 3090 결과가 아닙니다.

## 지금 판단할 수 있는 범위

- 실제 flow 모델의 생성 길이별 action 오차와 CPU action-expert 비용을 측정합니다. 짧은 H는 재학습하지 않은 분포 밖 조건입니다.
- 선행 계산은 4개 frame 전의 이미지와 past-only 등속 외삽 state를 사용합니다. 예상 좌표만으로 미래 카메라 feature를 정확히 만들 수 있다는 전제는 두지 않습니다.
- `oracle_proprio`는 실제 미래 state를 사용한 진단 조건이며 배포 가능한 예측기가 아닙니다.
- Offline expert action MSE와 teacher 일치는 실제 정밀 조작 성공률이 아닙니다. 데이터는 공개 train demonstration이고 pretrained 모델의 training overlap이 가능합니다.
- `C_ridge`/`B_ridge`는 같은 소형 Ridge 가중치를 사용하는 제한적인 baseline입니다. 학습 target은 fresh SmolVLA teacher이며, 강한 small VLA를 대체하지 않습니다.
- `C_flow_refine`/`D_partial`은 최신 관측 VLM 비용을 모두 포함합니다. Few-step refinement가 VLM까지 저렴하게 만드는 것은 아닙니다.
- Closed-loop 표는 추론 중 simulator가 멈추는 **zero-inference-delay 품질 평가**입니다. CPU 실시간 성능이나 PIM speedup을 뜻하지 않습니다. Timeout/censored는 task 실패율에 섞지 않습니다.

## 1. 선행 계산과 보정: test episode

{table([s for s in summary if s['experiment']=='prior' and s['split']=='test'],['method','samples','episodes','normalized_action_mse','teacher_nmse','gripper_accuracy','post_cpu_ms'])}

{table(pairs,['comparison','metric','episodes','mean_delta','ci95'])}

오차 지표는 음수 차이가 앞 방식에 유리하고 gripper accuracy는 양수가 유리합니다. Episode 3개 수준의 pilot 구간이므로 확정적 일반화 근거가 아닙니다.

## 2. 실제 생성 H: 공통 첫 action 비교

{table([s for s in summary if s['experiment']=='horizon' and s['K']==1],['H','split','samples','episodes','first_action_nmse','ae_ms','backbone_ms'])}

같은 관측의 H별 예측을 여러 K prefix로 채점한 offline 표본은 독립적인 제어 rollout이 아닙니다. K 효과는 아래 closed-loop에서 확인합니다.

## 3. Closed-loop H/K 품질

{table(cls,['H','K','completed','successes','success_rate','wilson_ci95','censored_or_errors'])}

## 추가 지표와 연구 결정 기준

1. B–C 차이는 episode로 짝지어 확인합니다. 이득이 없는 경우 강한 Small, 학습량, 미래 영상 불일치를 분리하기 전에는 Heavy prior 전체를 기각하지 않습니다.
2. `stale_prior` → `forecast_prior` → `oracle_proprio` → fresh A의 차이로 좌표 외삽 오차와 오래된 시각 조건의 영향을 나눕니다.
3. First-action NMSE, translation/rotation command MSE, gripper sign accuracy, action saturation을 따로 봅니다. Command 오차는 말단 mm 정밀도와 다릅니다.
4. H/K의 품질 이득이 확인되면 RTX 3090에서 stage p50/p95/p99, 실제 tensor shape, DRAM traffic, deadline miss, GPU concurrent 경합을 측정합니다.
5. PIM은 실제 지원 연산과 전송/동기화 비용을 포함해 GPU-only concurrent, GPU+NPU와 비교해야 합니다.

현재 원자료: `offline.jsonl`, `closed_loop.jsonl`, `offline_summary.json`, `closed_loop_summary.json`, `status.json`, `manifest.json`. 작업 오류는 `errors.jsonl`에 보존합니다. 결과가 적으면 결론은 미정입니다.
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    import html
    body=html.escape(text)
    (out/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>VLA local pilot</title><style>body{font:16px/1.65 system-ui;max-width:1200px;margin:40px auto;padding:20px}pre{white-space:pre-wrap}</style><pre>'+body+'</pre>',encoding='utf-8')
    write_json(out/'status.json',state)


def offline(model,out,state,deadline):
    from vla_eval.small import fit_ridge, RidgeSmall, observation_features
    cases=get_cases()
    write_json(out/'cases.json',[{k:v for k,v in c.items() if k not in {'actual','stale','forecast','oracle_proprio','target','mask'}} for c in cases])
    cache_file=out/'predictions.npz'
    cache={}
    if cache_file.exists():
        with np.load(cache_file,allow_pickle=False) as d:cache={k:d[k] for k in d.files}
    existing={(r['experiment'],r['case_id'],r['method'],r.get('H'),r.get('K')) for r in read_rows(out/'offline.jsonl')}
    teachers=[];priors=[];features=[];training_ids=[]
    def emit(case,method,a,h=50,k=4,experiment='prior',**extra):
        key=(experiment,case['id'],method,h,k)
        if key in existing:return
        mask=case['mask'][:k]
        metrics=scores(a[:k],case['target'][:k],mask,model.action_std)
        row={'experiment':experiment,'case_id':case['id'],'episode_id':case['episode_id'],
             'split':case['split'],'phase':case['phase'],'H':h,'K':k,
             'N':2 if method=='B_few_steps' else (None if 'ridge' in method else 10),'method':method,
             'diagnostic_only':True,'forecast_position_error_mm':case['forecast_position_error_mm'],
             **metrics,**extra}
        if 'fresh_'+case['id'] in cache:
            fresh=cache['fresh_'+case['id']]
            row['teacher_nmse']=float(np.mean(((a[:k]-fresh[:k])/model.action_std)**2))
        append(out/'offline.jsonl',row);existing.add(key)
    for index,case in enumerate(cases):
        if time.monotonic()>deadline:break
        state.update(phase='offline',completed_cases=index,total_cases=len(cases),updated_utc=now())
        report(out,state)
        print('OFFLINE',index+1,case['id'],case['phase'],flush=True)
        if 'done_'+case['id'] in cache:
            features.append(observation_features(case['actual']))
            teachers.append((cache['fresh_'+case['id']]-model.action_mean)/model.action_std)
            priors.append((cache['prior_'+case['id']]-model.action_mean)/model.action_std)
            training_ids.append(case['episode_id'])
            continue
        cond=model.condition(case['actual'])
        a,latent,ae=model.predict_condition(cond,50,10,0)
        cache['fresh_'+case['id']]=a
        emit(case,'A_fresh',a,ae_ms=ae,backbone_ms=cond.backbone_ms,
             post_cpu_ms=cond.preparation_ms+cond.backbone_ms+ae)
        few,_,few_ms=model.predict_condition(cond,50,2,0)
        emit(case,'B_few_steps',few,ae_ms=few_ms,backbone_ms=cond.backbone_ms,
             post_cpu_ms=cond.preparation_ms+cond.backbone_ms+few_ms)
        for h in [1,4,16,50]:
            if h==50:act,hm=a,ae
            else:act,_,hm=model.predict_condition(cond,h,10,0)
            for k in [1,4,16,50]:
                if k<=h:
                    emit(case,'flow',act,h,k,'horizon',ae_ms=hm,backbone_ms=cond.backbone_ms,
                         post_cpu_ms=cond.preparation_ms+cond.backbone_ms+hm)
        for variant in ['forecast','stale','oracle_proprio']:
            previous=model.condition(case[variant])
            if variant=='forecast':
                partial,pre8_ms=model.integrate(previous,50,10,0,stop_step=8)
                full,pre2_ms=model.integrate(previous,50,10,0,latent=partial,start_step=8)
                pre_ms=pre8_ms+pre2_ms
                prior=model.decode(full)
                cache['prior_'+case['id']]=prior
                resumed,resume_ms=model.integrate(cond,50,10,0,latent=partial,start_step=8)
                emit(case,'D_partial',model.decode(resumed),post_cpu_ms=cond.preparation_ms+cond.backbone_ms+resume_ms,
                     pre_cpu_ms=previous.preparation_ms+previous.backbone_ms+pre8_ms,remaining_steps=2)
                renoised=.8*full+.2*model.noise(50,0)
                refined,refine_ms=model.integrate(cond,50,10,0,latent=renoised,start_step=8)
                emit(case,'C_flow_refine',model.decode(refined),post_cpu_ms=cond.preparation_ms+cond.backbone_ms+refine_ms,
                     pre_cpu_ms=previous.preparation_ms+previous.backbone_ms+pre_ms,remaining_steps=2)
            else:
                prior,_,pre_ms=model.predict_condition(previous,50,10,0)
            emit(case,variant+'_prior',prior,post_cpu_ms=0.,
                 pre_cpu_ms=previous.preparation_ms+previous.backbone_ms+pre_ms,
                 privileged=variant=='oracle_proprio')
            del previous
        features.append(observation_features(case['actual']))
        teachers.append((a-model.action_mean)/model.action_std)
        priors.append((cache['prior_'+case['id']]-model.action_mean)/model.action_std)
        training_ids.append(case['episode_id'])
        cache['done_'+case['id']]=np.array(True)
        temporary=cache_file.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,**cache);temporary.replace(cache_file)
        state.update(completed_cases=index+1,updated_utc=now());report(out,state)
        del cond;gc.collect()
    train=[i for i,c in enumerate(cases[:len(features)]) if c['split']=='train']
    test=[i for i,c in enumerate(cases[:len(features)]) if c['split']=='test']
    if len(train)>=3 and test:
        fit_ridge(np.asarray(features)[train],np.asarray(priors)[train],np.asarray(teachers)[train],
                  np.asarray(training_ids)[train],out/'correction_ridge.npz',regularization=10.)
        small=RidgeSmall(out/'correction_ridge.npz')
        for j,i in enumerate(test):
            case=cases[i]
            t=time.perf_counter();b=small.predict(case['actual'],50,10,0)
            bms=(time.perf_counter()-t)*1000
            t=time.perf_counter();c=small.correct(case['actual'],priors[i],case['forecast'])
            cms=(time.perf_counter()-t)*1000
            other=next((v for v in test if cases[v]['episode_id']!=case['episode_id']),test[(j+1)%len(test)])
            shuffled=small.correct(case['actual'],priors[other],case['forecast'])
            emit(case,'B_ridge',b*model.action_std+model.action_mean,post_cpu_ms=bms)
            emit(case,'C_ridge',c*model.action_std+model.action_mean,post_cpu_ms=cms)
            emit(case,'C_shuffled_ridge',shuffled*model.action_std+model.action_mean)
    state['offline_finished']=len(features)==len(cases);report(out,state)
    return state['offline_finished']


def closed_loop(model,out,state,deadline,settings=None,task_ids=None,initial_states=None,
                flow_steps=10,max_actions=220,seed_offset=0,update=None):
    os.environ['LIBERO_CONFIG_PATH']=str(ROOT/'.runtime/libero_config')
    os.environ.setdefault('MUJOCO_GL','glfw' if os.name=='nt' else 'egl')
    sys.path.insert(0,str(ROOT/'.runtime/vendor/LIBERO-master'))
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite=benchmark.get_benchmark_dict()['libero_spatial']()
    # Interleave settings across matched states; small pilot, not a definitive SR claim.
    settings=settings if settings is not None else [(50,10),(4,4),(50,1),(4,1),(16,4),(50,4)]
    task_ids=task_ids if task_ids is not None else [0,1,2]
    initial_states=initial_states if initial_states is not None else [0,1]
    update=update or report
    if any(not (1<=k<=h<=model.native_horizon) for h,k in settings):
        raise ValueError('Every closed-loop setting must satisfy 1 <= K <= H <= native H')
    finished={(r['task_id'],r['initial_state'],r['H'],r['K']) for r in read_rows(out/'closed_loop.jsonl')
              if r.get('status')=='completed'}
    for initial in initial_states:
        for h,k in settings:
            for task_id in task_ids:
                if time.monotonic()>deadline:return
                if (task_id,initial,h,k) in finished:continue
                task=suite.get_task(task_id)
                state.update(phase='closed_loop',task=task.language,H=h,K=k,initial_state=initial,updated_utc=now())
                update(out,state)
                print('CLOSED_LOOP',task_id,initial,h,k,flush=True)
                env=None;trace=[];success=False;status='completed';steps=0;path_length=0.;last=None;trajectory=[];commands=[];peak_contacts=0;frames=[]
                begun=time.perf_counter()
                try:
                    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                                           camera_heights=256,camera_widths=256)
                    env.seed(0);env.reset();obs=env.set_init_state(suite.get_task_init_states(task_id)[initial])
                    for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
                    last=np.asarray(obs['robot0_eef_pos']).copy()
                    while steps<max_actions and not success:
                        if time.monotonic()>deadline:
                            status='censored_budget';break
                        pos=np.asarray(obs['robot0_eef_pos'])
                        statevec=np.r_[pos,quat_to_training_rotvec(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']].astype(np.float32)
                        current=Observation(statevec,{'image':np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),
                                                     'wrist_image':np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])},task.language)
                        cond=model.condition(current)
                        actions,_,ae=model.predict_condition(cond,h,flow_steps,seed_offset+len(trace))
                        trace.append({'call':len(trace),'step':steps,'backbone_ms':cond.backbone_ms,
                                      'preparation_ms':cond.preparation_ms,'ae_ms':ae,'H':h,'K':k})
                        for act in actions[:k]:
                            obs,_,done,_=env.step(act.tolist());steps+=1
                            measured=np.asarray(obs['robot0_eef_pos']).copy()
                            path_length+=float(np.linalg.norm(measured-last));last=measured
                            trajectory.append(measured);commands.append(act.tolist())
                            peak_contacts=max(peak_contacts,int(env.env.sim.data.ncon))
                            if steps%5==0:frames.append(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]))
                            success=bool(done)
                            if success or steps>=max_actions:break
                        if len(trace)%10==0:
                            write_json(out/'current_episode.json',{'task_id':task_id,'initial_state':initial,
                                       'H':h,'K':k,'steps':steps,'policy_calls':len(trace),'updated_utc':now()})
                            print('  steps',steps,'calls',len(trace),flush=True)
                            from PIL import Image
                            Image.fromarray(np.ascontiguousarray(obs['agentview_image'][::-1,::-1])).save(out/'latest_view.png')
                        del cond;gc.collect()
                except Exception as error:
                    status='error';append(out/'errors.jsonl',{'phase':'closed_loop','task_id':task_id,'H':h,'K':k,'error':str(error),'traceback':traceback.format_exc()})
                finally:
                    if env is not None:env.close()
                row={'task_id':task_id,'task':task.language,'initial_state':initial,'seed':seed_offset,'H':h,'K':k,'N':flow_steps,
                     'status':status,'success':success if status=='completed' else None,'steps':steps,
                     'policy_calls':len(trace),'wall_seconds':time.perf_counter()-begun,
                     'eef_path_length_m':path_length,'peak_sampled_contact_count':peak_contacts,
                     'eef_jerk_rms_m_s3':float(np.sqrt(np.mean(np.diff(np.asarray(trajectory),n=3,axis=0)**2))/.05**3) if len(trajectory)>3 else None,
                     'command_second_difference_rms':float(np.sqrt(np.mean(np.diff(np.asarray(commands),n=2,axis=0)**2))) if len(commands)>2 else None,
                     'latency_mode':'zero_inference_delay', 'trace':trace}
                if frames:
                    from PIL import Image
                    replay_dir=out/'replays';replay_dir.mkdir(exist_ok=True)
                    replay=replay_dir/f'task{task_id}_init{initial}_H{h}_K{k}.gif'
                    images=[Image.fromarray(f) for f in frames]
                    images[0].save(replay,save_all=True,append_images=images[1:],duration=250,loop=0)
                    row['replay']=str(replay.relative_to(out))
                append(out/'closed_loop.jsonl',row);update(out,state)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/research_pilot')
    parser.add_argument('--hours',type=float,default=6.)
    parser.add_argument('--offline-only',action='store_true')
    args=parser.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();deadline=start+args.hours*3600
    state={'status':'running','phase':'load_model','started_utc':now(),'pid':os.getpid(),'updated_utc':now(),
           'deadline_utc':(datetime.now(timezone.utc)+timedelta(hours=args.hours)).isoformat()}
    if (out/'offline_done.json').exists():
        state.update(offline_finished=True,completed_cases=24,total_cases=24)
    report(out,state)
    try:
        model=SmolPilot(ROOT/'checkpoints/smolvla_libero',ROOT/'checkpoints/smolvlm_config',threads=4)
        from vla_eval.dataset import PairedDataset
        dataset=PairedDataset(ROOT/'data/libero_pilot.npz')
        sanity=model.sanity(dataset.observation(0));dataset.close()
        manifest={**model.metadata,**sanity,'created_utc':now(),'python':sys.version,
                  'dataset_revision':'a4336d589d589045d1c56423ffdf3b88a0e19b1f',
                  'hardware':'GTX 1650 4GB available, inference on CPU; RTX3090 not connected',
                  'environment_patches':['disable eager unrelated policy registrations',
                    'Windows MuJoCo DLL path and GLFW rendering','workspace-local robosuite log path',
                    'official initial-state torch.load weights_only=False'],
                  'checkpoint_config_sha256':hashlib.sha256((ROOT/'checkpoints/smolvla_libero/config.json').read_bytes()).hexdigest(),
                  'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [ROOT/'scripts/research_pilot.py',ROOT/'vla_eval/smolvla.py',ROOT/'vla_eval/arrival_rollouts.py']}}
        write_json(out/'manifest.json',manifest)
        if not (out/'offline_done.json').exists():
            complete=offline(model,out,state,deadline)
            if complete:write_json(out/'offline_done.json',{'updated_utc':now()})
        if not args.offline_only and time.monotonic()<deadline:
            os.environ['LIBERO_CONFIG_PATH']=str(ROOT/'.runtime/libero_config')
            os.environ.setdefault('MUJOCO_GL','glfw' if os.name=='nt' else 'egl')
            sys.path.insert(0,str(ROOT/'.runtime/vendor/LIBERO-master'))
            from vla_eval.arrival_rollouts import run_arrival_pilot
            run_arrival_pilot(model,out,state,deadline,report)
            if time.monotonic()<deadline:
                # Keep the target-sum diagnostic separate from the improved causal
                # forecast. Fresh baselines are exactly the same interventions.
                import shutil
                variant=out/'arrival_calibrated';variant.mkdir(exist_ok=True)
                for source in [out/'correction_ridge.npz',*out.glob('arrival_plan_task*.npz')]:
                    if source.exists():shutil.copy2(source,variant/source.name)
                existing={(r['task_id'],r['method']) for r in read_rows(variant/'arrival_closed_loop.jsonl')}
                for r in read_rows(out/'arrival_closed_loop.jsonl'):
                    if r.get('status')=='completed' and r['method'] in {'A_fresh','B_few_steps','B_ridge'} and (r['task_id'],r['method']) not in existing:
                        # Baseline actions never consume a forecast. Do not reuse
                        # forecast-dependent timings/errors as variant evidence.
                        r=dict(r);r['baseline_reused_from']='../arrival_closed_loop.jsonl'
                        r['source_forecast_position_error_mm']=r.pop('forecast_position_error_mm',None)
                        r['forecast_position_error_mm']=None
                        append(variant/'arrival_closed_loop.jsonl',r)
                def variant_update(ignored,progress):
                    progress['arrival_forecast_variant']='past_calibrated_command'
                    report(out,progress)
                run_arrival_pilot(model,variant,state,deadline,variant_update,
                                  forecast_mode='past_calibrated_command',
                                  methods=['C_flow_refine','D_partial','C_ridge'])
        if not args.offline_only and time.monotonic()<deadline:
            closed_loop(model,out,state,deadline)
        state.update(status='finished' if time.monotonic()<deadline else 'budget_finished',phase='done',updated_utc=now(),wall_seconds=time.monotonic()-start)
    except BaseException as error:
        state.update(status='failed',error=str(error),updated_utc=now())
        append(out/'errors.jsonl',{'error':str(error),'traceback':traceback.format_exc()})
        raise
    finally:report(out,state)


if __name__=='__main__':main()
