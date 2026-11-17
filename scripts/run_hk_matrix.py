"""Real SmolVLA H/K matrix and controlled prior diagnostics, with live reports."""
from __future__ import annotations
import argparse,gc,hashlib,html,json,os,sys,time,traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from vla_eval.hk_matrix import matrix_cells
from vla_eval.statistics import wilson
from scripts.research_pilot import append,read_rows,write_json,now,get_cases,closed_loop


def table(rows,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for r in rows:lines.append('| '+' | '.join(f'{r[c]:.5g}' if isinstance(r.get(c),float) else str(r.get(c,'')) for c in columns)+' |')
    return '\n'.join(lines)


def render(out,state,config,cells):
    state['updated_utc']=now();write_json(out/'status.json',state)
    # The last attempt of each trial is shown. Errors/censoring never become failures.
    trials={}
    for r in read_rows(out/'closed_loop.jsonl'):
        trials[(r['H'],r['K'],r['task_id'],r['initial_state'],r['seed'])]=r
    grouped=defaultdict(list)
    for r in trials.values():grouped[(r['H'],r['K'])].append(r)
    summary=[]
    expected=len(config['task_ids'])*len(config['initial_states'])
    for cell in cells:
        rr=grouped[(cell['H'],cell['K'])]
        completed=[r for r in rr if r['status']=='completed'];wins=sum(r['success'] for r in completed)
        trace=[t for r in completed for t in r['trace']]
        costs=[t['preparation_ms']+t['backbone_ms']+t['ae_ms'] for t in trace]
        used=sum(r['steps'] for r in completed);generated=sum(r['H']*r['policy_calls'] for r in completed)
        summary.append({**cell,'completed':len(completed),'planned':expected if cell['applicable'] else 0,
            'successes':wins,'success_rate':wins/len(completed) if completed else None,
            'wilson_ci95':wilson(wins,len(completed)),'errors_or_censored':len(rr)-len(completed),
            'cpu_call_p50_ms':float(np.median(costs)) if costs else None,
            'cpu_call_p95_ms':float(np.quantile(costs,.95)) if costs else None,
            'cpu_call_p99_ms':float(np.quantile(costs,.99)) if costs else None,
            'generation_utilization':used/generated if generated else None,
            'command_second_difference_rms':float(np.mean([r['command_second_difference_rms'] for r in completed])) if completed else None,
            'eef_jerk_rms_m_s3':float(np.mean([r['eef_jerk_rms_m_s3'] for r in completed])) if completed else None})
    write_json(out/'matrix_summary.json',summary)
    prior=read_rows(out/'prior_diagnostics/trials.jsonl')
    pg=defaultdict(list)
    for r in prior:pg[(r['prior_condition'],r['method'],r['post_velocity_evaluations'])].append(r)
    ps=[]
    for (condition,method,budget),rr in sorted(pg.items()):
        ps.append({'prior_condition':condition,'method':method,'post_steps':budget,'cases':len(rr),
            'expert_nmse':float(np.mean([r['expert_nmse'] for r in rr])),
            'teacher_nmse':float(np.mean([r['teacher_nmse'] for r in rr])),
            'gripper_accuracy':float(np.mean([r['gripper_accuracy'] for r in rr]))})
    write_json(out/'prior_diagnostics_summary.json',ps)
    phase_groups=defaultdict(list)
    for r in prior:
        phase_groups[(r['phase'],r['prior_condition'],r['method'],r['post_velocity_evaluations'])].append(r)
    by_phase=[]
    for (phase,condition,method,budget),rr in sorted(phase_groups.items()):
        by_phase.append({'phase':phase,'prior_condition':condition,'method':method,'post_steps':budget,
            'cases':len(rr),'expert_nmse':float(np.mean([r['expert_nmse'] for r in rr])),
            'teacher_nmse':float(np.mean([r['teacher_nmse'] for r in rr]))})
    write_json(out/'prior_diagnostics_by_phase.json',by_phase)
    # Every contrast fixes all other prior/refinement inputs. Compare the same cases.
    index={(r['case_id'],r['prior_condition'],r['method'],r['post_velocity_evaluations']):r for r in prior}
    contrasts=[
        ('replace_RGB_at_actual_state',('arrival_rgb_arrival_state','renoise_original_noise',2),('stale_rgb_arrival_state','renoise_original_noise',2)),
        ('replace_RGB_at_predicted_state',('arrival_rgb_predicted_state','renoise_original_noise',2),('stale_rgb_predicted_state','renoise_original_noise',2)),
        ('replace_state_at_stale_RGB',('stale_rgb_arrival_state','renoise_original_noise',2),('stale_rgb_predicted_state','renoise_original_noise',2)),
        ('independent_noise_same_observation',('arrival_rgb_arrival_state','renoise_independent_noise',2),('arrival_rgb_arrival_state','renoise_original_noise',2)),
        ('self_renoise_vs_original_fresh',('arrival_rgb_arrival_state','renoise_original_noise',2),('none','fresh_full',10)),
        ('forecast_renoise_vs_fresh_2step',('stale_rgb_predicted_state','renoise_original_noise',2),('none','fresh_few',2)),
        ('forecast_continue_vs_renoise',('stale_rgb_predicted_state','continue_intermediate',2),('stale_rgb_predicted_state','renoise_original_noise',2)),
    ]
    comparisons=[]
    for label,a,b in contrasts:
        for metric in ['expert_nmse','teacher_nmse']:
            units=defaultdict(list)
            for (case,condition,method,budget),r in index.items():
                other=index.get((case,*b))
                if (condition,method,budget)==a and other:
                    units[r['episode_id']].append(r[metric]-other[metric])
            if units:
                values=np.array([np.mean(v) for v in units.values()]);rng=np.random.default_rng(442)
                boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
                comparisons.append({'contrast':label,'metric':metric,'cases':sum(map(len,units.values())),
                    'episodes':len(units),'mean_delta':float(values.mean()),
                    'ci95':np.quantile(boot,[.025,.975]).tolist() if len(values)>=3 else [None,None]})
    write_json(out/'prior_diagnostic_comparisons.json',comparisons)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    data=np.full((len(config['horizons']),len(config['execute'])),np.nan)
    lookup={(r['H'],r['K']):r for r in summary}
    for i,h in enumerate(config['horizons']):
        for j,k in enumerate(config['execute']):
            r=lookup[(h,k)]
            if r['success_rate'] is not None:data[i,j]=r['success_rate']
    cmap=colormaps['viridis'].copy();cmap.set_bad('#e5e7eb')
    fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(np.ma.masked_invalid(data),vmin=0,vmax=1,cmap=cmap)
    for i,h in enumerate(config['horizons']):
        for j,k in enumerate(config['execute']):
            r=lookup[(h,k)]
            label='N/A\nK > H' if not r['applicable'] else (f"{r['successes']}/{r['completed']}" if r['completed'] else 'pending')
            ax.text(j,i,label,ha='center',va='center',color='white' if r.get('success_rate') is not None and r['success_rate']<.5 else '#17212b')
    ax.set(xticks=range(len(config['execute'])),xticklabels=config['execute'],yticks=range(len(config['horizons'])),yticklabels=config['horizons'],xlabel='Executed K',ylabel='Generated H',title='Real SmolVLA-LIBERO / N=10 / zero-delay simulator')
    fig.colorbar(im,ax=ax,label='Task success rate');fig.tight_layout();fig.savefig(out/'matrix.png',dpi=150);plt.close(fig)
    complete=sum(r['completed'] for r in summary);planned=sum(r['planned'] for r in summary)
    focused=[r for r in ps if r['post_steps']==2 or r['method'] in {'fresh_full','prior_unmodified'}]
    device=config.get('device','cpu')
    md=f'''# H/K matrix and prior diagnosis

상태: **{state['status']} / {state.get('phase')}**. 갱신 UTC: {state['updated_utc']}.<br>
Closed-loop 완료 **{complete}/{planned}**. SmolVLA-LIBERO **{device} / float32**, N={config['flow_steps']}. PIM 측정이 아닙니다. `cpu_call_*` 필드는 이전 버전과 호환되는 이름이며 실제 장치는 manifest의 device를 확인합니다. CUDA는 stage 경계에서 동기화해 측정합니다.

## H/K 16칸 중 13개 유효 조합

H={config['horizons']}, K={config['execute']}. K>H 3칸은 N/A이며 실패율에 넣지 않습니다. Action 반복·padding·추가 chunk 생성으로 K의 의미를 바꾸지 않습니다.

![H/K matrix](matrix.png)

{table(summary,['H','K','applicable','completed','planned','successes','success_rate','wilson_ci95','cpu_call_p50_ms','generation_utilization','errors_or_censored'])}

같은 3 task × 2 initial states를 새로 실행합니다. 이전 batch 결과를 복사하지 않습니다. Simulator는 추론 중 정지하므로 제어 품질 평가입니다. Short H는 재학습하지 않은 조건입니다. 작은 표본의 최대 성공률을 최적 설정으로 확정하지 않습니다. SmolVLA causal attention의 공통 prefix 성질은 GR00T와 다를 수 있습니다.

## Prior가 나빠진 원인: 관측과 보정 방식을 분리

{table(focused,['prior_condition','method','post_steps','cases','expert_nmse','teacher_nmse','gripper_accuracy'])}

{table(comparisons,['contrast','metric','cases','episodes','mean_delta','ci95'])}

차이는 앞 조건−뒤 조건이며 NMSE는 음수면 앞 조건이 유리합니다. 독립 episode 단위 paired bootstrap이며, 3 episode 미만이면 CI를 표시하지 않습니다. 2-step 조건 외에 4/8-step 비교는 `prior_diagnostics_summary.json`, grasp 직전/직후 및 이동 구간별 결과는 `prior_diagnostics_by_phase.json`에 있습니다.

- 이미지: stale RGB vs 실제 arrival RGB. State: predicted vs 실제 arrival state. 실제 미래 관측을 prior에 넣은 조건은 **privileged 진단**입니다.
- `renoise_original_noise`: 기존 방식. 완성 prior와 원래 noise를 섞어 중간 flow time에서 다시 시작합니다.
- `renoise_independent_noise`: 다른 noise로 같은 방식을 반복하여 noise와 prior의 상관 영향을 분리합니다.
- `continue_intermediate`: 완성 action을 다시 섞지 않고 실제 중간 latent를 이어 계산합니다.
- `fresh_split_identity`: 같은 관측에서 중간 latent를 이어 계산했을 때 원래 full 결과와 일치해야 합니다.
- 같은 실제 관측의 prior조차 재보정하면 나빠지는지 먼저 확인합니다. 그것이 관찰되면 오래된 이미지만으로 모든 차이를 설명할 수 없습니다.

이 진단은 **미래 좌표에서 feature를 생성하는 모델을 구현한 실험이 아닙니다**. 정답 미래 RGB의 효용을 먼저 측정하는 oracle 조건을 포함합니다. 좌표만으로 미래 wrist RGB가 결정되지는 않으며, 미래 camera pose·scene geometry·가림을 다룰 표현이 필요합니다. Offline action NMSE는 정밀 조작 성공률이 아닙니다.

원자료: [closed_loop.jsonl](closed_loop.jsonl), [prior trials](prior_diagnostics/trials.jsonl), [status](status.json), [manifest](manifest.json), [오류](errors.jsonl). GIF는 `replays/`에 저장합니다. 오류/시간 예산 중단은 task 실패와 분리합니다.
'''
    (out/'RESULTS.md').write_text(md,encoding='utf-8')
    import markdown
    body=markdown.markdown(md,extensions=['tables','fenced_code'])
    (out/'RESULTS.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>H/K matrix</title><style>body{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:20px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd4dc;padding:6px}th{background:#eef3f7}img{max-width:100%}</style>'+body+'</html>',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smolvla_hk_matrix.json')
    p.add_argument('--output',default='results/hk_matrix_20261117');p.add_argument('--hours',type=float,default=12)
    p.add_argument('--plan-only',action='store_true');args=p.parse_args()
    if args.hours<=0:raise ValueError('hours must be positive')
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    plan_path=out/'config.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=config:
        raise ValueError('Different config: use a new output directory')
    write_json(plan_path,config)
    # The installed, pinned checkpoint defines native H; do not guess it.
    native=json.loads((ROOT/config['checkpoint']/'config.json').read_text())['chunk_size']
    cells=matrix_cells(config['horizons'],config['execute'],native)
    write_json(out/'matrix_plan.json',cells)
    state={'status':'planned' if args.plan_only else 'running','phase':'load_model','pid':os.getpid(),
           'started_utc':now(),'deadline_utc':(datetime.now(timezone.utc)+timedelta(hours=args.hours)).isoformat()}
    render(out,state,config,cells)
    if args.plan_only:return
    deadline=time.monotonic()+args.hours*3600
    def update(ignored,progress):render(out,progress,config,cells)
    try:
        from vla_eval.smolvla import SmolPilot
        from vla_eval.dataset import PairedDataset
        model=SmolPilot(ROOT/config['checkpoint'],ROOT/config['vlm_config'],config['threads'],device=config.get('device','cpu'))
        dataset=PairedDataset(ROOT/'data/libero_pilot.npz')
        parity=model.sanity(dataset.observation(0));dataset.close()
        write_json(out/'manifest.json',{**model.metadata,**parity,'config':config,'created_utc':now(),
            'source_archives':json.loads((ROOT/'.runtime/assets_manifest.json').read_text()) if (ROOT/'.runtime/assets_manifest.json').exists() else {'LIBERO.zip':'8f1084e3132a39270c3a13ebe37270a43ece2a01','lerobot':'0.4.4'},
            'data_revision':'a4336d589d589045d1c56423ffdf3b88a0e19b1f',
            'python':sys.version,'code_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [Path(__file__),ROOT/'scripts/research_pilot.py',ROOT/'vla_eval/smolvla.py',ROOT/'vla_eval/prior_diagnostics.py',ROOT/'vla_eval/hk_matrix.py']}})
        if config['prior_diagnostics']['enabled'] and time.monotonic()<deadline:
            from vla_eval.prior_diagnostics import run
            def progress(case,condition,completed,total):
                state.update(phase='prior_diagnostics',case_id=case,condition=condition,
                             prior_cases_completed=completed,prior_cases_total=total)
                render(out,state,config,cells)
                print('PRIOR',case,condition,completed,'/',total,flush=True)
            try:state['prior_diagnostics_finished']=run(model,out,get_cases(),config['prior_diagnostics'],deadline,progress)
            except Exception as error:
                state['prior_diagnostics_finished']=False
                append(out/'errors.jsonl',{'phase':'prior_diagnostics','error':str(error),'traceback':traceback.format_exc()})
                print('PRIOR ERROR',str(error),flush=True)
            gc.collect()
        settings=[(c['H'],c['K']) for c in cells if c['applicable']]
        closed_loop(model,out,state,deadline,settings=settings,task_ids=config['task_ids'],
            initial_states=config['initial_states'],flow_steps=config['flow_steps'],
            max_actions=config['max_actions'],seed_offset=config['flow_seed_offset'],update=update)
        final_status='finished' if time.monotonic()<deadline else 'budget_finished'
        if final_status=='finished' and (not state.get('prior_diagnostics_finished',True) or read_rows(out/'errors.jsonl')):
            final_status='finished_with_errors'
        state.update(status=final_status,phase='done')
    except BaseException as error:
        state.update(status='failed',error=str(error));append(out/'errors.jsonl',{'error':str(error),'traceback':traceback.format_exc()})
        raise
    finally:render(out,state,config,cells)


if __name__=='__main__':main()
