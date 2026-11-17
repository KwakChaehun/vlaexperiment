"""Render incremental real-model findings and scientific figures for the local pilot."""
from __future__ import annotations
import argparse,json,time,sys,os,html,traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np


def read_rows(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass # Writer may be between JSON bytes.
    return result


def table(rows,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in rows:
        lines.append('| '+' | '.join(f'{row[c]:.5g}' if isinstance(row.get(c),float) else str(row.get(c,'')) for c in columns)+' |')
    return '\n'.join(lines)


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from vla_eval.statistics import wilson
    state=json.loads((out/'status.json').read_text(encoding='utf-8'))
    rows=read_rows(out/'offline.jsonl');closed=read_rows(out/'closed_loop.jsonl')
    arrival=read_rows(out/'arrival_closed_loop.jsonl')
    calibrated=read_rows(out/'arrival_calibrated/arrival_closed_loop.jsonl')
    hrows=[r for r in rows if r['experiment']=='horizon' and r['K']==1]
    # Compare exactly the same observations even while a case is being written.
    cases_by_h={h:{r['case_id'] for r in hrows if r['H']==h} for h in [1,4,16,50]}
    common_cases=set.intersection(*cases_by_h.values())
    hrows=[r for r in hrows if r['case_id'] in common_cases]
    prior=[r for r in rows if r['experiment']=='prior' and r['split']=='test']
    hg=defaultdict(list)
    for r in hrows:hg[r['H']].append(r)
    horizons=[]
    for h,rr in sorted(hg.items()):
        horizons.append({'H':h,'observations':len(rr),'episodes':len({r['episode_id'] for r in rr}),
                         'first_action_nmse':float(np.mean([r['first_action_nmse'] for r in rr])),
                         'teacher_prefix_nmse':float(np.mean([r.get('teacher_nmse',0) for r in rr])),
                         'ae_cpu_ms':float(np.mean([r['ae_ms'] for r in rr])),
                         'backbone_cpu_ms':float(np.mean([r['backbone_ms'] for r in rr]))})
        stage=horizons[-1]
        stage['end_to_end_cpu_ms']=float(np.mean([r['post_cpu_ms'] for r in rr]))
        stage['ae_cpu_fraction']=stage['ae_cpu_ms']/stage['end_to_end_cpu_ms']
        stage['zero_ae_cpu_speedup_bound']=1/(1-stage['ae_cpu_fraction'])
    pg=defaultdict(list)
    for r in prior:pg[r['method']].append(r)
    methods=[]
    for name,rr in sorted(pg.items()):
        methods.append({'method':name,'observations':len(rr),'episodes':len({r['episode_id'] for r in rr}),
                        'expert_nmse':float(np.mean([r['normalized_action_mse'] for r in rr])),
                        'teacher_nmse':float(np.mean([r.get('teacher_nmse',0) for r in rr])),
                        'gripper_accuracy':float(np.mean([r['gripper_accuracy'] for r in rr])),
                        'post_cpu_ms':float(np.mean([r['post_cpu_ms'] for r in rr if 'post_cpu_ms' in r]))
                                      if any('post_cpu_ms' in r for r in rr) else None})
    arrival_groups=defaultdict(list)
    for row in arrival:
        if row['method']!='phase_selection':arrival_groups[row['method']].append(row)
    arrival_summary=[]
    for method,rr in sorted(arrival_groups.items()):
        completed=[r for r in rr if r['status']=='completed'];wins=sum(r['success'] for r in completed)
        arrival_summary.append({'method':method,'completed_tasks':len(completed),'successes':wins,
                                'success_rate':wins/len(completed) if completed else None,
                                'wilson_ci95':wilson(wins,len(completed)),
                                'errors_or_censored':len(rr)-len(completed),
                                'post_cpu_ms':float(np.mean([r['arrival_post_cpu_ms'] for r in completed])) if completed else None,
                                'analytical_cpu_first_delay_ms':float(np.mean([r['analytical_first_action_delay_cpu_ms'] for r in completed])) if completed else None})
    (out/'arrival_summary.json').write_text(json.dumps(arrival_summary,indent=2),encoding='utf-8')
    arrival_by_method=defaultdict(dict)
    for r in arrival:
        if r.get('status')=='completed':arrival_by_method[r['method']][(r['task_id'],r['initial_state'])]=r
    arrival_pairs=[]
    for a,b in [('C_ridge','B_ridge'),('C_flow_refine','B_few_steps'),('D_partial','A_fresh')]:
        keys=arrival_by_method[a].keys() & arrival_by_method[b].keys()
        if keys:
            arrival_pairs.append({'candidate':a,'reference':b,'matched_states':len(keys),
                'success_rate_delta':float(np.mean([int(arrival_by_method[a][k]['success'])-int(arrival_by_method[b][k]['success']) for k in keys]))})
    (out/'arrival_paired.json').write_text(json.dumps(arrival_pairs,indent=2),encoding='utf-8')
    calibrated_groups=defaultdict(list)
    for r in calibrated:calibrated_groups[r['method']].append(r)
    calibrated_summary=[]
    for method,rr in sorted(calibrated_groups.items()):
        cc=[r for r in rr if r['status']=='completed'];wins=sum(r['success'] for r in cc)
        errors=[r['forecast_position_error_mm'] for r in cc if r.get('forecast_position_error_mm') is not None]
        calibrated_summary.append({'method':method,'completed_tasks':len(cc),'successes':wins,
            'success_rate':wins/len(cc) if cc else None,'wilson_ci95':wilson(wins,len(cc)),
            'forecast_error_mm':float(np.mean(errors)) if errors else None,
            'post_cpu_ms':float(np.mean([r['arrival_post_cpu_ms'] for r in cc])) if cc else None,
            'errors_or_censored':len(rr)-len(cc)})
    (out/'arrival_calibrated_summary.json').write_text(json.dumps(calibrated_summary,indent=2),encoding='utf-8')
    endpoint_rows=read_rows(out/'arrival_endpoints.jsonl')
    reference={r['task_id']:r['endpoint'] for r in endpoint_rows if r['method']=='A_fresh'}
    endpoint_groups=defaultdict(list)
    for r in endpoint_rows:
        if r['task_id'] not in reference:continue
        e,ref=r['endpoint'],reference[r['task_id']]
        error=float(np.linalg.norm(np.asarray(e['position'])-ref['position'])*1000)
        qa,qb=np.asarray(e['quaternion']),np.asarray(ref['quaternion'])
        angle=float(np.rad2deg(2*np.arccos(np.clip(abs(np.dot(qa,qb)/(np.linalg.norm(qa)*np.linalg.norm(qb))),0,1))))
        endpoint_groups[(r['variant'],r['method'])].append((error,angle))
    endpoints=[]
    for (variant,method),rr in sorted(endpoint_groups.items()):
        values=np.asarray(rr)
        endpoints.append({'forecast':variant,'method':method,'tasks':len(rr),
            'teacher_position_error_mm':float(values[:,0].mean()),
            'teacher_rotation_error_deg':float(values[:,1].mean()),
            'within_2mm':float((values[:,0]<=2).mean()),'within_5mm':float((values[:,0]<=5).mean())})
    (out/'arrival_endpoint_summary.json').write_text(json.dumps(endpoints,indent=2),encoding='utf-8')
    clg=defaultdict(list)
    for r in closed:clg[(r['H'],r['K'])].append(r)
    cl=[]
    for (h,k),rr in sorted(clg.items()):
        completed=[r for r in rr if r['status']=='completed'];wins=sum(r['success'] for r in completed)
        trace=[t for r in completed for t in r['trace']]
        costs=[t['backbone_ms']+t['preparation_ms']+t['ae_ms'] for t in trace]
        cl.append({'H':h,'K':k,'episodes':len(completed),'successes':wins,
                   'success_rate':wins/len(completed) if completed else None,'wilson_ci95':wilson(wins,len(completed)),
                   'errors_or_censored':len(rr)-len(completed),
                   'cpu_call_p50_ms':float(np.median(costs)) if costs else None,
                   'cpu_call_p95_ms':float(np.quantile(costs,.95)) if costs else None,
                   'cpu_call_p99_ms':float(np.quantile(costs,.99)) if costs else None,
                   'mean_sim_steps':float(np.mean([r['steps'] for r in completed])) if completed else None})
    closed_by_setting=defaultdict(dict)
    for r in closed:
        if r['status']=='completed':closed_by_setting[(r['H'],r['K'])][(r['task_id'],r['initial_state'])]=r
    cl_pairs=[]
    for a,b in [((4,1),(50,1)),((4,4),(50,4)),((16,4),(50,4)),((50,1),(50,10)),((50,4),(50,10))]:
        keys=closed_by_setting[a].keys() & closed_by_setting[b].keys()
        if keys:
            differences=[int(closed_by_setting[a][key]['success'])-int(closed_by_setting[b][key]['success']) for key in keys]
            cl_pairs.append({'candidate':str(a),'reference':str(b),'matched_initial_states':len(keys),
                             'success_rate_delta':float(np.mean(differences))})
    (out/'closed_loop_paired.json').write_text(json.dumps(cl_pairs,indent=2),encoding='utf-8')
    prefix=max([r.get('teacher_nmse',0) for r in hrows],default=None)
    matched=[r for r in hrows if r['H']!=50]
    max_prefix=max([r.get('teacher_nmse',0) for r in matched],default=None)
    findings=[]
    if matched:
        findings.append(f'현재 {len(matched)}개의 짧은 H/관측 조합에서 native H=50 대비 공통 첫 action의 최대 NMSE는 {max_prefix:.3g}입니다.')
        findings.append('이 버전 SmolVLA의 action attention은 causal mask를 사용합니다. 수학적으로 같은 prefix noise와 조건에서 앞 action은 뒤 action에 의존하지 않습니다. 실제 tensor 길이에 따른 수치 차이는 위 NMSE로 남깁니다. H 단축의 주 효과와 K 단축의 피드백 효과를 나누어 봐야 하며, GR00T에도 같은 불변성이 있다고 가정하지 않습니다.')
    pairs=[]
    if (out/'paired_prior_comparisons.json').exists():pairs=json.loads((out/'paired_prior_comparisons.json').read_text())
    cb=[r for r in pairs if r['comparison']=='C_ridge - B_ridge' and r['metric']=='normalized_action_mse']
    if cb:
        r=cb[0];findings.append(f"C_ridge−B_ridge의 expert NMSE 차이는 {r['mean_delta']:.4g}, episode bootstrap 95% 구간 {r['ci95']}입니다. 독립 episode {r['episodes']}개 pilot이며, 이 결과를 강한 Small VLA나 GR00T의 결론으로 확대하지 않습니다.")
        if r['ci95'][0]<=0<=r['ci95'][1]:
            findings.append('현재 이 작은 보정기의 held-out 표본에서는 prior가 더 정확하다는 근거가 없습니다. 학습량·보정기 능력·forecast 오차를 분리해서 확인해야 합니다.')
    else:findings.append('B_ridge–C_ridge 비교는 correction 훈련/test 표본 처리가 끝난 뒤 자동으로 갱신됩니다. 현재는 판정 대기입니다.')
    if not cl:findings.append('Closed-loop 성공률은 아직 없습니다. Offline action 오차로 성공률을 대체하지 않습니다.')
    if cl and not sum(r['episodes'] for r in cl):findings.append('현재는 실행 오류/중단 기록만 있고 유효하게 완료된 closed-loop episode가 없습니다. Task 성공률은 미측정입니다.')
    elif cl and not any(r['successes'] for r in cl):findings.append('현재 완료된 closed-loop에서 성공이 관측되지 않았습니다. 먼저 native baseline의 재현과 카메라·action·state 전처리를 확인해야 하며, 이 상태에서 PIM/짧은 H의 품질 우위를 주장하지 않습니다.')
    if cl:findings.append('Closed-loop는 같은 공식 초기 상태를 H/K마다 재사용합니다. 아직 완료되지 않은 조합은 서로 다른 표본 수일 수 있으므로, 최종 paired 비교는 공통 완료 상태만 사용해야 합니다.')
    if horizons:
        short=next((r for r in horizons if r['H']==4),None)
        if short:
            findings.append(f"H=4에서 action expert는 CPU 총 추론 시간의 {100*short['ae_cpu_fraction']:.1f}%입니다. 다른 단계 비용을 그대로 두고 expert 비용만 0으로 만드는 CPU상의 이상적 속도 향상 상한도 {short['zero_ae_cpu_speedup_bound']:.2f}배입니다. GPU/PIM 결과는 아니지만, GPU에서 backbone까지 포함한 stage 비중을 확인해야 하는 이유입니다.")
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        x=[r['H'] for r in horizons]
        axes[0].plot(x,[r['first_action_nmse'] for r in horizons],'o-');axes[0].set(xlabel='Generated horizon H',ylabel='Common first-action NMSE',title='Real SmolVLA / recorded LIBERO')
        axes[0].set_ylim(0,max(r['first_action_nmse'] for r in horizons)*1.15)
        axes[1].plot(x,[r['ae_cpu_ms'] for r in horizons],'o-',label='action expert');axes[1].plot(x,[r['backbone_cpu_ms'] for r in horizons],'o-',label='backbone');axes[1].set(xlabel='Generated horizon H',ylabel='CPU milliseconds',title='Stage cost: CPU, not GPU/PIM');axes[1].legend()
        for ax in axes:
            ax.set_xscale('log',base=2);ax.set_xticks(x,labels=[str(h) for h in x]);ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(out/'horizon.png',dpi=150);plt.close(fig)
    if methods:
        fig,ax=plt.subplots(figsize=(11,5))
        ax.barh([r['method'] for r in methods],[r['expert_nmse'] for r in methods]);ax.set(xlabel='Expert action NMSE (first four actions)',title='Held-out pilot episodes; pretrained training overlap possible')
        fig.tight_layout();fig.savefig(out/'prior.png',dpi=150);plt.close(fig)
    forecast=[]
    if (out/'cases.json').exists():
        cases=json.loads((out/'cases.json').read_text());errors=np.asarray([r['forecast_position_error_mm'] for r in cases])
        forecast=[{'cases':len(cases),'mean_mm':float(errors.mean()),'median_mm':float(np.median(errors)),
                   'p95_mm':float(np.quantile(errors,.95)),'max_mm':float(errors.max())}]
    live_forecast=[]
    forecast_file=out/'forecast_diagnostics.json'
    if forecast_file.exists():
        diagnostic=json.loads(forecast_file.read_text(encoding='utf-8'))
        fg=defaultdict(list)
        for task in diagnostic['tasks']:
            for r in task['records']:fg[r['method']].append(r)
        for method,rr in sorted(fg.items()):
            errors=[r['position_error_mm'] for r in rr]
            selected=[r['position_error_mm'] for r in rr if r['is_selected_arrival']]
            live_forecast.append({'method':method,'tasks':len({r['task_id'] for r in rr}),
                'overlapping_windows':len(rr),'mean_mm':float(np.mean(errors)),
                'p95_mm':float(np.quantile(errors,.95)),
                'selected_arrival_mean_mm':float(np.mean(selected)) if selected else None})
    (out/'forecast_summary.json').write_text(json.dumps(live_forecast,indent=2),encoding='utf-8')
    md=f'''# VLA 로컬 실제 평가 — 내일 확인할 결과

갱신(KST): {datetime.now(timezone(timedelta(hours=9))).isoformat()}<br>
실행 상태: **{state.get('status')} / {state.get('phase')}**, offline 완료 {state.get('completed_cases',0)}/{state.get('total_cases',24)}.<br>
PID: {state.get('pid')}. 모델은 SmolVLA-LIBERO, native H=50/N=10, CPU float32입니다. GTX 1650/RTX 3090 성능 결과가 아닙니다.

## 현재 결과가 말하는 것

'''+ '\n\n'.join('- '+s for s in findings)+f'''

## 1. 선행 계산 + 작은 보정

### 동일한 reaching 후 실제 도착 상태에서 비교한 rollout

{table(arrival_summary,['method','completed_tasks','successes','success_rate','wilson_ci95','errors_or_censored','post_cpu_ms','analytical_cpu_first_delay_ms'])}

같은 task/도착 상태가 양쪽 모두 완료된 비교:

{table(arrival_pairs,['candidate','reference','matched_states','success_rate_delta'])}

현재 관측에서 알려진 reaching action 4개의 controller 목표 변화로 예상 도착 pose를 만들고, 과거 RGB와 결합해 prior를 계산합니다. 실제 미래 state/image는 prior 입력으로 사용하지 않습니다. 같은 초기 상태에서 동일한 reaching action을 재생해 도착 상태가 일치하는지 검증합니다. 물체 위치를 이용한 pre-grasp 구간 선택은 evaluator의 oracle 주석이며, 배포 가능한 구간 감지기를 검증한 것은 아닙니다.

각 방법은 도착 직후 첫 4개 action만 다르게 실행하고 이후에는 같은 Heavy controller를 사용합니다. 따라서 이 표는 **arrival-only intervention**이며 전체 fine manipulation을 Small만으로 수행하는 시스템이 아닙니다. CPU 선행 계산이 reaching의 200 ms를 초과하면 그 나머지를 analytical first delay에 포함합니다. 실제 concurrent 측정은 아닙니다. Replay는 `arrival_replays/`, 원자료는 `arrival_closed_loop.jsonl`입니다.

### 도착 위치 예측을 개선한 별도 조건

{table(calibrated_summary,['method','completed_tasks','successes','success_rate','wilson_ci95','forecast_error_mm','post_cpu_ms','errors_or_censored'])}

같은 reaching 경로에서 과거 4 step으로만 추정한 command gain으로 위치를 예측하고, 회전은 마지막 관측 SO(3) 속도로 예측합니다. 미래 관측은 입력하지 않습니다. A/B의 행동은 prior를 사용하지 않으므로 앞 표의 동일 baseline rollout을 재사용하고 원자료에 출처를 표시합니다. Prior를 사용하는 C/D만 다시 실행합니다. 이 조건도 첫 4 action 개입이며, 원자료·GIF·오류는 `arrival_calibrated/`에 따로 보존합니다.

### 보정 직후 실제 EEF 위치·회전의 fresh A 대비 차이

{table(endpoints,['forecast','method','tasks','teacher_position_error_mm','teacher_rotation_error_deg','within_2mm','within_5mm'])}

동일 도착 상태에서 첫 4 action을 실제로 실행한 endpoint를 fresh A의 endpoint와 비교합니다. 이 mm/degree는 **teacher 동작과의 일치도**이며 물체의 정답 grasp pose 오차가 아닙니다. 이후 Heavy가 복구해서 최종 task 성공이 같아지는 경우에도 첫 개입의 차이를 볼 수 있습니다. `arrival_endpoints.jsonl`에 실제 endpoint를 저장합니다.

### Offline 관측의 action 오차

{table(methods,['method','observations','episodes','expert_nmse','teacher_nmse','gripper_accuracy','post_cpu_ms'])}

`forecast_prior`: 4 frame 전 RGB + 과거 두 state의 등속 외삽. `stale_prior`: 과거 RGB/state 그대로. `oracle_proprio_prior`: 실제 미래 state + 과거 RGB인 privileged 진단 조건. `A_fresh`: 실제 도착 RGB/state로 처음부터 생성.

`C_flow_refine`: 완성 prior를 다시 noise와 섞고 최신 관측에서 2 step refine. `D_partial`: 과거 조건 8 step + 최신 조건 2 step. 두 방법은 최신 VLM 인코딩 비용을 포함합니다. `B_few_steps`: 최신 조건에서 noise부터 2 step.

`B_ridge/C_ridge`: 동일 가중치의 저비용 회귀기. 3 train episode에서 fresh teacher action을 학습하고 다른 3 episode에서 평가합니다. 강한 Small 모델과의 비교는 아직 필요합니다.

{table(pairs,['comparison','metric','episodes','mean_delta','ci95'])}

## 예상 좌표에서 feature를 만들면 되는가?

현재 forecast state 외삽 오차:

{table(forecast,['cases','mean_mm','median_mm','p95_mm','max_mm'])}

이 값은 **도착 state 예측 오차**이며 action 실행 후 정밀도 mm가 아닙니다. EEF 좌표만으로 미래 손목 카메라 이미지는 결정되지 않습니다. Joint configuration, object geometry와 가림도 필요합니다. 오늘의 stale-RGB+예측-state는 현실적인 출발점 하나이며, 미래 시각 feature가 정확하다고 가정하지 않습니다. `oracle_proprio_prior`와 A의 차이는 state가 맞아도 남는 시각 조건 차이를 진단합니다. 실제 미래 이미지를 사전 입력으로 사용해 성능을 올리는 것은 배포 가능한 방법이 아닙니다.

### 실제 reaching에서 좌표 예측기 추가 진단

{table(live_forecast,['method','tasks','overlapping_windows','mean_mm','p95_mm','selected_arrival_mean_mm'])}

이미 저장한 reaching 경로를 재생해 4 step 뒤의 실제 위치와 비교합니다. Controller 목표 delta의 합, 과거 속도 외삽, 과거 4 step으로만 보정한 command gain을 비교합니다. 미래 simulator state는 채점에만 사용합니다. 겹치는 window는 독립 trial이 아닙니다. 이 진단은 현재 rollout의 prior 생성기를 교체하지 않습니다. Controller 목표를 achieved pose로 취급해서 생기는 오차와 보정 정책 자체의 실패를 구분하기 위한 결과입니다.

## 2. 생성 H와 실행 K

{table(horizons,['H','observations','episodes','first_action_nmse','teacher_prefix_nmse','ae_cpu_ms','backbone_cpu_ms'])}

CPU stage 비중과 AE만 가속할 때의 비용 상한:

{table(horizons,['H','end_to_end_cpu_ms','ae_cpu_fraction','zero_ae_cpu_speedup_bound'])}

`zero_ae_cpu_speedup_bound = total / (total − AE)`는 다른 단계와 호출 횟수를 고정한 이상적 CPU 비용 계산입니다. PIM 성능 예측이나 GPU memory-bound 판정이 아닙니다.

![Horizon](horizon.png)

H 단축은 실제 expert tensor 길이를 바꾼 실험입니다. N=10과 공통 초기 noise prefix를 유지합니다. Native 이외 길이는 재학습하지 않은 조건입니다. K의 피드백 효과는 아래 live simulator 결과에서만 판단합니다.

## 실제 LIBERO closed-loop

{table(cl,['H','K','episodes','successes','success_rate','wilson_ci95','errors_or_censored','mean_sim_steps','cpu_call_p50_ms','cpu_call_p95_ms'])}

같은 task/초기 상태가 양쪽 모두 완료된 결과만 비교:

{table(cl_pairs,['candidate','reference','matched_initial_states','success_rate_delta'])}

Task 0/1/2, 공식 초기 상태 0/1, H/K 여섯 조합을 최대 7시간 동안 실행합니다. 20 Hz simulator는 추론 중 정지하므로 이는 **지연을 제거한 품질 실험**입니다. 완료 실패, 시간 예산에 따른 censored, 실행 오류를 구분합니다. GIF는 `replays/`, 진행 중 화면은 `latest_view.png`입니다.

## 추가로 기록한 지표

- Translation/rotation command MSE, gripper sign 정확도, action 범위 초과 비율.
- Fresh teacher와의 차이, 다른 episode prior를 넣는 대조군, episode 단위 paired bootstrap.
- CPU backbone/AE 시간, call p50/p95, policy 호출 수, 실행 step 수.
- 매 제어 step에서 측정한 EEF 경로 길이·jerk, command 불연속성, sampled contact 수. 정상 grasp 접촉을 모두 충돌 실패로 분류하지 않습니다.

## 다음 연구 판단

1. C가 B보다 유리한지, 그리고 그 이득이 weak Ridge baseline 때문에 생긴 것인지 분리합니다. Task 성공 검증 전에는 작은 action 오차를 정밀 조작 성공으로 부르지 않습니다.
2. 작은 H에서 같은 prefix가 유지되면 H는 정확도보다 낭비와 실행 비용을 줄이는 knob일 수 있습니다. K를 줄였을 때 실제 성공률과 부드러움이 어떻게 바뀌는지 봅니다.
3. Forecast와 exact-state 진단의 차이가 작고 A와 차이가 크면 미래 시각 feature가 주요 문제일 가능성이 있습니다. 반대라면 도착 상태 예측/좌표계 보정부터 개선합니다.
4. RTX 3090 연결 후 π0.5/GR00T native baseline, GPU tensor shape·DRAM traffic·p95/p99·deadline 및 concurrent 경합을 측정합니다. CPU 시간만으로 PIM speedup을 주장하지 않습니다.

LIBERO Spatial의 bowl grasp/place 성공은 좁은 공차의 insertion 정밀도를 입증하지 않습니다. 이 pilot으로 방법을 선별한 뒤, 실제 정밀 작업과 도착 오차 2/5/10/20 mm·회전 perturbation에서 검증해야 합니다. A는 이 checkpoint의 fresh 기준선이며 정확도의 수학적 상한이 아닙니다. B_few_steps는 작은 parameter 모델이 아니라 같은 모델의 적은 flow step 조건입니다.

원자료: [offline.jsonl](offline.jsonl), [도착 후 rollout](arrival_closed_loop.jsonl), [H/K rollout](closed_loop.jsonl), [status.json](status.json), [manifest.json](manifest.json), [세부 보고서](REPORT.md). 미완료/오류가 있으면 `errors.jsonl`과 상위 폴더의 stderr log를 확인하세요.
'''
    (out/'RESULTS.md').write_text(md,encoding='utf-8')
    try:
        import markdown
        body=markdown.markdown(md,extensions=['tables','fenced_code'])
    except ImportError:body='<pre>'+html.escape(md)+'</pre>'
    (out/'RESULTS.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>VLA real-model local pilot</title><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#17212b}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccd4dc;padding:6px 10px}th{background:#eef3f7}img{max-width:100%}code{background:#f0f2f5}h2{margin-top:36px}</style>'+body+'</html>',encoding='utf-8')
    return state.get('status')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/research_pilot');p.add_argument('--watch',action='store_true');args=p.parse_args()
    out=ROOT/args.output
    while True:
        try:
            # Only load the simulator if a new committed reaching plan appeared.
            plans={int(p.stem.removeprefix('arrival_plan_task')) for p in out.glob('arrival_plan_task*.npz')}
            diag=out/'forecast_diagnostics.json'
            done={r['task_id'] for r in json.loads(diag.read_text())['tasks']} if diag.exists() else set()
            if plans-done:
                import subprocess
                subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/forecast_diagnostics.py'),
                                '--output',str(out)],check=True)
            from scripts.arrival_endpoint_diagnostics import pending
            if pending(out):
                import subprocess
                subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/arrival_endpoint_diagnostics.py'),
                                '--output',str(out)],check=True)
            status=render(out)
        except Exception:
            (out/'dashboard_error.log').write_text(traceback.format_exc(),encoding='utf-8');status='retry'
        if not args.watch or status in {'finished','budget_finished','failed'}:break
        time.sleep(60)


if __name__=='__main__':main()
