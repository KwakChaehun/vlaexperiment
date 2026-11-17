"""Export oracle/causal arrival diagnostics without treating teacher agreement as precision."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone,timedelta
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np
import run_gr00t_wrist_pilot as base


def table(rows,cols):
    def format(v):return f'{v:.4g}' if isinstance(v,float) else str(v)
    return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(format(r.get(c,'')))+'</td>' for c in cols)+'</tr>' for r in rows)+'</table>'


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='results/arrival_gr00t_20261117')
    p.add_argument('--output',default='reports/arrival_gr00t_20261117')
    p.add_argument('--control',default='results/arrival_gr00t_stale_control_20261117');args=p.parse_args()
    source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'config.json').read_text());status=json.loads((source/'status.json').read_text())
    diagnostics=base.rows(source/'diagnostics.jsonl');trials=base.rows(source/'rollouts.jsonl')
    methods=list(config['methods']);control=Path(args.control)
    control_status=json.loads((control/'status.json').read_text())
    control_diag=base.rows(control/'diagnostics.jsonl')
    original={r['case']:r for r in diagnostics}
    for r in control_diag:
        a=np.load(source/original[r['case']]['observation_file'],allow_pickle=False)
        b=np.load(control/r['observation_file'],allow_pickle=False)
        for key in ['prefix','sim_state','arrival_external','arrival_wrist','before_state','forecast_state']:
            np.testing.assert_array_equal(a[key],b[key])
    diagnostics+=control_diag;trials+=base.rows(control/'rollouts.jsonl');methods+=['stale_replay']
    timings={r['case']:r['cpu_p50_ms'] for r in json.loads((source/'forecast_cpu_timings.json').read_text())}
    groups=defaultdict(dict)
    for r in trials:
        if r['status']=='completed':
            if r['case'] in groups[r['method']]:raise ValueError('duplicate case')
            groups[r['method']][r['case']]=r
    common=set.intersection(*(set(groups[m]) for m in methods))
    rows=[];per_case=[]
    forecast_methods={'forecast_resume3_1','oracle_rgb_resume3_1','forecast_full_refine1','forecast_replay'}
    for method in methods:
        dd=[r for r in diagnostics if r['method']==method and r['case'] in common]
        rr=[groups[method][c] for c in sorted(common)]
        cpu=[timings[r['case']] if method in forecast_methods else 0. for r in dd]
        pre=[r['precompute_ms']+t for r,t in zip(dd,cpu)]
        post=[r['post_ms'] for r in dd]
        endpoint=[float(np.linalg.norm(np.array(r['endpoint_state'])[:3]-
                  np.array(groups['fresh4'][r['case']]['endpoint_state'])[:3])*1000) for r in rr]
        rows.append(dict(method=method,success=f"{sum(r['success'] for r in rr)}/{len(rr)}",
            immediate_goal=f"{sum(r['goal_after_intervention'] for r in rr)}/{len(rr)}",
            precompute_ms=float(np.mean(pre)),post_p50_ms=float(np.median(post)),
            total_model_plus_forecast_ms=float(np.mean(np.array(pre)+post)),
            analytical_delay_ms=float(np.mean([max(0.,a-config['lookahead_steps']*50)+b for a,b in zip(pre,post)])),
            teacher_nmse16=float(np.mean([r['teacher_nmse_first16'] for r in dd])),
            teacher_endpoint_mm=float(np.mean(endpoint)),
            privileged=any(r['privileged'] for r in dd)))
        for r in rr:per_case.append(dict(method=method,task=r['task'],initial_state=r['initial_state'],
            success=r['success'],immediate_goal=r['goal_after_intervention'],steps=r['steps']))
    paired=[]
    for a,b in [('forecast_resume3_1','fresh1'),('forecast_resume3_1','fresh2'),
                ('forecast_resume3_1','fresh4'),('oracle_rgb_resume3_1','forecast_resume3_1'),
                ('forecast_full_refine1','fresh1'),('forecast_replay','fresh4'),('forecast_replay','stale_replay')]:
        win=[c for c in common if groups[a][c]['success'] and not groups[b][c]['success']]
        loss=[c for c in common if groups[b][c]['success'] and not groups[a][c]['success']]
        paired.append(dict(comparison=a+' - '+b,pairs=len(common),wins=len(win),losses=len(loss),
                           win_cases=win,loss_cases=loss))
    hashes={}
    for filename in ['config.json','manifest.json','status.json','diagnostics.jsonl','rollouts.jsonl','forecast_cpu_timings.json']:
        shutil.copy2(source/filename,out/filename);hashes[filename]=hashlib.sha256((out/filename).read_bytes()).hexdigest()
    (out/'stale_control').mkdir(exist_ok=True)
    for filename in ['config.json','manifest.json','status.json','diagnostics.jsonl','rollouts.jsonl']:
        dest=out/'stale_control'/filename;shutil.copy2(control/filename,dest)
        hashes[str(dest.relative_to(out))]=hashlib.sha256(dest.read_bytes()).hexdigest()
    code=out/'source';code.mkdir(exist_ok=True)
    for filename in ['scripts/run_arrival_gr00t_probe.py','scripts/profile_arrival_forecast.py',
                     'scripts/summarize_arrival_probe.py','vla_eval/gr00t16_arrival.py','vla_eval/gr00t16_pilot.py']:
        shutil.copy2(base.ROOT/filename,code/Path(filename).name)
    result=dict(as_of_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),status=status,
        matched_cases=len(common),summary=rows,paired=paired,per_case=per_case,hashes=hashes,
        control_status=control_status,total_rollouts=len(trials),
        source_directory=str(source.resolve()),forecast_cpu_mean_ms=float(np.mean(list(timings.values()))),
        forecast_position_mean_mm=float(np.mean([r['forecast_position_error_mm'] for r in diagnostics if r['method']=='fresh4'])))
    terminal=[];terminal_status=[]
    for i in range(3):
        folder=source.parent/f'arrival_gr00t_terminal_{i}_20261117'
        if not (folder/'status.json').exists():continue
        terminal_status.append(json.loads((folder/'status.json').read_text()))
        terminal+=base.rows(folder/'rollouts.jsonl')
        dest=out/f'terminal_{i}';dest.mkdir(exist_ok=True)
        for name in ['config.json','manifest.json','status.json','diagnostics.jsonl','rollouts.jsonl']:
            shutil.copy2(folder/name,dest/name)
            hashes[str((dest/name).relative_to(out))]=hashlib.sha256((dest/name).read_bytes()).hexdigest()
    terminal_summary=[]
    for method in sorted({r['method'] for r in terminal}):
        rr=[r for r in terminal if r['method']==method and r['status']=='completed']
        terminal_summary.append(dict(method=method,completed=len(rr),successes=sum(r['success'] for r in rr),
            goal_at_end=sum(r['goal_after_intervention'] for r in rr)))
    result['terminal_diagnostic']=dict(statuses=terminal_status,total_rollouts=len(terminal),summary=terminal_summary,
        per_case=[{k:r[k] for k in ['task','initial_state','method','success','goal_after_intervention','steps']} for r in terminal],
        selection='First successful native reference per task; its last native chunk start. No recovery beyond the next 16 actions. Exploratory selection, not unbiased success-rate estimate.')
    base.save(out/'summary.json',result)
    content='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>GR00T arrival precompute probe</title>
<style>body{max-width:1400px;margin:32px auto;padding:0 24px;font:16px/1.6 system-ui;color:#17202a}table{border-collapse:collapse;font-size:13px;margin:20px 0}td,th{border:1px solid #ccd3da;padding:7px;text-align:left}th{background:#eef2f6}img{max-width:100%}</style>
<h1>GR00T N1.6: 선행 계산 + 도착 후 작은 보정 진단</h1>'''
    content+=f'<p>{result["as_of_kst"]}. 상태 {status["status"]}, 추가 대조군 {control_status["status"]}, 공통 도착 조건 {len(common)}개, 총 {len(trials)}회.</p>'
    content+='''<p>H=50 고정. 같은 로봇 이동 prefix를 실행한 뒤 첫 16개 action만 방법별로 바꾸고, 이후 모두 동일한 H50/N4/K16 제어기로 복귀합니다.
따라서 전체 episode를 보정기로 수행한 성공률이 아닙니다. 이후 Heavy의 복구가 첫 개입의 품질 차이를 가릴 수 있습니다.
Goal 2작업과 LIBERO-10 1작업 × 초기 상태 0/4/8의 탐색 평가이며, 초기 상태마다 하나의 flow seed를 사용합니다.
도착 시점은 Goal 64 steps, LIBERO-10 160 steps로 고정했으며 접촉 단계를 검출한 것이 아닙니다.</p>
<p>선행 계산 시점은 도착 4 steps(200ms) 전입니다. 모든 rollout에서 전체 simulator state와 두 카메라 이미지가 동일하게 도착했는지 검사합니다.
공식 sampler 대 확장 sampler 및 4-step 대 3+1-step의 수치 차이가 0인지 먼저 검증했습니다.</p>
<h2>방법</h2><ul>
<li>fresh4/2/1: 도착 후 실제 관측에서 처음부터 4/2/1 flow steps.</li>
<li>oracle_resume3_1: 정답 미래 RGB/state에서 3단계 선행, 도착 후 실제 관측을 다시 인코딩해 마지막 1단계. 동일 조건 분할이므로 fresh4와 일치하는 수치 검증 대조군입니다.</li>
<li>stale_resume3_1: 과거 RGB/state 그대로 3단계 선행 후 최신 1단계.</li>
<li>forecast_resume3_1: 과거 RGB + 과거 4-step 속도로 외삽한 도착 state에서 3단계 선행 후 최신 1단계. 미래 정답 입력 없는 조건입니다.</li>
<li>oracle_state_resume3_1: 정답 미래 state + 과거 RGB. oracle_rgb_resume3_1: 정답 미래 RGB + 외삽 state. 시각/state 오차 분리용 privileged 조건입니다.</li>
<li>bias5mm/20mm_resume3_1: 정답 미래 RGB와 xyz 중 x축에 ±5/20mm 오차를 넣은 미래 state. 실제 영상 기하 변화가 아닌 state 입력 오차 민감도입니다.</li>
<li>forecast_full_refine1: 완성 예측 prior와 원래 noise를 0.75/0.25로 섞고 최신 관측에서 마지막 1단계. 이 절차로 재학습하지 않았습니다.</li>
<li>forecast_replay: 예측 prior를 그대로 실행. oracle_replay: 정답 미래 관측의 완성 prior를 그대로 실행하는 비현실적인 재사용 하한 대조군입니다.</li>
<li>stale_replay: 과거 RGB/state에서 만든 완성 prior를 그대로 실행. 초기 7개 조건을 본 뒤 state 예측의 추가 가치를 분리하려고 추가한 탐색 대조군입니다. 기존 조건과 prefix·전체 도착 state·양쪽 RGB가 동일함을 검증했습니다.</li></ul>
<h2>결과</h2>'''
    content+=table(rows,['method','success','immediate_goal','precompute_ms','post_p50_ms','total_model_plus_forecast_ms',
                         'analytical_delay_ms','teacher_nmse16','teacher_endpoint_mm','privileged'])
    content+='''<p><strong>고정 도착 시점 실험에서는 모든 방법의 immediate_goal이 0/9였습니다. 최종 성공은 이후 공통 제어기의 복구를 포함합니다.</strong>
아래 별도 마지막 구간 진단은 이 문제를 분리하기 위해 사후 추가했습니다. 서로 다른 평가이므로 두 성공률을 합치지 않습니다.</p>
<h2>추가 진단: 작업을 마치는 마지막 16개 동작만 평가</h2>
<p>작업별 첫 번째 성공한 fresh4 reference를 고르고, 그 reference가 처음 목표를 달성한 마지막 native chunk의 시작점을 도착 시점으로 사용합니다.
세 작업 각 1개 조건 × 7방법 = 21회이며, 16개 action 이후 복구 제어를 붙이지 않습니다. 성공한 기준선과 미래 완료 시각을 이용한 평가용 선택입니다.
배포 가능한 phase detector, 무작위 test set, 전체 작업 성공률을 검증한 것은 아닙니다. successes는 구간 중 한 번 이상 목표 달성, goal_at_end는 16개 실행 직후 목표 유지입니다.</p>'''
    content+=table(terminal_summary,['method','completed','successes','goal_at_end'])
    content+=table(result['terminal_diagnostic']['per_case'],['task','initial_state','method','success','goal_after_intervention'])
    content+='''<p>precompute_ms는 도착 전에 수행한 모델 계산과 해당 방법의 CPU state 외삽 비용입니다. CPU 외삽은 원래 prefix를 다시 실행해 동일한 과거 입력임을 검증한 후 100회 별도로 재측정했습니다.
실제 영상/feature predictor는 구현하지 않았으며 oracle 입력을 얻는 비용은 계산하지 않습니다. oracle 결과를 배포 가능한 속도 향상으로 읽지 않습니다.</p>
<p>analytical_delay_ms = max(0, precompute_ms−200ms)+post_ms. GPU·제어를 실제로 동시에 실행한 시간이 아닙니다.
전처리/VLM/AE/decode는 포함하지만 simulator rendering, command transport는 제외합니다. 선행 계산으로 도착 후 대기는 줄어도 총연산은 늘 수 있습니다.
replay의 post 비용은 이미 준비한 action 배열을 복사하는 시간으로, 실제 제어 전달 지연이 아닙니다.</p>
<p>teacher_nmse16과 teacher_endpoint_mm는 같은 도착 조건의 fresh4와의 일치도입니다. 정답 조작 정밀도 또는 삽입 공차가 아닙니다.
Oracle future observation이 완벽해도 native policy 자체는 실패할 수 있으므로 oracle는 task 성공률의 수학적 상한이 아닙니다.
이 고정 조건·기존 가중치의 작은 진단을 새로운 task 또는 학습된 보정기 전체로 일반화하지 않습니다.</p>
<h2>짝지은 승패</h2>'''
    content+=table(paired,['comparison','pairs','wins','losses'])
    content+='<h2>조건별 결과</h2>'+table(per_case,['method','task','initial_state','success','immediate_goal','steps'])
    content+='<p><a href="summary.json">집계</a> · <a href="diagnostics.jsonl">action·latency 원자료</a> · <a href="rollouts.jsonl">rollout 원자료</a> · <a href="config.json">설정</a> · <a href="manifest.json">검증·provenance</a></p></html>'
    (out/'RESULTS.html').write_text(content)
    print(json.dumps({k:result[k] for k in ['status','control_status','matched_cases','total_rollouts','summary','paired','forecast_cpu_mean_ms','forecast_position_mean_mm','terminal_diagnostic']},ensure_ascii=False))


if __name__=='__main__':main()
