"""Matched K comparison, real-time response, and hypothetical PIM breakdown."""
import argparse
import csv
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import statistics

os.environ.setdefault('MPLCONFIGDIR','/tmp/vla-matplotlib-cache')


def rows(path):return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
def mean(values):return statistics.mean(values)
def table(headers,data):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in data)+'</tbody></table>'


def save_csv(path,records):
    fields=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--k',required=True);p.add_argument('--pim',required=True)
    p.add_argument('--realtime',required=True);p.add_argument('--output',required=True)
    p.add_argument('--repeat')
    args=p.parse_args();ks=Path(args.k);ps=Path(args.pim);rt=Path(args.realtime);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for source in [ks,rt]:
        status=json.loads((source/'status.json').read_text())
        if status['status']!='completed' or status['errors']:raise ValueError(f'Incomplete run {source}')
    rawk=rows(ks/'rollouts.jsonl');calls=rows(ks/'calls.jsonl');rawrt=rows(rt/'rollouts.jsonl')
    repeated=[]
    if args.repeat:
        rp=Path(args.repeat);rstatus=json.loads((rp/'status.json').read_text())
        if rstatus['status']!='completed' or rstatus['errors']:raise ValueError('Repeat is incomplete')
        repeated=rows(rp/'rollouts.jsonl');assert len(repeated)==48
        folder=out/'realtime_repeat';folder.mkdir(exist_ok=True)
        for path in rp.glob('*.json*'):shutil.copy2(path,folder/path.name)
        for name,digest in json.loads((rp/'manifest.json').read_text())['source_hashes'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    assert len(rawk)==72 and len(rawrt)==96
    for source,name in [(ks,'k_sweep'),(ps,'pim'),(rt,'realtime')]:
        folder=out/name;folder.mkdir(exist_ok=True)
        for path in source.glob('*.json*'):
            if path.name!='usage.json':shutil.copy2(path,folder/path.name)
    if (ps/'render_contention/summary.json').exists():
        shutil.copytree(ps/'render_contention',out/'pim/render_contention',dirs_exist_ok=True)
        shared_manifest=json.loads((ps/'render_contention/manifest.json').read_text())
        assert hashlib.sha256(Path('scripts/profile_render_contention.py').read_bytes()).hexdigest()==shared_manifest['source_hash']
        target=out/'source/scripts/profile_render_contention.py';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2('scripts/profile_render_contention.py',target)
    hashes={}
    for source in [ks,rt]:
        manifest=json.loads((source/'manifest.json').read_text())
        for name,digest in manifest['source_hashes'].items():
            actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if actual!=digest:raise ValueError(f'Source changed {name}')
            hashes[name]=actual
    profile_manifest=json.loads((ps/'manifest.json').read_text())
    assert hashlib.sha256(Path('scripts/profile_action_expert_pim.py').read_bytes()).hexdigest()==profile_manifest['source_hash']
    for name in list(hashes)+['scripts/profile_action_expert_pim.py','scripts/summarize_k_pim_followup.py']:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    (out/'source_verification.json').write_text(json.dumps(hashes,indent=2))
    kgroups=defaultdict(list);rgroups=defaultdict(list)
    for r in rawk:kgroups[r['method']].append(r)
    for r in rawrt:rgroups[r['method']].append(r)
    kstats=[]
    for method,rr in sorted(kgroups.items()):
        cc=[r for r in calls if r['method']==method]
        kstats.append(dict(method=method,n=len(rr),success=sum(r['success'] for r in rr),end_success=sum(r['goal_at_end'] for r in rr),
            post_ms=statistics.median(c['post_ms'] for c in cc),post_per_action=mean(r['post_ms_per_action'] for r in rr),
            pre_model_and_post_per_action=mean(r['model_plus_post_ms_per_action'] for r in rr),
            mean_pre_model_s=mean(sum(p['model_ms'] for p in r['plans']) for r in rr)/1000,
            mean_startup_s=mean(r['plan_wall_ms'] for r in rr)/1000))
    rtstats=[]
    for method,rr in sorted(rgroups.items()):
        cc=[c for r in rr for c in r['calls']]
        times=[c['compute_ms'] for c in cc]
        rtstats.append(dict(method=method,n=len(rr),success=sum(r['success'] for r in rr),stable=sum(r['stable_success'] for r in rr),
            compute_p50_ms=statistics.median(times),mean_hold_steps=mean(r['hold_steps'] for r in rr),
            mean_phase_s=mean(r['phase_wall_s'] for r in rr),mean_inclusive_s=mean(r['phase_wall_s']+r['startup_s'] for r in rr),
            mean_startup_s=mean(r['startup_s'] for r in rr),
            worker_ms_per_action=mean(sum(c['compute_ms'] for c in r['calls'])/max(1,r['action_steps']) for r in rr),
            total_model_ms_per_action=mean((r['startup_model_ms']+sum(c['compute_ms'] for c in r['calls']))/max(1,r['action_steps']) for r in rr),
            gate_accepts=sum(c['path']=='accepted_prediction' for c in cc),calls=len(cc),
            future_gate_accepts=sum(c['path']=='accepted_prediction' and c['offset']>0 for c in cc),
            future_calls=sum(c['offset']>0 for c in cc),
            tick_p95_mean_ms=mean(r['tick_lateness_p95_ms'] for r in rr),
            max_tick_lateness_ms=max(max(r['tick_lateness_ms']) for r in rr)))
    paired=[]
    for group,label in [(kgroups,'synchronous'),(rgroups,'realtime')]:
        for reference in ['fresh4_k16','fresh1_k16']:
            baseline={(r['task'],r['initial_state']):r for r in group[reference]}
            for method,rr in group.items():
                if method==reference:continue
                wins=losses=0;shared=[]
                for r in rr:
                    other=baseline[(r['task'],r['initial_state'])]
                    wins+=int(r['success'] and not other['success']);losses+=int(not r['success'] and other['success'])
                    if label=='realtime' and r['success'] and other['success']:
                        shared.append(dict(task=r['task'],initial_state=r['initial_state'],phase_delta_s=r['phase_wall_s']-other['phase_wall_s'],
                            inclusive_delta_s=r['phase_wall_s']+r['startup_s']-other['phase_wall_s']-other['startup_s']))
                paired.append(dict(experiment=label,method=method,reference=reference,wins=wins,losses=losses,n=len(rr),
                    shared_success_pairs=len(shared),shared_success_phase_delta_s=mean(x['phase_delta_s'] for x in shared) if shared else None,
                    shared_success_inclusive_delta_s=mean(x['inclusive_delta_s'] for x in shared) if shared else None,shared=shared))
    aggregate=dict(k_sweep=kstats,realtime=rtstats,paired=paired)
    target=next(r for r in paired if r['experiment']=='realtime' and r['reference']=='fresh4_k16' and r['method']=='whole_s25_k16')
    def overlap_delta(seconds):
        return mean(max(0.,x['inclusive_delta_s']-x['phase_delta_s']-seconds)+x['phase_delta_s'] for x in target['shared'])
    low,high=0.,max(x['inclusive_delta_s']-x['phase_delta_s'] for x in target['shared'])
    for _ in range(60):
        mid=(low+high)/2
        if overlap_delta(mid)>0:low=mid
        else:high=mid
    overlap=dict(scope='Optimistic accounting only, NOT concurrent precomputation measurement; identical preparation time, free overlap and no GPU/CPU/render contention or prediction degradation assumed. Same 8 joint-success pairs.',
        break_even_shared_lookahead_s=high,
        scenarios=[dict(available_lookahead_s=x,predicted_mean_inclusive_delta_s=overlap_delta(x)) for x in [0.,.8,1.6,3.2,4.,6.4]])
    (out/'overlap_bounds.json').write_text(json.dumps(overlap,indent=2))
    aggregate['optimistic_overlap']=overlap
    if repeated:
        repeat_stats=[];first={(r['task'],r['initial_state'],r['method']):r for r in rawrt}
        for method in sorted({r['method'] for r in repeated}):
            rr=[r for r in repeated if r['method']==method]
            repeat_stats.append(dict(method=method,n=len(rr),successes=sum(r['success'] for r in rr),
                stable_successes=sum(r['stable_success'] for r in rr),
                goal_flips=sum(r['success']!=first[(r['task'],r['initial_state'],method)]['success'] for r in rr),
                stable_flips=sum(r['stable_success']!=first[(r['task'],r['initial_state'],method)]['stable_success'] for r in rr),
                mean_hold_steps=mean(r['hold_steps'] for r in rr),mean_phase_wall_s=mean(r['phase_wall_s'] for r in rr),
                mean_inclusive_wall_s=mean(r['phase_wall_s']+r['startup_s'] for r in rr)))
        aggregate['repeat']=repeat_stats
    (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    save_csv(out/'k_comparison.csv',kstats);save_csv(out/'realtime_comparison.csv',rtstats)
    profile=json.loads((ps/'summary.json').read_text());scenarios=json.loads((ps/'pim_scenarios.json').read_text())
    partition=json.loads((ps/'partition.json').read_text())
    save_csv(out/'breakdown.csv',profile);save_csv(out/'pim_scenarios.csv',scenarios)
    payload=max(x['input_bytes']+x['output_bytes'] for x in partition['payloads'])
    # Exportable scientific charts; scenario panel is explicitly hypothetical.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    order=['fresh4','fresh1','predicted25_resume1'];lookup={r['mode']:r for r in profile}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    colors=['#94a3b8','#3b82f6','#8b5cf6','#f59e0b']
    stages=[('Other / input','other'),('Vision encoder','vision_ms'),('Vision-language','vlm_ms'),('Action expert','ae_ms')]
    left=np.zeros(3)
    for (label,key),color in zip(stages,colors):
        values=[lookup[m][key] if key!='other' else lookup[m]['total_ms']-sum(lookup[m][k] for k in ['vision_ms','vlm_ms','ae_ms']) for m in order]
        axes[0].barh(order,values,left=left,label=label,color=color);left+=np.array(values)
    axes[0].set_xlabel('Measured mean latency (ms)');axes[0].set_title('RTX 3090 breakdown (48 samples/mode)');axes[0].legend(fontsize=8)
    x=np.arange(3)
    for i,(label,speed,color) in enumerate([('GPU measured',1,'#334155'),('AE 2x',2,'#0ea5e9'),('AE 4x',4,'#14b8a6'),('Ideal AE=0',float('inf'),'#a7f3d0')]):
        values=[]
        for m in order:
            r=lookup[m];values.append(r['total_ms'] if i==0 else r['total_ms']-r['ae_ms']+r['ae_ms']/speed+(payload/32e6+.1 if i<3 else 0))
        axes[1].bar(x+(i-1.5)*.19,values,.19,label=label,color=color,hatch='' if i==0 else '//')
    axes[1].set_xticks(x,order,rotation=12);axes[1].set_ylabel('Latency (ms)');axes[1].set_title('Hypothetical PIM: 32 GB/s + 0.1 ms round trip');axes[1].legend(fontsize=8)
    fig.savefig(out/'breakdown.png',dpi=180);fig.savefig(out/'breakdown.svg');plt.close(fig)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>K16 기준 성능·실시간 제어·PIM breakdown</title>
<style>body{font:16px/1.65 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#17212b}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto;margin:18px 0}td,th{border:1px solid #ccd4dc;padding:8px;white-space:nowrap}th{background:#e9f0f7}.note{padding:16px;background:#fff2d6}img{max-width:100%}label{display:block;margin:12px 0}input{width:240px}output{font-weight:bold}</style>
<h1>K=16 기준선 복원 · 실시간 대기 · action expert PIM</h1>
<p>H=50 고정, GR00T N1.6 LIBERO, RTX 3090. 앞선 feature 탐색(초기 상태 0·4)에서 제외했던 초기 상태 1·2·3·5 × 3개 작업 = 12개 조건. 모델 학습 데이터와 분리된 일반화 검증을 뜻하지는 않습니다. 접촉·배치·서랍 작업이며 정밀 삽입 benchmark가 아닙니다.</p>
<p><strong>표기:</strong> fresh4의 4는 액션 생성 계산 단계 N, k16의 16은 한 번에 실행하는 액션 수 K입니다. fresh는 현재 실제 관측으로 전부 새로 계산한다는 뜻이며 별도 모델이 아닙니다.</p>
<p class="note">한 호출의 지연 감소, 액션당 계산 비용, 실제 완료 시간은 서로 다릅니다. 준비 비용 포함 여부와 성공률을 함께 비교해야 합니다. 미래 예측기는 현재 simulator 상태를 아는 nominal simulator이며 실제 미래는 읽지 않지만 배포 가능한 학습 예측기는 아닙니다.</p>
<h2>이번 결과의 판단</h2><p><strong>K16 기준선을 유지할 근거가 더 강합니다.</strong> 동일 12개 조건에서 fresh4/K16은 동기·실시간 모두 8/12, fresh4/K4는 모두 7/12였습니다. 예측+25% 보정/K16도 8/12이고, 실시간 안정 유지에서는 8/12 대 기준선 7/12였으나 소표본의 한 사례 차이입니다.</p>
<p>부분 보정은 실제 대기 tick을 평균 5.9→0.1로 줄였습니다. 하지만 양쪽 모두 성공한 8개 조건에서 phase 단축은 평균 0.188초이고, 준비를 포함하면 오히려 평균 3.364초 느렸습니다. 현재 구현의 전체 시간 이득은 없습니다. 같은 조건에서 준비를 숨기지 않고 이기려면 전체 준비 비용이 평균 약 0.188초 이하여야 한다는 관측상 기준이 생겼습니다. 이는 배포 보장이나 일반적 임계값은 아닙니다.</p>
<p>실제 PIM은 미검증입니다. 다만 N4 action expert가 전체 지연의 약 절반이므로, 이 구간의 가속은 N4 계산을 유지하며 응답을 줄이는 후보입니다. 이미 N1로 줄인 방식에서는 PIM의 추가 이득 상한이 작아집니다.</p>
<h2>1. K4/K16 비교 — 72회</h2><p>동일한 prefix 이후 같은 물리적 시작 상태와 영상에서 비교했습니다. 이 표의 시뮬레이터는 추론 중 멈춥니다. 기본 성공은 한 번 이상 목표 달성, 종료 유지는 K-action chunk 종료 시 목표 유지입니다. 액션당 수치는 각 trial의 비율을 평균했습니다.</p>'''
    page+=table(['방식','성공','chunk 종료 유지','호출 p50 ms','도착 후 ms/action','선행 모델+도착 후 ms/action','준비 모델 평균 s','준비 전체 평균 s'],[
        [r['method'],f"{r['success']}/{r['n']}",f"{r['end_success']}/{r['n']}",f"{r['post_ms']:.2f}",f"{r['post_per_action']:.2f}",f"{r['pre_model_and_post_per_action']:.2f}",f"{r['mean_pre_model_s']:.2f}",f"{r['mean_startup_s']:.2f}"] for r in kstats])
    page+='''<h2>2. 실제 20Hz 제어 루프 — 96회</h2><p>GPU 추론은 별도 단일 worker에서 수행하고, 환경은 50ms마다 진행합니다. 다음 액션이 준비되지 않으면 이동 0·기존 gripper 명령을 실행합니다. 렌더링과 추론은 같은 3090을 사용합니다. 초기 예측 bank 준비 중에는 환경을 멈추며, 이 준비 시간을 포함한 열을 별도로 표시합니다. 미리 계산하는 시간을 공짜로 숨기지 않습니다.</p>
<p>whole_gate는 state/영상 오차가 작은 경우 보관한 예측 액션을 수용하고, 나머지는 25% feature 보정합니다. 임계값은 사전 고정한 휴리스틱이며 학습된 안전 판정기가 아닙니다. 안정 성공은 첫 목표 달성 뒤 이동 0·동일 gripper를 10스텝 유지한 별도 진단입니다.</p>'''
    page+=table(['방식','성공','0.5초 유지','worker p50 ms','평균 대기 ticks','평균 phase s*','준비 포함 s*','worker ms/action','선행 모델 포함 ms/action','예측 수용/호출'],[
        [r['method'],f"{r['success']}/{r['n']}",f"{r['stable']}/{r['n']}",f"{r['compute_p50_ms']:.2f}",f"{r['mean_hold_steps']:.1f}",f"{r['mean_phase_s']:.2f}",f"{r['mean_inclusive_s']:.2f}",f"{r['worker_ms_per_action']:.2f}",f"{r['total_model_ms_per_action']:.2f}",f"{r['gate_accepts']}/{r['calls']}"] for r in rtstats])
    gate=next(r for r in rtstats if r['method']=='whole_gate_k16')
    page+=f'<p>gate 수용에는 처음부터 정확한 준비 시작 프레임이 포함됩니다. 이를 제외한 실제 미래 호출에서는 {gate["future_gate_accepts"]}/{gate["future_calls"]}회 수용했습니다. gate의 성공률은 부분 보정과 같았으나 안정 유지가 7/12 대 8/12였습니다. 이 임계값의 정확도 우위는 확인하지 못했습니다.</p>'
    page+='<p>* 전체 평균에는 실패 시 고정 시간 예산까지 실행한 trial도 포함됩니다. 성공률이 다른 방법의 평균 시간만으로 빠르다고 판단하지 않습니다. 아래는 양쪽 모두 성공한 같은 초기 조건의 시간 차이입니다. 음수는 method가 더 빠름을 뜻합니다.</p>'
    page+=table(['방식','기준','성공 이김/짐','양쪽 성공 쌍','phase 차이 s','준비 포함 차이 s'],[
        [r['method'],r['reference'],f"{r['wins']}/{r['losses']}",r['shared_success_pairs'],
         '-' if r['shared_success_phase_delta_s'] is None else f"{r['shared_success_phase_delta_s']:.3f}",
         '-' if r['shared_success_inclusive_delta_s'] is None else f"{r['shared_success_inclusive_delta_s']:.3f}"] for r in paired if r['experiment']=='realtime'])
    page+='<h3>앞선 이동 구간에 준비를 겹칠 수 있다면?</h3><p class="note">아래는 실제 동시 선행 계산의 측정값이 아닙니다. 준비가 기존과 같은 속도로 진행되고 GPU·CPU·렌더링 경합과 예측 정확도 저하 없이 겹쳐진다는 낙관적 시간 계산입니다. 준비가 끝난 뒤의 실제 제어 구간은 위에서 측정했습니다.</p>'
    page+=table(['미리 사용할 수 있는 시간 s','기준선 대비 예상 총 시간 차이 s'],[
        [r['available_lookahead_s'],f"{r['predicted_mean_inclusive_delta_s']:+.3f}"] for r in overlap['scenarios']])
    page+=f'<p>동일하게 성공한 8개 조건 평균에서 손익분기 선행 여유는 이 가정하에 약 <strong>{high:.2f}초</strong>입니다. K16의 0.8초만큼 겹치는 것으로는 부족합니다. 더 긴 확정된 접근 동작이 있거나, 준비 비용 자체를 크게 줄이는 예측기가 필요합니다. 이 조건의 실제 동시 실행은 미검증입니다.</p>'
    if repeated:
        page+='<h3>같은 조건의 실시간 재실행 — 48회</h3><p>핵심 K16 방식 네 개를 같은 12개 초기 조건에서 역순으로 다시 실행했습니다. 새 독립 초기 조건 12개가 추가된 것은 아닙니다. 성공 판정이 바뀐 경우와 안정 유지 판정이 바뀐 경우를 각각 표시합니다.</p>'
        page+=table(['방식','두 번째 성공','두 번째 안정 유지','성공 판정 변동','안정 유지 변동','평균 대기 tick','평균 phase s','준비 포함 s'],[
            [r['method'],f"{r['successes']}/{r['n']}",f"{r['stable_successes']}/{r['n']}",r['goal_flips'],r['stable_flips'],f"{r['mean_hold_steps']:.1f}",f"{r['mean_phase_wall_s']:.2f}",f"{r['mean_inclusive_wall_s']:.2f}"] for r in repeat_stats])
    page+='<h2>3. GPU 실측 breakdown과 PIM 가정</h2><img src="breakdown.png"><p>각 방식 12개 관측 × 4회 = 48회. 표는 단계 평균이며 합산 가능한 수치입니다. 예측 보정은 도착 후 비용만 포함합니다. K는 같은 호출의 연산량을 바꾸지 않고 호출 빈도와 실행할 action 수를 바꿉니다.</p>'
    page+=table(['방식','입력 ms','영상 ms','영상·언어 ms','state ms','expert ms','변환 ms','전체 평균 ms'],[
        [r['mode']]+[f"{r[k]:.2f}" for k in ['preprocess_ms','vision_ms','vlm_ms','state_and_fusion_ms','ae_ms','decode_ms','total_ms']] for r in profile])
    page+='<p class="note">PIM 결과는 실측이 아닌 가정입니다. action encoder·DiT·action decoder·latent 갱신 전체를 PIM으로 옮기고, weights는 이미 상주하며 conditioning/latent는 호출당 한 번 왕복한다고 가정합니다. state 결합과 backbone은 GPU에 남습니다. 선택한 dtype의 정확도 유지, nonlinear 연산 지원, 내부 메모리 대역폭 및 실제 AE 가속률은 검증하지 않았습니다.</p>'
    page+=f'<p>현재 관측에서 가장 큰 왕복 activation payload: <strong>{payload:,} bytes</strong>. 32GB/s 링크 + 고정 왕복 0.1ms 예시에서는 전송 약 {payload/32e6+.1:.3f}ms입니다. 모델 weight 전송은 포함하지 않습니다. 상주 parameter bytes는 실제 DRAM traffic이 아닙니다.</p>'
    pimrows=[]
    for r in profile:
        rest=r['total_ms']-r['ae_ms'];transfer=payload/32e6+.1
        pimrows.append([r['mode'],f"{r['total_ms']:.2f}",f"{rest+r['ae_ms']/2+transfer:.2f}",f"{rest+r['ae_ms']/4+transfer:.2f}",f"{rest:.2f}",f"{r['total_ms']/rest:.2f}x"])
    page+=table(['방식','GPU 실측 ms','AE 2x 가정 ms','AE 4x 가정 ms','AE=0·전송=0 ms','이상적 최대 가속'],pimrows)
    if (ps/'render_contention/summary.json').exists():
        shared=json.loads((ps/'render_contention/summary.json').read_text())
        page+='<h3>20Hz 렌더링과 GPU를 공유하는 별도 latency 진단</h3><p>정지한 한 장면에서 방식당 16회 측정했습니다. 실제 작업 전체의 시간이나 성공률은 아닙니다. 고립된 profile과 실제 제어 worker 지연을 같은 값으로 취급하지 않기 위한 추가 검사입니다. PIM 가정에서는 측정된 non-AE 시간을 고정하므로, 실제 offload가 GPU 경합을 바꾸는 효과는 모델링하지 않습니다.</p>'
        page+=table(['방식','현재 실측 ms','expert ms','AE 2x 가정 ms','AE 4x 가정 ms','AE=0·전송=0 ms'],[
            [r['mode'],f"{r['total_ms']:.2f}",f"{r['ae_ms']:.2f}",
             f"{r['total_ms']-r['ae_ms']/2+payload/32e6+.1:.2f}",
             f"{r['total_ms']-.75*r['ae_ms']+payload/32e6+.1:.2f}",f"{r['total_ms']-r['ae_ms']:.2f}"] for r in shared])
        page+='<p>카메라 관측을 얻은 뒤 다음 제어 시점까지 남는 시간도 고려해야 합니다. 50ms 주기라도 모델이 50ms 전체를 사용할 수 있는 것은 아닙니다. PIM 4x 가정만으로 실제 제어 deadline 충족을 보장하지 않습니다.</p>'
    page+='''<h2>PIM 가정 바꿔 보기</h2><p>아래는 측정값을 바탕으로 한 계산기이며 실제 PIM 측정이 아닙니다.</p><label>방식 <select id="mode"></select></label><label>AE 가속률 <input id="speed" type="range" min="0.5" max="8" step="0.5" value="2"><output id="speedval"></output></label><label>링크 대역폭 GB/s <input id="bandwidth" type="range" min="8" max="128" step="8" value="32"><output id="bwval"></output></label><label>고정 왕복 비용 ms <input id="fixed" type="range" min="0" max="5" step="0.1" value="0.1"><output id="fixedval"></output></label><p id="calc"></p>'''
    page+='<script>const profiles='+json.dumps(profile)+';const bytes='+str(payload)+''';const mode=document.querySelector('#mode');profiles.forEach((r,i)=>mode.add(new Option(r.mode,i)));mode.value=String(profiles.findIndex(r=>r.mode==='fresh4'));function update(){const r=profiles[+mode.value],s=+document.querySelector('#speed').value,b=+document.querySelector('#bandwidth').value,f=+document.querySelector('#fixed').value;const transfer=bytes/(b*1e6)+f,total=r.total_ms-r.ae_ms+r.ae_ms/s+transfer;document.querySelector('#speedval').textContent=s+'x';document.querySelector('#bwval').textContent=b;document.querySelector('#fixedval').textContent=f;document.querySelector('#calc').textContent='가정된 전체 '+total.toFixed(2)+'ms / GPU 대비 '+(r.total_ms/total).toFixed(2)+'배 / K16 액션당 '+(total/16).toFixed(2)+'ms / K4 액션당 '+(total/4).toFixed(2)+'ms (선행 예측 비용 제외)';}document.querySelectorAll('input,select').forEach(x=>x.addEventListener('input',update));update();</script>'''
    page+='''<h2>검증과 제한</h2><ul><li>기준선과 예측 시작 state/RGB 일치 및 소스 hash를 검사했습니다. 실패·실행 오류·시간 예산 종료를 섞지 않습니다.</li><li>고정된 task phase 시작점과 nominal simulator 예측을 사용했습니다. phase detector, 학습된 world model, 실제 로봇이나 실제 PIM은 아닙니다.</li><li>12개 조건의 작은 표본입니다. 실시간 trial은 조건당 1회여서 scheduler jitter에 따른 결과 변동 가능성이 남습니다.</li><li>부정확한 초기 디버그 실행과 4-step 경계 테스트는 집계에서 제외했습니다.</li></ul>
<p><a href="aggregate.json">전체 집계·짝 비교</a> · <a href="k_sweep/rollouts.jsonl">K 실험 원자료</a> · <a href="realtime/rollouts.jsonl">실시간 원자료·실행 action trace</a> · <a href="pim/pim_scenarios.json">PIM 전체 가정 표</a> · <a href="pim/partition.json">텐서·parameter bytes</a> · <a href="breakdown.svg">벡터 그림</a></p><p>CSV: <a href="k_comparison.csv">K 비교</a> · <a href="realtime_comparison.csv">실시간 비교</a> · <a href="breakdown.csv">breakdown</a> · <a href="pim_scenarios.csv">PIM 가정 전체</a></p></html>'''
    if repeated:
        page=page.replace('실시간 trial은 조건당 1회여서 scheduler jitter에 따른 결과 변동 가능성이 남습니다.',
            '핵심 K16 네 방식은 같은 12개 조건을 두 번, 나머지는 한 번 실행했습니다. 더 넓은 조건에서의 검증이 필요합니다.')
    (out/'RESULTS.html').write_text(page)
    print(json.dumps(aggregate,ensure_ascii=False))


if __name__=='__main__':main()
