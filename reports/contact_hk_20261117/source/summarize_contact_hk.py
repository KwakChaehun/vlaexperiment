"""Export an auditable contact-task report and fixed-task paired comparisons."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
import run_contact_hk as runner


def table(rows,columns):
    def fmt(v):
        if isinstance(v,float):return f'{v:.3g}'
        return str(v)
    return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in columns)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(c,'')))+'</td>'
        for c in columns)+'</tr>' for r in rows)+'</tbody></table>'


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='results/contact_hk_20261117')
    p.add_argument('--output',default='reports/contact_hk_20261117');args=p.parse_args()
    source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/'summary.json').read_text())
    data=runner.base.rows(source/'trials.jsonl');status=json.loads((source/'status.json').read_text())
    comparisons=runner.comparisons(data)
    # Family: the two primary claims needed to assert H16/K4 beats both alternatives.
    primary=[r for r in comparisons if r['scenario']=='contact' and r['metric']=='success'
             and r['comparison'] in ['16/4 - 50/4','16/4 - 50/16']]
    adjusted=0.
    for i,r in enumerate(sorted(primary,key=lambda x:x['exact_mcnemar_p'])):
        adjusted=max(adjusted,min(1.,(len(primary)-i)*r['exact_mcnemar_p']))
        r['holm_p_primary_two']=adjusted
    summary['paired']=comparisons
    summary['status']=status
    summary['source_directory']=str(source.resolve())
    summary['as_of_kst']=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
    hashes={}
    for name in ['config.json','manifest.json','status.json','trials.jsonl','calls.jsonl']:
        shutil.copy2(source/name,out/name)
        hashes[name]=hashlib.sha256((out/name).read_bytes()).hexdigest()
    code=out/'source';code.mkdir(exist_ok=True)
    for name in ['scripts/run_contact_hk.py','scripts/summarize_contact_hk.py',
                 'scripts/run_gr00t_wrist_pilot.py','vla_eval/gr00t16_pilot.py']:
        dest=code/Path(name).name;shutil.copy2(runner.base.ROOT/name,dest)
        hashes[str(dest.relative_to(out))]=hashlib.sha256(dest.read_bytes()).hexdigest()
    summary['hashes']=hashes
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    columns=['scenario','suite','task','initial_state','H','K','N','status','success','sustained_success','steps','wall_s']
    with (out/'trials.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(data)
    groups=summary['summary']
    results=[dict(scenario=r['scenario'],HK=f"{r['H']}/{r['K']}",success=f"{r['successes']}/{r['n']}",
            rate_percent=100*r['successes']/r['n'],sustained=f"{r['sustained']}/{r['n']}",
            p50_ms=r['p50_ms'],p95_ms=r['p95_ms'],amortized_ms=r['amortized_ms'],
            perturbations_applied=r['perturbations_applied']) for r in groups]
    content='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>GR00T contact H/K</title>
<style>body{max-width:1200px;margin:36px auto;padding:0 24px;font:16px/1.65 system-ui;color:#17202a}table{border-collapse:collapse;font-size:14px;margin:20px 0}td,th{border:1px solid #ccd3da;padding:7px 10px;text-align:left}th{background:#edf2f7}img{max-width:100%}code{background:#eef;padding:2px}a{color:#0369a1}</style>
<h1>RTX 3090: 접촉·수납 작업 H/K 비교</h1>'''
    content+=f'<p>갱신 {summary["as_of_kst"]}. 상태: {html.escape(status["status"])}. 완료 {status["completed"]}회.</p>'
    contact=[r for r in results if r['scenario']=='contact']
    content+='<p><strong>관측된 접촉 성공률 순위는 아래 표와 같습니다. 이 표만으로 일반적인 우열이나 동등성을 확정하지 않습니다.</strong></p>'
    content+=table(sorted(contact,key=lambda r:-r['rate_percent']),['HK','success','rate_percent','p50_ms','amortized_ms'])
    content+='<p>앞선 Spatial·카메라·SmolVLA 216회 평가는 <a href="../gpu_3090_wrist_20261117/RESULTS.md">별도 보고서</a>에 보존합니다. 이번 평가와 독립 표본 수를 늘리기 위해 합산하지 않습니다.</p>'
    content+='''<p>주 지표는 동일 초기 조건에서 공식 LIBERO 목표를 제한 control steps 안에 한 번 이상 달성하는 비율입니다.
GR00T N1.6, BF16, flow N=4를 고정했습니다. H는 action latent 생성 길이, K는 실제 실행 후 재관측 간격입니다.
50/4 대 16/4가 H 효과, 50/16 대 50/4가 K 효과를 분리합니다.</p>
<p>접촉 실험은 결과를 보기 전에 고른 Goal 4개 작업과 LIBERO-10 1개 작업 × 초기 상태 12개 × H/K 3개 = 180회입니다.
스트레스 실험은 별도의 bowl-on-plate 작업에서 6개 초기 상태 × 접시 이동 0/−20/+20mm × H/K 3개 = 54회입니다.
각 묶음 내 모델 실행 순서를 섞었습니다. 노이즈는 base seed + initial state × 10000 + control step으로 결정하며 H별 native noise prefix를 공유합니다.</p>
<p><strong>이 결과는 좁은 공차의 핀 삽입 성능이 아닙니다.</strong> LIBERO의 공식 On/In/Close 조건을 그대로 사용합니다.
예를 들어 cream-cheese-in-bowl 작업의 실제 목표는 On이며, 일부 In 조건은 가상 region의 containment입니다.
물리적 접촉을 모든 목표에서 따로 요구하거나 성공 기준을 임의로 강화하지 않았습니다. 목표 predicate는 각 원자료에 기록했습니다.</p>
<p>추론 중 simulator는 멈춥니다. 실제 비동기 제어, 관측 지연, PIM 장치 성능은 평가하지 않았습니다.
H=16은 native H=50 모델을 재학습 없이 바꾼 조건입니다. Checkpoint는 공개 커뮤니티 all-four-LIBERO fine-tune이므로 학습 task와 겹칩니다.</p>
<h2>성공률과 비용</h2>'''
    content+=table(results,['scenario','HK','success','rate_percent','sustained','p50_ms','p95_ms','amortized_ms','perturbations_applied'])
    content+='''<p>sustained는 최초 성공 직후 10 control steps(0.5초) 동안 위치·회전 명령을 0으로, gripper 명령을 직전 값으로 유지할 때 목표가 계속 참인지입니다.
정책을 더 실행하거나 강제로 놓게 하지 않습니다. 이는 보조 안정성 검사이며 자유로운 release 성공률이 아닙니다.
amortized_ms는 성공 여부와 관계없이 실제 정책 control action당 추론 총비용입니다. 보조 10-step 검사는 분모에 포함하지 않습니다.</p>
<h2>짝지은 비교</h2><p>delta_pp &gt; 0이면 왼쪽 설정의 성공률이 높습니다. wins/losses는 같은 초기 조건에서 한쪽만 성공한 횟수입니다.
95% bootstrap 구간은 지정된 작업 안에서 초기 상태를 재표집한 구간으로, 다른 작업에 대한 일반화 구간은 아닙니다.
둘 다 전부 성공해서 bootstrap 구간이 0인 경우에도 동등성이 입증되지는 않습니다.
주 비교 두 개에는 Holm 보정을 적용합니다. 그 외 비교는 탐색 결과이며 다중 비교를 보정하지 않았습니다.</p>'''
    content+=table(comparisons,['scenario','comparison','metric','pairs','wins','losses','ties','delta_pp','bootstrap_ci95_pp','exact_mcnemar_p','holm_p_primary_two'])
    content+='<h2>작업별 성공</h2>'+table(summary['per_task'],['scenario','task','H','K','successes','n'])
    content+='''<h2>외란 조건의 범위</h2><p>접시를 world X축으로 ±20mm 즉시 이동하는 합성 스트레스입니다. 실제 물리적 힘을 가한 실험이 아닙니다.
시작 후 41/45/49/53/57/61번째 action 직전에 한 번 적용하며, chunk 도중이어도 실행합니다. +0mm에도 같은 재관측·속도 초기화 처리를 합니다.
이벤트 직후 정책에 강제 관측하지 않고 각 K의 예정된 시점에만 관측합니다. 이미 성공해서 이벤트 전에 끝난 trial은 perturbations_applied에서 제외되지만 전체 성공률 분모에는 유지합니다.</p>
<h2>PIM 판단과 구분</h2><p>16/4가 두 대조군보다 접촉 성공률이 높아야 '짧은 H/K가 더 정확하다'는 현재 가설을 지지합니다.
50/4와 16/4가 함께 좋아지면 먼저 K 효과를 고려해야 합니다. 정확도 이득이 없을 경우 정확도 우위를 PIM 동기로 주장할 수 없습니다.
정확도 유지와 지연·에너지 이득이라는 별도 가설에는 실제 장치 또는 검증된 비용 모델이 필요합니다.</p>
<p><a href="summary.json">집계 JSON</a> · <a href="trials.csv">trial CSV</a> · <a href="trials.jsonl">전체 trial</a> · <a href="calls.jsonl">호출 시간</a> · <a href="config.json">사전 설정</a> · <a href="manifest.json">모델·코드 provenance</a></p>
<p>trial의 replay 경로는 summary.json의 source_directory를 기준으로 합니다. 원본 GIF는 results의 replays에 보관하며 이 스냅샷에 중복 복사하지 않습니다.</p>'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scenarios=sorted({r['scenario'] for r in groups})
    fig,axes=plt.subplots(1,len(scenarios),figsize=(max(5,4*len(scenarios)),4),squeeze=False)
    for ax,scenario in zip(axes[0],scenarios):
        rr=[next((r for r in groups if r['scenario']==scenario and (r['H'],r['K'])==cell),None)
            for cell in [(50,16),(50,4),(16,4)]]
        for i,r in enumerate(rr):
            if r is None:continue
            y=100*r['successes']/r['n'];lo,hi=np.array(r['ci95'])*100
            ax.bar(i,y,color=['#64748b','#d97706','#0284c7'][i],alpha=.85)
            ax.errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],color='black',capsize=4)
            ax.text(i,3,f"{r['successes']}/{r['n']}",ha='center',va='bottom',color='white',weight='bold')
        ax.set(xticks=range(3),xticklabels=['50/16','50/4','16/4'],ylim=(0,105),title=scenario,
               xlabel='Generated H / Executed K',ylabel='Official task success (%)')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('GR00T N1.6 / RTX 3090 / N=4 — Wilson 95% intervals',fontsize=12)
    fig.tight_layout();fig.savefig(out/'success_rates.png',dpi=180);plt.close(fig)
    content+='<img src="success_rates.png" alt="H/K success rates and confidence intervals"></html>'
    (out/'RESULTS.html').write_text(content)
    print(json.dumps(dict(output=str(out),status=status,primary=primary),ensure_ascii=False))


if __name__=='__main__':main()
