"""Export the completed/partial GPU pilot without combining repeated controls as new trials."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def read(path):
    if not path.exists():return []
    if path.suffix=='.jsonl':return [json.loads(x) for x in path.read_text().splitlines() if x]
    return json.loads(path.read_text())


def table(data,columns):
    out=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for r in data:
        out.append('| '+' | '.join(f'{r[c]:.4g}' if isinstance(r.get(c),float)
                   else str(r.get(c,'')) for c in columns)+' |')
    return '\n'.join(out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='results/wrist_gr00t_20261117')
    parser.add_argument('--output',default='reports/gpu_3090_wrist_20261117')
    args=parser.parse_args();source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    live=[];offline=[];hashes={};status={}
    for run in ['run','refresh8','adaptive','small','small_n2','small_n4','requested_hk','requested_hk_extra','small_n2_extra']:
        folder=source/run
        if not folder.exists():continue
        target=out/run;target.mkdir(exist_ok=True)
        for filename in ['config.json','manifest.json','status.json','small_status.json','offline_summary.json',
                         'closed_loop_summary.json','closed_loop.jsonl','calls.jsonl','offline.jsonl','small_offline.jsonl']:
            p=folder/filename
            if p.exists():
                shutil.copy2(p,target/filename)
                hashes[str(Path(run)/filename)]=hashlib.sha256(p.read_bytes()).hexdigest()
        status[run]=read(folder/('small_status.json' if run.startswith('small') else 'status.json'))
        for r in read(folder/'closed_loop_summary.json'):
            live.append(dict(run=run,model='SmolVLA' if run.startswith('small') else 'GR00T',
                mode=r['mode'],H=r['H'],K=r['K'],N=r['N'],success=f"{r['successes']}/{r['completed']}",
                successes=r['successes'],completed=r['completed'],ci95=r['wilson95'],
                p50_ms=r['p50_ms'],p95_ms=r['p95_ms'],amortized_ms=r['amortized_ms_per_action']))
        if run.startswith('small'):
            groups=defaultdict(list)
            for r in read(folder/'small_offline.jsonl'):
                if r['K']==4:groups[(r['H'],r['N'])].append(r)
            for (h,n),rr in groups.items():
                offline.append(dict(run=run,model='SmolVLA',mode='both_fresh',H=h,K=4,N=n,lag=0,
                    samples=len(rr),expert_nmse=float(np.mean([r['expert_nmse'] for r in rr])),
                    p50_ms=float(np.median([r['timing']['total_ms'] for r in rr]))))
        else:
            for r in read(folder/'offline_summary.json'):
                if r['K']==4:
                    offline.append(dict(run=run,model='GR00T',mode=r['mode'],H=r['H'],K=4,N=r['N'],lag=r['lag'],
                        samples=r['samples'],expert_nmse=r['expert_nmse'],p50_ms=r['call_p50_ms']))
    paired=[]
    for model,runs,h,k,n in [('GR00T',['requested_hk','requested_hk_extra'],50,16,4),
                            ('GR00T',['requested_hk','requested_hk_extra'],50,4,4),
                            ('GR00T',['requested_hk','requested_hk_extra'],16,4,4),
                            ('SmolVLA',['small_n2','small_n2_extra'],16,4,2)]:
        trials=[r for run in runs for r in read(source/run/'closed_loop.jsonl')
                if (r['mode'],r['H'],r['K'],r['N'],r['status'])==('both_fresh',h,k,n,'completed')]
        if not trials:continue
        keys=[(r['task'],r['initial_state']) for r in trials]
        assert len(keys)==len(set(keys)), 'Do not pool repeated controls'
        times=[t for r in trials for t in r['call_ms']]
        successes=sum(r['success'] for r in trials);count=len(trials)
        z=1.959963984540054;p=successes/count;den=1+z*z/count
        center=(p+z*z/(2*count))/den
        radius=z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
        paired.append(dict(model=model,H=h,K=k,N=n,success=f'{successes}/{count}',
            ci95=[float(center-radius),float(center+radius)],
            p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
            amortized_ms=sum(times)/sum(r['steps'] for r in trials)))
    stamp=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
    (out/'snapshot.json').write_text(json.dumps(dict(as_of_kst=stamp,source=str(source.resolve()),
        statuses=status,offline=offline,closed_loop=live,paired_hk=paired,hashes=hashes),indent=2))
    for name in ['gr00t_environment.txt','smol_environment.txt']:
        if (source/name).exists():shutil.copy2(source/name,out/name)
    for version in ['source_v1','source_v2']:
        if (source/version).exists():shutil.copytree(source/version,out/version,dirs_exist_ok=True)
    text=f'''# RTX 3090 손목 갱신·H/K pilot

갱신: {stamp}. 원자료: `{source.resolve()}`.

## 측정 범위

- RTX 3090 24GB, PyTorch 2.9/CUDA 12.8. GR00T N1.6 BF16와 SmolVLA BF16 autocast를 순차 실행했습니다.
- GR00T: 공개 커뮤니티 `0xAnkitSingh/GR00T-N1.6-LIBERO`, revision `d690a226ad06e81736786f56cf879d2ed1dd3f0f`. NVIDIA 공식 N1.6 코드 revision `9b37aa1ce69c73c6d165233fa88128283bba4508`. N1.7은 Cosmos backbone 접근 승인이 없어 실행하지 않았습니다.
- GR00T 공식 sampler 최대 절대 차이 약 3e-9, GPU allocated memory peak 6.34 GiB. SmolVLA 공식 sampler 차이 0.
- GR00T native는 **50개 생성 후 16개 decode**입니다. H=16/8/4 실험은 실제 action latent 길이를 바꿉니다. 재학습하지 않은 H 변경입니다.
- Offline: 공개 LIBERO episode 1274/1269/1285의 12관측. 시연과 checkpoint 학습의 overlap 가능성이 있습니다. NMSE는 동일 std로 정규화한 앞 4개 action 오차이며 성공률이 아닙니다.
- Closed loop: 기본 LIBERO Spatial task 0/1/2 × initial state 0/1, 설정당 6회, 최대 220 action. requested_hk_extra/small_n2_extra는 task 3/4/5입니다. 최신 관측 → H개 생성 → K개 실행 → 실제 시뮬레이터 이미지·state 재관측입니다. 미래 이미지를 예측하거나 oracle 미래 frame을 넣지 않았습니다.
- 시뮬레이터는 추론 중 멈춥니다. 지연은 전처리·vision·VLM·AE·decode를 포함하고 renderer/control은 제외합니다. 실제 지연을 반영한 제어 성공률 또는 sub-mm 정밀도 결과가 아닙니다.
- 여러 실행의 동일 fresh 대조군은 반복 확인이며 성공 횟수를 합쳐 독립 표본 수를 늘리지 않습니다. 6/6의 Wilson 95% 구간도 약 61–100%입니다.

## 요청한 세 조합: 서로 다른 12개 초기 조건

'''
    text+=table(paired,['model','H','K','N','success','p50_ms','p95_ms','amortized_ms'])
    text+='''

첫 6개 조건의 천장 효과를 보고 task 3/4/5를 추가한 탐색 결과입니다. 같은 task/initial state를 중복 합산하지 않았습니다. GR00T 12/12의 Wilson 95% 구간은 약 76–100%, SmolVLA 9/12는 약 47–91%입니다. 관측된 성공률 차이는 일반적인 우월성이나 정밀도 개선의 증명이 아닙니다. 정책별 학습 데이터/학습 예산도 통제하지 않았습니다.

H=50→16의 호출 시간 감소는 약 0.6ms에 그쳤습니다. K=16은 재계획 빈도를 줄여 action당 추론 비용을 줄이지만, 이 실험에서는 접촉 정밀도·외란 복구·추론 중 움직이는 환경을 평가하지 않았습니다. 미래 feature 예측 또는 미리 계산한 action의 작은 보정은 이 GPU batch에 포함되지 않았습니다.

## H/K와 작은 모델 전체 비교

'''
    hk=[r for r in live if r['run'] in ('run','small','small_n2','small_n4','requested_hk','requested_hk_extra','small_n2_extra') and r['mode']=='both_fresh']
    text+=table(hk,['run','model','H','K','N','success','p50_ms','p95_ms','amortized_ms'])
    text+='\n\n`amortized_ms`는 실제 실행 action당 총 추론 시간입니다. N은 flow 횟수이며 모델 크기와 별개입니다.\n\n## 외부/손목 갱신\n\n'
    text+=table([r for r in live if r['model']=='GR00T' and (r['mode']!='both_fresh' or (r['H']==16 and r['K']==4))],
                ['run','mode','H','K','N','success','p50_ms','p95_ms','amortized_ms'])
    text+='''

- `run`: 외부/손목 stale 조건은 16 control steps마다 갱신. K=4에서 사용 영상 age는 0/4/8/12 steps, 20Hz 기준 최대 600ms입니다.
- `refresh8`: 8 steps마다 갱신, K=4에서 age 0/4 steps, 최대 200ms. 캐시 miss 때 두 카메라를 묶어 인코딩합니다.
- `adaptive`: 최대 16 steps, 외부 영상과 저장 영상의 평균 RGB 차이가 5/255를 초과하면 조기 갱신. 탐색 후 추가한 조건으로 독립 검증 결과는 아닙니다.
- 모든 재사용 조건에서 state는 최신입니다. 외부 cache는 vision encoder 출력만 재사용하며 VLM은 전체 재계산합니다.
- 단일 카메라 인코딩과 두 카메라 batch의 BF16 연산 차이 때문에 action 최대 차이 0.00549가 관찰됐습니다. 캐시 hit 자체의 반복 차이는 0이며, 전체 fresh 계산과 비트 단위 동등하다고 주장하지 않습니다.
- 최초 Small 실행 원자료의 `mean_camera_age_steps`는 사용하지 않은 과거 후보 snapshot의 age를 기록한 legacy 필드입니다. Small은 실제로 두 카메라 모두 최신 입력을 사용했습니다. 후속 실행에는 카메라별 실제 age가 별도로 기록됩니다.
- 최초 Small config의 `flow_steps=4`는 복사한 Heavy template의 잔여 필드입니다. 실제 N=10은 `closed_cells`, 각 trial의 N, timing과 manifest에 기록돼 있습니다. 후속 Small 설정은 이 필드도 실제 N에 맞춥니다.

## Offline: 같은 앞 4개 action 비교

'''
    text+=table([r for r in offline if r['run'] in ('run','small','small_n2','small_n4') and
                 (r['mode']=='both_fresh' or r['lag']==8)],
                ['model','mode','H','N','lag','samples','expert_nmse','p50_ms'])
    text+='\n\n## 상태와 재현\n\n```json\n'+json.dumps(status,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='원자료·config·manifest·환경 목록을 같은 폴더에 보존했습니다. source_v1/source_v2는 GR00T batch 실행 당시 코드이며 각 manifest hash와 대응합니다. 계정 주간 사용량은 로컬 세션 한도 메타데이터에서 별도로 확인해 대화에 보고합니다.\n'
    (out/'RESULTS.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[r for r in offline if r['H']==16 and ((r['run'] in ('run','small','small_n2','small_n4') and
            (r['mode']=='both_fresh' or (r['mode']=='external_stale_cached' and r['lag']==8))) or
            (r['run']=='refresh8' and r['mode']=='both_fresh' and r['N']==2))]
    fig,ax=plt.subplots(figsize=(11,5))
    markers=['o','s','^','D','P','v','X','*','h']
    for i,r in enumerate(chosen):
        label=f"{r['model']} N{r['N']}"+(' ext-cache' if 'cached' in r['mode'] else '')
        ax.scatter(r['p50_ms'],r['expert_nmse'],s=70,marker=markers[i%len(markers)],label=label,
                   color='#b45309' if r['model']=='SmolVLA' else '#0369a1')
    ax.set(xlabel='Policy call p50 latency (ms)',ylabel='Expert action NMSE (lower is better)',
           title='RTX 3090: H16 / first 4 actions, 12 recorded observations')
    ax.margins(x=.1,y=.15);ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=8)
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'latency_accuracy.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(output=str(out),closed_conditions=len(live),statuses=status),ensure_ascii=False))


if __name__=='__main__':main()
