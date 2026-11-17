"""Publish a reproducible, explicitly limited forecast-feature diagnostic report."""
import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();src=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    def read(name):return json.loads((src/name).read_text())
    rows=[json.loads(s) for s in (src/'rollouts.jsonl').read_text().splitlines()]
    calls=[json.loads(s) for s in (src/'calls.jsonl').read_text().splitlines()]
    config=read('config.json');manifest=read('manifest.json');status=read('status.json')
    expected=len(config['tasks'])*len(config['initial_states'])*len(config['gains'])*len(config['methods'])
    if status['status']!='completed' or len(rows)!=expected or any(r['status']!='completed' for r in rows):
        raise ValueError('Incomplete evaluation cannot be published as completed')
    verified={}
    for name,digest in manifest['source_hashes'].items():
        actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual!=digest:raise ValueError(f'Source changed since run: {name}')
        verified[name]=actual
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    for name in ['config.json','manifest.json','parity.json','status.json','summary.json','rollouts.jsonl','calls.jsonl']:
        shutil.copy2(src/name,out/name)
    for name in ['residuals.jsonl','residual_summary.json']:
        if (src/name).exists():shutil.copy2(src/name,out/name)
    if (src/'replays/verification.json').exists():
        shutil.copy2(src/'replays/verification.json',out/'replay_verification.json')
    pairs=src/'observation_pairs'
    if pairs.exists():
        (out/'observation_pair_hashes.json').write_text(json.dumps({str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pairs.glob('*.npz'))},indent=2))
    for name in ['scripts/measure_prediction_residuals.py','scripts/render_speculative_replays.py',
                 'scripts/summarize_speculative_features.py','tests/test_speculative_features.py']:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    (out/'source_verification.json').write_text(json.dumps(verified,indent=2))
    cells=defaultdict(list)
    for r in rows:cells[(r['gain'],r['method'])].append(r)
    summary=[]
    for (gain,method),rr in sorted(cells.items()):
        cc=[c for c in calls if c['gain']==gain and c['method']==method]
        initial=[r['plans'][0]['wall_ms'] for r in rr if r['plans']]
        summary.append(dict(gain=gain,method=method,success=sum(r['success'] for r in rr),n=len(rr),
            end_success=sum(r['goal_at_end'] for r in rr),
            post_median_ms=statistics.median(c['post_ms'] for c in cc),
            median_per_trial_post_ms=statistics.median(statistics.median(c['post_ms'] for c in cc if c['trial']==r['trial']) for r in rr),
            mean_initial_plan_s=statistics.mean(initial)/1000 if initial else 0,
            mean_all_plans_s=statistics.mean(r['plan_wall_ms'] for r in rr)/1000,
            mean_sequential_service_s=statistics.mean(r['plan_wall_ms']+r['post_sum_ms'] for r in rr)/1000,
            max_bank_MiB=max(r['max_cache_MiB'] for r in rr),
            mean_forecast_position_mm=statistics.mean(c['prediction_position_mm'] for c in cc if 'prediction_position_mm' in c)
                if any('prediction_position_mm' in c for c in cc) else None))
    (out/'aggregate.json').write_text(json.dumps(summary,indent=2))
    paired=[]
    for gain in config['gains']:
        for reference in ['fresh4','fresh1']:
            baseline={r['case']:r for r in cells[(gain,reference)]}
            if not baseline:continue
            for method in config['methods']:
                if method==reference:continue
                aa=cells[(gain,method)]
                paired.append(dict(gain=gain,method=method,reference=reference,
                    wins=sum(r['success'] and not baseline[r['case']]['success'] for r in aa),
                    losses=sum(not r['success'] and baseline[r['case']]['success'] for r in aa),n=len(aa)))
    (out/'paired.json').write_text(json.dumps(paired,indent=2))
    def table(headers,data):
        return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</table>'
    page='''<!doctype html><meta charset="utf-8"><title>미래 feature 부분 보정 — RTX 3090</title>
<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#202830}table{border-collapse:collapse;margin:20px 0;font-size:14px}td,th{border:1px solid #ccd3da;padding:8px;text-align:left}th{background:#eaf0f5}.note{background:#fff3d6;padding:16px}code{background:#eef2f5}</style>
<h1>예측한 미래 feature와 실제 관측의 차이만 갱신</h1>
<p>GR00T N1.6 LIBERO / RTX 3090 / H=50, K=4. 기존 checkpoint를 고정하고 예측 길이와 MLP 갱신 비율을 비교했습니다.</p>
<p class="note"><strong>범위:</strong> 현재 simulator 상태를 아는 별도 nominal simulator가 미래를 예측합니다. 실제 미래 관측을 읽지는 않지만, 학습된 영상 예측기보다 유리한 조건입니다. 실제 실행에만 이동 명령 0.98배를 적용합니다. Attention은 전체 재계산, MLP만 부분 갱신합니다. 실제 비동기 제어·PIM 평가가 아닙니다.</p>
<h2>방법</h2>
<p>예측된 각 미래 프레임의 영상/VLM MLP 입력·출력 및 액션 생성 3단계 latent를 보관합니다. 도착 후 실제 입력과 예측 입력의 차이가 큰 토큰을 각 층에서 골라 MLP를 다시 계산하고, 실제 state를 넣어 마지막 액션 생성 1단계를 수행합니다. 25%/50%는 MLP 갱신 예산이며 관측된 오차의 희소성을 미리 증명한 수치가 아닙니다.</p>
<ul><li>early16: 초반 16 control steps(0.8초)만 예측 후 나머지는 fresh4.</li><li>rolling16: 현재 상태에서 다음 16 steps를 반복해서 예측.</li><li>whole: 남은 평가 구간 128/192 steps(6.4/9.6초)를 한 번에 예측. 이후 미래 bank는 고정.</li><li>dense: feature 전체 재계산 후 예측 latent의 마지막 1단계. replay: 보정 없이 예측 action 실행.</li></ul>
<h2>성공률과 측정 시간</h2>
<p>동일 초기 상태의 짝지은 소규모 평가입니다. 과제별 초기 상태 0·4를 모두 포함하며 성공 사례만 고르지 않았습니다. 첫 성공 chunk 또는 고정 종료 시점까지 실행합니다. 성공률 우위의 통계적 확증은 아닙니다.</p>'''
    page+=table(['gain','method','성공','종료 목표 유지','도착 후 p50 ms','첫 준비 평균 s','모든 준비 평균 s','준비+보정 총 평균 s','최대 bank MiB'],[
        [r['gain'],r['method'],f"{r['success']}/{r['n']}",f"{r['end_success']}/{r['n']}",f"{r['post_median_ms']:.2f}",
         f"{r['mean_initial_plan_s']:.2f}",f"{r['mean_all_plans_s']:.2f}",f"{r['mean_sequential_service_s']:.2f}",f"{r['max_bank_MiB']:.1f}"] for r in summary])
    page+='''<p class="note">도착 후 p50만으로 전체 시스템이 빨라졌다고 판단하면 안 됩니다. 준비에는 simulator 초기화·과거 replay·미래 rendering·모델 계산이 포함됩니다. 이 구현은 준비 중 환경을 멈춥니다. 준비+보정 시간은 측정된 순차 서비스 비용이며 물리적 동작 시간과 실제 로봇 통신 비용을 포함하지 않습니다. replay의 도착 후 시간은 배열 복사 시간입니다.</p>
<h2>작업별 결과</h2>'''
    taskgroups=defaultdict(dict)
    for r in rows:taskgroups[(r['task'],r['initial_state'],r['gain'])][r['method']]=r
    page+=table(['task','init','gain']+config['methods'],[
        list(key)+[('성공' if dd[m]['success'] else '실패')+f" ({dd[m]['steps']} step)" for m in config['methods']] for key,dd in sorted(taskgroups.items())])
    page+='<h2>짝지은 승패</h2>'+table(['gain','method','reference','이김','짐','쌍 수'],[
        [r[k] for k in ['gain','method','reference','wins','losses','n']] for r in paired])
    if (src/'residual_summary.json').exists():
        residual=read('residual_summary.json')
        page+='<h2>예측 오차의 feature 집중도</h2><p>저장한 예측/실제 관측 쌍을 모두 dense로 다시 계산한 별도 진단입니다. 이 진단은 제어기의 토큰 선택에 사용되지 않습니다. 아래는 MLP 입력 residual 제곱 에너지 기준, 프레임·층별 비율의 평균입니다. 완전히 일치하는 예측 시작 프레임은 제외합니다.</p>'
        page+=table(['method','층 종류','프레임','상위 25%가 포함한 오차 에너지','90% 에너지에 필요한 토큰','99% 에너지에 필요한 토큰'],[
            [r['method'],r['group'],r['frames'],f"{r['top25_energy_fraction']*100:.1f}%",f"{r['fraction_for_90pct_energy']*100:.1f}%",f"{r['fraction_for_99pct_energy']*100:.1f}%"] for r in residual])
        page+='<p>오차가 일부 토큰에 집중되는 현상은 관찰되지만, 나머지 토큰의 오차가 0이거나 접촉 성공에 무관하다는 뜻은 아닙니다. <a href="residuals.jsonl">층별 집중도 및 fresh4 대비 action 오차</a>. 원본 관측 NPZ는 로컬 results의 observation_pairs에 보관하며 이 보고서에는 hash만 포함합니다.</p>'
    page+='''<h2>해석상의 제한</h2><ul>
<li>긴 예측은 실제 보정 action과의 불일치까지 누적됩니다. 정확한 simulator를 사용해도 이후 계획이 자동으로 맞춰지지는 않습니다.</li>
<li>기존 MLP 캐시만 부분 보정합니다. Sparse attention, 학습된 residual predictor, 보정 학습, 적응형 예측 길이는 구현하지 않았습니다.</li>
<li>성공 판정은 LIBERO의 기본 접촉·배치·서랍 조건입니다. 정밀 peg insertion이나 삽입 공차를 검증하지 않습니다.</li>
<li>짧은 계획의 반복 준비 비용에는 과거 동작 replay가 누적됩니다. 이는 정확한 복제 검증을 위한 구현 비용으로, 실제 배포 예측기의 비용으로 일반화할 수 없습니다.</li>
<li>실제 추론 지연 때문에 움직이는 물체를 놓치는 효과는 이 동기식 simulator 평가에 포함되지 않습니다.</li></ul>
<h2>검증·원자료</h2><p>native/split 및 전체 갱신 action 일치, 실제/복제 시작 state·RGB 일치, 예측 캐시 불변성, 실제 MLP 입력 행 감소를 검증했습니다.</p>
<p><a href="aggregate.json">집계</a> · <a href="paired.json">짝 비교</a> · <a href="rollouts.jsonl">rollout 원자료</a> · <a href="calls.jsonl">호출별 시간·예측 오차</a> · <a href="config.json">설정</a> · <a href="manifest.json">구현·범위</a> · <a href="source_verification.json">소스 hash 검증</a></p>'''
    page=page.replace('실제 실행에만 이동 명령 0.98배를 적용합니다.',
        '실제 실행의 이동 명령 배율: '+', '.join(map(str,config['gains']))+'. '+html.escape(config['perturbation']))
    page=page.replace('과제별 초기 상태 0·4를 모두 포함하며',
        '과제별 초기 상태 '+', '.join(map(str,config['initial_states']))+'를 포함하며')
    (out/'RESULTS.html').write_text(page)
    print(json.dumps(dict(completed=len(rows),summary=summary),ensure_ascii=False))


if __name__=='__main__':main()
