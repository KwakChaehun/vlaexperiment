"""Export fixed predictor/corrector experiments with honest paired accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]


def paired(rr,baseline='fresh4'):
    reference={r['episode']:r for r in rr if r['method']==baseline};result=[]
    for method in sorted({r['method'] for r in rr}):
        group=[r for r in rr if r['method']==method]
        common=[r for r in group if r['success'] and reference[r['episode']]['success']]
        result.append(dict(method=method,baseline=baseline,
            wins=sum(r['success'] and not reference[r['episode']]['success'] for r in group),
            losses=sum(not r['success'] and reference[r['episode']]['success'] for r in group),
            joint_successes=len(common),
            joint_mean_steps_delta=float(np.mean([r['steps']-reference[r['episode']]['steps'] for r in common])) if common else None,
            joint_mean_service_ms_delta=float(np.mean([r['total_service_ms']-reference[r['episode']]['total_service_ms'] for r in common])) if common else None))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='reports/direct_features_20261117');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    roots={'dataset':Path('results/direct_features_20261117'),
           'predictor':Path('results/direct_feature_model_20261117'),
           'corrector':Path('results/direct_feature_corrector_20261117'),
           'refresh_first':Path('results/direct_feature_evaluation_20261117'),
           'arrival_first':Path('results/direct_feature_arrival_evaluation_20261117')}
    patterns={
        'dataset':['config.json','manifest.json','parity.json','episodes.jsonl','status.json'],
        'predictor':['training.jsonl','training_summary.json'],
        'corrector':['training.jsonl','training_summary.json'],
        'refresh_first':['offline.jsonl','offline_summary.json','offline_manifest.json','offline_status.json',
                         'closed_loop.jsonl','closed_summary.json','closed_manifest.json','closed_status.json'],
        'arrival_first':['closed_loop.jsonl','closed_summary.json','closed_manifest.json','closed_status.json']}
    validations={}
    for name,root in roots.items():
        dest=out/name;dest.mkdir(exist_ok=True)
        for file in patterns[name]:shutil.copy2(root/file,dest/file)
        manifests=[root/f for f in patterns[name] if f.endswith('manifest.json') or f=='training_summary.json']
        verified=0
        for manifest in manifests:
            for file,digest in read(manifest).get('source_hashes',{}).items():
                source=root/'source'/file
                if not source.exists():source=Path(file)
                assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,(name,file)
                target=dest/'source'/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);verified+=1
        validations[name]=dict(source_hashes_verified=verified)
    cfg=read(roots['dataset']/'config.json');eps=rows(roots['dataset']/'episodes.jsonl')
    assert len(eps)==36 and not any(r['already_successful_at_start'] for r in eps)
    splits=cfg['splits'];allids=sum(splits.values(),[]);assert len(allids)==len(set(allids))
    validations['split']={s:dict(episodes=sum(r['split']==s for r in eps),pairs=sum(r['pairs'] for r in eps if r['split']==s),initial_states=ids)
                          for s,ids in splits.items()}
    for name in ['refresh_first','arrival_first']:
        rr=rows(roots[name]/'closed_loop.jsonl');assert len(rr)==72
        assert len({(r['episode'],r['method']) for r in rr})==72
        assert all(r['paired_images_equal'] and r['paired_sim_max_abs']<1e-7 and len(r['action_trace'])==r['steps'] for r in rr)
        assert all(abs(r['total_service_ms']-sum(c['charged_ms'] for c in r['calls']))<1e-5 for r in rr)
        if name=='arrival_first':
            forecast=[r for r in rr if r['method'] in ['stale','direct','corrected']]
            assert all(r['calls'][0]['path']=='arrival' and r['initial_plan_reproduction_max_abs']<=1e-6 for r in forecast)
        validations[name].update(completed=72,paired_state_RGB=True,service_accounting=True)
    first=rows(roots['refresh_first']/'closed_loop.jsonl');arrival=rows(roots['arrival_first']/'closed_loop.jsonl')
    baseline_matches=[]
    for r in first:
        if not r['method'].startswith('fresh'):continue
        other=next(x for x in arrival if (x['episode'],x['method'])==(r['episode'],r['method']))
        assert r['action_trace']==other['action_trace'] and r['success']==other['success'] and r['stable_success']==other['stable_success']
        baseline_matches.append((r['episode'],r['method']))
    validations['fresh_repeated_baselines']=dict(exact_trace_and_success_matches=len(baseline_matches),independent_extra_conditions=0)
    comparisons={name:{b:paired(rows(root/'closed_loop.jsonl'),b) for b in ['fresh4','stale','direct']}
                 for name,root in roots.items() if name in ['refresh_first','arrival_first']}
    off=read(roots['refresh_first']/'offline_summary.json')
    summary=dict(dataset=validations['split'],predictor=read(roots['predictor']/'training_summary.json'),
        corrector=read(roots['corrector']/'training_summary.json'),offline=off,
        refresh_first=read(roots['refresh_first']/'closed_summary.json'),
        arrival_first=read(roots['arrival_first']/'closed_summary.json'),paired=comparisons)
    (out/'aggregate.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps(validations,indent=2)+'\n')
    report='# 미래 VLM feature 직접 예측: 첫 검증\n\n2026-11-17 KST. 기존 GR00T N1.6 / RTX 3090 / H50/K16. **단순 재사용보다 feature/action 오차는 줄었지만, 첫 정밀 action부터 적용한 성공률은 Fresh4 10/12 대비 직접 예측5/12, 보정7/12로 낮았습니다. 가벼운 예측의 실행 가능성과 접촉 품질 달성은 구분해야 합니다.**\n\n'
    report+='36개 teacher episode를 초기 상태별 학습18/검증6/평가12로 분리했습니다. 학습270쌍, 검증96쌍, 평가165쌍입니다. 평가 초기 상태는3/7/9/11이며 작업3종은 학습 때와 같습니다. GR00T 학습 데이터나 이전 실험에서 미관측이었다는 뜻은 아닙니다. Predictor/보정기/PCA/모델 선택에서 평가 episode를 읽지 않았습니다.\n\n'
    report+='## 1. 무엇을 바꿨나\n\n미래 simulator와 GR00T를 실행 중 반복하는 Whole-bank를 제거했습니다. 과거·현재 관측 feature/state와 이미 정해진 action prefix로 **다음 한 시점**을 예측합니다. 약49k 파라미터 predictor와 rank32 고정 basis를 사용했습니다. 고정 basis는 학습 residual의 image-token 에너지 약39.8%를 설명합니다. 도착 보정기는 두 카메라의16×16 RGB 변화와 실제state를 쓰는 약148k MLP입니다. Sparse attention이나 변화 patch만 재계산한 구현은 아닙니다.\n\n'
    report+='모든 예측 계열 action expert는 새 noise에서 N4 전체 계산을 유지합니다. 기존3+1 flow continuation을 섞지 않았습니다. Fresh1/2도 같은 크기의 모델이며 flow 횟수만 다릅니다. Same-feature bypass는 native action과 최대 차이0, backbone 호출0으로 검증했습니다.\n\n'
    report+='## 2. 평가용 미래 관측 쌍에서의 결과\n\nFeature 열은 단순 재사용의 오차 에너지를1로 정규화한 값입니다. Action RMSE는 Fresh4와의 일치도이며 작업 정확도가 아닙니다. 지연은 중앙값(ms), 현재 feature가 이미 있는 상태의 추가 예측 준비와 도착 처리입니다.\n\n|예측 시간|방법|feature 오차 비율|action RMSE|예측 준비|도착 처리|\n|---|---|---:|---:|---:|---:|\n'
    for h in [4,8,16]:
        for method in ['stale','direct','corrected','fresh1','fresh2','fresh4']:
            r=next(x for x in off if x['horizon']==h and x['method']==method)
            feat=f"{r['feature_energy_vs_stale']:.3f}" if 'feature_energy_vs_stale' in r else '—'
            report+=f"|{h/20:.1f}s|{method}|{feat}|{r['action_rmse']:.4f}|{r['prepare_ms_p50']:.3f}|{r['arrival_ms_p50']:.2f}|\n"
    report+='\n검증에서 선택한 temporal extrapolation 계수는0으로, 단순 재사용과 동일해졌습니다. Feature predictor는 weighted feature loss만으로 학습했고 action loss를 쓰지 않았습니다. 미래 feature의 평균화와 low-rank 제약이 policy에 미치는 영향은 미해결입니다. Fresh1은 예측 N4보다 낮은 teacher-action 오차를 보였습니다.\n\n'
    report+='## 3. 접촉·배치·서랍 closed-loop\n\n모두 동일한12조건의 paired 평가입니다. 물리가 추론 중 정지하는 **동기 평가**이며, 실제20Hz deadline이나 비동기 완료시간을 측정하지 않았습니다. 각 방식을 phase 종료까지 유지하고 공통 Fresh rescue를 붙이지 않았습니다.\n\n'
    report+='첫 Fresh 방식은 phase의 첫16 action을 Fresh로 만들고 다음 chunk를 예측합니다. 1조건은 첫 chunk에서 끝나 예측을 사용하지 않아, **공통 접근 prefix에서 예측해 첫 정밀 action부터 사용하는 도착 예측 방식**을 추가했습니다. 모델과 하이퍼파라미터는 변경하지 않았습니다. 추가 방식의 첫 예측 origin은 학습 pair의 시작보다16step 빠르므로 이 시점의 분포 차이도 포함합니다. 두 방식 모두 이후2chunk마다 full VLM을 다시 계산합니다.\n\n'
    for title,name in [('도착 첫 action부터 예측 사용 (12조건 모두 적용)','arrival_first'),('첫 chunk는 Fresh, 이후 교대로 예측','refresh_first')]:
        report+='### '+title+'\n\n|방법|성공|0.5초 상당의 성공 유지|실행action당 계산ms|예측 도착 처리p50 ms|\n|---|---:|---:|---:|---:|\n'
        for r in summary[name]:
            delay=f"{r['arrival_only_p50_ms']:.2f}" if r['arrival_only_p50_ms'] is not None else '—'
            report+=f"|{r['method']}|{r['successes']}/{r['trials']}|{r['stable_successes']}/{r['trials']}|{r['charged_ms_per_executed_action']:.3f}|{delay}|\n"
        report+='\n준비 비용은 전부 더하고, 쓰이지 않고 종료된 예측도 포함합니다. Action당 계산 감소가 더 적은 action으로 작업을 완료한다는 뜻은 아닙니다. 성공률이 다른 방식의 전체 평균 완료시간만으로 우열을 판단하지 않습니다.\n\nFresh4와 **둘 다 성공한 동일 조건**에서는:\n\n|방법|공통 성공 조건|평균action 수 차이|평균계산 시간 차이 ms|Fresh4 대비 승/패|\n|---|---:|---:|---:|---:|\n'
        for r in comparisons[name]['fresh4']:
            if r['method']=='fresh4':continue
            report+=f"|{r['method']}|{r['joint_successes']}|{r['joint_mean_steps_delta']:+.1f}|{r['joint_mean_service_ms_delta']:+.2f}|{r['wins']}/{r['losses']}|\n"
        report+='\n'
    report+='Fresh1/2/4의36 rollout은 두 방식 사이에 **전체 action trace와 성공 판정이 일치**했습니다. 이는 같은12조건의 반복이며 독립 조건을 늘린 것이 아닙니다. Closed rollout은144회, 고유한 평가 초기 조건은12개입니다.\n\n'
    report+='## 4. 작업별 결과 (도착 첫 action부터 예측)\n\n|작업|방법|성공/4|성공 유지/4|\n|---|---|---:|---:|\n'
    for task in sorted({r['task'] for r in arrival}):
        for method in ['fresh4','fresh2','fresh1','stale','direct','corrected']:
            rr=[r for r in arrival if r['task']==task and r['method']==method]
            report+=f"|{task}|{method}|{sum(r['success'] for r in rr)}|{sum(r['stable_success'] for r in rr)}|\n"
    report+='\n## 5. 해석과 한계\n\n작은 action-conditioned feature predictor의 구현 가능성은 확인했지만, 일반적인 정밀 삽입 성능이나 PIM 전체의 우위를 보여준 결과는 아닙니다. 낮은 feature MSE만으로 grasp/contact에 중요한 오류를 배제할 수 없습니다. 2chunk마다 full VLM을 갱신하는 비용, 느린 복구 동작, 실패까지 포함해 판단해야 합니다. 약0.33ms의 한 시점 예측을 수초 뒤까지 전체 bank를 만드는 기존 구현과 같은 작업량의 speedup으로 비교하지 않습니다.\n\n'
    report+='다음 단계는 실제 제어 성공을 떨어뜨리는 feature 오류를 분석하고, frozen action expert의 출력 일치를 사용하는 학습을 검증하는 것입니다. 그 다음 실제20Hz 비동기 실행을 평가해야 합니다. 이번 결과만으로 full-VLM 갱신 빈도를 줄이거나 PIM에서의 성공률·속도를 주장하지 않습니다.\n\n'
    report+='## 원자료·재현\n\n[설계·제약](../../docs/direct_feature_experiment.md), [기계 판독 집계](aggregate.json), [검증](validation.json). 각 하위 폴더에 raw rows, 학습 곡선, split/episode hash, 소스 snapshot이 있습니다. 모델 checkpoint와 수집 영상/feature는 로컬 results에 보존하고 Git에서 제외합니다.\n\n```bash\n.venv/bin/python scripts/collect_direct_features.py --config configs/direct_feature_probe.json --output results/direct_features_new\n.venv/bin/python scripts/train_direct_features.py --data results/direct_features_new --output results/direct_model_new\n.venv/bin/python scripts/train_feature_corrector.py --data results/direct_features_new --predictor results/direct_model_new/predictor.pt --output results/direct_corrector_new\n.venv/bin/python scripts/evaluate_direct_features.py --data results/direct_features_new --predictor results/direct_model_new/predictor.pt --corrector results/direct_corrector_new/corrector.pt --output results/direct_eval_new --phase offline\n.venv/bin/python scripts/evaluate_direct_features.py --data results/direct_features_new --predictor results/direct_model_new/predictor.pt --corrector results/direct_corrector_new/corrector.pt --output results/direct_eval_new --phase closed\n.venv/bin/python scripts/evaluate_direct_features.py --data results/direct_features_new --predictor results/direct_model_new/predictor.pt --corrector results/direct_corrector_new/corrector.pt --output results/direct_arrival_new --phase closed --arrival-start\n```\n'
    report+='\n![Feature error and contact success](comparison.png)\n'
    (out/'RESULTS.md').write_text(report)
    print(json.dumps(dict(arrival_first=summary['arrival_first'],paired=summary['paired']['arrival_first']['fresh4']),indent=2))


if __name__=='__main__':main()
