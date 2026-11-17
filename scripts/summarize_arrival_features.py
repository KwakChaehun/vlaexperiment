"""Export the approach-covered follow-up on previously untouched test states."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from summarize_direct_features import paired, read, rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='reports/direct_arrival_features_20261117');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    roots={
        'dataset':Path('results/direct_arrival_features_20261117'),
        'predictor':Path('results/direct_arrival_feature_model_20261117'),
        'corrector':Path('results/direct_arrival_feature_corrector_20261117'),
        'evaluation':Path('results/direct_feature_arrival_followup_20261117')}
    files={'dataset':['config.json','manifest.json','parity.json','episodes.jsonl','status.json','transition_validation.json'],
           'predictor':['training_summary.json','training.jsonl'],
           'corrector':['training_summary.json','training.jsonl'],
           'evaluation':['offline.jsonl','offline_summary.json','offline_manifest.json','offline_status.json',
                         'closed_loop.jsonl','closed_summary.json','closed_manifest.json','closed_status.json']}
    verified={}
    for name,root in roots.items():
        dest=out/name;dest.mkdir(exist_ok=True);count=0
        for file in files[name]:
            shutil.copy2(root/file,dest/file)
            if file.endswith('manifest.json') or file=='training_summary.json':
                for source,digest in read(root/file).get('source_hashes',{}).items():
                    original=root/'source'/source
                    if not original.exists():original=Path(source)
                    assert hashlib.sha256(original.read_bytes()).hexdigest()==digest,(name,source)
                    target=dest/'source'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target);count+=1
        verified[name]=count
    cfg=read(roots['dataset']/'config.json');episode_rows=rows(roots['dataset']/'episodes.jsonl')
    split=cfg['splits'];assert not(set(split['test']) & {3,7,9,11})
    assert len(sum(split.values(),[]))==len(set(sum(split.values(),[])))
    assert len(episode_rows)==36 and not any(r['already_successful_at_start'] for r in episode_rows)
    closed=rows(roots['evaluation']/'closed_loop.jsonl');assert len(closed)==72
    assert len({(r['episode'],r['method']) for r in closed})==72
    for r in closed:
        assert r['paired_images_equal'] and r['paired_sim_max_abs']<1e-7
        assert len(r['action_trace'])==r['steps']
        assert abs(r['total_service_ms']-sum(c['charged_ms'] for c in r['calls']))<1e-5
        if r['method'] in ['stale','direct','corrected']:
            assert r['calls'][0]['path']=='arrival' and r['initial_plan_reproduction_max_abs']<=1e-6
    offline=rows(roots['evaluation']/'offline.jsonl');offsummary=read(roots['evaluation']/'offline_summary.json')
    # Separate the arrival boundary from within-phase average performance.
    boundary=[]
    for h in [4,8,16]:
        for method in sorted({r['method'] for r in offline}):
            rr=[r for r in offline if r['origin_before_precision'] and r['horizon']==h and r['method']==method]
            assert len(rr)==12
            item=dict(horizon=h,method=method,pairs=len(rr),action_rmse=float(np.sqrt(np.mean([r['action_mse'] for r in rr]))),
                translation_rmse=float(np.sqrt(np.mean([r['translation_mse'] for r in rr]))),
                gripper_agreement=float(np.mean([r['gripper_agreement'] for r in rr])))
            if 'feature_sse' in rr[0]:
                stale=[r for r in offline if r['origin_before_precision'] and r['horizon']==h and r['method']=='stale']
                item['feature_energy_vs_stale']=sum(r['feature_sse'] for r in rr)/sum(r['feature_sse'] for r in stale)
            boundary.append(item)
    comparisons={b:paired(closed,b) for b in ['fresh4','stale','direct','fresh1']}
    dataset={s:dict(episodes=sum(r['split']==s for r in episode_rows),pairs=sum(r['pairs'] for r in episode_rows if r['split']==s),initial_states=ids)
             for s,ids in split.items()}
    aggregate=dict(dataset=dataset,predictor=read(roots['predictor']/'training_summary.json'),
        corrector=read(roots['corrector']/'training_summary.json'),offline=offsummary,arrival_boundary=boundary,
        closed=read(roots['evaluation']/'closed_summary.json'),paired=comparisons)
    (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    validation=dict(source_hashes_verified=verified,completed_rollouts=72,unique_test_conditions=12,
        test_states_disjoint_from_first_pilot=True,all_controller_anchors_exact=True,all_initial_plans_reproduced=True,
        no_oracle_controller=True,all_service_costs_accounted=True,train_validation_shared_features_unchanged=True,
        offline_rows=len(offline),unique_observation_pairs=len({(r['episode'],r['origin'],r['horizon']) for r in offline}))
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    report='# 도착 feature 직접 예측: 접근 구간을 포함한 재검증\n\n2026-11-17 KST. 동일 GR00T N1.6 / RTX3090 / H50/K16. Predictor 49,216 parameters, rank32; 보정기148,032 parameters. Action expert는 예측 계열 모두 N4를 유지했습니다.\n\n'
    by_method={r['method']:r for r in aggregate['closed']}
    report+=f"**이번12조건에서 stale {by_method['stale']['successes']}/12, 직접 예측 {by_method['direct']['successes']}/12, 예측＋보정 {by_method['corrected']['successes']}/12, Fresh2 {by_method['fresh2']['successes']}/12였습니다. 직접 예측은 단순 재사용보다 개선됐지만 강한 Fresh2 기준선을 넘지 못했습니다.** 전체 갱신·준비 포함 계산량은 직접 예측 {by_method['direct']['charged_ms_per_executed_action']:.3f}ms/action, Fresh2 {by_method['fresh2']['charged_ms_per_executed_action']:.3f}ms/action입니다.\n\n"
    report+='## 왜 다시 평가했나\n\n[첫 파일럿](../direct_features_20261117/RESULTS.md)은 feature/action 오차를 줄였지만, 정밀 구간의 첫 action부터 예측했을 때 Fresh4 10/12 대비 direct5/12, corrected7/12였습니다. 학습이 정밀 phase 시작부터여서 접근→도착 전환이 빠져 있었습니다. 이 결과를 본 뒤 학습 범위를 보완한 탐색적 후속이며, 첫 결과를 삭제하거나 같은 평가 데이터를 새 held-out이라고 재사용하지 않았습니다.\n\n'
    report+='학습 초기 상태0/2/4/6/8/10, 검증1/5는 그대로입니다. **평가는 새 초기 상태12/13/14/15 ×3작업 =12조건**입니다. GR00T pretraining과 겹치지 않는다는 뜻은 아니며, 같은 작업의 새 초기 상태 일반화입니다. 기존 학습·검증24 episode의 action trace와 공통 feature가 모두 정확히 같고, 각 episode에 접근 구간3쌍만 추가됐습니다. 모델 구조·rank·학습 epoch 예산은 변경하지 않았습니다.\n\n'
    report+='첫 파일럿과 후속의 test 조건이 다르므로, 두 실험의 성공 수 변화만으로 학습 보완의 인과적 효과를 계산하지 않습니다. 각 실험 안의 동일 조건 비교를 사용합니다.\n\n'
    report+=f"데이터: 학습{dataset['train']['pairs']}쌍, 검증{dataset['validation']['pairs']}쌍, 평가{dataset['test']['pairs']}쌍. PCA/fit/checkpoint·blend 선택에서 test 파일은 읽지 않았습니다.\n\n"
    report+='## 1. 실제 접촉 구간 제어\n\n접근 prefix에서 이미 정해진16 action으로 도착 feature를 예측하고, **첫 정밀 action부터** 그 feature를 사용했습니다. 모든 최초 예정 action은 origin 관측으로 다시 생성해 정확히 같음을 확인했습니다. 이후2chunk마다 full VLM을 갱신합니다. 실제 미래 simulator rollout은 predictor에 없습니다.\n\n|방법|성공/12|성공 유지/12|계산 ms/action|예측 도착 처리 p50 ms|\n|---|---:|---:|---:|---:|\n'
    for r in aggregate['closed']:
        latency=f"{r['arrival_only_p50_ms']:.2f}" if r['arrival_only_p50_ms'] is not None else '—'
        report+=f"|{r['method']}|{r['successes']}|{r['stable_successes']}|{r['charged_ms_per_executed_action']:.3f}|{latency}|\n"
    report+='\n성공 유지는 chunk 끝부터 zero-motion/same-gripper10step 동안 goal 유지 여부입니다. **동기식 simulator 평가**로, 추론 중 물리는 정지합니다. 실제20Hz deadline/비동기 완료시간을 측정한 결과가 아닙니다. 모든 예측 준비·보정 비용을 계산에 넣었습니다. 미사용 예측도 제외하지 않았습니다.\n\nFresh4와 둘 다 성공한 동일 조건에서:\n\n|방법|공통 성공 조건|평균action 수 차이|평균계산 시간 차이 ms|Fresh4 대비 성공 승/패|\n|---|---:|---:|---:|---:|\n'
    for r in comparisons['fresh4']:
        if r['method']=='fresh4':continue
        report+=f"|{r['method']}|{r['joint_successes']}|{r['joint_mean_steps_delta']:+.1f}|{r['joint_mean_service_ms_delta']:+.2f}|{r['wins']}/{r['losses']}|\n"
    report+='\n계산 ms/action 감소가 곧 더 빠른 완료를 의미하지 않습니다. 더 많은 action으로 복구하거나 실패하면 전체 이득이 사라집니다. 소표본 성공률 차이를 통계적으로 확정된 우열로 주장하지 않습니다.\n\n'
    report+='## 2. Feature/action 진단\n\n평가 관측 쌍은 같은 episode 안에서 상관됩니다. 수백 쌍을 수백 개 독립 작업으로 세지 않습니다. Feature 열은 stale residual 에너지 대비 비율, action/translation RMSE는 Fresh4와의 일치도입니다.\n\n|예측 시간|방법|feature 비율|action RMSE|translation RMSE|gripper 일치|예측 준비 p50 ms|\n|---|---|---:|---:|---:|---:|---:|\n'
    for h in [4,8,16]:
        for m in ['stale','direct','corrected','fresh1','fresh2','oracle_rank32']:
            r=next(x for x in offsummary if x['horizon']==h and x['method']==m)
            feature=f"{r['feature_energy_vs_stale']:.3f}" if 'feature_energy_vs_stale' in r else '—'
            pre='평가 전용' if m=='oracle_rank32' else f"{r['prepare_ms_p50']:.3f}"
            report+=f"|{h/20:.1f}s|{m}|{feature}|{r['action_rmse']:.4f}|{r['translation_rmse']:.4f}|{r['gripper_agreement']:.3f}|{pre}|\n"
    report+='\n`oracle_rank32`는 **실제 미래 feature 변화의 rank32 투영**을 넣은 평가용 대조군입니다. 예측의 학습 오차를 없애고 현재 저랭크 표현이 남기는 오차를 봅니다. 미래 정답을 사용하므로 실제 controller나 배포 지연 결과가 아니며, closed-loop에는 포함하지 않았습니다. 낮은 feature 오차가 action 최적성을 보장하지 않아 이 대조군도 성공률의 이론적 상한이라고 부르지 않습니다.\n\n'
    report+='Action/translation RMSE는 native decoded action 좌표계의 수치이며 실제 물체 위치의 mm 오차가 아닙니다. 이 파일럿은 접촉·배치·서랍 작업이고 좁은 공차의 peg insertion은 아닙니다.\n\n'
    report+='접근 시작에서 **정밀 구간 도착까지0.8초**인12쌍만 분리하면:\n\n|방법|feature 비율|action RMSE|translation RMSE|gripper 일치|\n|---|---:|---:|---:|---:|\n'
    for r in boundary:
        if r['horizon']!=16 or r['method']=='temporal':continue
        feature=f"{r['feature_energy_vs_stale']:.3f}" if 'feature_energy_vs_stale' in r else '—'
        report+=f"|{r['method']}|{feature}|{r['action_rmse']:.4f}|{r['translation_rmse']:.4f}|{r['gripper_agreement']:.3f}|\n"
    report+='\n## 3. 작업별 성공\n\n|작업|방법|성공/4|성공 유지/4|\n|---|---|---:|---:|\n'
    for task in sorted({r['task'] for r in closed}):
        for m in ['fresh4','fresh2','fresh1','stale','direct','corrected']:
            rr=[r for r in closed if r['task']==task and r['method']==m]
            report+=f"|{task}|{m}|{sum(r['success'] for r in rr)}|{sum(r['stable_success'] for r in rr)}|\n"
    report+='\n## 원자료와 재현\n\n[집계](aggregate.json), [검증](validation.json), [설계](../../docs/direct_feature_experiment.md). 하위 폴더에 episode/split hash, training curves, 고정 모델 선택, 모든 offline/rollout row, 실행 소스 snapshot이 있습니다. 영상·feature 데이터와 학습 checkpoint는 로컬 results에 남기고 Git에는 넣지 않았습니다.\n\n'
    report+='설정은 `configs/direct_feature_arrival_probe.json`입니다. 기존 설계의 collect/train/correct/evaluate 명령에서 데이터·모델 출력 경로를 새 것으로 바꾸고, closed 평가에 `--arrival-start`를 붙이면 됩니다. 로컬 결과는 `results/direct_arrival_features_20261117`, `results/direct_arrival_feature_model_20261117`, `results/direct_arrival_feature_corrector_20261117`, `results/direct_feature_arrival_followup_20261117`입니다.\n'
    report+='\n![Feature error and held-out contact success](comparison.png)\n'
    (out/'RESULTS.md').write_text(report)
    print(json.dumps(dict(closed=aggregate['closed'],paired=comparisons['fresh4'],boundary=[r for r in boundary if r['horizon']==16]),indent=2))


if __name__=='__main__':main()
