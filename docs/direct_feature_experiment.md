# 도착 시점 VLM feature 직접 예측: 파일럿 설계

2026-11-17. Whole-bank 준비를 최적화하는 대신, 현재 관측의 post-vlln backbone feature와 state, 과거 feature/state, 이미 정해진 action prefix로 도착 시점의 GR00T conditioning feature를 직접 예측합니다. Predictor 입력에 미래 이미지·실제 미래 state·simulator full state를 넣지 않습니다. Simulator와 frozen GR00T는 offline 학습 정답 생성 및 평가에만 사용합니다.

## 고정한 비교와 데이터 분리

- 모델: 기존 GR00T N1.6 LIBERO checkpoint, RTX 3090.
- 기본 action expert: H50/N4/K16, 새 noise에서 시작. 기존 3+1 flow continuation을 섞지 않습니다.
- 예측 시간: 4/8/16 control steps = 0.2/0.4/0.8초 @20Hz. 이번 offline/closed 평가는 동기식이며 실제 20Hz deadline 평가가 아닙니다.
- 3개 기존 접촉·수납 작업, 기존에 정한 phase 시작/종료 시점.
- 학습 초기 상태: 0/2/4/6/8/10, 검증: 1/5, 평가: 3/7/9/11. Predictor 관점의 episode 분리이며 GR00T 학습 데이터나 이전 H/K 실험에서 미관측이었다는 뜻은 아닙니다. 평가 작업 종류도 학습 작업과 같습니다.
- 예측 당시 이미 정해진 action 중 예측 시간 안의 prefix만 입력하고 나머지는 0과 validity mask로 가립니다. 학습 target을 만들기 위한 실제 applied action/gain 또는 미래 재계획 action은 입력하지 않습니다.
- 실제 실행 translation gain .98; 나머지는 기존 nominal controller. 이 gain은 predictor 입력에 제공하지 않습니다.

비교는 Fresh4/Fresh2/Fresh1, stale feature, validation으로 선택한 temporal extrapolation, learned direct forecast, forecast + arrival correction입니다. Fresh few-step은 같은 크기의 GR00T이며 작은 별도 모델은 아닙니다. 실제 미래 feature는 학습 target 및 offline teacher-action 진단에만 쓰며 배포용 controller로 간주하지 않습니다.

## 구현

학습 데이터는 전체 36개 episode이며 성공과 실패 모두 보존합니다. Fresh H50/N4/K16으로 phase까지 실행하고 각 chunk의 origin과 +4/+8/+16에서 backbone feature를 기록합니다. 도착 feature가 action expert에 들어가는 인터페이스를 그대로 사용합니다. Same-feature bypass는 기존 GR00T action과 최대 차이0, backbone 호출0으로 검증했습니다. 도착 시 native processor와 state encoder는 실행하므로 그 비용은 측정에 포함합니다.

학습용 residual만으로 rank32 basis를 만들고, token별 작은 MLP로 그 basis의 residual 계수를 예측합니다. 약49k trainable parameters이며, basis는 별도 고정 buffer입니다. 현재/과거 feature projection, pooled feature, token 위치, image mask, state 및 예정 action을 사용합니다. Feature 학습 loss는 image token 1, text token .2 가중치입니다. 예측 모델은 action loss로 학습하지 않았습니다. 검증 split으로 checkpoint와 residual blend를 선택합니다. Test 파일은 fit/PCA/선택에서 열지 않습니다.

도착 보정기는 두 카메라 각각의 16×16 RGB 변화와 실제 state 변화, forecast context를 입력으로 하는 별도 약148k residual MLP입니다. 고정 rank32 basis를 공유합니다. 실제 도착 영상은 이 시점에만 사용하며 vision encoder/VLM은 실행하지 않습니다. 학습·검증 split으로만 보정기와 blend를 선택합니다. 이것은 **저랭크 feature 보정**이며 sparse attention 또는 변화한 patch만 재계산한 구현은 아닙니다.

## Offline와 closed-loop의 구분

Offline에서는 12개 평가 episode의 공통 관측 쌍에서 feature 잔차와 Fresh4 action 일치도를 비교합니다. 동일 실제 도착 state와 noise seed를 모든 action expert에 사용합니다. Feature MSE나 action agreement는 접촉 성공률이 아닙니다. 샘플은 episode 안에서 상관되므로 수백 쌍을 수백 개 독립 작업으로 세지 않습니다.

Closed-loop에서는 12개 평가 조건을 각 방법으로 다시 실행합니다. Phase 이전 action을 동일하게 재생해 simulator state와 두 카메라의 정확한 일치를 검사합니다. Fresh는 매16 action마다 전체 관측을 계산합니다. Forecast 계열은 다음처럼 두 chunk마다 갱신합니다.

1. 실제 현재 feature를 전체 계산하고 첫16 action을 생성합니다.
2. 그16 action 뒤의 feature 하나를 작은 모델로 예측합니다. 준비 비용을 별도로 기록하고 총비용에 더합니다.
3. 16 action 뒤 실제 state를 받아 예측/보정 feature로 새 N4 action을 생성합니다. 미래 simulator를 실행하지 않습니다.
4. 다음16 action 뒤에는 실제 전체 feature로 다시 갱신합니다.

처음의 과거 feature는 공통 prefix에서 실제 관측한 것입니다. 이후 predictor의 과거 입력은 이전 arrival에서 예측하거나 보정한 feature이므로 학습 때와 다른 오차가 누적될 수 있습니다. 이 분포 차이를 평가에 남겼습니다. 예측 feature를 계속 재귀적으로 사용해 full VLM을 영구히 없애는 정책은 아닙니다.

Phase 종료까지 같은 방법으로 실행하고 공통 Fresh rescue를 붙이지 않습니다. Native 성공, chunk 끝의 성공, 추가10 zero-motion/same-gripper step 동안의 성공 유지를 구분합니다. 준비·도착 처리 비용, 전체 호출당/실행 action당 비용을 보고합니다. 물리는 추론 중 정지하므로 이 비용을 실제 비동기 완료 시간 또는 deadline 개선으로 주장하지 않습니다.

## 재현 파일

- 설정: `configs/direct_feature_probe.json`
- 수집: `scripts/collect_direct_features.py`
- 예측 학습: `scripts/train_direct_features.py`
- 보정 학습: `scripts/train_feature_corrector.py`
- 평가: `scripts/evaluate_direct_features.py`
- 구현: `vla_eval/direct_feature_prediction.py`

데이터와 작은 모델 checkpoint는 로컬 `results/direct_features_20261117`, `results/direct_feature_model_20261117`, `results/direct_feature_corrector_20261117`에 저장합니다. 평가 출력은 `results/direct_feature_evaluation_20261117`입니다. 원본 GR00T 가중치·수집 영상/feature 데이터는 Git에 넣지 않습니다. 학습·평가 source hash, episode hash와 선택된 설정은 보고서에 보존합니다.

## 첫 결과 뒤 보완한 접근→도착 검증

첫 모델은 정밀 phase 내부 pair만 학습했습니다. 첫 Fresh chunk를 거친 rolling 평가에서 native 성공은 Fresh4 10/12, stale8/12, direct9/12, corrected10/12였지만, 실제 첫 정밀 action부터 예측하는 평가에서는 Fresh4 10/12, stale5/12, direct5/12, corrected7/12였습니다. 이 결과를 보고 학습 범위의 빠진 전환을 보완했습니다. 첫 평가 결과는 삭제하지 않고 별도 보고서에 보존합니다.

`configs/direct_feature_arrival_probe.json`은 `collect_lead_steps=16`을 추가합니다. 기존 공통 접근 prefix의 마지막16step에서도 +4/+8/+16 pair를 수집합니다. Precision phase 전 gain은1, 이후는 기존 .98로 유지해 기존 학습·검증 trajectory를 바꾸지 않았습니다. 실제로 24개 train/validation episode의 전체 action trace와 공유 feature가 모두 정확히 같고 각3쌍만 추가됐음을 검증했습니다.

학습0/2/4/6/8/10, 검증1/5는 유지하고 **새 평가 상태12/13/14/15**를 사용합니다. 앞선 평가 상태3/7/9/11로 새 모델을 선택하지 않습니다. 모델 구조·rank32·epoch120 예산은 그대로이며, 새 validation에서 checkpoint/blend를 선택합니다. 이 재검증은 첫 테스트 결과를 보고 설계를 보완한 탐색적 후속임을 명시합니다.

평가는 학습과 같은 의미의 접근 origin에서 미래 feature를 생성하고, 공통 마지막16 접근 action을 실제 실행한 뒤 첫 정밀 action부터 사용합니다. 예정 접근 action은 origin에서 기존 GR00T로 다시 생성하여 기록과 최대 차이1e-6 이하를 확인합니다. 과거 feature cache 재구성/plan 재현은 공통 prefix를 복원하는 진단이고, predictor의 실제 추가 준비 시간은 별도로 측정해 phase 비용에 모두 더합니다.

추가 offline `oracle_rank32`는 실제 미래 feature 변화의 고정 basis 투영으로, 학습 오차와 표현 제약을 분리하는 평가 전용 대조군입니다. 미래 정답을 쓰며 controller에는 절대 넣지 않습니다. 낮은 feature MSE가 최적 action을 보장하지 않아 이 대조군을 작업 성공률의 수학적 상한이라고 해석하지 않습니다.
