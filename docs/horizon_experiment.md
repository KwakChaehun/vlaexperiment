# 생성 H·실행 K가 정밀도와 PIM 필요성에 주는 영향

## 2026-11-17 RTX 3090 접촉 성공률 후속

사용자가 정밀도 기준으로 **접촉·삽입 작업 성공률**을 선택했습니다. 앞선 Spatial 12조건에서 H50/K16, H50/K4, H16/K4가 모두 성공했으므로 해당 결과만으로 H16/K4의 정확도 우위를 주장할 수 없습니다. 전체 216회 GPU pilot은 [앞선 보고서](../reports/gpu_3090_wrist_20261117/RESULTS.md)에 보존합니다.

추가 설정은 [gr00t16_contact_hk.json](../configs/gr00t16_contact_hk.json), 실행기는 [run_contact_hk.py](../scripts/run_contact_hk.py)입니다. GR00T N1.6/N4를 고정하고 Goal의 bowl placement, drawer containment, rack placement, pushing 4개 작업 및 LIBERO-10의 drawer containment+closing 1개 작업에 각각 초기 상태 12개를 사용합니다. 세 H/K 조합으로 총 180회입니다. Goal은 최대 280, LIBERO-10은 520 control steps를 허용하며 simulator는 추론 중 멈춥니다. 실제 타이트한 peg insertion 환경·정책을 준비한 것은 아닙니다.

별도 접시 이동 스트레스는 Goal bowl-on-plate의 6개 초기 상태 × 0/−20/+20mm X축 이동 × 세 H/K = 54회입니다. 이벤트는 control step 41/45/49/53/57/61에 chunk 중간이어도 적용합니다. 모델에 추가 관측을 강제로 주지 않으며 해당 K의 다음 관측까지 기존 action을 실행합니다. +0mm도 동일한 상태 재생성·대상 속도 초기화를 거칩니다. 합성 위치 점프이며 실제 물리적 push가 아닙니다.

같은 초기 상태·control step에서 noise seed를 공유하고 H=50 noise의 prefix로 H=16을 만듭니다. 각 조건의 실행 순서를 섞습니다. 이는 이전 GPU pilot의 call index 기반 seed 규칙과 다르므로 기존 결과와 반복 표본으로 합치지 않습니다.

주 지표는 공식 task success, 보조 지표는 최초 성공 직후 위치·회전 명령 0 및 직전 gripper 명령 유지 상태에서 10 control steps 동안 성공 유지입니다. 보조 지표는 release 성능이 아닙니다. 작업별 실제 goal predicate도 저장합니다. 일부 이름이 `in`이어도 BDDL 조건이 `On`일 수 있고, 가상 site의 `In`은 접촉 검사 대신 containment만 확인할 수 있으므로 정밀 삽입 성공률로 재명명하지 않습니다.

16/4 대 50/4로 H 효과, 50/4 대 50/16으로 K 효과를 분리합니다. 같은 task/initial state의 승패 쌍과 exact McNemar 검정을 기록하고, 16/4의 두 주 비교에는 Holm 보정을 적용합니다. Bootstrap은 고정된 작업 내부에서 초기 상태를 재표집하므로 새로운 task 일반화 구간은 아닙니다. 중간 결과를 보고 표본 수나 성공 기준을 변경하지 않습니다.

결과: [접촉 평가 보고서](../reports/contact_hk_20261117/RESULTS.html), [실시간 상태](../results/contact_hk_20261117/status.json). 보고서가 실행 중인 스냅샷이면 최종 결과로 취급하지 않습니다. H16/K4의 정확도 이득과 PIM의 지연·전력 이득은 별개 검증 항목입니다.

**완료 결과 (16:09 KST):** 총 234회, 오류·중단 0회. 접촉 5개 작업 × 12초기 상태에서 50/16은 **53/60 (88.3%)**, 50/4는 **50/60 (83.3%)**, 16/4는 **47/60 (78.3%)**였습니다. 호출 p50은 각각 65.51/65.18/64.70ms, action당 추론 비용은 4.34/16.41/16.27ms입니다. 16/4 대 50/16은 6승·12패·42동률, 차이 −10%p, exact McNemar p=0.238 (두 주 비교 Holm 보정 p=0.476)입니다. 관측 순위는 50/16이 높지만 일반적 우위를 확정하지 않으며, 16/4가 더 정확하다는 주장은 지지하지 않습니다.

아래 서랍에 넣고 닫기에서는 16/4가 10/12, 두 H50 조건이 각각 8/12였습니다. 그러나 위 서랍 열고 넣기는 7/12 대 10/12(50/4)·11/12(50/16)로 반대였습니다. 작업별 소표본을 보고 유리한 작업에만 H16을 선택하는 정책이 검증된 것은 아닙니다. 최초 성공 뒤 0.5초 고정 명령 유지 보조 검사도 전체 50/16=43/60, 50/4=42/60, 16/4=41/60으로 H16의 우위를 보이지 않았습니다. 접시 이동 0/−20/+20mm는 각 H/K 각각 6/6으로 천장 효과였습니다.

## 2026-11-17 후속 평가: H={32,16,8,4}, K={16,8,4,2}

실제 로컬 모델은 **SmolVLA-LIBERO, CPU**입니다. 아래 π0.5 선정 내용은 목표 GPU 실험 계획이며, 완료된 SmolVLA pilot과 구분합니다. 11월 16~17일 첫 batch는 H/K 36회와 도착 후 개입 27회를 완료했습니다. 후속 batch는 기존 결과를 복사하지 않고 새로 실행합니다.

| 생성 H \\ 실행 K | 16 | 8 | 4 | 2 |
|---|---|---|---|---|
| 32 | 평가 | 평가 | 평가 | 평가 |
| 16 | 평가 | 평가 | 평가 | 평가 |
| 8 | N/A | 평가 | 평가 | 평가 |
| 4 | N/A | N/A | 평가 | 평가 |

16칸 중 `K <= H`인 **13개 조합**이 유효합니다. H=4/K=16처럼 생성한 수보다 많이 실행하는 경우는 표준 action chunk 평가로 정의할 수 없습니다. Action 반복·hold·추가 chunk 호출로 채우면 제어 규칙이 달라지므로 이번 표에서는 N/A로 둡니다.

- 설정: [configs/smolvla_hk_matrix.json](../configs/smolvla_hk_matrix.json).
- N=10 고정, 3 task × 2 initial states = 설정당 6회, 총 **78회**.
- 실제 noise/action-expert 길이를 H로 변경하고 공통 native-H noise prefix를 사용합니다.
- 추론 중 simulator 정지: 제어 품질을 분리한 zero-inference-delay 실험입니다.
- 성공률과 구간, CPU 호출 p50/p95/p99, 생성 action 사용 비율, command 불연속성과 EEF jerk를 기록합니다.
- 결과: [후속 H/K matrix와 prior 진단](../results/hk_matrix_20261117/RESULTS.html).

```powershell
& .\.runtime\py311\python.exe -X utf8 -u scripts/run_hk_matrix.py --config configs/smolvla_hk_matrix.json --output results/hk_matrix_20261117 --hours 12
```

이미 실행 중이면 추가로 같은 명령을 실행하지 않습니다. PID와 로그 위치는 `results/hk_matrix_20261117/launch.json`에 있습니다. 중단 후 같은 설정으로 다시 실행하면 완료 trial을 이어 사용합니다. 다른 설정은 새 output 디렉터리를 사용합니다.

## Prior + 2-step이 불리했던 이유를 분리하는 실험

첫 pilot의 C-flow와 fresh 2-step은 작업 성공률이 각각 2/3으로 같았습니다. C의 불리함은 주로 expert action 오차와 fresh 동작과의 endpoint 차이에서 관찰됐습니다. 이것만으로 정밀 조작 능력의 열등함을 확정하지 않습니다.

기존 C는 **학습된 correction policy가 아닙니다**. 예상 도착 state와 stale RGB로 완성한 normalized flow latent `a_prior`에 noise를 다시 섞어 `x_0.2 = 0.8*a_prior + 0.2*epsilon`을 만들고, 최신 관측에서 남은 Euler step 2개를 실행한 휴리스틱입니다. Fresh 2-step은 noise에서 전체 flow time 1→0을 두 step으로 적분합니다. 두 방식의 post velocity 평가 횟수는 같지만 초기 분포와 적분 구간은 다릅니다.

현재 실험으로 오래된 이미지, 예상 state의 오차, prior/noise의 상관, 보정 step 부족을 하나의 원인으로 단정할 수 없습니다. 후속 진단은 다음 prior 조건을 교차합니다. 최종 refinement의 관측은 항상 실제 도착 관측입니다.

| Prior 이미지 | Prior state | 의미 |
|---|---|---|
| Stale RGB | Predicted arrival state | 기존 예측 조건 |
| Stale RGB | Actual arrival state | State 예측 오차를 제거한 privileged 진단 |
| Actual arrival RGB | Predicted arrival state | 시각 예측 오차를 제거한 privileged 진단 |
| Actual arrival RGB | Actual arrival state | 같은 관측의 prior도 재보정하면 나빠지는지 확인 |

Stale RGB/state 그대로인 조건도 대조군으로 둡니다. 각 prior에 원래 noise 재사용·독립 noise·실제 중간 latent continuation을 적용하고, 도착 후 평가 횟수 2/4/8에서 fresh 2/4/8과 비교합니다. 같은 관측에서 실제 중간 latent를 이어 계산했을 때 full 결과와 일치하는지도 검증합니다. Held-out 3 episode의 12개 관측을 사용하고 episode 단위 paired 차이를 기록합니다.

Oracle RGB/state를 선행 계산에 쓰는 조건은 **배포 가능한 미래 예측기가 아닙니다**. Image를 바꿨을 때의 개선은 미래 시각 조건 예측의 잠재적 가치이고, pose만 바꿨을 때의 개선은 state 예측의 가치입니다. 같은 실제 관측의 prior도 재보정 후 나빠진다면, 이미지 staleness만으로 모든 차이를 설명할 수 없습니다.

## 예상 좌표에서의 feature를 만드는 방법

지금까지는 미래 proprioception만 예측하고 과거 RGB를 다시 인코딩했습니다. 예상 pose에서 바라본 미래 시각 feature를 생성한 것은 아닙니다. EEF 위치만으로 미래 wrist image나 VLM feature가 유일하게 정해지지 않으므로 camera pose, robot configuration, 물체 geometry와 가림을 함께 고려해야 합니다.

다음 구현 후보는 현재 RGB-D/scene 표현과 예상 camera pose로 future view를 렌더링·워핑한 뒤 인코딩하거나, 현재 feature·state·committed reaching actions로 future feature를 예측하는 방법입니다. Simulator의 정답 scene state나 실제 미래 프레임을 사용한 렌더링은 privileged 진단으로 구분합니다. 먼저 위 oracle 시각 조건에 이득이 있는지 확인한 뒤 predictor 구현·학습 비용을 투입합니다.

성능은 feature cosine similarity만으로 판단하지 않고 action/teacher 오차, 실제 정밀 task 성공, 도착 오차와 step budget에 대한 민감도로 평가합니다. Controller 목표 delta의 합을 achieved pose로 간주하지 않습니다.

## 실험의 목적

**정확도를 유지하거나 높이는 짧은 생성/실행 설정이 존재하는가, 그리고 그 설정의 빈번한 재계획을 GPU보다 PIM이 효율적으로 처리할 수 있는가?**

첫 질문은 VLA 제어 실험, 두 번째 질문은 HW 실험입니다. 짧은 H/K에서 성공률이 높다는 결과는 PIM의 동기를 강화하지만 PIM speedup을 직접 입증하지 않습니다. 정확도가 비슷하면서 지연·에너지가 줄어드는 경우도 유용합니다. 원하는 결론에 맞춰 설정을 고르지 않고 손실이 발생하는 조건까지 보고합니다.

## 1차 시뮬레이터·모델·데이터 선정

2026-11-16 선정: **MuJoCo/robosuite 기반 LIBERO + π0.5-LIBERO**로 첫 H/K 실험을 진행합니다. GR00T 같은 flow-matching 계열에서 먼저 가설을 검증하되, 공식 공개된 task-finetuned checkpoint로 시작해 별도 대형 모델 학습을 피합니다. GR00T는 LIBERO checkpoint의 기준 성공률과 버전 호환성을 확인한 뒤 교차 검증합니다.

| 항목 | 1차 선택 |
|---|---|
| 시뮬레이터/benchmark | LIBERO의 `OffScreenRenderEnv`, MuJoCo/robosuite 기반 |
| 모델 | openpi의 π0.5-LIBERO |
| 체크포인트 | `gs://openpi-assets/checkpoints/pi05_libero/` |
| 데모 데이터 | Hugging Face `physical-intelligence/libero` |
| 첫 task suite | LIBERO-Spatial, 이후 Object/Goal/10(Long)으로 확장 |
| 실제 평가 입력 | 환경이 매번 렌더링하는 외부·손목 카메라 RGB, proprioception, task 언어 |
| 초기 조건 | 공식 task별 initial states를 H/K 설정마다 동일하게 재사용 |

시뮬레이터 control timestep은 선택한 공식 평가 wrapper에서 확인해 고정합니다. 변환된 데모의 `fps` metadata를 그대로 simulator 제어 주기로 가정하지 않으며, H/K를 초 단위 horizon이나 deadline으로 바꿀 때 실제 제어 주기를 사용합니다.

공식 openpi `pi05_libero` 설정의 action horizon은 10입니다. 따라서 이 모델의 첫 K 실험은 **H=10 고정, K∈{1,2,4,8,10}**으로 시작합니다. 실제 H 축을 바꿀 때는 같은 checkpoint에서 길이 변경을 지원하는지 확인하고 H∈{1,2,4,8,10}을 검토합니다. 아래의 H=16 비교표와 CPU 설정 파일은 일반적인 설계 예시이며, π0.5-LIBERO의 native 설정을 의미하지 않습니다. N도 먼저 선택한 공식 sampler의 기본값으로 고정합니다.

데모는 observation/action 형식 확인, 공통 prefix MSE, 이후 Small/보정기 학습에 사용합니다. 모델이 이미 학습한 데모를 held-out 성공률 데이터라고 부르지 않습니다. 추가 학습 시 episode 단위 분리를 지키고, pretrained checkpoint의 training overlap을 포함한 한계를 기록합니다.

**성공률의 주 평가는 live closed-loop rollout**입니다. 같은 초기 상태에서 모델이 만든 action을 실제 환경에 실행하고 새로운 관측으로 재계획합니다. 저장된 expert action을 그대로 재생하는 실험은 모델 성공률 평가가 아닙니다. 일반적인 simulator는 정책 추론 중 자동으로 시간이 흐르지 않으므로, 측정 지연 반영 조건에는 환경 advance와 대기 중 controller 동작을 명시적으로 구현해야 합니다.

정밀 조작 확장은 **ManiSkill 3의 PegInsertionSide-v1**, 이후 PlugCharger-v1을 후보로 둡니다. 공식 demonstration은 제공되지만 LIBERO checkpoint를 그대로 옮겨 정확한 조작이 가능하다고 가정하지 않습니다. 해당 task 정책 학습/적응 후 평가하며, 삽입 깊이·정렬 오차·접촉 실패를 기록합니다. 기본 task 성공 조건만으로 sub-mm precision을 입증하지 않습니다.

현재 실제 로컬 검증은 SmolVLA-LIBERO에서 수행했습니다. π0.5 native offline adapter와 GR00T용 실험적 adapter는 준비했지만, 해당 checkpoint의 GPU 추론과 성공률 재현은 아직 수행하지 않았습니다. CPU fixture나 adapter 작성만으로 이 검증을 완료했다고 해석하지 않습니다. 목표 모델의 실제 실행 환경은 공식 지원 경로를 확인해 정합니다.

근거: [openpi LIBERO 평가 및 checkpoint](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md), [openpi 모델·데이터 설정](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py), [LIBERO initial-state API](https://github.com/Lifelong-Robot-Learning/LIBERO#task), [ManiSkill task 목록](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html).

## 용어와 실제 생성량

- `H`: 한 번의 정책 호출이 생성하는 action timestep 수.
- `K`: 그중 실행하고 다시 관측하는 timestep 수. `1 <= K <= H`.
- `N`: chunk를 갱신하는 flow 적분 step 수. H/K와 별개입니다.
- `D`: 각 action의 제어 차원. D를 줄이는 실험은 아닙니다.

GR00T 계열의 flow head는 `[batch,H,D]` latent를 여러 번 함께 갱신합니다. 여기서 H는 미래 action 위치 수이며, autoregressive vocabulary token 수와 동일한 개념으로 취급하지 않습니다. 실제 action embedding과 GEMM shape도 확인해야 합니다. [NVIDIA N1.6 action head](https://github.com/NVIDIA/Isaac-GR00T/blob/n1d6/gr00t/model/gr00t_n1d6/gr00t_n1d6.py)

H=16으로 추론하고 앞의 4개만 실행하면 H=16/K=4입니다. 결과를 잘라 반환하거나 나머지를 mask했지만 내부가 16개를 계속 계산하는 구현도 H=4의 연산 감소 근거가 아닙니다. H=4라면 실제 noise/action embedding/DiT의 해당 길이가 줄어드는지 기록해야 합니다.

## 최소 비교표

다음은 학습 H=16인 모델을 가정한 예시입니다. Native H가 다르면 기준을 그 길이로 바꿉니다. 모델 checkpoint, N, 이미지/history, control timestep, action 단위와 task는 고정합니다.

| 설정 | 생성 H | 실행 K | 무엇을 확인하는가 |
|---|---:|---:|---|
| 기준 | 16 | 16 | 긴 생성·긴 실행 |
| 피드백 빈도 | 16 | 4 | H를 고정하고 자주 재관측하는 효과 |
| 생성 길이 | 4 | 4 | 같은 실행 길이에서 실제 생성 H를 줄이는 효과 |
| 더 빈번한 피드백 | 4 | 1 | 짧은 H를 유지하며 K를 더 줄이는 효과 |

핵심 대비는 `16/4 vs 16/16`, `4/4 vs 16/4`, `4/1 vs 4/4`입니다. `4/4 vs 16/16`만으로는 H 효과와 K 효과를 구분할 수 없습니다. 전체 grid에서는 H/K를 각각 1,2,4,8,16 등으로 바꾸되 K>H는 제외합니다. 실제 지원하는 길이만 실험하고 나머지는 지원 여부/분포 밖 여부를 명시합니다.

처음에는 N을 하나로 고정합니다. N 감소로 인한 오차와 H 감소 효과가 섞이지 않게 하기 위해서입니다. 이후 별도의 N sweep으로 확장합니다. 설정 파일의 N=8은 fixture 예시이며, 실제 모델에서는 먼저 공식 기본 N을 사용합니다.

## 두 번의 closed-loop 평가

1. **추론 지연을 제거한 평가:** 동일 관측 갱신/제어 규칙에서 H/K 자체가 task 성공률과 정밀도에 주는 영향을 봅니다. 환경의 제어 timestep은 유지합니다. `latency_mode=zero`가 이 조건입니다.
2. **측정 지연을 반영한 평가:** 같은 H/K grid에서 추론 대기 중 물체 이동과 controller 동작을 반영합니다. `latency_mode=wall`이며, 실제 로봇 adapter에서는 이미 흐른 시간을 중복으로 기다리지 않아야 합니다.

첫 평가에서 유리한 K가 두 번째에서 불리해지면, 빈번한 재계획의 비용이 제어 이득을 상쇄하는지 분석할 수 있습니다. GPU와 PIM의 속도 차이는 실제 측정이나 명시한 PIM 모델로 넣어야 하며, 임의로 빠른 지연을 넣은 결과를 PIM 실측으로 제시하지 않습니다.

성공률 외에 말단 위치/회전 오차, grasp·insertion/contact 실패, 완료 시간을 봅니다. 정적 물체뿐 아니라 도착 오차·물체 이동·접촉 직전 변화 조건이 필요합니다. 일반 task 성공만으로 sub-mm 정밀성을 주장하지 않습니다. 긴 H는 strategy와 trajectory 일관성에 도움이 될 수도 있으므로, H를 줄일수록 정확도가 오른다고 가정하지 않습니다.

모든 조건은 동일 initial-state/episode seed를 짝지어 비교하고, 가능한 범위에서 flow noise seed도 맞춥니다. 같은 seed라도 생성 shape가 다르면 초기 noise의 공통 prefix가 같다는 보장은 없으므로, 모델별 sampler의 noise 생성을 확인합니다. H별 학습을 하게 되면 같은 checkpoint의 inference-only 실험과 분리해 training budget도 맞춥니다.

## Offline 평가의 역할

같은 도착 관측에서 H를 바꾸고 공통 prefix의 action 오차를 비교하면 모델 출력 품질을 먼저 볼 수 있습니다. K를 바꿔 다른 길이의 expert MSE를 재는 것만으로 재관측의 효과를 판단할 수는 없습니다. Offline snapshot에는 실행 이후의 상태 변화와 피드백이 없기 때문입니다.

현재 `first_action_mse`는 모든 H에서 같은 첫 timestep을 비교합니다. K를 고정한 H 비교에는 `action_mse`도 사용할 수 있습니다. H가 다른 full-chunk MSE는 비교 시간 범위가 다르므로 주된 인과 비교로 쓰지 않습니다. 최종 결론은 closed-loop 성공률·정밀도·실패 유형으로 확인합니다.

## PIM 주장으로 연결하는 증거

| 단계 | 필요한 결과 | 지지하는 주장 |
|---|---|---|
| 제어 | 짧은 H/K가 성공률·정밀도를 유지/개선 | 짧고 빈번한 계획을 지원할 필요 |
| GPU profile | 해당 H에서 AE GEMM shape와 실제 bandwidth/실행 병목 확인 | PIM 대상으로 적절한 workload인지 판단 |
| 플랫폼 비교 | 전송·동기화·unsupported 연산을 포함해 GPU 대비 deadline/에너지 개선 | PIM 배치의 정량적 이득 |
| 시스템 | 같은 제어 품질에서 GPU 동시 실행/GPU+NPU와 비교 | PIM을 선택할 이유 |

첨부 [Beyond LLM Serving §4.2](https://arxiv.org/html/2610.05062v1#S4.SS2)는 작은 action tensor의 제한된 weight reuse와 memory 병목을 관찰합니다. 우리 실험에서는 이 관찰을 RTX 3090과 선택한 H/N에 대해 다시 확인하고, 그 제어상 필요한 실행 빈도까지 연결해야 합니다.

한 weight matrix에 대한 이론적 설명은 다음과 같습니다. `m`은 batch×해당 token 수, `q×r`은 weight shape, `b`는 원소당 byte입니다.

`FLOPs = 2 m q r`

`최소 tensor traffic = b (m q + q r + m r)`

Weight 항이 지배한다면 arithmetic intensity는 대략 `2m/b`입니다. H가 작아져도 state token과 VLM cross-attention의 모든 길이가 함께 줄지는 않습니다. 실제 cache·fusion·kernel launch 때문에 작은 H가 곧 DRAM 병목이라는 보장은 없으며, 이론 traffic은 계측한 DRAM byte와 다릅니다.

K를 줄이면 동일한 실행 action 수를 만드는 데 정책 호출이 늘어납니다. 종료에 따른 폐기를 무시한 근사로 호출당 H개 생성/ K개 실행이므로 생성/실행 비율은 `H/K`이고, 실행 action당 지연 비용은 대략 `T_call(H,N)/K`입니다. 따라서 “한 번에 덜 계산한다”와 “task 전체 비용이 줄어든다”도 따로 검증해야 합니다. Shared weight cache가 있다면 호출 횟수 증가를 그대로 DRAM 읽기 횟수 증가로 환산하지 않습니다.

PIM에서 AE만 옮기는 경우 핵심 비교는 `T_AE,PIM + 추가 전송/동기화/왕복 비용`과 `T_AE,GPU`입니다. 최신 관측을 반영하는 VLM 비용이 더 크다면 AE 가속만으로 원하는 K를 달성하지 못할 수 있습니다. Stage별 실제 critical path와 GPU 자원 경합을 함께 봅니다.

## 실행 및 출력

```powershell
python -m vla_eval run --config configs/horizon_accuracy.json
python -m vla_eval run --config configs/horizon_realtime.json
```

두 설정은 현재 CPU fixture backend를 사용하며 각 300 trial입니다. Toy controller는 학습 모델이 아니므로 이 결과로 GR00T의 H 의존성이나 PIM 효과를 판단할 수 없습니다. 실제 환경과 모델을 연결한 뒤 같은 설정 구조를 사용하세요. 기존 `horizon_sweep.json`은 N까지 교차하는 후속 실험입니다.

`horizon_analysis=true`인 실행은 `horizon_comparisons.json`도 저장합니다.

- `axis=generation`: K/N 고정, H 변화.
- `axis=execution`: H/N 고정, K 변화. Offline snapshot에는 생성하지 않음.
- `axis=joint`: H=K인 설정들의 결합 효과. 각각의 효과를 분리할 수 없음.

기준은 그 축에서 **실제로 평가한 가장 큰 값**입니다. Native H라고 자동 추정하지 않습니다. 동일 seed/episode가 있는 조건만 짝지으며 episode 단위 bootstrap 95% 구간을 계산합니다. 여러 H/K를 탐색한 뒤 가장 좋은 값의 구간만으로 결론내리지 말고, val에서 설정을 정해 test에서 검증합니다.

`mean_delta`는 작은 설정−큰 설정이며 성공률은 양수, 오차/시간은 음수일 때 작은 설정에 유리합니다. `summary.json`과 trial 원자료에는 생성량 중 사용 비율, 실행 action당 생성량, 실행 action당 측정 추론 비용을 함께 저장합니다. 측정 추론 비용은 모델 호출 범위의 비용이며, 카메라/전송/controller 전체 시간을 대신하지 않습니다.
