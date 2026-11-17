# VLA–PIM 연구 인수인계: 대화, 결정, 구현, 결과

기준일: 2026-11-17. 지금까지 대화에서 나온 연구 목표·수정 방향·실험·실측·미해결 사항을 이 파일에 모았습니다. 향후 SSH/GPU 세션에서는 이 문서와 [실행 절차](SSH_GPU_RUNBOOK.md)를 먼저 읽으면 됩니다. 이 문서는 대화의 연구 맥락을 정리한 것이며, 검증되지 않은 아이디어를 구현/실측 완료로 취급하지 않습니다.

## 빠른 시작과 현재 상태

- **최신 논의 추가:** [카메라별 feature 재사용과 GPU 후속 평가](camera_feature_reuse.md). 입력 구성, state와 카메라의 예측 한계, attention 의존성, 손목만 갱신하는 두 수준의 구현, 네 카메라 조건의 실험과 연구 진행 기준을 정리했습니다. 해당 카메라별 실험은 아직 미구현·미실행입니다.
- 사용자 목표 GPU: **RTX 3090 24GB**. 로컬에서 실제 확인한 GPU는 GTX 1650 4GB이며, 지금까지 보고한 모델 추론은 **CPU float32**입니다.
- 실제 실행 모델/환경: **HuggingFaceVLA/smolvla_libero + LIBERO Spatial + MuJoCo/robosuite**.
- 첫 batch: 2026-11-17 06:49 KST 정상 종료. Offline 24관측, 도착 후 개입 27회, H/K 36회 = live rollout 63회.
- 후속 batch: H={32,16,8,4}, K={16,8,4,2}. 16칸 중 유효 13조합, 설정당 6회 = 78회. Prior 원인 분리 12관측을 먼저 수행합니다. 로컬 background worker는 진행 중이며 실제 최신 상태는 로컬 `results/hk_matrix_20261117/status.json`을 확인합니다.
- Git에는 코드·문서·작은 **결과 스냅샷**을 보존합니다. 로컬의 계속 갱신되는 `results/`, model/data/.runtime는 Git 대상이 아닙니다. Push 시점 스냅샷은 `reports/`에 있으며 이후 결과와 다를 수 있습니다.
- [SSH GPU 실행 절차](SSH_GPU_RUNBOOK.md), [H/K 실험 설계](horizon_experiment.md), [선행연구와 novelty](research_direction.md).

## 1. 최초 연구 아이디어

주제는 정밀 조작을 위한 VLA–PIM HW/SW co-design입니다. 빠른 이동과 정확한 접촉/조작을 함께 달성하고자 합니다.

1. **Reaching:** 목표 pre-grasp 위치를 미리 정하고 로봇이 이동하는 동안 미래 정밀 action을 GPU에서 생성해 고비용 추론 지연을 숨깁니다.
2. **Arrival/manipulation:** 실제 상태와 예측 상태의 차이를 작은 계산으로 보정하고, 최신 관측을 반영하도록 짧은 chunk를 짧게 실행합니다.
3. **PIM:** VLM backbone은 GPU에 두고 action expert의 작은 action tensor/GEMM을 PIM에서 실행합니다. H가 작을 때 weight reuse가 낮아질 가능성을 병목으로 연결합니다. 양자화도 후보입니다.
4. **협업:** GPU의 현재/미래 추론, PIM의 reactive action expert, control을 겹쳐 end-to-end 지연을 줄입니다.

초기 contribution 후보는 미래 정밀 action 선행 계산, short-horizon PIM dataflow, GPU–PIM 협업 실행이었습니다. PIM hardware, 실제 concurrent 실행, 양자화와 에너지 측정은 아직 구현하지 않았습니다.

## 2. 대화에서 수정한 연구 질문

사용자도 동의한 중심 질문은 다음입니다.

> 무거운 VLA의 높은 수준 조작 능력을 유지하면서, 실시간 제어에 필요한 연산만 현재 관측에 맞춰 자주 갱신할 수 있는가?

완성 미래 action을 조금 이동 보정하는 접근만으로는 충분하지 않을 수 있습니다. 선행 계산 대상을 semantic context/strategy, projection/KV, flow latent와 진행 시각으로 넓히고 최신 관측에 따라 이어 계산하는 방향을 검토합니다.

- **Computation reuse:** 미래 행동의 어떤 연산을 먼저 수행하고 유지할 것인가?
- **Reactive refinement:** 관측이 바뀌면 무엇을 재사용/재계산할 것인가?
- **HW-aware scheduling:** observation 변화·불확실성·deadline·queue·traffic에 따라 부분 재사용/추가 denoising/전체 재생성을 선택할 것인가?

사용자가 제시한 논의의 5/10(완성 action+위치 보정), 7/10(strategy prior), 8/10(부분 재사용+rolling refinement)은 **검증 전 주관적 가능성 평가**였고 실험 점수가 아닙니다. Adaptive scheduling이나 stale→fresh 전환 자체의 novelty도 확정하지 않습니다. GPU-only concurrent와 GPU+NPU 비교가 필요합니다.

## 3. 반드시 검증하려던 두 가지

1. 비정밀/reaching 동작 중 정밀 action을 선행 생성한 뒤 작은 보정으로 높은 정확도를 얻을 수 있는가? 예상 도착 좌표와 그 위치에서의 feature를 어떻게 만들 것인가? 기존 방법 대비 성공률·오차·지연 비교가 필요합니다.
2. Flow-matching 모델에서 **생성한 action 수 H**와 **실행한 수 K**에 따라 정확도와 비용이 어떻게 달라지는가? 작은 H/K의 품질 이득/보존이 실제로 존재하는지 먼저 확인하고 PIM workload 필요성을 판단합니다.

로컬 데이터 일부만으로 먼저 평가해 연구 방향을 잡고, GPU를 연결한 뒤 목표 모델과 hardware를 검증하기로 했습니다. 사용자는 추가로 필요한 대조군과 지표도 자율적으로 평가하도록 요청했습니다.

## 4. A/B/C/D와 실제 로컬 구현의 차이

| 원래 실험 | Reaching 중 | 도착 후 | 질문 |
|---|---|---|---|
| A Heavy fresh/oracle | 없음 | Heavy + 실제 관측 | 강한 기준선 |
| B Small-only | 없음 | Small + 실제 관측 | 가장 중요한 비용 경쟁 상대 |
| C Heavy + Small | 완성 prior | 최신 관측 Small 보정 | Heavy prior의 추가 가치 |
| D Partial Heavy + Small | 중간 계산 | 최신 관측으로 resume/보정 | 계산 재사용 가능성 |

A는 수학적인 정확도 상한이 아닙니다. B–C 이득이 없다고 같은 Heavy의 latent 재사용까지 반증되는 것은 아닙니다. 별도 `resume_only` 대 A 비교가 필요합니다.

실제 SmolVLA pilot은 자원 제약 때문에 다음 진단을 실행했습니다.

- `A_fresh`: 같은 SmolVLA, 최신 관측, flow 10-step.
- `B_few_steps`: 같은 parameter 모델, 최신 관측, flow 2-step. **작은 모델이 아닙니다.**
- `C_flow_refine`: 완성 prior를 다시 noise와 섞고 최신 조건으로 마지막 2-step.
- `D_partial`: 과거 조건 8-step의 실제 latent를 최신 조건으로 나머지 2-step.
- `B_ridge/C_ridge`: 동일 가중치의 약한 Ridge 모델에 prior 유무를 함께 학습한 Small/보정 진단. Language를 처리하지 않으므로 정식 Small VLA를 대신하지 않습니다.
- stale prior, forecast prior, 실제 미래 state+stale RGB oracle, shuffled-prior 대조군.

Ridge는 3개 train episode의 fresh teacher를 학습하고 다른 3개 episode에서 평가했습니다. Heavy pretraining과 공개 demo overlap은 가능합니다. 첫 flow 비교에서 최신 VLM을 다시 인코딩하므로 AE step을 줄였다고 최신 관측 경로 전체가 가벼워진 것은 아닙니다.

## 5. 모델·환경·데이터 결정 과정

목표 모델은 GR00T 계열 또는 π0.5입니다. 처음에는 공식 task checkpoint를 활용할 수 있는 π0.5-LIBERO + LIBERO Spatial을 GPU 1차 후보로 정했습니다. π0.5-LIBERO의 native H는 10이므로 SmolVLA H=50 또는 일반 예시 H=16과 혼동하지 않습니다.

RTX 3090이 아직 로컬에 없어 실제 CPU pilot에는 공개 SmolVLA-LIBERO를 사용했습니다.

| 항목 | 실제 local pilot |
|---|---|
| SmolVLA checkpoint | `HuggingFaceVLA/smolvla_libero` |
| Weight revision | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| Weight SHA256 | `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f` |
| VLM config/tokenizer | `HuggingFaceTB/SmolVLM2-500M-Instruct`, revision `4cde9d208e19519c36917aa170fffc7e381aa0ad` |
| Native inference | H=50, N=10, action 7차원, 내부 padded action dimension 32 |
| 입력 | 외부/손목 RGB 2개, proprioception 8차원, task instruction |
| LeRobot | 0.4.4, source archive `8fff0fde7c79f23a93d845d1a50e985de01f8b8a` |
| LIBERO source | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Dataset | `physical-intelligence/libero`, revision `a4336d589d589045d1c56423ffdf3b88a0e19b1f` |
| Episode IDs | 1261,1274,1267,1269,1280,1285 |
| Ridge train/test | train 1261/1267/1280, test 1274/1269/1285 |
| Offline 선택 | episode당 gripper close/open 직전과 uniform motion 4개 = 24개 관측 |
| Live suite | LIBERO Spatial task 0/1/2, H/K 실험 initial state 0/1 |
| 제어 | 20Hz, 최대 220 action, 시작 시 10 dummy settling action |

외부/손목 이미지를 공식 convention에 맞춰 180도 반전합니다. Quaternion→axis-angle은 LIBERO training convention을 사용합니다. SciPy canonical rotvec를 그대로 쓰면 pi 부근에서 state 표현이 달라질 수 있어 수정했습니다. Checkpoint normalizer/tokenizer/이미지 전처리를 사용합니다.

공식 sampler와 custom sampler의 native H/N 최대 차이는 **CPU에서 0**으로 검증했습니다. H를 바꿀 때 noise/action expert 실제 tensor 길이를 바꾸며 출력만 crop한 비용을 H 감소로 계산하지 않습니다.

## 6. 첫 batch: 선행 계산/보정 결과

동일한 reaching action을 재생해 각 방법의 실제 도착 state 최대 차이가 0인지 확인했습니다. Pre-grasp 구간 선택은 object 위치 또는 예정 gripper 전환을 활용한 **평가 oracle 주석**입니다. 배포 가능한 phase detector를 검증한 것이 아닙니다.

첫 4개 action만 방법마다 다르게 실행하고 이후에는 공통 Heavy controller로 복귀했습니다. 따라서 **arrival-only intervention**이며 전체 정밀 작업을 Small만으로 실행한 실험이 아닙니다.

### 예측 도착 pose를 개선한 조건

| 방법 | Task 성공 | Fresh A endpoint와의 위치 차이 평균 |
|---|---:|---:|
| A fresh 10-step | 2/3 | 기준 |
| Fresh 2-step | 2/3 | 1.36mm |
| 완성 prior + fresh 2-step | 2/3 | 4.90mm |
| Old 8 + fresh 2-step | 2/3 | 4.40mm |
| B Ridge | 2/3 | 19.27mm |
| C prior + Ridge | 1/3 | 22.53mm |

이 mm는 **정답 grasp pose 오차가 아니라 fresh A 동작과의 일치도**입니다. 이후 공통 Heavy의 복구가 task 성공 차이를 가릴 수 있습니다. 성공률은 3회뿐이어서 큰 불확실성이 있습니다.

Offline held-out 12관측/3episode의 expert NMSE는 B Ridge 0.67973, C Ridge 0.67809. Episode paired 차이 C−B는 −0.001637, bootstrap 95% 구간 [−0.07278,0.05105]로 prior 이득의 근거가 약했습니다. C-flow와 fresh 2-step의 expert NMSE는 0.60635 대 0.40631이었습니다. 이것은 한 expert 시퀀스와의 action 오차이며 가능한 grasp strategy의 전체 정답 집합이나 task 성공을 대신하지 않습니다.

### 도착 좌표 예측

실제 reaching에서 controller 목표 delta를 단순 합하면 선택 arrival의 평균 위치 오차가 **121.36mm**였습니다. 과거 4 step의 command gain으로 보정하면 **8.53mm**, 등속 외삽은 **11.82mm**였습니다. Controller 목표는 achieved pose가 아닙니다.

Offline 24관측의 4-frame 등속 state 외삽 위치 오차는 평균 8.06mm, p95 14.85mm였습니다. 이 값도 조작 성공 후의 정밀도가 아닙니다. Pose 예측 개선이 C의 task 성공/동작 일치도 향상으로 바로 이어지지는 않았습니다.

CPU prior 계산은 200ms reaching window에 들어가지 않았습니다. 순차 측정 뒤 `max(0, precompute−reaching) + post`로 계산한 지연 회계이며 실제 concurrent 실행을 측정한 것이 아닙니다. GPU에서 overlap 가능성을 다시 확인해야 합니다.

## 7. 첫 batch: H/K 결과

H는 생성 action 시점 수, K는 실행 후 재관측까지의 시점 수, N은 flow 적분 횟수입니다. H/K/N은 서로 다른 축입니다. Flow는 chunk를 함께 갱신하므로 autoregressive vocabulary token 조기 종료와 구분합니다.

N=10, 동일 3 task × 2 initial states, 설정당 6회입니다.

| H | K | 성공 | 호출 1회 CPU p50 |
|---:|---:|---:|---:|
| 50 | 10 | 3/6 | 8.01s |
| 50 | 4 | 5/6 | 8.05s |
| 16 | 4 | 6/6 | 5.29s |
| 4 | 4 | 4/6 | 4.40s |
| 50 | 1 | 1/6 | 8.01s |
| 4 | 1 | 1/6 | 4.40s |

- **짧을수록 항상 좋다는 결과가 아닙니다.** K=1이 이번 표본에서 가장 나빴습니다.
- H=50에서 K=4→1은 command 2차 차분 RMS 약 2.1배, EEF jerk 약 1.9배 증가. 새 noise에서의 빈번한 재생성이 시간적 일관성을 해칠 가능성은 있으나 원인은 아직 확정하지 않았습니다.
- H16/K4의 6/6은 최적값의 확정이 아닙니다. 95% Wilson 구간은 약 61~100%, H50/K4보다 성공 1회 많은 정도입니다. H16/K4 대 H50/K4의 paired exact McNemar p=1.0이었습니다.
- SmolVLA 버전의 action mask는 causal입니다. Offline 24관측에서 H16/H50의 첫 4 action은 동일했고 H4의 차이도 매우 작았습니다. Closed-loop SR 차이를 H 감소의 본질적 정확도 이득으로 바로 해석하지 않습니다. 수치 차이 누적·접촉 민감성·반복 재현을 확인해야 합니다.
- 각 호출의 noise는 native H=50에서 만든 공통 prefix를 사용합니다. K가 다르면 호출 횟수와 관측이 달라지는 것은 제어 실험의 일부입니다. 여러 seed의 검증이 더 필요합니다.

### 비용과 PIM에 대한 중요한 결과

Offline H50→H4에서 AE CPU 시간은 4.446s→0.631s, 약 7배 감소했지만 전체 호출은 8.579s→4.764s, 약 1.8배 감소했습니다. H4의 AE 비중은 **13.3%**였습니다. 다른 단계가 고정이고 AE 비용만 0으로 만드는 이상적 CPU 상한도 **1.15배**입니다. GPU/PIM 성능 예측이 아니며 GPU에서 critical path를 다시 측정해야 합니다.

H50/K1은 생성 action 사용률 약 2%, H50/K4 약 7.9%, H4/K4 약 99.5%였습니다. 그러나 작은 H로 호출 수가 늘면 task 전체 비용은 커질 수 있으므로 `T_call(H,N)/K`, 총 생성/사용량, 완료 시간과 품질을 함께 봅니다.

## 8. Prior가 나빠진 원인과 미래 feature 가설

사용자는 오래된 이미지 때문에 prior가 부정확할 수 있고, 정밀 작업 직전의 예상 pose에서 feature를 생성해야 한다고 제안했습니다. 현재 구현은 **예상 proprioception + stale RGB**이며 미래 시각 feature 예측기가 아닙니다.

기존 C는 `x_0.2 = 0.8*prior_full + 0.2*original_noise`를 만들고 최신 조건에서 2번 적분합니다. Fresh 2-step은 noise에서 1→0 전체를 2번 적분합니다. 계산 횟수는 같지만 초기 분포·적분 구간은 다릅니다. 기존 모델을 이 재보정 절차에 맞춰 학습하지 않았습니다.

따라서 image staleness, state 예측 오차, re-noising의 분포/상관 문제, step 부족을 분리해야 합니다. 잘못된 grasp mode의 완성 prior를 2-step만으로 크게 바꾸기 어려울 수 있습니다.

후속 prior 진단의 첫 1관측에서는 예측 state를 고정하고 stale→actual arrival RGB로 바꾸자 fresh teacher NMSE가 0.343→0.008로 크게 줄었습니다. 이것은 **단일 privileged 진단의 초기 결과**였고 전체 결론이 아닙니다. 누적 snapshot에서는 12관측 중 실제 완료 수를 함께 보고, expert NMSE와 teacher 일치를 구분합니다. 미래 RGB가 teacher 일치도를 높여도 expert action 오차나 실제 task 성공이 반드시 개선되지는 않습니다.

미래 feature 경로 후보는 다음입니다.

`현재 RGB-D/scene + 예상 robot/camera pose → 예상 future view/feature → prior → 실제 도착 관측으로 refinement`

EEF 좌표만으로 이미지가 정해지지는 않습니다. 회전, wrist camera extrinsic, robot configuration, 물체 geometry/가림이 필요합니다. Scene/RGB-D view warp나 learned feature predictor를 검토할 수 있습니다. Simulator의 정답 scene·실제 미래 frame은 **oracle**로 구분하고 deployable 방법이라고 주장하지 않습니다.

먼저 oracle future feature에 이득이 있는지 확인한 뒤 predictor를 구현/학습합니다. Feature cosine similarity만으로 결론내리지 않고 action 품질·정밀 task 성공·forecast 오차 민감도·추가 예측 비용까지 포함합니다.

## 9. 후속 batch의 정확한 설계

설정: `configs/smolvla_hk_matrix.json`(CPU), `configs/smolvla_hk_matrix_cuda.json`(CUDA), 실행기: `scripts/run_hk_matrix.py`.

| H \\ K | 16 | 8 | 4 | 2 |
|---|---|---|---|---|
| 32 | O | O | O | O |
| 16 | O | O | O | O |
| 8 | N/A | O | O | O |
| 4 | N/A | N/A | O | O |

`K>H`를 action 반복/padding/추가 chunk 생성으로 채우지 않습니다. 그 경우 제어 규칙이 달라집니다. 16칸을 보존하고 13개 유효 조합만 실행합니다. N=10 고정, 78 rollout을 이전 결과 복사 없이 새로 실행합니다.

먼저 held-out episode 3개에서 12관측의 prior 입력을 교차합니다.

| 이미지 | State | 역할 |
|---|---|---|
| stale | predicted arrival | 실제 예측 조건 |
| stale | actual arrival | state 오차 제거 oracle |
| actual arrival | predicted arrival | 시각 오차 제거 oracle |
| actual arrival | actual arrival | 자기 prior 재보정 진단 |
| stale | stale | 오래된 입력 그대로 대조군 |

각각 2/4/8 post step에서 original-noise re-noising, independent-noise re-noising, 실제 중간 latent continuation을 비교합니다. Fresh 2/4/8 및 full 10을 기준으로 둡니다. 같은 실제 조건의 latent continuation은 full과 정확히 일치해야 합니다.

원자료는 `prior_diagnostics/trials.jsonl`, 요약은 `prior_diagnostics_summary.json`, 조건 차이는 `prior_diagnostic_comparisons.json`, 작업 구간별 결과는 `prior_diagnostics_by_phase.json`입니다. 3 episode 미만에서는 bootstrap CI를 표시하지 않습니다. 3개에서도 충분한 통계 검증으로 간주하지 않습니다.

### Push에 포함한 후속 스냅샷: 10/12관측

`reports/hk_matrix_20261117_snapshot/`은 prior 진단 10관측/3episode가 완료되고 H/K rollout은 아직 0회인 시점입니다. 초기 단일 관측의 큰 시각 효과만을 전체 결과로 기억하지 않도록 다음 누적값도 보존합니다.

- Actual state에서 stale→actual RGB의 expert NMSE 차이는 −0.00053, episode bootstrap 구간 [−0.01885,0.00921]로 이 표본에서 expert action 오차 개선을 확인하지 못했습니다. Teacher 일치도와 구분해야 합니다.
- 같은 실제 관측에서 만든 prior를 다시 noise와 섞어 2-step 실행하면 원래 fresh full보다 expert NMSE가 +0.07327 증가했습니다. 구간은 [0.05227,0.09509]입니다. 이미지 staleness 외에 re-noising 방식도 문제일 수 있음을 보여주는 진단입니다.
- Forecast 조건에서 실제 intermediate continuation은 completed-prior re-noising보다 expert NMSE가 −0.07625 낮았습니다. 구간은 [−0.10854,−0.04808]입니다. 강한 최종 일반화 근거가 아닌 3episode pilot입니다.

이후 관측·rollout은 로컬에서 계속 실행되며 이 고정 스냅샷에 자동으로 추가되지 않습니다. 반드시 source/export 시각과 실제 최신 결과를 확인하세요.

## 10. 평가와 주장에 반드시 유지할 구분

1. Offline expert NMSE, fresh teacher 일치도, 실제 성공률, 정답 pose 정밀도는 서로 다릅니다.
2. LIBERO Spatial bowl grasp/place가 좁은 insertion 공차나 sub-mm 제어를 입증하지 않습니다. 이후 ManiSkill peg/plug 등은 맞는 정책 학습/적응 후 별도 검증합니다.
3. Simulator는 추론 중 멈춥니다. 현재 live SR은 **zero-inference-delay 품질 평가**입니다. 실제 지연 반영과 대기 중 control은 구현해야 합니다.
4. CPU thin-GEMM 시간/이론 arithmetic intensity만으로 GPU DRAM-bound나 PIM 우위를 입증하지 않습니다.
5. GPU concurrent, GPU+NPU, GPU+PIM은 전송·동기화·unsupported 연산·queue·전력까지 비교해야 합니다.
6. H를 바꾼 inference-only 실험과 H별 재학습을 구분합니다. SmolVLA causal 성질을 bidirectional 모델에 일반화하지 않습니다.
7. C의 이득 부재가 모든 Heavy prior/strategy 또는 동일 Heavy partial reuse의 부재를 뜻하지 않습니다.
8. 현재 Ridge는 작고 약하며, 진짜 Heavy–Small 능력 차이를 비교한 실험이 아닙니다.
9. 오류/censored는 task 실패로 집계하지 않습니다. 가장 좋은 작은 표본 설정을 test로 확정하지 않습니다.

## 11. 선행연구와 novelty 주의

사용자가 제공한 PDF는 다음 7편입니다. 로컬 PDF 자체는 Git에 올리지 않으며 검토 내용과 링크는 [research_direction.md](research_direction.md)에 있습니다.

1. A Motion-Aware Vector Quantization Framework with Centroid Reuse for Efficient VLA Inference (VQVLA).
2. Algorithm–Architecture Co-Design for Efficient VLA Inference via Speculative Inference and Verification (SpecVLA).
3. Beyond LLM Serving: Characterizing Vision-Language-Action Workloads for Embodied AI System Design.
4. Characterizing Vision-Language-Action Models across XPUs: Constraints and Acceleration for On-Robot Deployment.
5. DaDu-Corki: Algorithm-Architecture Co-Design for Embodied AI-powered Robotic Manipulation.
6. Deltoris: Enabling Real-time VLA Inference in Embodied AI via Bit-level Sparsity and Speculative Inference.
7. DiTPA: A DiT-based Action Planner Accelerator Exploiting Action-Denoising-Multimodality Redundancy for Embodied Artificial Intelligence.

특히 XPU 논문의 **V-AEFusion은 stale KV로 초기 denoising 후 fresh KV로 전환**하므로 D와 직접 비교해야 합니다. REACT/RAVEL은 rolling latent·비동기 VLM·최신 관측 경로·학습 정렬을, VLA-Rolling은 tail reuse/confidence budget을 다룹니다. SpecVLA의 Heavy+verifier와 DiTPA의 여러 redundancy 활용도 구분해야 합니다.

우리 contribution 후보는 어떤 값을 언제 재사용할지와 어떤 HW에서 실행할지를 함께 정해 정확도/deadline/traffic을 개선하는 데 있습니다. 기존 방법을 PIM에 옮겼다는 설명만으로 novelty를 확정하지 않습니다.

## 12. 코드 지도와 재현

| 경로 | 역할 |
|---|---|
| `vla_eval/smolvla.py` | 공식 math 기반 전처리, native sampler parity, 실제 short H, latent split/resume, CPU/CUDA device |
| `scripts/research_pilot.py` | 첫 실제 offline/arrival/HK batch, 공유 closed-loop 실행 |
| `vla_eval/arrival_rollouts.py` | 동일 reaching 후 첫 chunk 개입, forecast 조건 |
| `scripts/run_hk_matrix.py` | 후속 13조합과 prior 원인 분리, 자동 보고서 |
| `vla_eval/prior_diagnostics.py` | RGB/state 교차와 re-noise/continuation |
| `scripts/download_libero_subset.py` | 지정 episode만 받아 NPZ와 source manifest 생성 |
| `scripts/prepare_remote_assets.py` | Linux SSH 환경의 pinned model/data/LIBERO 준비 |
| `scripts/check_smolvla_runtime.py` | 선택 device에서 공식 sampler와 short-H 실제 tensor 검증 |
| `vla_eval/runner.py`, `toy.py` | CPU fixture 기반 전체 계약/로그 검증; 실제 학습 VLA 결과가 아님 |
| `vla_eval/openpi.py` | π0.5 native H=10 offline adapter, 실제 checkpoint 실행 미검증 |
| `vla_eval/gr00t.py` | N1.7 실험적 adapter, 실제 checkpoint/sampler 미검증 |
| `vla_eval/profiling.py` | GEMM/policy profiler, 이론 traffic과 실측을 구분 |
| `tests/` | 데이터/실행/flow/통계/invalid H/K 계약 검사 |

현재 단위 검사는 35개가 통과했습니다. 실제 모델 CPU parity도 0으로 검증했습니다. CUDA 경로를 추가할 때 timing 경계에 동기화를 넣고 model/input/flow time을 같은 device로 이동했습니다. Linux/RTX 3090은 현 세션에서 실행하지 못했으므로 서버에서 smoke 검증을 통과해야 합니다.

기존 Windows의 `.runtime/py311`은 Python 3.11.9/Torch 2.7.1+cpu/NumPy 1.26.4/SciPy 1.13.1/MuJoCo 2.3.7/robosuite 1.4.0/transformers 4.57.1입니다. Windows DLL 경로·GLFW·로그 경로와 Torch initial-state 로드 호환 패치를 적용했습니다. Linux에서 Windows 경로 패치를 그대로 복사하지 않습니다.

실행 중인 로컬 worker는 시작 시 로드한 코드를 계속 사용합니다. GPU 이식 코드 추가를 이유로 그 평가를 중지하지 않았습니다. Local output을 GPU output에 합치거나 새 코드의 hash로 과거 실행을 재표기하지 않습니다.

## 13. 다음 세션 우선순위

1. `reports/` 스냅샷과 실제 로컬/서버 output의 as-of/완료 수를 확인합니다. 후속 batch가 끝났다고 추정하지 않습니다.
2. SSH 서버에서 [runbook](SSH_GPU_RUNBOOK.md)에 따라 CUDA/renderer/native sampler를 확인합니다. 서버 주소/계정은 현재 문서에 없습니다. 이번 요청은 Git push까지이며 실제 서버 접속/설치는 아직 하지 않았습니다.
3. 목표 모델을 확정하기 전 SmolVLA GPU 재현을 먼저 확인하고 별도 output에 기록합니다.
4. GR00T/π0.5의 공식 task baseline, common-noise prefix와 H 변경의 진짜 tensor shape를 검증합니다.
5. Fresh few-step을 필수 기준선으로 두고 prior/partial reuse를 같은 post 계산 예산에서 비교합니다. 원인 분리 결과로 미래 feature predictor와 re-noising 방식 중 먼저 고칠 대상을 정합니다.
6. K=1/2의 시간적 일관성, noise/latent warm-start, cache age와 step budget을 실험합니다. 여러 initial state/seed 및 정밀 task로 확대합니다.
7. GPU AE/backbone p50/p95/p99, 실제 DRAM bytes·transfer·critical path·동시 실행 경합을 측정한 뒤 PIM target/dataflow를 정합니다.

현재 유망한 방향은 **Heavy 능력 보존 여부를 확인하면서 최신 관측 갱신과 부분 재사용의 품질–비용 관계를 정량화하는 것**입니다. 완성 prior+작은 보정, 미래 feature 예측, PIM 선택의 이득은 각각 검증해야 합니다.
