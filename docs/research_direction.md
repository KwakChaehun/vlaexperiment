# Heavy 능력 보존을 위한 연산 재사용과 reactive refinement

기준일: 2026-11-16. 사용자 제공 PDF 7편의 방법 및 평가 부분을 읽고, REACT/RAVEL/VLA-Rolling의 공개 원문을 추가 확인했습니다. 아래 구분과 실험 제안은 논문 결과를 바탕으로 한 연구 설계 판단이며, 우리 시스템의 검증된 성능이나 novelty를 뜻하지 않습니다.

## 연구 질문

**Heavy VLA가 제공하는 조작 능력을 유지하면서, 최신 관측에 민감한 연산을 필요한 빈도로 갱신하고 deadline을 만족할 수 있는가?**

선행 계산의 대상 후보는 완성 action뿐 아니라 VLM semantic context, 조건부 projection/KV, flow latent와 진행 시각, action expert의 재사용 가능한 feature입니다. 각각에 관측 시각·좌표계·조건 version·flow 시각을 붙이고, 오래된 값이 언제까지 유용한지는 측정해야 합니다.

Reaching/grasping 구분은 후보 신호입니다. 작은 움직임이 항상 재사용 안전성을 뜻하지는 않습니다. 작은 접촉 변화가 제어를 바꿀 수 있고, 큰 카메라 변화가 task와 무관할 수도 있습니다. 관측 변화량, proprioception 오차, context age, 예측 불확실성, 사용 가능한 contact 정보, 하드웨어 queue 상태를 함께 검토합니다.

## 첨부 논문에서 이미 다루는 범위

페이지는 제공된 PDF의 1부터 시작하는 파일 페이지입니다. 학회 판본·후속 개정과 내용이 다를 수 있습니다.

| 제공 논문 | 확인한 방법·평가 위치 | 이번 연구와의 관계 |
|---|---|---|
| [VQVLA: Motion-Aware Vector Quantization](https://arxiv.org/abs/2607.24148v1) | p.1, p.4, §7 p.8–9: 동작 상태에 따른 VQ 정밀도, centroid 병합/재사용. OpenVLA/OFT/RDT/π0/GR00T와 LIBERO/ManiSkill | 단계별 정밀도 선택만으로 차별화하기 어렵습니다. 정확도 보존과 메모리 접근량을 연결한 HW 설계가 필요합니다. |
| [SpecVLA: Speculative Inference and Verification](https://arxiv.org/abs/2608.15636v1) | p.1–2, §5 p.8–9: 긴 Heavy action 예측, 작은 verifier, 실패 시 재추론, GPU와 전용 모듈 병렬화. OpenVLA/RDT, LIBERO/ManiSkill | Heavy+Small과 상태별 실행 길이 조절은 이미 강한 비교 대상입니다. C의 이득과 verifier 방식과의 차이를 분리해야 합니다. |
| [Beyond LLM Serving](https://arxiv.org/abs/2610.05062v1) | Table 2 p.5, §4.2 p.7: GR00T N1.6/π0.5의 작은 action tensor와 bandwidth 병목. LIBERO closed-loop 평가 | thin-GEMM/memory-bound라는 관찰만을 새로운 contribution으로 삼기 어렵습니다. 반복 계획과 자원 경합에서 PIM의 추가 이득을 검증해야 합니다. |
| [VLA across XPUs](https://proceedings.mlr.press/v306/zhou26e.html) | §5.2 p.7, §5.3.1 p.8, Appendix A.4 p.14: DP-Cache와 V-AEFusion. 후자는 stale KV로 초기 denoising, fresh KV로 나머지 denoising | **현재 D와 가장 직접적인 중복**입니다. stale→fresh 전환을 PIM에 옮겼다는 설명만으로는 충분한 차별화를 입증하지 못합니다. |
| [DaDu-Corki](https://arxiv.org/abs/2407.04292v5) | p.1, §4.4/§5 p.9–10: trajectory 예측, inference/control/communication 분리, control matrix 재사용. RoboFlamingo+CALVIN | 통신·제어·추론 overlap 자체는 기존 접근입니다. 작은 flow action expert의 지속적 refinement와 구분해야 합니다. |
| [Deltoris](https://arxiv.org/abs/2608.04428v1) | p.1, §7 p.9–10: 시간적 bit sparsity, speculative inference로 loading 상각, bit-serial HW. PAD/DP/UVA, Meta-World/PushT/LIBERO | 재사용은 연산을 줄이면서 activation 보관·전송을 늘릴 수 있습니다. Weight traffic 감소만으로 전체 traffic 감소를 가정하면 안 됩니다. |
| [DiTPA](https://github.com/fengbintu/ISCA2026-DiTPA) | p.1, §V-A p.8, §V-G p.12–13: action/denoising/modality 재사용과 전용 accelerator. 주 평가 Dita+LIBERO-Long, 추가 π0.5/GR00T N1.5 | action expert의 여러 종류의 redundancy를 이미 함께 다룹니다. 보고된 accelerator 수치의 범위를 전체 VLM+control latency와 구분해야 합니다. |

첨부 PDF의 논문별 최대 speedup은 서로 다른 모델·hardware·연산 범위에서 얻은 수치이므로 직접 줄 세우지 않습니다. DiTPA 공개 repository의 기본 재현 경로는 Dita와 LIBERO를 사용합니다. 논문의 GR00T 확장 실험과 현재 작성된 N1.7 어댑터가 곧바로 같은 설정이라고 간주하지 않습니다.

## 추가로 확인한 가까운 방법

- [REACT](https://arxiv.org/html/2610.12007v1), §3 및 §6: per-block flow 시각을 가진 rolling buffer, 최신 관측에 따른 반복 refinement와 dual decoupling. 저자들은 학습한 rolling schedule을 임의로 바꾸는 zero-shot 지원을 주장하지 않습니다.
- [RAVEL](https://arxiv.org/html/2609.34170v1), §3.1–3.3 및 §4.4: rolling latent, 비동기 VLM context, Fast Observation Pathway, 시간 offset을 반영한 학습. 기존 checkpoint에 rolling 절차만 적용하는 조건의 실패도 분석합니다.
- [VLA-Rolling](https://arxiv.org/abs/2610.25043), abstract: 남은 action tail을 재사용하고 confidence에 따라 renoising과 denoising budget을 정합니다. 따라서 adaptive reuse/budget 자체도 새로움으로 가정하지 않습니다.

이 근거에 따른 판단은 다음과 같습니다. Rolling refinement와 HW-aware 선택의 가능성은 있지만, **무엇을 언제 재사용할지와 이를 어느 하드웨어에서 실행할지의 결합**에서 기존 방법보다 나은 결과를 보여야 합니다. 알고리즘 이름을 바꾸거나 PIM에 배치하는 것만으로 novelty가 확정되지는 않습니다.

## 먼저 수행할 두 실험

| 실험 | 비교 | 검증 대상 | 현재 지원 |
|---|---|---|---|
| Prior 가치 | B vs C, A와 prior-only/shuffled-prior 대조 | Heavy prior가 최신 관측 Small에 추가로 주는 조작 정보 | `configs/prior_value_gate.json` |
| Heavy 재사용 가능성 | A vs resume_only, D 추가 | 같은 Heavy의 전체 fresh 재생성 대비 부분 latent 재사용의 정확도/남은 비용 | `configs/reuse_probe.json` |

첫 실험은 현재 방식의 forecast+action prior를 평가합니다. 여기서 성공해도 추상 grasp strategy의 재사용까지 입증한 것은 아닙니다. Strategy prior로 바꾸려면 grasp pose/mode/접근 방향 등 표현을 명시하고 별도로 평가해야 합니다.

두 번째 실험은 Small의 능력과 독립적인 질문입니다. `resume_only`는 도착 때 한 번 resume하고 이후에는 Heavy를 사용합니다. 모든 cycle에서 미래 action을 이어가는 rolling buffer는 아직 구현하지 않았습니다.

C−B의 성공률 차이와 실패 유형을 먼저 봅니다. 최소 의미 있는 차이와 허용 가능한 정확도 손실은 val/pilot 이후 test 전에 고정합니다. 신뢰구간이 넓거나 0을 포함한다는 이유만으로 효과 부재를 확정하지 않습니다. 충분한 검증에서도 이득이 작으면 reaching-time prior contribution을 줄이고, A 대비 재사용의 이득과 short-horizon action expert 가속을 따로 판단합니다.

## 다음 구현의 분리 원칙

1. **고정 재사용 기준선:** 매번 전체 fresh 재생성, 고정 stale-step 수로 fresh 조건 전환, 고정 rolling refinement를 각각 구현합니다. V-AEFusion에 대응하는 실제 VLM/AE overlap을 별도로 계측합니다.
2. **최신 관측 경로:** full VLM 갱신, proprioception만 갱신, lightweight visual/proprioceptive 경로를 구분합니다. 현재 D는 full VLM을 다시 수행하므로 저비용 reactive path의 증거가 아닙니다. 최신 state만 주는 조건을 최신 이미지 반영이라고 부르지 않습니다.
3. **HW-aware 선택:** 유효한 cache 범위에서 재사용, 추가 refinement, 전체 재생성을 선택합니다. 관측 변화만 쓰는 선택기와 deadline/queue/전송 비용까지 쓰는 선택기를 비교합니다. 접촉·task mode 변화 등에서 full regeneration 경로가 필요할 수 있습니다.
4. **지원되는 계산만 변경:** 모델이 학습한 flow 시각·rolling profile·horizon 조건을 지킵니다. heterogeneous flow times나 새로운 fast path에는 추가 학습이 필요한지 확인하고 training-free 실험과 분리합니다.

유효한 cache는 같은 조건 version에서 재사용하는 projection/KV와 flow 진행 상태 등으로 정의합니다. 서로 다른 이미지의 중간 hidden/KV를 임의로 조합해서 최신 관측에 대한 정확한 cache라고 주장하지 않습니다. Stale feature를 의도적으로 사용하는 근사 경로는 stale임을 표시하고 오류를 측정합니다.

## HW 비교 매트릭스

알고리즘과 플랫폼을 따로 바꿔 원인을 분리합니다. 아래 플랫폼에서 동일한 항상 재생성·고정 재사용·adaptive 알고리즘을 가능한 범위에서 교차합니다.

| 배치 | 확인할 것 |
|---|---|
| GPU-only 순차 | 공식 sampler 및 stage별 baseline |
| GPU-only concurrent | VLM과 AE의 실제 stream overlap, bandwidth/SM/cache 경합, naive concurrent 대비 최적화 |
| GPU+NPU | AE 지원 연산/정밀도, 전송·동기화, 최신 관측 경로의 실행 위치 |
| GPU+PIM | PIM에 남길 weight/state, bank/dataflow 배치, unsupported 연산의 GPU 왕복, 변경된 cache의 이동 비용 |

측정 항목은 task 성공률·위치/회전 오차·contact 실패, 관측→실행 지연 p50/p95/p99, context age, deadline miss, 처리량, 전체 energy, DRAM 및 장치 간 traffic, cache footprint입니다. Nominal control Hz가 높아도 새로운 관측이 실제 action에 반영되는 빈도가 낮을 수 있으므로 둘을 따로 봅니다.

현재 코드가 이 항목을 모두 기록하지는 않습니다. 기존 지연/정밀도/action 수 외에 timestamp·cache/version·device trace를 추가해야 합니다. Concurrent 경합은 순차 GEMM 시간의 합/최댓값으로 대체하지 않습니다. NPU/PIM 장비가 없으면 분석/시뮬레이션 결과로 표시하고, RTX 3090 실측 결과와 구분합니다. PIM의 정량적 이득은 특정 PIM의 메모리 용량·bandwidth·연산 지원·전력 모델을 정한 이후에 평가할 수 있습니다.

## 모델·환경 선정에 대한 잠정 판단

후속 선정(2026-11-16): 첫 H/K 실험은 공식 task checkpoint가 제공되는 **π0.5-LIBERO + LIBERO-Spatial**로 시작합니다. 데이터는 `physical-intelligence/libero`, 성공률은 공식 initial states에서 live closed-loop rollout으로 측정합니다. GR00T와 ManiSkill precision 확장은 아래 후보로 유지합니다. 상세 구분은 [horizon_experiment.md](horizon_experiment.md)의 선정 항목을 따릅니다.

1차 공통 benchmark는 첨부 논문과 비교 가능한 **LIBERO**를 우선 검토합니다. 일반 task 성공만으로 정밀성을 입증하지 않으며, 별도로 도착 오차와 contact/위치 허용 오차를 제어할 수 있는 task가 필요합니다. ManiSkill의 insertion/plug 계열은 precision 확장 후보이며, 맞는 pretrained checkpoint가 있는지 먼저 확인해야 합니다.

모델 후보는 **GR00T N1.6**와 **π0.5**입니다. 전자는 제공 workload 논문의 DiT/작은 action tensor와 맞춰볼 수 있고, 후자는 V-AEFusion 및 REACT/RAVEL과 비교하기 좋습니다. [공식 GR00T n1d6 branch](https://github.com/NVIDIA/Isaac-GR00T/tree/n1d6)는 LIBERO 평가 경로를, [공식 openpi](https://github.com/Physical-Intelligence/openpi)는 `pi05_libero` checkpoint와 평가 경로를 제공합니다. 이는 후보 선정이며 설치·checkpoint 실행·수치 재현을 마쳤다는 뜻은 아닙니다.

RTX 3090에서는 먼저 pretrained batch=1 추론부터 확인합니다. Openpi의 공개 요구사항은 추론과 full fine-tuning의 메모리 규모가 크게 다르며, Ubuntu 환경을 기준으로 합니다. 이 저장소의 Windows CPU fixture 통과로 모델의 Windows GPU 실행이나 RTX 3090 학습 가능성을 보증하지 않습니다. 기존 N1.7 어댑터를 N1.6용으로 검증 없이 재사용하지 않습니다.
