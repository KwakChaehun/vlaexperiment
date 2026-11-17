# 카메라별 feature 재사용: GPU 평가 전 연구 메모

기준일: 2026-11-17. 최초 push 이후 대화에서 추가된 가설과 평가 계획입니다. **아래 카메라별 ablation과 cache 최적화는 아직 구현·실행하지 않았습니다.** 기존 prior 진단의 RGB 교체는 두 카메라를 함께 교체하므로 이 실험을 대신하지 않습니다.

## 1. 현재 feature와 예측의 의미

현재 실제 평가 모델은 SmolVLA-LIBERO이며 GR00T가 아닙니다. `vla_eval/smolvla.py`의 `batch()`와 `condition()`이 다음 입력을 처리합니다.

| 입력 | 현재 구성 | 미래 정보에 대한 한계 |
|---|---|---|
| State | 말단 위치 3, axis-angle 회전 3, 그리퍼 관절 위치 2 = 8차원. 정규화·패딩 후 선형 투영 | 목표 상태는 알 수 있지만 실제 도착 상태는 오차가 있으므로 측정값으로 갱신해야 함 |
| 사용자 입력 | 작업 지시 문자열의 토큰·embedding | 작업이 같으면 입력은 동일하지만 영상과 결합한 VLM 출력은 영상에 의존 |
| 외부 카메라 | `image` RGB의 이미지 인코더 출력 | 카메라가 고정되어도 팔 이동·가림 때문에 미래 영상 전체가 고정되지는 않음 |
| 손목 카메라 | `wrist_image` RGB의 이미지 인코더 출력 | 시점 변화·가림·새롭게 보이는 면 때문에 미래 영상 예측이 어려움 |

VLM은 이 토큰들을 처리하고 층별 K/V cache를 생성합니다. Action expert는 이 조건과 noisy action 및 flow timestep을 사용합니다. Ridge Small baseline의 feature는 별도로 state + 카메라별 4×4 RGB 평균 + prior action이며 학습된 VLM feature가 아닙니다.

현재 forecast는 **과거 RGB + 예상 도착 state + 동일 지시**입니다. 예상 도착 좌표에서 보일 이미지나 그 feature를 생성한 것이 아닙니다. 목표 state embedding을 정확히 계산할 수 있다는 사실은 실제 도착 state 또는 미래 시각 feature의 정확성을 보장하지 않습니다. State 투영만 선행 계산하면 절약되는 연산도 작습니다.

접촉 전 정지 물체의 월드 pose와 예상 손목 pose가 있으면 `T_wrist_object = inverse(T_world_wrist) @ T_world_object`를 계산할 수 있습니다. 이는 pose 추정·정지 가정에 조건부인 기하 정보이며 미래 RGB/VLM feature가 아닙니다. 현재 checkpoint에는 이 상대 pose를 별도로 넣는 경로가 없어 adapter/학습 등이 필요합니다.

## 2. 손목 부분만 나중에 재계산할 수 있는가?

설치된 LeRobot 0.4.4의 `modeling_smolvla.py`에서 `embed_prefix()` 및 `make_att_2d_masks()`를 확인했습니다. 이미지·언어 토큰은 같은 attention block에 있고 서로 참조하며, state는 뒤의 block에 있어 이미지·언어 토큰이 state를 참조하지 않습니다.

- **동일 이미지·지시에서 state만 갱신:** 이미지·언어 K/V를 유지하고 state 관련 계산만 갱신할 구조적 근거가 있습니다. 현재 adapter는 전체 조건을 재계산하며 최적화 구현과 수치 일치 검증은 남아 있습니다. 오래된 이미지가 실제 조작에 충분한지는 별도 문제입니다.
- **이미지 인코더 출력 재사용:** 외부 이미지 embedding과 지시 입력 embedding을 먼저 저장하고, 도착 후 손목 이미지와 실제 state를 처리한 뒤 전체 VLM을 실행할 수 있습니다. 재학습 없이 구현 가능한 출발점이지만 VLM·action expert 비용은 남습니다.
- **VLM K/V까지 재사용:** 최신 손목 영상은 외부 영상·언어의 후속 hidden state에도 영향을 줍니다. 손목 K/V만 교체하는 것은 현재 모델에서 전체 재계산과 동등하지 않은 근사입니다. 토큰 순서만 바꾸어 해결되지 않습니다.
- **구조 변경 후보:** 외부 영상·지시의 무거운 context를 먼저 계산하고, 최신 손목 영상·state가 이 context를 읽는 별도 갱신 경로를 둡니다. Context가 손목 입력에 역으로 의존하지 않게 해야 합니다. Attention 변경에 대한 fine-tuning 또는 fusion module 학습과 정확도 검증이 필요할 수 있습니다.

따라서 외부 영상의 완벽한 미래 예측보다 **조금 오래된 외부 정보로도 정확도를 유지할 수 있는가**를 먼저 검증합니다. 이는 배포 가능한 입력만으로 시험할 수 있습니다.

## 3. 우선 평가: 두 카메라의 관측 시차 분리

아래 A–D는 **카메라 실험 안의 이름**이며 기존 Heavy/Small/prior의 A–D와 다릅니다. 로그에는 표의 고유 method 이름을 사용해야 합니다.

| 조건 / method | 외부 RGB | 손목 RGB | State |
|---|---|---|---|
| A / `camera_both_fresh` | 최신 | 최신 | 최신 |
| B / `camera_external_stale` | 과거 | 최신 | 최신 |
| C / `camera_wrist_stale` | 최신 | 과거 | 최신 |
| D / `camera_both_stale` | 과거 | 과거 | 최신 |

먼저 모든 조건에서 VLM을 전체 재계산합니다. 완성 prior, re-noising, 2-step correction은 섞지 않고 같은 H/K/N, 동일 지시, 동일 초기 상태 및 공통 noise seed를 사용합니다. 이렇게 해야 관측 시차의 영향과 근사 연산의 영향을 분리할 수 있습니다.

권장 순서:

1. 도착 관측과 이전 관측을 고정한 offline paired 비교로 action 변화와 fresh teacher 일치도를 확인합니다. Teacher 일치는 정답 정확도나 성공률이 아닙니다.
2. 같은 reaching 명령을 재생하고 동일한 도착 simulator state에서 네 조건을 분기합니다. 우선 첫 chunk만 다르게 실행하고 이후 모두 동일한 fresh 정책을 사용해 arrival-only 효과를 봅니다. 이후 회복이 차이를 가릴 수 있음을 명시합니다.
3. 과거 관측의 시차를 예를 들어 0/2/4/8/16 control steps로 바꾸고 실제 ms를 함께 기록합니다. 같은 초기 상태·seed로 paired 평가하고 여러 task와 initial state로 확대합니다.
4. 유망하면 외부 영상은 느리게, 손목·state는 빠르게 갱신하는 지속적인 closed-loop 정책을 별도 평가합니다. Arrival-only 결과와 합산하지 않습니다.
5. 외부 영상 재사용의 품질이 확인되면 이미지 embedding 캐시를 구현하고, 같은 stale-external 입력을 전체 재계산한 결과와 수치 일치를 검증합니다. VLM K/V 근사 재사용 또는 구조 변경은 그 다음 별도 실험입니다.

필수 기록: task/initial-state/seed, 두 영상의 관측 timestamp와 age, 실제 state, H/K/N, 모델·코드 revision, 성공률과 paired 차이, confidence interval, 충돌·재시도(환경에서 측정 가능한 경우), 접촉 직전 위치·회전 오차(정답 target pose 정의 가능한 경우), action 불연속성, 각 단계 지연 p50/p95/p99. 정답 pose가 없는 task에서 fresh action과의 차이를 정밀도라고 부르지 않습니다.

GPU timing은 외부 vision encoder, wrist vision encoder, VLM, state 처리, action expert 및 전송·동기화 경계를 분리합니다. Warm-up과 CUDA 동기화를 적용하고 최초 호출과 정상 상태를 구분합니다. 추론 중 simulator가 정지하는 품질 실험과 실제 지연 반영 제어 실험은 별도입니다.

## 4. 진행 여부를 판단하는 기준

- B가 A에 가깝고 C/D보다 좋으면 외부 정보 재사용 + 최신 손목 관측 가설을 지지합니다. 허용 가능한 성공률·정밀도 저하는 task 기준으로 사전 정의해야 하며 작은 표본에서 유의차가 없다는 사실만으로 동등성을 주장하지 않습니다.
- B도 A보다 크게 나쁘면 외부 영상 갱신이 필요하므로 이 분리의 매력이 줄어듭니다. 더 짧은 cache age나 변화 감지 갱신을 별도 검토할 수 있습니다.
- B/C/D가 모두 A와 비슷하면 현재 task가 시각 갱신에 둔감할 수 있습니다. 더 정밀하거나 관측 변화가 큰 작업으로 확인해야 합니다.
- B의 품질이 좋아도 이미지 인코더 절약이 전체 지연에서 작으면 큰 시스템 기여는 입증되지 않습니다. 무거운 VLM 계산까지 재사용 가능한지와 그 정확도 비용이 다음 쟁점입니다.

연구 가능성이 확정된 것은 아닙니다. 기존 completed-prior의 이득 부재가 모든 reuse의 실패를 뜻하지는 않지만, 이번 분리도 실패하면 미래 prior를 억지로 유지하지 않고 최신 관측 기반 short-horizon action expert 가속으로 집중할 수 있습니다. PIM 채택은 GPU 실제 memory traffic, 경합, 전송·동기화, GPU-only concurrent 및 GPU+NPU 비교 이후 판단합니다. GPU 계산 전체가 아닌 재사용 가능한 연산과 남는 연산의 양을 먼저 측정합니다.

## 5. GPU 세션 인수인계

사용자는 이제 실제 GPU를 사용할 예정입니다. 먼저 [SSH GPU runbook](SSH_GPU_RUNBOOK.md)의 환경·EGL·CUDA native sampler smoke를 통과한 뒤 기존 H/K 실험을 실행하세요. 이 문서의 네 카메라 조건은 **새 runner/config 구현이 필요한 후속 평가**이며 기존 `run_hk_matrix.py`를 실행해도 자동으로 수행되지 않습니다. 기존 H/K 및 Heavy prior 실험 목표를 대체하지 않습니다.

모델 weight/data/runtime는 Git에 없으므로 runbook대로 준비해야 합니다. GPU output은 CPU 결과와 다른 디렉터리에 기록합니다. 이 메모는 새로운 실측 결과나 RTX 3090 검증 완료를 의미하지 않습니다.

운영 관련 논의: GPU에서 독립 실행되는 평가 script 자체는 OpenAI 모델 토큰을 사용하지 않습니다. 코드 작성·디버깅·로그 해석은 Codex 사용량을 소모합니다. Astra high의 주간 사용 비율은 계정 한도와 작업량 없이 추정할 수 없으므로, 설계 검토/최종 분석과 독립 실행을 분리하고 Settings → Usage의 전후 차이로 확인합니다. [공식 사용량 안내](https://help.openai.com/en/articles/20001516-managing-usage-with-gpt-6-astra-in-work-and-codex).
