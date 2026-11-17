# 실험 프로토콜과 판단 기준

2026-11-16 방향 수정: 우선 B–C로 Heavy prior의 가치를 검증하고, 독립적으로 A–resume_only로 Heavy 연산 재사용의 가치를 검증합니다. 장기 목표는 최신 관측에 따른 rolling refinement와 HW-aware scheduling입니다. 첨부 7편과 추가 선행연구를 확인한 근거는 [research_direction.md](research_direction.md)에 있습니다. 아래 A/B/C/D는 초기 가설 검증과 대조군으로 유지합니다.

## 검증하려는 가설

1. 도착 전 정보로 만든 prior가 최신 관측만 사용하는 Small보다 유용한 정보를 제공하는가?
2. 실제 도착 오차가 커질 때 prior의 이득이 유지되는가? 어떤 오차에서 해로운가?
3. 중간 flow latent를 최신 관측으로 이어가면 완성 prior 보정보다 정확도/도착 후 지연이 나은가?
4. 짧은 실행 K의 피드백 이득과 짧은 생성 H의 추론 비용 변화가 각각 얼마나 되는가?
5. 실제 action expert의 전체 지연에서 PIM으로 옮길 수 있는 연산의 비중은 얼마인가?

현재 코드는 1–4의 공통 평가 경로와 5의 프로파일링 도구를 준비합니다. PIM 실행기·양자화·GPU와 PIM의 동시 실행은 구현하지 않았습니다.

## GPU pilot에서 확정할 항목

| 항목 | 선택 기준 |
|---|---|
| Heavy checkpoint | 공개 가중치/추론 코드, RTX 3090 메모리, 선택한 embodiment 지원 |
| 환경 | 실제 이미지·proprioception, contact-rich task, 상태 복원과 perturbation 지원 |
| 데이터 | checkpoint의 관측·action 형식 일치, episode split, pre-grasp 경계 식별 가능 |
| Small | 같은 task로 학습할 수 있는 작은 정책, C에서 prior를 입력받는 구조 |
| precision 판정 | task 성공 외에 위치/회전 허용 오차, 삽입/접촉 실패의 명시적 정의 |
| 예측 입력 | stale 카메라 + 목표 proprioception, 학습한 미래 관측, 예측 상태의 렌더링 중 선택 |

정밀 조작을 판단하려면 일반 물체 이동 성공률만으로는 부족합니다. Task가 허용하는 오차를 바꿔도 결론이 유지되는지 확인합니다. 관절 MSE는 말단 위치 mm와 동일한 지표가 아닙니다.

## 평가 순서

1. 모델의 공식 추론을 기본 설정으로 재현하고 action 좌표계·정규화·단위·history 길이를 확인합니다. 기본 H로 비교하는 custom sampler가 공식 sampler와 같은 초기 noise에서 일치하는지 먼저 검증합니다.
2. 도착 전/후 관측을 episode 단위로 모읍니다. 실제 도착 이미지를 사전 추론의 입력으로 사용하지 않습니다. 미래 영상 생성 비용도 계측합니다.
3. train으로 Small을 학습하고 val로 설정을 고릅니다. Test에서 B–C, A, prior-only, shuffled-prior를 먼저 비교합니다. 지연 없는 품질 평가와 실제 지연 반영 평가를 구분하고, 가능성이 보이는 조건은 closed-loop에서 확인합니다.
4. 같은 Heavy에서 fresh 재생성 A와 `resume_only`를 직접 비교합니다. D를 함께 두면 Small의 추가 이득을 분리할 수 있습니다. 중단 비율과 관측 변화를 val에서 바꾸고, test에서 정확도·남은 연산량·지연을 확인합니다. `resume_only`는 rolling 시스템의 재현이 아닙니다.
5. 고정 H/N에서 K를 바꾼 뒤, 실제 tensor 길이 H와 flow step N을 교차합니다. 학습 H와 다른 조건의 결과를 따로 표시합니다.
6. 재사용이 가능한 조건과 실패 조건이 보이면 rolling buffer와 최신 관측 경로를 구현합니다. 고정 재사용·항상 재생성·adaptive 선택을 같은 모델과 학습 조건에서 비교합니다. Action MSE를 실제 성공률로 대체하지 않습니다.
7. 같은 알고리즘을 GPU 순차/GPU 동시 실행/GPU+NPU/GPU+PIM에서 비교해 플랫폼 효과를 분리합니다. GPU 자원 경합·전송·동기화까지 반영한 뒤 PIM 대상 연산과 데이터플로를 정합니다. 측정하지 못한 플랫폼은 모델링 결과로 명시합니다.

RTX 3090에서 첫날은 batch=1, 모델의 기본 H, 기본 정밀도로 시작합니다. OOM이면 먼저 모델·이미지/history·실행 환경을 확인합니다. 양자화는 정확도 비교 조건을 바꾸므로 baseline 재현 이후 별도 축으로 추가합니다.

## A/B/C/D의 공정한 비교

- 같은 task, episode/seed, 실제 도착 상태, perturbation, 센서 설정, action 형식, H/K/N을 사용합니다. Flow noise seed도 첫 호출에 맞춥니다.
- A는 도착 후 전체 Heavy 연산을 수행합니다. 동일 H/K/N의 정확도 기준선이며 절대 상한이 아닙니다.
- B를 핵심 경쟁 상대로 둡니다. C의 correction model이 더 많은 학습량·파라미터·이미지 인코더를 사용하면 비용 차이를 별도로 보고합니다. 제공한 Ridge는 B/C에 같은 가중치를 사용합니다.
- C는 forecast로 완성한 prior를 실제 도착 관측으로 보정합니다. prior-only와의 차이는 보정의 필요성을, shuffled-prior와의 차이는 prior 내용의 유용성을 봅니다. 실제 causal prior 효과를 더 분리하려면 같은 Small에 zero-prior 입력도 추가하세요.
- D의 기본 정의는 flow 시각 `m/N`에서 중단한 `x_t`를 보존하고, 실제 관측을 새로 인코딩한 뒤 `m ... N-1`의 나머지 step을 수행하고 Small residual을 적용하는 것입니다. **관측에 의존하는 오래된 KV/hidden state를 최신 영상의 유효한 cache로 취급하지 않습니다.**
- D는 현재 최신 VLM 인코딩과 나머지 Heavy step을 모두 도착 후 수행합니다. 따라서 C보다 도착 후 비용이 클 수 있습니다. Partial Heavy가 자동으로 저비용이라는 전제는 두지 않습니다.
- `resume_only`는 D의 중간 latent 재사용 경로에서 Small을 제거하고, 이후 재계획도 Heavy로 수행합니다. A와 비교하는 데 B가 필요하지 않습니다. `configs/reuse_probe.json`과 paired `resume_only-A` 결과를 사용하세요.
- D의 conditioning 변경은 ODE 분포를 바꿉니다. C prior로만 학습한 corrector가 D latent의 출력을 잘 보정한다는 보장은 없습니다. D-specific 훈련 또는 혼합 prior 훈련은 별도 ablation입니다.
- 현재 closed-loop에서 C/D의 개입은 첫 chunk이며 이후는 B와 같은 Small입니다. 반복해서 reaching을 예측하는 시스템을 평가하려면 스케줄러와 정책을 확장하고 개입 범위를 다시 명시해야 합니다.

## 상태 오차·신선도 ablation

목표 위치 오차와 회전 오차, 도착한 로봇팔 자체의 오차를 구분합니다. 실제 환경 adapter에서 목표 물체 이동·도착 시간 오차·카메라 가림·접촉 상태 변화를 독립적으로 넣어야 합니다. 현재 toy는 2D 위치와 평면 회전 정렬만 구현하며, 로봇팔·접촉·삽입 물리를 제공하지 않습니다.

`partial_fraction`은 val에서 여러 값으로 반복 실행해 결정합니다. N이 작으면 `floor(N × fraction)`이 같은 split으로 중복될 수 있으므로 실제 `partial_steps`를 확인합니다. 완성한 latent에서 최신 관측을 바꾸기만 하는 것은 남은 flow step이 없으므로 D로 사용하지 않습니다.

도착 전 wrist-camera 이미지와 도착 후 이미지는 시점이 다릅니다. 목표 pre-grasp 상태를 정했다고 미래 시각 입력이 확보되는 것은 아닙니다. Stale image를 쓰는 조건, 예측 영상을 쓰는 조건, ground-truth 미래 영상을 쓰는 진단용 oracle 조건을 구분해야 합니다. 현재 기본 데이터 loader는 생산 가능한 forecast 세 종류만 받습니다.

## 지연·통계 해석

첫 액션 지연은 forecast 생성 비용을 포함한 선행 계산 중 남은 대기와 도착 후 연산의 합입니다. D의 post 시간에는 최신 관측 인코딩, Heavy continuation, Small correction이 모두 포함됩니다. 현재 데이터 I/O·카메라 전송·환경 wrapper 처리·안전 제어기의 비용은 별도 계측해야 하며, 중첩 시간은 분석적으로 계산합니다.

Warmup을 측정에서 제외하며, 지연 p95는 현재 summary에서 trial별 평균 지연들의 p95입니다. Call별 긴 꼬리는 `trials.jsonl`의 `trace`를 분석하세요. Microbenchmark와 end-to-end 정책 지연은 다른 지표입니다.

Closed-loop 성공률에 Wilson 95% 구간을, 동일 조건에서 B 대비 차이에 paired bootstrap 95% 구간을 제공합니다. Offline은 같은 episode의 frame와 flow seed를 한 단위로 묶어 재표집합니다. 소수 episode의 좁은 구간을 일반화 근거로 해석하지 마세요. 최종 seed/episode 수는 pilot 변동성과 필요한 차이 크기를 보고 정합니다.

## PIM 주장으로 넘어가기 전

생성 H가 작아지면 action expert의 일부 행렬 연산이 얇아질 수 있지만, state token, VLM 조건의 길이, cross-attention K/V, weight cache, kernel launch, batch, N도 영향을 줍니다. `K`만 줄여서는 모델이 생성하는 H가 바뀌지 않습니다.

GEMM 반복 microbenchmark는 weight cache의 이득을 포함합니다. Memory-bound를 확인하려면 실제 action expert의 실행 시간, 메모리 트래픽과 bandwidth, 연산 사용률을 측정하고 profiler의 operator shape와 함께 봐야 합니다. PIM 평가에는 데이터 이동, 지원하지 않는 연산, 비선형/attention 처리, 양자화 정확도, 동기화 비용까지 포함해야 합니다.

## 참고한 공개 1차 자료

- [NVIDIA GR00T 정책 문서](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md): predicted action horizon과 execution horizon의 구분, 관측/action 인터페이스.
- [NVIDIA N1.7 action head 소스](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/gr00t_n1d7.py): flow latent 생성, Euler 적분, state와 action embedding의 결합. 실험적 어댑터는 이 API를 대상으로 작성했습니다.
- [NVIDIA 정책 구현](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/policy/gr00t_policy.py): processor, batching, action decode 연결.
- [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339): 비동기 chunk 실행과 추론 지연을 함께 다뤄야 한다는 관련 연구. 현재 구현이 이 논문을 재현한다는 의미는 아닙니다.

공식 저장소의 `main`은 변할 수 있습니다. 실제 GPU 실험을 시작할 때 upstream commit과 checkpoint revision을 고정하고, 첨부 논문과의 차이를 업데이트해야 합니다.
