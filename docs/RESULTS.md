# 로컬 실제 모델 평가 결과

**2026-11-17 16:43 선행 계산·보정 진단 완료:** [GR00T arrival 보고서](../reports/arrival_gr00t_20261117/RESULTS.html). 본 평가 126회 + 마지막 구간 진단 21회 = 147회입니다. 9개 고정 도착 조건에서 fresh1과 선행 3+최신 1은 모두 8/9, 도착 후 약 41ms로 같았고 후자는 선행 계산 약 58ms가 추가됐습니다. 공통 제어기의 복구를 제거한 3개 마지막 구간에서는 fresh1=3/3, 과거 영상+예측 state의 3+1=2/3입니다. 작은 보정의 독립 이득은 확인하지 못했습니다. 실제 비동기/PIM 측정이나 학습된 미래 영상 예측기 결과는 아닙니다.

**2026-11-17 16:09 접촉 후속 완료:** [접촉·수납 및 접시 이동 234회](../reports/contact_hk_20261117/RESULTS.html). 접촉 작업에서는 50/16=53/60, 50/4=50/60, 16/4=47/60입니다. 앞선 Spatial의 천장 효과를 벗어났지만 16/4의 정확도 우위는 확인하지 못했습니다. 실제 정밀 핀 삽입 결과는 아닙니다.

**2026-11-17 오후 GPU 후속:** [RTX 3090 GR00T/SmolVLA·손목 갱신·H/K 비교](../reports/gpu_3090_wrist_20261117/RESULTS.md). 요청한 H50/K16, H50/K4, H16/K4는 각 12/12 성공, 약 65ms/호출입니다. 같은 12개 조건의 SmolVLA H16/K4/N2는 9/12입니다. 이는 LIBERO 잡기·놓기 소표본 성공률이며 미래 feature 예측이나 정밀 보정의 검증은 아닙니다. 아래는 이전 CPU 세션의 결과입니다.

첫 batch는 2026-11-17 오전 06:49에 정상 완료했습니다. 아래는 해당 batch의 결과입니다. H={32,16,8,4}/K={16,8,4,2} 후속 평가와 prior 원인 분리 진단은 [별도 보고서](../results/hk_matrix_20261117/RESULTS.html)에 저장합니다.

- [최신 결과와 연구 판단 — HTML](../results/research_pilot/RESULTS.html)
- [최신 결과 — Markdown](../results/research_pilot/RESULTS.md)
- [진행 상태](../results/research_pilot/status.json)
- [원자료: 선행 추론·H별 action 오차](../results/research_pilot/offline.jsonl)
- [원자료: 실제 LIBERO H/K rollout](../results/research_pilot/closed_loop.jsonl)
- [원자료: 동일 reaching 후 보정 방식 비교](../results/research_pilot/arrival_closed_loop.jsonl)
- [입력·모델·sampler 검증](../results/research_pilot/manifest.json)
- [현재 실행의 PID·로그 위치](../results/research_pilot/launch.json)
- [도착 위치 예측기 추가 진단](../results/research_pilot/forecast_summary.json)
- [첫 4 action 실행 후 fresh A와의 EEF 차이](../results/research_pilot/arrival_endpoint_summary.json)

사용 모델은 공개 SmolVLA-LIBERO이며 로컬 **CPU**에서 실행합니다. GTX 1650 4GB만 확인되어 RTX 3090을 사용하지 않았습니다. Native H=50/N=10에서 공식 sampler와 최대 오차 0으로 일치함을 확인했습니다. π0.5/GR00T나 GPU/PIM 성능 결과로 해석하지 않습니다.

첫 질문은 stale RGB + 과거 state 외삽으로 만든 prior, exact-state oracle 진단, 소형 Ridge 보정, 2-step flow refinement, partial-flow continuation으로 나누어 봅니다. Action 오차와 teacher 일치는 실제 정밀 조작 성공률과 구분합니다. 현재 소형 Ridge의 held-out episode는 3개여서 결과는 pilot입니다.

추가로 실제 simulator에서 알려진 reaching action으로 예상 도착 pose를 만들고 prior를 계산한 뒤, 같은 초기 상태에서 동일 reaching 동작을 재생해 보정 방식을 비교합니다. 첫 4 action만 개입하고 이후 Heavy controller를 같게 둡니다. Phase 선택의 object-position oracle과 배포 가능한 forecast 입력을 구분하며, CPU 선행 계산의 미완료 대기도 지연 계산에 포함합니다. GIF는 `arrival_replays/`에 저장합니다.

Controller 목표 delta의 합이 achieved pose와 크게 다른 문제가 관찰되어, 과거 움직임으로 command gain을 보정한 causal forecast도 `arrival_calibrated/`에 별도로 평가합니다. 미래 관측을 사전 입력으로 넣지 않습니다. 최종 성공률 외에 첫 4 action의 실제 EEF endpoint가 fresh A와 얼마나 다른지도 mm/degree로 기록합니다. 이것은 teacher agreement이며 정답 grasp pose에 대한 정밀도가 아닙니다.

둘째 질문은 실제 expert 생성 H=1/4/16/50의 공통 prefix action 오차와 비용, 실제 LIBERO의 H/K 여섯 설정으로 봅니다. Closed-loop는 추론 지연을 제거한 품질 실험이며, 같은 task/초기 상태를 짝지어 비교합니다. 완료된 episode의 GIF는 `results/research_pilot/replays/`에 저장됩니다.

SmolVLA의 causal action mask 때문에 H와 공통 prefix의 관계는 GR00T와 다를 수 있습니다. 또한 H를 줄이면 AE의 전체 비용 비중도 줄어듭니다. 보고서에 stage 비중과 AE 비용만 0으로 만들었을 때의 이상적 CPU 상한을 함께 표시합니다. 이것을 PIM speedup 예측으로 해석하지 않습니다.

결과 파일이 작거나 `running`이면 아직 진행 중입니다. `failed`이면 `errors.jsonl`과 stderr를 확인합니다. `budget_finished`는 설정한 시간 예산에 도달한 것이며, 미완료 rollout을 task 실패로 집계하지 않습니다.

RTX 3090 연결 후에는 같은 실제 모델 조건을 먼저 재현하고, 목표 π0.5/GR00T로 확장해 GPU stage latency, memory traffic, deadline과 동시 실행을 검증해야 합니다. 오늘의 CPU 측정만으로 PIM speedup을 주장하지 않습니다.

현재 B_few_steps는 같은 parameter 모델의 2-step 조건이고 B_ridge는 매우 약한 보정기입니다. 큰 Heavy와 학습된 Small의 정식 A/B/C/D 비교는 아직 아닙니다. LIBERO Spatial grasp/place도 좁은 공차의 insertion 정밀도 증거가 아니므로, 이 pilot으로 방법을 선별한 뒤 목표 정밀 작업과 강한 Small에서 검증해야 합니다.

중단된 batch를 다시 실행하려면 다음 명령을 사용합니다. 이미 평가 프로세스가 살아 있으면 중복 실행하지 않으며, 저장된 관측/완료 rollout을 이어서 사용합니다. 실행마다 새 로그를 만들고 `launch.json`에 위치를 남깁니다. 이 명령은 CPU 환경을 사용하며, 3090 연결만으로 CUDA 실행으로 바뀌지 않습니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local_pilot.ps1 -Hours 7
```

환경 버전은 `results/research_pilot/environment-freeze.txt`, checkpoint/data/code hash는 `manifest.json`과 `artifact-hashes.json`에 기록합니다. 설치 과정의 Windows 호환 패치는 `manifest.json`의 `environment_patches`에 명시되어 있습니다.
