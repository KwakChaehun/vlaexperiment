# HARDWARE_ANALYSIS — GPU 지연, 메모리 트래픽, 저비트 kernel, PIM

작성 2026-11-24. 표의 "측정"은 이 기계(RTX 3090)에서 잰 값이다. "추정"은 계산이고, "인용"은 앞선 캠페인의 측정이다.

## 1. 실제 GPU 지연(측정)

- 대상: GR00T N1.5 tactile(작업 27 checkpoint), BF16(배포 그대로), batch 1.
- 입력: 카메라 4대 + 촉각 지도 10개 + 상태 52차원. 출력은 action 16×52, flow 4 step.
- 방법: 20회 반복의 중앙값, module hook의 CUDA event로 잼(`scripts/ci23_latency.py` → `hardware/latency_task27.json`).

| 부분 | 호출 수/정책 호출 | 지연 ms | 비율 | 파라미터 |
|---|---|---|---|---|
| **전체 get_action** | 1 | **105.2**(p10 104.9, p90 106.2) | 100% | — |
| VLM backbone(Eagle ViT+LLM, 1104 token) | 1 | 61.4 | 58% | 1,655M |
| `vl_self_attention` | 1 | 8.7 | 8% | 201M |
| DiT action expert | 4 | 25.0 | 24% | 550M |
| action encoder / decoder / state encoder | 4 / 4 / 1 | 0.72 / 0.19 / 0.05 | 1% | 230M / 36M / 52M(embodiment별 MLP 묶음 전체. 실제 사용은 1/32 slice) |
| 촉각 encoder + cross-attn + norm | 1 + 4 + 8 | 0.46 + 0.25 + 0.12 = **0.83** | **0.8%** | 5.6M(0.2%) |
| 기타(전처리, 호스트) | — | 약 8 | 8% | — |

- 정책 호출 최대 메모리: 5.9 GB.
- 폐루프에서는 16스텝마다 한 번 호출한다. 1190스텝 에피소드에서 약 75회 호출이다(`policy_query_count 75`, 측정).

## 2. 메모리 트래픽(추정, BF16 기준, 호출당)

| 부분 | 가중치 읽기 | activation |
|---|---|---|
| VLM(ViT+LLM, embedding table 제외 약 1.34B) | 약 2.7 GB(1회) | hidden [1, 1104, 2048] = 4.5 MB/층 |
| `vl_self_attention` | 0.4–0.8 GB(bf16/fp32) | 같은 크기 |
| DiT | 1.1 GB × 4 step = **4.4 GB** | [1, 49, 1536]. cross-attn 대상 [1, 1104, 2048] |
| 촉각 경로 | 11 MB(cross-attn) × 4 + 0.6 MB | 입력 1.15 MB, token 123 KB |
| 민감 위치(X, X+T, 디코더 은닉) | 1 MB 미만 | step당 [16, 1024] = 32 KB, 호출당 128 KB |

- 합계는 약 7.5–8 GB/호출이다. 3090 대역폭(936 GB/s)으로 나누면 하한 약 8 ms다. 실측 105 ms보다 훨씬 작다. batch 1에서 이 정책은 대역폭보다 kernel 실행, 짧은 GEMM, 호스트 작업이 지배한다(추정).
- DiT는 M = 49 token으로 가중치를 4번 다시 읽는다. 산술 강도는 약 25 FLOP/byte로 3090의 ridge(약 38)보다 낮다. 그래서 원리적으로는 가중치 압축 이득이 가장 클 수 있는 부분이다. 하지만 실측 25 ms는 대역폭 하한(4.7 ms)의 5배라, 압축만으로 줄어드는 몫은 제한적이다(추정).

## 3. 저비트 kernel 실행 결과

| 실험 | 결과 | 구분 |
|---|---|---|
| 이 정책(GR00T N1.5 tactile) | 실제 저비트 kernel은 실행하지 않았다. 사전 등록 우선순위 7(GPU HW 평가)은 1–4(접촉 의존 정밀도)가 유망할 때만 진행하는데, 1–4가 부정이었다 | — |
| GR00T N1.6 LIBERO(같은 날 앞선 캠페인) | Triton W8A8을 ViT·VLM·AE 전체에 넣으면 55.4 → 44.3 ms(1.25배), GEMM 합계 37.1 → 20.2 ms. W4A16·W4A8 Triton은 BF16과 같거나 느렸다. tinygemm W4A16은 1.7배 느렸다. torch 내장 W8A16 GEMM은 사용 불가. W8A8 폐루프 손실은 검출되지 않았다 | 인용(`reports/wa_quant_20261123/RESULTS.md`) |

- 이 정책에서도 같은 W8A8 경로를 쓰면, 지연의 82%를 차지하는 VLM과 DiT가 대상이다. 비슷한 비율의 이득을 기대할 수 있다(추정, 미측정).
- 접촉 조건 정밀도 전환은 대상 자체가 없다. 접촉이 정밀도 요구를 바꾸지 않기 때문이다.

## 4. 정밀도 요구가 있는 곳과 HW 비용

- 민감한 곳은 action head 출력 경로(X, X+T, 디코더 은닉, 속도)와 마지막 flow step이다. 접촉과 무관하다(`TACTILE_INJECTION_AND_PRECISION_PROPAGATION.md`).
- 이 부분은 호출당 데이터 128 KB, 디코더 연산 0.19 ms(0.2%)다. 이 부분을 BF16으로 두고 나머지를 INT4/INT8로 두는 **정적 위치별 혼합 정밀도**는 하드웨어 지원 없이 소프트웨어(kernel 선택)로 충분하다(추정).
- 동적 channel 선택(DiT 동적 35% 대 정적 50%)은 activation 기준으로는 이득이 있다. 하지만 가중치 열까지 쌍으로 바꾸면 두 형식을 모두 저장해야 한다(정적 대비 2.2배, `JOINT_WA_RESULTS.md` §4). 앞선 캠페인에서 token별 합집합이 가중치 열의 30%라 bytes 이득이 없었다.

## 5. PIM 적용 가능성

- 명세대로, 정밀도 민감도의 변화만으로 PIM 필요성을 주장하지 않는다. 이번에는 접촉에 따른 변화 자체가 없다.
- 앞선 캠페인(11/17)의 PIM 가설은 모두 측정에서 기각됐다(`memory: VLA-PIM null 결과`).
- 이 정책에서 PIM 친화적인 부분은 DiT다(낮은 산술 강도, 가중치 4회 재사용). 이것은 촉각·접촉과 무관한 기존 관찰이고, 측정된 PIM 이점은 없다.

## 6. 예상되는 HW/SW contribution

| 후보 | 근거 | 평가 |
|---|---|---|
| 접촉 인지 동적 정밀도 HW | 접촉이 민감도·r\*·중요 channel을 바꾸지 않음(측정) | **근거 없음** |
| 촉각 branch 전용 정밀도 HW | branch가 지연 0.8%, 파라미터 0.2%이고 INT4로 충분(측정) | 이득 없음 |
| 정적 위치별 혼합 정밀도(출력 경로 고정밀) | 민감 위치가 작고 고정적(측정) | 유효하지만 소프트웨어로 충분하고, 기존 기법 |
| flow step별 정밀도(마지막 step만 고정밀) | step 0–2는 INT4에서 noise의 0.07–0.08배, step 3만 3.0배(측정) | 유효한 관찰. 다만 diffusion의 timestep별 양자화(TDQ 등)와 같은 계열이라 새롭지 않음 |
| W8A8 kernel | 1.25배(인용, 다른 정책) | 기존 기법 |
