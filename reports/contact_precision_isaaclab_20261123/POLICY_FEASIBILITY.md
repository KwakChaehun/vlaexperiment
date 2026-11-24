# POLICY_FEASIBILITY — 촉각을 실제로 쓰는 학습된 정책이 있는가

작성 2026-11-24. "측정"은 이 기계에서 실행한 결과, "인용"은 외부 자료, "정적 분석"은 코드 읽기다.

## 1. 후보와 결정

| 후보 | 상태 | 판단 |
|---|---|---|
| A. DeCAL | 공개 가중치 없음, 실로봇 embodiment. 공식 코드만 읽음(정적 분석) | 실행 불가. 구조 비교에만 사용(§5) |
| B. Bench2Dex GR00T N1.5 tactile (`GR00T_n15_Tactile_Cross`) | 작업별 학습 checkpoint 26개 공개. 작업 09·26·27 checkpoint를 받음 | **채택** |
| C. DECO(UniVTAC, 약 80M) | `DECO_sim/univtac-p.pth`. 원 논문 표에서 촉각 추가 효과 0.0 pp(인용). Bench2Dex 작업이 아님 | 예비로만 기록, 사용하지 않음 |

새 정책 학습은 하지 않았다(명세상 금지 사항).

## 2. 사용 가능한 checkpoint와 촉각 입력 경로(측정, 정적 분석)

- 받은 checkpoint: `policy_ckpt/{09,26,27}/<robot>/{gr00t_n15_tactile, gr00t_n15}`. 이 중 촉각판만 썼다.
- 학습 설정(checkpoint `config.json`): `tactile_dropout_prob 0.3`. 학습할 때 30% 확률로 촉각을 지운다.
- 촉각 경로(모듈 trace, `prop/task26/trace.json`):

| 단계 | 모양, dtype | 파라미터 |
|---|---|---|
| TacMap 깊이 지도 10개 | [1, 10, 1, 240, 240] bf16(미터 단위 침투 깊이, 최대 15 mm) | — |
| `tactile_encoder` | → [1, 40, 1536] bf16(사이트당 4 token) | 0.31M |
| `tactile_cross_attn`(nn.MultiheadAttention, 8 head) | Q = LayerNorm(DiT 출력의 action token) [1, 16, 1024] fp32, K/V = 촉각 token [1, 40, 1536] → T [1, 16, 1024] | 5.25M |
| 합류 | X + T. gate 없는 residual(`post_dit`). flow step 4번마다 반복 | — |
| `action_decoder` | 1024 → 1024 → 64(속도) | 사용 slice 약 1.1M |

- VLM(Eagle, ViT+LLM 1,655M)과 `vl_self_attention`(201M)은 촉각을 보지 않는다. 촉각은 **VLM 바깥, action head 출력 직후**에 들어간다.
- 촉각 경로는 연산의 0.8%(호출당 약 0.83 ms / 105 ms), 파라미터의 0.2%다(측정, `HARDWARE_ANALYSIS.md`).

## 3. 데이터에 촉각 신호가 있는가(측정, `stage0/tactile_survey.jsonl`)

| 작업 | 로봇 | 촉각 신호 |
|---|---|---|
| 09 cleaner | xArm7 + LEAP | **전부 0**(접촉 프레임 없음) |
| 62 | LEAP | 0 |
| 51 | LEAP | 사이트-프레임 10개뿐 |
| 26 canned food tray | iiwa7 + Sharpa | 접촉 있음 |
| 27 ball box loading | UR5 + Wuji | 시연 50개(성공), 39,967 프레임 중 접촉 33%, 접촉 시작 429회, 손끝 사이트 10개 |

LEAP 손 작업의 리더보드 "촉각 이득"(작업 09: 비전 22/50 → 촉각 35/50, 인용)은 촉각 신호가 0인데도 나타났다. 따라서 이 이득은 촉각 효과가 아니라 **따로 학습된 checkpoint 사이의 편차**다. 그래서 작업을 09에서 27(주)과 26(보조)으로 바꿨다(NOTES 21:03).

## 4. 촉각 ablation 결과

### 4.1 오프라인(시연 replay 관측, 측정)

방법:
- 같은 관측과 같은 flow seed로, 촉각만 바꿨을 때 행동이 얼마나 변하는지 쟀다.
- 기준은 "flow seed만 바꿨을 때의 변화"(noise floor)다. 표의 값은 그 배수다.
- 행동 변화는 시연 명령의 차원별 표준편차로 정규화한 NMSE다.

| 작업 | 상태(관측 수) | 촉각 0 | shuffled(다른 시연의 촉각) | stale(16 프레임 전) | 촉각 기여 ‖T‖/‖X‖ |
|---|---|---|---|---|---|
| 27 | 접촉 818 | ×0.05 [0.04, 0.07] | ×0.09 [0.08, 0.09] | ×0.06 [0.05, 0.08] | 0.0805 |
| 27 | 비접촉 1,316 | ×0.00 | ×0.04 [0.04, 0.05] | ×0.00 | 0.0806 |
| 26 | 접촉 914 | ×0.04 [0.04, 0.05] | ×0.05 [0.04, 0.05] | ×0.05 [0.04, 0.05] | 0.0862 |
| 26 | 비접촉 884 | ×0.00 | ×0.05 [0.04, 0.05] | ×0.00 | 0.0741 |

(대괄호는 에피소드 단위 bootstrap 95% 구간. 파일: `stage0/task{26,27}/stage0_summary.json`)

- 접촉 상태에서 촉각을 완전히 지워도 행동 변화는 **flow 무작위성의 4–9%**다.
- 사전 등록 판정(NOTES 20:49)에 따라 **"이 정책은 오프라인에서 촉각을 거의 쓰지 않는다"**로 기록한다. 촉각 정밀도 연구의 근거로는 약하다.
- 학습할 때의 촉각 dropout 0.3이 이런 결과에 기여했을 수 있다(추정).

### 4.2 폐루프(Isaac Sim 6.2 이식, 측정)

native와 zero 촉각을 같은 장면·같은 seed로 짝지어 작업 27을 50 에피소드씩 돌렸다. 자세한 내용은 `CLOSED_LOOP_RESULTS.md`에 있다.

| arm | 작업 성공 | 단계 완료(LSCR) |
|---|---|---|
| native 촉각 | 29/50(58%) | 72.5% |
| zero 촉각 | 21/50(42%) | 63.0% |
| 차이(zero − native) | −16%p, 짝 bootstrap [−30, −2]. **exact McNemar p = 0.077**(엇갈린 쌍 12 대 4) | −9.5%p [−19.0, 0.0] |

- 촉각을 지우면 폐루프 성공률이 떨어지는 방향이 일관된다. 그러나 사전 등록한 주 검정은 5% 수준에서 유의하지 않다.
- 오프라인 호출당 영향(seed noise의 0.05배)보다 폐루프 영향이 크다.
- 6.2는 5.1과 물리가 다르다(`ENVIRONMENT_STATUS.md` §4). 그래서 6.2 native 성공률은 새 환경의 값이다.
- 5.1 리더보드는 참고로만 쓴다(인용, channel None, 50 에피소드).

| 작업 | 비전판 | 촉각판 |
|---|---|---|
| 27 | 21/50 | 23/50 |
| 26 | 10/50 | 14/50 |

두 판은 따로 학습한 모델이라, 이 차이는 촉각 ablation이 아니다.

## 5. DeCAL과의 구조 차이(정적 분석 + 논문 인용)

| | DeCAL | Bench2Dex GR00T N1.5 tactile |
|---|---|---|
| 주입 위치 | VLM(Qwen3-VL) 이해 layer **마다** gated cross-attention | VLM 바깥, action head DiT 출력 뒤(post-DiT), flow step마다 |
| gate | contact-aware gate σ. 전역 촉각 token에서 샘플당 scalar 하나를 만들어 모든 layer가 공유. 공개 기본 설정에서는 gate가 꺼져 있음(코드 확인) | 없음(항상 더함) |
| 촉각이 이후 계산에 미치는 범위 | 이후 VLM layer, action expert 전체 | 디코더와 다음 flow step의 DiT 입력만 |
| 가중치 | 비공개 | 공개 |

결과적으로 이 정책으로는 **"촉각이 VLM 은닉 상태에 섞인 뒤 정밀도 요구가 VLM 안에서 전파되는가"를 시험할 수 없다**. 이 정책에서 시험할 수 있는 것은 action head 쪽 전파뿐이며, 그 결과는 `TACTILE_INJECTION_AND_PRECISION_PROPAGATION.md`에 있다.

## 6. 결론

- 실행 가능한 학습된 촉각 정책은 있다(B).
- 다만 촉각을 VLM 밖에서 gate 없이 작은 residual(‖T‖/‖X‖ ≈ 0.08)로 더한다.
- 오프라인에서는 촉각을 지워도 호출당 행동이 거의 변하지 않는다(seed noise의 0.05배).
- 폐루프에서는 촉각을 지우면 성공률이 58%에서 42%로 떨어졌다(p = 0.077). 이 정책은 촉각을 쓰는 것으로 보지만, 근거의 강도는 중간이다.
- 촉각 정밀도 연구의 전제(촉각을 실제로 쓰는 정책)는 오프라인만 볼 때보다 더 잘 충족된다. 다만 촉각이 VLM 밖에 들어가는 구조라는 한계는 그대로다(§5).
