# Contact-dependent precision demand in a tactile-aware VLA — Isaac Sim / Bench2Dex (2026-11-23 밤~)

연구 명세: 사용자가 붙여 준 "Codex Research Task: Contact-Dependent Precision Demand in Tactile-Aware VLA". 환경 구축 기록은 `results/bench2dex_20261123/NOTES.md`. 이전 LIBERO 결과(`results/contact_precision_20261123/`, `results/wa_quant_20261123/`)는 negative baseline으로만 쓰고 다시 대량 실행하지 않는다.

## 20:49 Phase A 현황

- 환경: Rocky 9.9 host, apptainer 1.5.5 rootless container(Ubuntu 22.04, glibc 2.35), RTX 3090 24 GB, driver 620.70.06, RAM 62 GB.
  - 시뮬레이터 venv `env_isaaclab`: Isaac Sim 5.1.0.0, Isaac Lab 0.54.2(v2.3.2, commit 37ddf626), torch 2.7.0(PyPI cu126 빌드; README의 cu128과 다름 — isaacsim 설치 시 먼저 들어와 같은 버전으로 간주), numpy 1.26.4(isaacsim-kernel은 1.26.0 요구, README대로 1.26.4). flatdict는 setuptools<70 + --no-build-isolation으로 빌드. Bench2Dex requirements 중 pinocchio(원격조작·replay 도구 전용)는 PyPI 이름 문제로 미설치.
  - 정책 venv `env_gr00t`: GR00T_n15_Tactile_Cross[base], torch 2.5.1+cu124, transformers 4.51.3, flash-attn 2.7.4.post1(공식 사전 빌드 wheel; [base]에 빠져 있으나 Eagle backbone이 import).
  - Bench2Dex commit c269f24b. 체크포인트: `policy_ckpt/09/multi_xarm7_with_leap/{gr00t_n15_tactile,gr00t_n15}`(Hugging Face Bench2Dex/policy_ckpt). 데이터: teleopdata 작업 09 replay-generalization 100개(50 시연 × 외형 2종), origin 100개.
- **Isaac Sim 실행은 보류**: NVIDIA Omniverse 라이선스(EULA) 동의 환경변수를 내가 대신 켜려다 권한 정책에 막혔다. 사용자 동의를 기다린다. 사용 데이터 수집 동의는 켜지 않는다.
- **Policy 후보(Stage 0)**:
  - A. DeCAL(AureleoPKU/DeCAL, CoRL 2026): 코드 공개, README가 안내하는 가중치는 초기화용 InternVLA-A1-3B뿐, 학습된 DeCAL 가중치 안내 없음. 실제 로봇(UR5+SharpaWave) 데이터 → Bench2Dex와 embodiment 불일치. 사용 불가.
  - B. **Bench2Dex GR00T N1.5 촉각(`GR00T_n15_Tactile_Cross`)**: 작업별 학습 체크포인트 공개. 코드 경로 확인: 시뮬레이터 촉각(`tactile.distance_along_normal_m`, 8 사이트 240×240) → `RemoteGR00TTactileDeployment` → `stack_tactile_mapping`(d_max 15 mm로 [0,1]) → `TactileTokenEncoder` → `nn.MultiheadAttention`(query = DiT 출력 action token, LayerNorm; key/value = 촉각 token) → **residual 덧셈(post_dit), gate 없음** → action decoder. 4 flow step마다 적용. 학습 때 촉각 dropout 0.3. leaderboard(저자 보고): 작업 09 matched scene 50회 기준 시각 전용 22/50, 촉각 35/50.
  - C. DECO(BAAI-Humanoid/DECO, ICML 2026): 코드와 sim 체크포인트 공개(`DECO_sim/univtac-p.pth` 촉각, 약 80M parameter). 지원 sim은 LIBERO·DexJoCo·UniVTAC·RoboTwin(Isaac Sim/Bench2Dex 아님). 저자 표에서 UniVTAC 촉각 추가 효과 평균 0.0%p. 대안 후보로만 기록.
  - 주 대상: B. DeCAL식 contact gate가 없으므로 gate를 가정하지 않고, 촉각 residual 크기 ‖Δ_tac‖/‖feature‖를 촉각 기여도로 잰다.
- 작업: 09 Cleaner Moisturizer Box Loading(xArm7 + LEAP, 양손으로 세정제를 들어 상자에 세우고 오른손으로 비누를 넣음). 접촉이 지속되는 양손 들기 + 집어 넣기.
- GPU 공유: 20:39경부터 20:54 전까지 다른 사용자의 vLLM 서버가 GPU 19.2 GB를 써서 3B 정책(약 7 GB)을 올리지 못했다. 다른 사용자의 작업은 건드리지 않고 GPU가 빌 때까지 기다렸다. 이후에도 GPU를 함께 쓸 수 있으므로 실행 전에 여유 메모리를 확인한다.

## 20:49 사전 등록 — Stage 0 오프라인 촉각 ablation (시연 replay 관측)

- 관측: 작업 09 replay-generalization 기본 외형 50 시연, 유효 프레임 중 16 프레임(정책 1 chunk) 간격. 입력은 시뮬레이터 client와 같은 형식(카메라 4대, 활성 관절, 촉각 깊이 지도). 시연은 이 정책의 학습 데이터일 수 있다(내부 민감도 측정용이며 일반화 평가가 아님).
- 조건(같은 관측, 같은 flow noise seed): 정상 촉각, seed만 바꾼 정상 촉각(noise floor), 촉각 0, 다른 시연의 무작위 프레임 촉각(shuffled), 16 프레임 전 촉각(stale).
- 지표: action NMSE(시연 commanded action의 차원별 표준편차로 정규화, 팔/손 관절 분리), 촉각 residual 비율 ‖Δ_tac‖/‖DiT 출력‖(4 flow step 평균), 접촉 상태(센서 수준 contact_mask: none/onset/sustained/release, 직전 16 프레임 기준).
- 판정: 촉각 0·shuffled의 action 변화가 noise floor보다 뚜렷이 크지 않으면 "이 정책은 촉각을 거의 쓰지 않는다"로 보고 촉각 정밀도 연구 근거 부족으로 기록한다. 폐루프 촉각 ablation은 Isaac Sim 실행이 가능해진 뒤 수행.

## 21:03 Stage 0 발견: 작업 09에는 촉각 신호가 없다 → 작업 교체

- 정책 smoke(20:55): 공개 촉각 체크포인트 로딩 성공, action 16×40(활성 관절 40), 같은 seed 재실행 최대 차 0(결정적), seed 변경 RMS 0.0084, 호출 0.106 s, GPU 5.5 GB.
- 작업 09 replay 데이터(기본 외형 50 시연 26,100 프레임, 외형 변형 50 시연 26,100 프레임)의 8개 촉각 사이트 깊이 지도와 contact_mask가 **전부 0**이다. 메타데이터상 센서는 tacmap, d_max 0.015 m. 시연 중 손이 물체를 들고 옮기는데도 xArm7+LEAP의 촉각 사이트가 한 번도 반응하지 않았다.
- 전 작업 조사(`stage0/tactile_survey.jsonl`, 작업마다 첫 기본 replay 1개): LEAP 손 작업 09(0), 62(0), 51(10 사이트-프레임)은 촉각이 사실상 없고, 나머지 23개 작업은 있다(예: 27 Ball Box 993, 26 Canned Food 900, 34 Fridge Wine 1187).
- 함의: 공개 촉각 체크포인트가 이 데이터로 학습되었다면 작업 09·62의 촉각 branch는 0 입력만 보았다. 그런데도 leaderboard에서 촉각 체크포인트가 +13(09), +6(62) 높았다 → leaderboard의 촉각 대 시각 차이에는 체크포인트별 학습 편차가 그 정도 크기로 섞여 있다. 촉각 사용 여부는 작업별 ablation으로만 판단한다.
- 작업 교체(Stage 0 결과에 따른 결정, Stage 1 이전): 주 작업 **27 Ball Box Loading**(UR5 + Wuji 손; 탁구공·골프공·미니 축구공·테니스공을 순서대로 상자에 넣기; leaderboard 시각 21/50, 촉각 23/50; 촉각 신호 풍부), 보조 작업 **26 Canned Food Tray Line Arrangement**(IIWA7 + Sharpa 손, DeCAL과 같은 Sharpa 계열 손; 시각 10/50, 촉각 14/50). 삽입·회전 작업(73 Jigsaw)은 성공률이 4–10/50로 낮아 짝지은 비교 검출력이 없어 제외.
- 다운로드(`.runtime/bench2dex/download3.sh`): 두 작업의 체크포인트(촉각·시각 전용), replay·origin 데이터, 에셋.
- 위 Stage 0 사전 등록(촉각 ablation 설계·판정)은 작업만 27(그리고 26)로 바꿔 그대로 적용한다.

## 21:32 사전 등록 — Stage 2(양자화 민감도)와 Stage 3(최소 고정밀 비율 r*), 작업 27, 오프라인

공통: 작업 27 replay 기본 외형 시연, 같은 flow seed로 native와 짝지어 action NMSE(시연 commanded action의 차원별 표준편차로 정규화) 측정. 접촉 상태는 센서 수준 contact_mask(직전 16 프레임 기준 none/onset/sustained/release). 이 정책에는 contact gate가 없으므로 촉각 기여도는 ‖Δ_tac‖/‖DiT 출력‖(4 flow step 평균)로 잰다. Isaac Sim 폐루프가 막혀 있는 동안 오프라인 결과만으로 다음 단계 진행 여부를 판단하지 않는다(폐루프는 사용자 EULA 동의 후).

**Stage 2** (`scripts/ci23_sens.py`; 시연 0–19, 24 프레임 간격): 위치 tac_kv / tac_delta / fused / dit_out / dit_linear / vl_cond / llm_linear × INT8/4/3/2(tile 32, FP16 scale).
- 질문 1(접촉 → 촉각 기여 증가?): tac_ratio를 접촉(onset+sustained) 대 none으로 비교, 시연 단위 bootstrap 95%.
- 질문 2(촉각 기여 → 양자화 민감도 증가?): 촉각 위치(tac_kv, tac_delta, fused)의 INT4·INT3 NMSE와 tac_ratio의 Spearman 상관, 그리고 접촉 대 none 비.
- 질문 3(모델 전체): 각 위치의 INT4·INT3 NMSE를 접촉 대 none으로 비교. 장면 변화 통제를 위해 시연 안에서 짝지은 비교(시연별 접촉 평균 − none 평균의 로그 비)를 주 지표로 한다.
- 판정: 어느 위치든 접촉/none 민감도 비의 95% 구간이 1을 넘으면 "접촉 시 민감도 증가"로 기록. 질문 1이 성립하지 않으면(기여도가 접촉과 무관) 질문 2의 "촉각 기여 증가 → 민감도" 경로는 이 정책에서 검증 불가로 기록한다.

**Stage 3** (`scripts/ci23_rstar.py`; 시험 시연 0–11, 32 프레임 간격; 정적 선택 보정은 시연 12–49의 16 관측):
- 범위: dit(DiT linear 112개 입력 전체), llm(VLM 언어 모델 linear 85개 입력 전체). 고정밀 INT8, 저정밀 INT4(activation tile). 비율 r ∈ {0, 3, 5, 10, 20, 35, 50, 75, 100}%.
- 선택: rand_s(정적 무작위), axw_s(정적 |x|·‖W열‖), axw_d(호출마다 동적). 가중치: A(activation만), WA(선택 열 W8, 나머지 W4 g128; 정적 선택만).
- r*(s, ε) = 시험한 비율 중 NMSE ≤ ε인 최소값. ε = 그 관측 자신의 seed 변경 NMSE × {0.1, 0.25, 0.5}(세 값 모두 보고). 100%에서도 못 미치면 ">100%"(INT8로도 부족)로 따로 센다.
- 주 비교: r*의 접촉 상태별 분포(none, onset, sustained, release), 시연 단위 bootstrap. 그리고 같은 r에서 접촉 대 none NMSE 비.
- 판정: 접촉 상태의 r* 중앙값이 none보다 크고 시연 단위 구간이 겹치지 않을 때만 "접촉 시 필요한 고정밀 channel 수 증가"로 기록한다. 그렇지 않으면 Stage 4 이후(폐루프 동적 정밀도)를 진행하지 않는다(명세 7.2).

## 21:56 Isaac Sim 라이선스 동의

- 사용자가 NVIDIA Omniverse/Isaac Sim EULA 동의를 명시적으로 알렸다(이 시각). 이후 시뮬레이터 실행에 `OMNI_KIT_ACCEPT_EULA=YES`만 설정한다. 사용 데이터 수집 동의(PRIVACY_CONSENT)는 설정하지 않는다.

## 22:02 폐루프 차단: Isaac Sim 5.1 RTX 렌더러가 이 드라이버에서 시작 직후 crash

- headless 기본 시험(`.runtime/bench2dex/test_isaacsim.py`, EULA 동의 변수만 설정)이 세 번 모두 segmentation fault(exit 139): "app ready" 직후 `librtx.scenedb.plugin.so`(carbOnPluginStartup) 안. 로그 `logs/test_isaacsim{,2,3}.log`.
  - 1회: GPU 여유 5.9 GB. 2회: 여유 12 GB(메모리 문제 아님). 3회: renderer=RaytracedLighting, multi_gpu=False(같은 crash).
  - container 안 NVIDIA 라이브러리(rtcore, nvoptix, glvkspirv, gpucomp, GLX/EGL)와 nvoptix.bin은 정상 연결, Vulkan은 RTX 3090 인식.
- NVIDIA 포럼(직원 답변, 2026-04·06): Isaac Sim 5.1.0은 595.x 이후 드라이버에서 같은 위치 crash가 알려진 문제이고, 검증 드라이버는 Linux 580.65.06. 설정으로 피하는 방법은 안내되지 않았다. 이 장비 드라이버는 620.70.06.
- 해결하려면 검증 드라이버로 내리거나(관리자 권한·재부팅 필요, 같은 GPU를 쓰는 다른 사용자에게도 영향) Isaac Sim 6.x로 옮겨야 한다(Bench2Dex는 5.1·Isaac Lab 2.3.2 기준이라 이식 필요, 6.x가 이 드라이버에서 되는지도 미확인). 사용자 결정 대기. 그동안 오프라인 Stage 2·3을 진행한다.
- Stage 3 시작(이 시각, `logs/stage3_task27.log`).

## 22:40 2번 경로(Isaac Sim 6.x 이식) 결정과 첫 관문

- 사용자 결정: 드라이버 교체(1번) 대신 Isaac Sim 6.x로 이식(2번).
- Isaac Sim 6.1·6.2의 검증 드라이버는 Linux 595.58.03(R595 계열). Isaac Lab v3.0.0(commit e3f7d9ff)은 Isaac Sim 6.2, Python ≥3.12 대상.
- container에 Python 3.12.16(deadsnakes) 추가, 별도 venv `env_isaacsim6`에 `isaacsim[all,extscache]==6.2.0`(torch 2.11.0+cu130 동반) 설치. 기존 5.1 환경은 그대로 둠.
- **시험 결과: Isaac Sim 6.2는 driver 620.70에서 정상 시작**(app ready 9.4 s, 30 frame 갱신, exit 0; `logs/test_isaacsim6.log`). NGX(DLSS) 비활성 오류만 있음.
- Isaac Lab 3.0 주요 변경(공식 migrating.rst): quaternion WXYZ→XYZW, 데이터가 ProxyArray(`.torch`), 쓰기 함수 `_index`/`_mask` 분리, backend 패키지(isaaclab_physx 등) 분리. Bench2Dex가 쓰는 모듈(sim, assets, actuators, sensors.camera, MultiMeshRayCaster, utils)은 3.0에 존재, ray_cast_utils만 이동.
- 이식 검증 계획: 원본 저장소는 수정하지 않고 복사본(`.runtime/bench2dex/Bench2Dex_is6`)에서 이식. Bench2Dex의 GT_REPLAY(기록된 commanded action 재생)로 작업 27 시연을 재생해 물체 궤적·성공 판정을 HDF5 기록과 비교한 뒤에만 정책 폐루프를 돌린다. Isaac Sim 5.1↔6.2 물리 차이로 재생이 갈라질 수 있으므로 그 크기도 보고한다.

## 22:54 Stage 3 설정 순서 변경(결과를 보기 전)

- 사용자가 추가 분석(촉각 주입 위치와 정밀도 요구 전파)을 요청해 우선순위가 바뀌었다. Stage 3 설정 순서를 dit activation-only → dit 가중치 쌍(WA) → llm으로 바꿨다(이전: dit → llm → WA). llm·WA 결과는 아직 하나도 나오지 않은 시점이다. 이미 끝난 19개 설정(dit activation-only)은 그대로 재사용(관측 단위로 재개). 이전 log: `logs/stage3_task27_part1.log`.

## 22:56 사전 등록 — 추가 분석: 촉각 주입 위치와 정밀도 요구 전파 (사용자 추가 요청)

- 이 정책의 실제 구조(코드 확인): 촉각은 VLM이 아니라 action head의 DiT 출력 뒤에 gate 없이 residual로 더해진다(post_dit, 4 flow step마다). 그래서 요청의 위치를 다음으로 대응시킨다. A 촉각 token, B=C cross-attention 출력 T(gate가 없어 B와 C가 같음), D 결합 X+T, E 결합 이후 decoder 은닉층·속도 출력과 flow 반복을 통한 다음 DiT 입력(dit_lin_s123 대 촉각 영향 전인 dit_lin_s0), F action expert 조건(vl_cond)과 DiT 내부 linear. "VLM 중간 hidden"은 이 정책에 존재하지 않으므로 측정하지 않으며, DeCAL 구조는 공식 코드 정적 분석으로만 다룬다.
- σ 대체 개입: X + α·T, α ∈ {0, 0.5, 1, 2}(모델 내부 개입, 물리 접촉과 구분). 결합 feature INT4를 동적 tile scale / 전체 보정 고정 channel scale(fixG) / 접촉 상태별 보정 고정 scale(fixS)로 나눠, 오차 전달 효과와 분포 변화 효과를 분리. T INT4·INT2는 배율 전에 양자화(오차가 α배로 전달).
- r*: T, X+T, decoder 은닉층; INT4 바탕 + 보호 channel INT8 또는 BF16; 무작위(rand_s) 대 동적 중요도(dyn); 비율 0–100%. ε는 Stage 3과 같게 관측별 noise floor × {0.1, 0.25, 0.5}.
- 대상: 작업 26(먼저, 접촉 시 촉각 기여가 커진 작업), 작업 27. 시험 시연 0–7, 40 프레임 간격. 고정 scale 보정은 시연 8–49 중 12개 × 6 프레임.
- 판정(가설 A/B/C):
  - A(촉각 국소): 촉각 위치(tac_tok, T)의 민감도만 접촉 시 커지고 X+T·이후 위치는 접촉/비접촉 비가 1과 구분되지 않음.
  - B(결합 유도): X+T의 접촉/비접촉 민감도 비 또는 r*가 접촉 시 커짐(시연 단위 구간이 1 또는 0을 넘음).
  - C(하류 전파): B에 더해 dec_h·vel 또는 dit_lin_s123에서도 접촉 효과가 유지되고, dit_lin_s0에는 없음.
  - 모두 아니면 "촉각 정보가 중요하더라도 INT4로 충분/접촉 무관"으로 기록하고 가설을 기각한다. 오프라인 r*는 진단 지표이며 폐루프 성공률을 대신하지 않는다.

## 22:57 Stage 2 일시 중지

- GPU 메모리(24 GB)를 Stage 2·Stage 3·추가 분석·Isaac Sim이 나눠 쓸 수 없어, 우선순위가 가장 낮은 Stage 2를 시연 11/20(관측 388)에서 멈췄다. 추가 분석(작업 26·27)이 같은 위치들을 더 자세히 재므로 중복이 크다. Stage 2는 관측 단위로 재개 가능하며, 재개하지 못하면 시연 11개 결과로 보고한다(사전 등록 20개 대비 축소를 명시).

## 23:17 우선순위 조정(결과를 보기 전)

- GPU 경쟁으로 실험이 모두 느려졌다(추가 분석 작업 26: 분당 약 1관측). 사용자 추가 명세의 우선순위(1–4 먼저, Joint W/A·폐루프 정밀도·HW는 1–4가 유망할 때만)에 맞춰, Stage 3은 dit activation-only 27개 설정이 끝나면 멈춘다. WA(가중치 쌍)와 llm 범위는 조건부로 미룬다. WA·llm 결과는 아직 하나도 보지 않았다.

## 2026-11-23 23:40 — Isaac Sim 6.2 GT replay vs 5.1 demonstration (task 27, anchor 0) — measured
Recording: closed_loop/gtreplay_is6_rec1/rec/failure/episode_000001.hdf5 (1190 policy steps, stage completion 0.25).
Comparison: scripts/ci23_compare_replay.py -> closed_loop/gtreplay_is6_rec1/compare_vs_demo.json (frames aligned by index;
the rollout recording has no joint_names, so the 52 qpos columns are taken in the demo order — same articulation, same width).
- Commanded actions are identical to the demo (max |diff| 0 over 916 frames; demo frame 0 is NaN, the rollout applies
  the frame-1 command at frame 0).
- Robot tracking: arm |q_rollout - q_demo| median 0.004 rad, p95 0.014-0.018 rad in frames 1-300; hand median 0.002 rad.
  The robot follows the commands as in 5.1 -> no actuator/joint-order port bug.
- Objects: gelatin box, cups, basket start within 0.02 mm; golf / table-tennis within 0.5 mm. Mini soccer ball differs
  by 4.9 cm (geometric centre) already at frame 0: from the same spawn placement it rolled ~5 cm during the pre-episode
  settle in 5.1 but stayed ~0.2 cm from the spawn in 6.2 (its origin is at the ball's bottom, so the origin offset
  2.9 cm understates it). Racquetball (distractor) 1.3 mm.
- First >1 cm divergences after contact begins (demo first sensor contact at frame 79): basket 0.9 cm at 80,
  table-tennis 1.1 cm at 173, golf 1.4 cm at 300, soccer 25 cm by 630.
Interpretation: the 6.2 port reproduces the robot but not the 5.1 object physics (initial settling of a ball and contact
outcomes). An open-loop replay cannot recover from that, so GT-replay success is not a usable port check; closed-loop
policy success in 6.2 must be reported as its own (new) environment, not compared one-to-one with the 5.1 leaderboard.

## 2026-11-23 23:45 — TacMap tactile sensor ported to Isaac Lab 3.0
The tactile client needs collector/tacmap_sensor/SharpaTacmap, a subclass of the Isaac Lab 2.x MultiMeshRayCaster.
Isaac Lab 3.0 rewrote the ray caster (warp-first buffers, backend factory, ClonePlan), so the subclass cannot be ported
by API renames. Bench2Dex_is6/collector/tacmap_rig.py now computes all 10 sites with one batched re-implementation
(_Is6TacMapBatch) that keeps the original computation: npy surface points (mm x correction_scale) and flipped unit
normals in the site link frame, all Mesh/primitive prims under each target merged in the target's rigid-body frame,
live PhysX rigid-body transforms (xyzw), first cast within max_distance, CPD back-cast from hit-1e-4 m within 0.5 m,
keep iff both hit, depth 0 elsewhere, uint8 quantize + Gaussian blur. Policy input is distance_along_normal_m (raw).
Tactile client patches: PhysxCfg import, T() around joint_pos reads, camera diagnostic quaternion. Runner:
.runtime/bench2dex/run_tactile_is6.sh (server overrides copied from eval_double_env.sh).
Validation plan (fixed before seeing results): GT replay of anchor 0 for 300 steps with IS6_TACMAP_DUMP_DIR; compare with
the demo's recorded distance_along_normal_m (scripts/ci23_compare_tactile.py, offsets 0/1/2 all reported) over demo
frames 60-172, i.e. before the first >1 cm object divergence. Pass if contact appears at the same sites and frames with
mask IoU clearly above chance; exact depth equality is not expected (robot states differ by ~mm).

## 2026-11-23 23:52 — TacMap validation: two failed attempts explained (no result yet)
- 23:45 run (tacval_gtreplay_is6) and 23:47 debug run: zero tactile contact. Cause found in the server log: the GT_REPLAY
  server consumes one demo row per query but the tactile client requests 16-step chunks (and update_obs calls), so the
  demo was exhausted within the first queries and the robot held the demo's final pose (hand 40 cm above the table;
  debug ray centroids constant from capture 40). The sensor was never near an object; these runs say nothing about
  the TacMap port.
- Hardening kept: site poses now come from the robot articulation's link transforms (root_view.get_link_transforms)
  instead of rigid-body views of articulation links.
- Fix for validation: client-side replay hook IS6_GT_REPLAY_HDF5 (Bench2Dex_is6 tactile client only): the demo's
  commanded row k is executed at policy step k (invalid rows held at the nearest valid row, as GT_REPLAY does).
  Rerun: closed_loop/tacval2_gtreplay_is6 (300 steps, dump + debug), compared as pre-registered at 23:45.

## 2026-11-23 23:55 — Closed-loop pre-registration (before any closed-loop policy result)
Environment: Isaac Sim 6.2 port (Bench2Dex_is6, GPU physics), task 27 (multi_ur5_wuji_with_flange), checkpoint
policy_ckpt/27/multi_ur5_wuji_with_flange/gr00t_n15_tactile served by the original Bench2Dex policy_model_server with the
eval_double_env.sh overrides; ported tactile client; generalization profile none; anchors replay-generalization;
budget 1190 policy steps (official ceil(expert_time x 1.5 / 3)); chunk 16; warmup 60; base seed 100000000.
Arms (same episodes and seeds): native (tactile as sensed) vs zero (IS6_TACTILE_ABLATION=zero: every tactile map is
zero at the policy input; sensing and logging unchanged).
N: episodes 0-19 per arm first; more episodes only by wall-clock availability, never by looking at results.
Primary: task success (scene success checker). Secondary: stage/partial completion if the client reports it; per-step
tactile contact log (IS6_TACMAP_LOG). Test: exact McNemar on paired success, bootstrap CI of the paired difference.
The 6.2 native success is a new environment's number; the 5.1 leaderboard (tactile 23/50, vision 21/50, channel None)
is quoted only as context.
Stage 4 gate: contact-conditioned precision arms run only if Stage 2/3 show contact-dependent demand. Current offline
evidence (Stage 2 interim, Stage 3 A|dit, propagation task 26 interim) shows contact/none ratios ~1, so they are not
planned.

## 2026-11-23 23:54 — TacMap port validation (measured, criterion fixed at 23:45)
Run closed_loop/tacval2_gtreplay_is6 (client-side replay of anchor 0, 300 policy steps, dump of all captures);
scripts/ci23_compare_tactile.py vs the demo's recorded distance_along_normal_m.
Demo frames 60-172 (offsets 0 / 1 / 2): site-frames with contact 124 / 123 / 124; in both 99 / 100 / 99; port only
6 / 5 / 6; demo only 19 / 18 / 19; mask IoU median (both) 0.66 / 0.69 / 0.72; depth-sum correlation 0.62 / 0.74 / 0.86;
total depth port/demo 1.12; median max depth 15.0 mm in both. Frames 0-59: no contact in either.
-> PASS. Contact appears at the same sites and frames with IoU far above chance; residual differences match the
known physics divergence (robot ~mm, soccer ball start 4.9 cm). Files: compare_vs_demo_60_172.json, compare_vs_demo_0_59.json.
Debug hook (IS6_TACMAP_DEBUG) stays off in all policy runs.

## 2026-11-23 23:54 — Closed loop started: native arm, task 27, episodes 0-19 (closed_loop/t27_native, port 9301)

## 2026-11-24 00:02 폐루프 첫 시도 중단과 재시작 계획(분석에 쓰지 않음)
- 정정: 위 "23:55 — Closed-loop pre-registration" 항목은 실제로 23:53 무렵, 촉각 검증 결과(23:54)를 보기 전에 썼다.
- 23:54/23:58에 시작한 native·zero 실행에서 두 가지 환경 결함을 발견했다.
  1. 작업 27 시연 100개 전부가 배경으로 iTHOR 방 USD(scenes/ithor/FloorPlan*_physics)를 쓰는데, 이 자산을 받지 않아 배경 없이 렌더링되고 있었다(client log: "USD scene file not found"). 스테레오 카메라 입력이 학습 분포와 크게 다르다.
  2. 이식한 client가 에피소드 사이 장면 재구성에서 멈춘다(utils/episode_runtime.py `sim._timeline` 없음, Isaac Lab 3.0). native는 1개만 돌고 종료됐다.
- 조치: 두 실행을 멈추고 `closed_loop/discarded_nobg_singleproc/`로 옮겼다(삭제하지 않음, 분석 제외). 본 결과는 native 1개(실패)뿐이며, 중단 이유는 환경 결함이지 결과가 아니다.
- iTHOR 장면 61개(~200 MB)를 받는다(.runtime/bench2dex/download6_ithor.sh). runner는 에피소드마다 client를 새로 띄우고(공식 eval_double_env.sh의 chunk 재시작과 같은 방식) 서버는 유지한다. 23:55 사전 등록 조건(arm, 에피소드 0-19, 지표, 검정)은 그대로 둔다.

## 2026-11-24 00:08 폐루프 표본 크기 확정(유효한 폐루프 결과를 하나도 보기 전)
- 에피소드당 약 5.7분(client 재시작 포함), 두 arm 병렬. 사전 등록의 "0-19 먼저, 추가는 시간에 따라"에 따라 지금 0-49 전체(작업 27 anchor 50개, 리더보드와 같은 수)로 정한다.
- 멈춤 규칙: 에피소드 49까지 끝나거나 06:30이 되면 멈춘다. 분석은 두 arm 모두 끝난 에피소드만 짝지어 쓴다(McNemar, 짝 bootstrap). 결과를 보며 N을 바꾸지 않는다.
- 그림 5개(전파 보고서용)를 figures/에 만들었다(scripts/ci23_figures.py): 주입 위치 도식, 위치별 INT4/INT3 오차, 접촉/비접촉 비 forest, α 개입, 보호 비율 곡선.

## 2026-11-24 00:20 폐루프 에피소드 번호 문제(결과와 무관한 실행 문제)
- 촉각 client의 `--start-episode`는 1부터 센다(`episode_idx = max(0, start-1)` 후 +1). runner가 0부터 넘겨서 k=0과 k=1이 모두 에피소드 1(seed 102700000, anchor 0)을 돌렸다. k=2..49는 에피소드 2..49로 맞다. 에피소드 50은 빠진다.
- 처리 규칙(지금 정함): 에피소드 번호별 첫 기록만 집계한다. 에피소드 1의 두 번째 실행은 같은 조건의 반복성 확인으로만 보고한다(두 arm 모두 첫 실행은 실패, 단계 완료 0). 에피소드 50은 현재 실행이 끝난 뒤 수정한 runner(`run_tactile_is6_v2.sh`, START를 1-based로 넘김)로 arm마다 하나씩 돌린다. 실행 중인 runner 파일은 bash가 읽는 중이라 수정하지 않았다.

## 2026-11-24 00:21 폐루프 두 번째 실행 문제와 조치(결과와 무관한 실행 문제)
- 00:07 실행에서 k ≥ 2인 client가 시작 직후 모두 종료됐다. `--start-episode`가 2 이상이면 프로세스의 첫 에피소드인데도 "장면 재구성" 경로(`episode_idx > 1`)로 들어가 `SimulationContext._timeline`(Isaac Lab 3.0에 없음)에서 멈췄다. 유효한 결과는 두 arm의 에피소드 1(각 2회)뿐이다. runner 두 개를 멈췄다.
- 수정(`[is6 port]`): 재구성은 이 프로세스가 이미 장면을 만든 경우(`_episodes_run > 0`)에만 한다. 첫 에피소드는 빈 stage에서 새로 만든다. 장면 생성 경로는 에피소드 1과 같다.
- 카메라 영상 점검을 추가한다(사전 등록 아님, 진단). 첫 정책 에피소드에서 접촉 캡처가 1.7%로 시연(33%)보다 훨씬 적었고, 손목 fisheye 카메라에서 "OmniLensDistortionOpenCvFisheyeAPI unavailable → fisheyeKannalaBrandtK3 대체" 경고가 있다. 카메라 자세(외부 행렬)는 시연과 0.1 mm, 회전 행렬까지 같음을 확인했다. 영상 자체를 시연 0번 0/80/150 프레임과 비교한다(`closed_loop/camcheck_gtreplay_is6`, hook `IS6_CAM_DUMP_DIR`, 기본 꺼짐).
- 폐루프는 카메라 점검 뒤 v2 runner로 에피소드 2–50을 다시 시작한다(에피소드 1은 기존 첫 기록 사용). 사전 등록 조건은 바꾸지 않는다.

## 2026-11-24 00:27 렌더링 충실도 복구와 폐루프 재시작(유효한 폐루프 결과를 보기 전)
카메라 점검(진단, `closed_loop/camcheck*_gtreplay_is6`, 시연 0번 같은 프레임과 영상 비교)에서 6.2 영상이 시연과 크게 달랐다. 원인과 조치:
1. 테이블 텍스처가 검게 나옴: 앞선 HF 다운로드에서 `--include`를 여러 번 넘겨 마지막 묶음만 적용됐다(텍스처 195개 중 1개만 받음; iTHOR 장면 누락도 같은 원인). 195개를 다시 받았다(`download7_textures.sh`). ENVIRONMENT_STATUS의 "텍스처 195개" 기록은 이 시점에서야 사실이 됐다.
2. 영상 잡음: apptainer `--nv`가 드라이버의 NGX 라이브러리(libnvidia-ngx, DLSS-D builtin)를 bind하지 않아 "Failed to create NGX context"(DLSS 디노이저 없음). `run_in_container_ngx.sh`로 두 파일을 bind하니 오류가 사라지고 영상이 깨끗해졌다.
3. 손목 fisheye 시야가 좁음: Isaac Sim 6.2가 OpenCV fisheye 렌즈 스키마(`omni.usd.schema.omni_lens_distortion`)를 기본으로 올리지 않아, Bench2Dex가 폐기된 fisheyeKannalaBrandtK3로 대체했다. 실행 중 활성화로는 스키마가 등록되지 않아, client 시작 인자(`--kit_args "--enable ..."`)로 켰다(`[is6 port]`, `IS6_LENS_EXT=0`으로 끔). 수정 후 스테레오·손목 영상이 시연과 거의 같다(`camcheck3_gtreplay_is6/compare_t0080.png`).
- 이전 폐루프 기록(에피소드 1, 두 arm × 2회, 모두 실패)은 깨진 렌더링에서 나온 것이라 `closed_loop/discarded_broken_render/`로 옮기고 분석에서 뺀다(삭제하지 않음).
- 재시작: v2 runner, NGX 컨테이너, 에피소드 1–50, arm native(port 9311)·zero(9312). 사전 등록 조건(arm, N=50, 지표, 검정, 멈춤 규칙 06:30)은 그대로다. 06:30 규칙 때문에 50개를 다 못 채울 수 있으며, 그 경우 두 arm 모두 끝난 에피소드만 짝지어 분석한다.

## 2026-11-24 00:33 폐루프 재개(에피소드 2부터)
- 00:30 에피소드 2가 clutter 물체 USD 누락(042_adjustable_wrench, 043_phillips_screwdriver; 같은 `--include` 문제)으로 장면 생성에서 멈췄다. 두 runner를 멈췄다(에피소드 3은 중간에 끊겨 결과 없음, 촉각 로그에 부분 기록만 남음).
- 100개 anchor 파일이 참조하는 물체 USD를 전부 점검했다. 빠진 것은 위 두 개뿐이었고(anchor 6개가 사용), 받았다. 배경 61개·텍스처 195개·나머지 물체 44개는 있다.
- 촉각 로그 줄마다 에피소드 번호(`ep`, runner가 `IS6_EPISODE_INDEX`로 넘김)를 기록하게 했다. 분석은 `ep`가 있는 줄을 우선 쓴다. `ep`가 없는 기존 줄은 첫 구간만 에피소드 1로 쓰고, 중단된 에피소드 3의 부분 구간은 버린다.
- 에피소드 1 결과(두 arm 모두 고친 환경에서 완주)는 유지하고, v2 runner로 에피소드 2–50을 이어 돌린다. 조건은 같다.

## 2026-11-24 00:36 사후 점검(사전 등록 아님): 접촉/비접촉 비의 분해
- 정규화 비 = 원 오차 비 / noise 바닥 비. noise 바닥(seed 변경 NMSE)의 접촉/비접촉 비: 작업 27 0.966 [0.909, 1.027], 작업 26 0.941 [0.79, 1.11].
- 원 오차 비(INT4, 작업 27): X+T 1.005 [0.987, 1.023], X 1.008, DiT linear 0.986, VLM linear 0.992. 작업 26: 디코더 은닉 1.056 [1.010, 1.110], X+T 1.04 [0.99, 1.09].
- 정규화 비의 약 5% 상승은 주로 분모에서 온다. CONTACT_SENSITIVITY.md §3에 반영.

## 2026-11-24 03:21 폐루프 종료, 작업 27 전파 분석 시작
- 폐루프: 두 arm 모두 에피소드 1–50을 마쳤다(native 03:12, zero 03:20). arm마다 50행이고, 에피소드 번호 중복과 error가 없다. 사전 등록한 멈춤 규칙(에피소드 50 또는 06:30) 가운데 에피소드 50으로 끝났다. 무결성 점검 출력에 성공 횟수(native 29, zero 21)가 함께 찍혔다. 분석은 사전 등록대로(짝 McNemar, 짝 bootstrap) `scripts/ci23_analyze_closed_loop.py`로 하고, 조건은 바꾸지 않는다.
- 전파 분석 작업 27(22:56 사전 등록, GPU 경쟁으로 미뤄 둠)을 지금 시작한다. 조건은 작업 26과 같다(시험 시연 0–7, 40 프레임 간격, 고정 scale 보정은 시연 8–49 중 12개 × 6 프레임). log: `logs/prop_task27.log`.

## 2026-11-24 04:20 작업 27 전파 분석 결과(측정, 22:56 사전 등록 조건 그대로)
- 03:22–04:15 실행, 오류 없음. 관측 164개(시연 0–7; 접촉 59 = onset 21 + sustained 38, 비접촉 78, release 27). `prop/task27/prop_summary.json`.
- 촉각 기여 ‖T‖/‖X‖ 접촉/비접촉(시연 내) 1.01 [0.97, 1.05].
- INT4·INT3 접촉/비접촉 비(정규화): 모든 위치의 구간이 1을 포함한다(0.96–1.04). INT2에서는 T 1.13 [1.00, 1.26], DiT linear step 1–3 1.12 [1.04, 1.22], 대조 step 0 DiT 1.05 [1.01, 1.11], X+T 0.99.
- r*(ε = 0.25): T 0% 대 0%, X+T 75% 대 75%(동적 INT8, 평균 차 −0.001 [−0.031, +0.023]), 디코더 75% 대 75%. 디코더 INT8 보호의 평균 r* 차이만 +0.039 [+0.015, +0.070]인데, 작업 26은 −0.12 [−0.22, −0.01]로 반대 방향이고 BF16 보호에서는 −0.009 [−0.024, +0.006]이다.
- 상위 32 channel 겹침: X+T 0.73 / 0.74 / 0.73, T 0.91 / 0.91 / 0.91(접촉–접촉 / 비접촉–비접촉 / 접촉–비접촉).
- 판정: 가설 A/B/C 모두 기각(작업 26과 같다).
- 00:36과 같은 사후 분해(정규화 전 원 오차 비, INT4): X+T 1.001 [0.979, 1.024], 디코더 1.022 [1.007, 1.037], 속도 1.052 [1.003, 1.106], noise 바닥 비 1.009 [0.951, 1.076]. 작업 26에서는 촉각이 닿지 않는 대조 위치도 원 오차 비가 1을 벗어났다(VLM 조건 1.045 [1.005, 1.106], step 0 DiT 0.920 [0.878, 0.960]). 그래서 디코더의 2–6% 차이는 관측 변화 수준으로 본다. 그림 `figures/figS4_prop_two_tasks.png`, 수치 `prop/two_tasks_ratios.json`(`scripts/ci23_prop_two_tasks_figure.py`).
