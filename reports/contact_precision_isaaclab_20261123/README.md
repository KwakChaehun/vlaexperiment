# 촉각 VLA의 접촉-정밀도 요구 평가 (Isaac Sim 6.2, 2026-11-23~24)

- 대상 정책: Bench2Dex GR00T N1.5 tactile.
- 실행 환경: Isaac Sim 6.2 / Isaac Lab 3.0 이식본.

**결론: 가설을 기각하고 이 방향은 중단을 권고한다.** 근거는 다음과 같다.
- 폐루프에서 촉각을 지우면 성공률이 58%에서 42%로 떨어진다(50쌍, McNemar p = 0.077). 즉 촉각은 쓰인다.
- 그러나 접촉은 필요한 수치 정밀도를 바꾸지 않는다. INT4·INT3의 접촉/비접촉 오차 비는 0.96–1.06으로 INT8 바닥과 같고, r\*와 중요 channel도 같다. 작업 26과 27에서 같은 결과가 나왔다.
- 정밀도 요구는 접촉이 아니라 위치가 정한다. 민감한 곳은 action head 출력 경로와 마지막 flow step이다.

## 문서

| 파일 | 내용 |
|---|---|
| [FINAL_ASSESSMENT.md](FINAL_ASSESSMENT.md) | 최종 판단, Q1–Q9 답변, 점수 |
| [CLOSED_LOOP_RESULTS.md](CLOSED_LOOP_RESULTS.md) | 폐루프 50쌍: 촉각 원래대로 대 촉각 0 |
| [TACTILE_INJECTION_AND_PRECISION_PROPAGATION.md](TACTILE_INJECTION_AND_PRECISION_PROPAGATION.md) | 촉각 주입 위치와 정밀도 요구 전파(작업 26·27) |
| [CONTACT_SENSITIVITY.md](CONTACT_SENSITIVITY.md) | Stage 2: 위치별 양자화 민감도와 접촉 효과 |
| [PRECISION_DEMAND.md](PRECISION_DEMAND.md) | Stage 3: 최소 고정밀 비율 r\* |
| [POLICY_FEASIBILITY.md](POLICY_FEASIBILITY.md) | Stage 0–1: 정책 동작과 촉각 ablation |
| [ENVIRONMENT_STATUS.md](ENVIRONMENT_STATUS.md) | 설치, 이식, 실행 방법 |
| [JOINT_WA_RESULTS.md](JOINT_WA_RESULTS.md) | 가중치·activation 결합 양자화(부분) |
| [HARDWARE_ANALYSIS.md](HARDWARE_ANALYSIS.md) | 하드웨어·PIM 분석(추정) |
| [RELATED_WORK.md](RELATED_WORK.md) | DeCAL, DyQ-VLA, QbD-VLA 등과의 비교 |
| [NOTES.md](NOTES.md) | 실험 기록. 보고서가 인용하는 사전 등록 시각("NOTES 21:32" 등)이 여기 있다 |

## 이 스냅숏에 없는 것

결과 보고서, 그림, 실험 기록만 담았다. 다음은 로컬 `results/contact_precision_isaaclab_20261123/`에만 있다(git 제외 폴더).
- 원자료와 요약 JSON(`stage*/`, `prop/`, `closed_loop/`), 로그.
- 분석 스크립트(`scripts/ci23_*.py`)도 아직 커밋하지 않았다.
