# RELATED_WORK — 선행 연구 비교와 중복 위험

작성 2026-11-24.
- VLA 관련 2026년 논문은 웹 검색 결과(초록, 요약)로만 확인했고 본문은 읽지 않았다. 해당 항목에 "(검색 확인)"으로 표시한다.
- 가속기와 양자화 고전 논문은 기존 지식으로 인용한다.
- DeCAL은 사용자가 준 PDF와 공식 코드를 읽었다.

## 1. 촉각 VLA

| 연구 | 촉각 주입 | 이번 연구와의 관계 |
|---|---|---|
| DeCAL (arXiv 2609.09119) | Mixture-of-Transformers. 이해 layer마다 촉각 cross-attention을 residual로 더한다. contact-aware gate(전역 촉각 token에서 샘플당 scalar)로 비접촉 phase의 촉각을 억제한다. 공개 코드 기본 설정에서는 gate가 꺼져 있다(코드 확인) | 가중치가 없어 실행하지 못했다. VLM 내부 주입 구조이므로, 이번 결론(정책이 촉각을 VLM 밖에서 받음)을 그대로 옮길 수 없다 |
| TacVLA (arXiv 2603.12665, 검색 확인) | contact-aware gate. 접촉이 감지될 때만 촉각 token을 활성화 | gate는 정보 주입 제어이지 정밀도 제어가 아니다 |
| AT-VLA (arXiv 2605.07308, 검색 확인) | 촉각 gate + adaptive cross-attention(gate가 켜질 때만 query를 촉각 token으로) | 같음 |
| ForeTac-VLA (arXiv 2609.20980, 검색 확인) | 촉각 예측 기반 | 같음 |
| Bench2Dex GR00T N1.5/N1.7 tactile | action head DiT 출력 뒤 gate 없는 residual(코드 확인) | 이번 실험 대상 |

## 2. VLA 양자화

| 연구 | 핵심 | 이번 연구와 겹치는 부분 |
|---|---|---|
| QVLA (ICLR 2026, arXiv 2602.03782, 검색 확인) | action 공간 민감도에 따른 channel별 bit 할당. OpenVLA-OFT LIBERO에서 VRAM 29.2%로 성능 98.9%, 1.49배 | channel 단위 중요도 기반 혼합 정밀도는 이미 VLA에서 발표됐다 |
| DyQ-VLA (arXiv 2603.07904, 검색 확인) | 시간에 따라 민감도가 크게 변한다는 관찰. 실시간 운동학 지표로 activation 정밀도를 바꾼다(가중치는 INT4 고정). 메모리 30.9%, 성능 99.5%, 1.49배(버전마다 수치가 다름) | **"상태에 따라 정밀도를 동적으로 바꾼다"는 아이디어 자체가 선점됐다.** 접촉 인지 정밀도는 그 전환 신호를 접촉으로 바꾼 특수한 경우다 |
| QbD-VLA (arXiv 2610.38661, 검색 확인) | 폐루프 성공률로 bit를 고르는 혼합 정밀도(학습·보정 불필요). 가중치 약 4비트, activation 6비트 미만. 균일 W4A4는 LIBERO-Spatial 0%, 선택 할당은 97–98.5%. 실로봇에서는 정책에 따라 실패 | 오프라인 민감도가 폐루프를 잘 예측하지 못한다는 관찰은 우리 LIBERO 결과와 같다. 폐루프 기반 할당은 이미 있다 |
| SQAP-VLA (arXiv 2509.09090, 검색 확인) | 양자화와 pruning을 함께 | 간접 |

## 3. LLM 양자화와 outlier channel

- LLM.int8()(NeurIPS 2022): outlier channel을 16비트로 분리한다.
- SmoothQuant(ICML 2023), AWQ(MLSys 2024), GPTQ(ICLR 2023): activation과 가중치의 크기 기반 처리다.
- Atom(MLSys 2024): W4A4 + INT8 outlier channel 융합 kernel.
- QServe(MLSys 2025): W4A8KV4.

이번과 앞선 캠페인에서 쓴 중요도 점수(|x|·‖W 열‖)는 이 계열과 같다. VLA 고유 신호(접촉, 촉각)는 이 점수를 바꾸지 않았다.

Diffusion의 timestep별 양자화(Q-Diffusion ICCV 2023, PTQ4DM CVPR 2023, TDQ NeurIPS 2023)는 flow step마다 민감도가 다르다는 이번 관찰(마지막 step만 민감)과 같은 계열이다.

## 4. 혼합·동적 정밀도 가속기

| 연구 | 내용 |
|---|---|
| BitFusion (ISCA 2018) | 층별 가변 bit fusion |
| OLAccel (ISCA 2018) | 4비트 + 약 3% 고정밀 outlier, 가중치·activation 모두 |
| DRQ (ISCA 2020) | 입력 영역 민감도에 따른 동적 혼합 정밀도 |
| ANT (MICRO 2022) | 적응형 수 형식 |
| OliVe (ISCA 2023) | outlier-victim pair 부호화 |
| MicroScopiQ (ISCA 2025) | outlier 인지 microscaling |

"중요 channel과 대응 가중치만 고정밀, 나머지 4비트" 구조와 "입력에 따른 동적 정밀도"는 모두 이 계열에서 다뤄졌다.

## 5. 이미 해결된 부분

1. VLA의 channel 단위 혼합 정밀도(QVLA)
2. VLA의 시간 축 동적 정밀도 전환(DyQ-VLA)
3. 폐루프 기반 bit 할당(QbD-VLA)
4. outlier channel + 대응 가중치 고정밀 HW(OLAccel, OliVe, MicroScopiQ, Atom)
5. timestep별 정밀도(diffusion 양자화)

## 6. 새로운 contribution 후보(이번 결과 기준)

| 후보 | 근거 | 평가 |
|---|---|---|
| 접촉 인지 동적 정밀도 | 측정에서 기각(접촉이 민감도·r\*·중요 channel을 바꾸지 않음) | 성립하지 않음 |
| "촉각 VLA에서도 정밀도 요구는 위치(출력 경로, 마지막 flow step)가 정한다"는 부정적 증거 | 측정(두 작업) | 학술적 부정 결과로는 의미가 있다. 하지만 HW contribution은 아니다. 촉각을 VLM 밖에서 받는 한 정책에 한정된다 |
| VLM 내부 촉각 주입(DeCAL형)의 정밀도 전파 | 미측정(가중치 없음) | 여전히 열린 질문이다. 가중치가 공개되거나, 직접 학습하는 비용(명세상 금지)을 감수해야 한다 |

## 7. 중복 위험

- "동적 정밀도 VLA": **높음**(DyQ-VLA).
- "channel 쌍 고정밀 HW": **높음**(OLAccel, OliVe, MicroScopiQ, Atom).
- "접촉 신호를 정밀도 전환에 쓰기": 선행이 직접 있지는 않다(검색 범위 안에서). 그러나 이번 측정에서 신호 자체가 없어, 새로움을 주장할 근거가 없다.
