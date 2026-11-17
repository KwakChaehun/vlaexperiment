# VLA–PIM 정밀 조작 평가 준비

**처음 읽을 문서:** [지금까지의 대화·연구 결정·전체 결과·후속 작업 인수인계](docs/RESEARCH_HANDOFF.md). SSH GPU 서버에서 이어 실행하는 방법은 [GPU runbook](docs/SSH_GPU_RUNBOOK.md), Git에 포함된 결과 스냅샷은 [reports/](reports/)를 참고하세요.

**11/17 후속 평가:** H={32,16,8,4}, K={16,8,4,2}의 유효 13개 조합(나머지 3칸 N/A)과 prior의 이미지·state·재보정 방식 진단을 실행합니다. [새 결과 보고서](results/hk_matrix_20261117/RESULTS.html), [설정](configs/smolvla_hk_matrix.json), [실험 설계](docs/horizon_experiment.md)를 확인하세요. CPU 실제 모델 평가이며 이전 batch와 별도 보관합니다.

**첫 로컬 실제 평가 완료:** [결과와 진행 상태](docs/RESULTS.md). SmolVLA-LIBERO의 선행 계산·보정 진단과 실제 LIBERO H/K rollout을 완료했습니다. CPU pilot이며 RTX 3090/π0.5/GR00T 검증과 구분합니다.

Reaching 중 선행 추론이 도착 후 정확도에 도움이 되는지, action 생성 길이와 실행 길이가 정확도·지연에 어떤 영향을 주는지 검증하기 위한 코드입니다. **실제 SmolVLA-LIBERO의 CPU pilot 결과가 저장되고 있습니다.** 아래 기본 CLI 설정은 CPU용 수치 제어 fixture이며 실제 모델 결과와 구분합니다. 목표 GPU 평가는 RTX 3090에서 진행합니다.

## 연구 방향 업데이트: 2026-11-16

주요 연구 질문은 **Heavy VLA의 조작 능력을 유지하면서, 최신 관측이 필요한 연산만 자주 갱신할 수 있는가**입니다. 완성한 미래 action의 위치 보정은 초기 검증용 가설로 남기고, 계산 재사용·최신 관측 기반 refinement·HW 자원과 deadline을 고려한 선택을 주된 설계 대상으로 둡니다.

평가를 두 갈래로 나눕니다. 먼저 B–C와 대조군으로 Heavy prior의 가치를 확인합니다. 별도로 같은 Heavy의 fresh 재생성 A와 `resume_only`를 비교해 Small 보정 없이 부분 추론 재사용이 성립하는지 봅니다. **C의 이득 부재가 Heavy 재사용까지 반증하지는 않습니다.**

```powershell
python -m vla_eval run --config configs/prior_value_gate.json
python -m vla_eval run --config configs/reuse_probe.json
```

두 설정도 현재는 CPU fixture입니다. 첫 설정의 `latency_mode=zero`는 우선 품질 차이를 분리하기 위한 조건입니다. 지연까지 비교할 때는 같은 설정의 `wall` 조건을 별도로 실행하세요. Offline 평가는 원래 측정 지연을 기록하며 이 옵션으로 지연을 제거하지 않습니다.

첨부 논문 중 V-AEFusion은 stale 조건에서 시작한 denoising을 fresh 조건으로 이어가므로 직접 비교 대상입니다. REACT/RAVEL과 adaptive tail reuse도 함께 구분해야 합니다. **현재 코드는 arrival 시점의 한 번 재사용만 지원하며, rolling buffer·HW-aware scheduler·GPU concurrent execution·NPU/PIM 실행은 아직 구현하지 않았습니다.** 선행연구와의 관계와 하드웨어 비교 계획은 [docs/research_direction.md](docs/research_direction.md)에 정리했습니다.

## 오늘 실행할 수 있는 것

Python 3.10 이상과 NumPy가 필요합니다. 실제 모델 환경은 해당 모델의 Python·PyTorch 요구사항에 맞춰 별도로 만드세요.

```powershell
python -m pip install -e .
python -m vla_eval doctor
python -m unittest discover -s tests -v
python -m vla_eval run --config configs/comparison.json
python -m vla_eval run --config configs/horizon_sweep.json
```

현재 이 PC에서는 PATH의 `python`이 실행되지 않는 Windows 앱 별칭이어서 다음 실제 실행 파일로 검증했습니다. 다른 머신에서는 그 머신의 Python을 사용하면 됩니다.

```powershell
$pythonExe = 'C:\Users\Administrator\AppData\Local\Python\pythoncore-3.14-64\python.exe'
& $pythonExe -m vla_eval doctor
& $pythonExe -m unittest discover -s tests -v
```

## 실험 1: A/B/C/D

| 방식 | Reaching 중 | 실제 도착 후 첫 추론 | 이후 재계획 |
|---|---|---|---|
| A: Heavy 기준선 | 없음 | Heavy + 최신 관측 | Heavy |
| B: Small-only | 없음 | Small + 최신 관측 | Small |
| C: Heavy + Small | 예상 도착 관측으로 완성한 action prior | Small residual 보정 | Small |
| D: Partial Heavy + Small | flow latent `x_t`, 완료 step, 전체 step 저장 | 최신 관측을 다시 인코딩하고 남은 Heavy step + Small 보정 | Small |
| resume_only: Heavy 재사용 진단 | D와 같은 중간 latent | 최신 관측으로 남은 Heavy step, Small 미사용 | Heavy |
| prior-only 대조군 | C와 같은 prior | 보정 없이 실행 | Small |
| shuffled-prior 대조군 | 다른 trial의 prior | C와 같은 Small 보정 | Small |

선행 계산의 개입 범위는 **도착 직후 첫 chunk**입니다. 이후에도 계속 Heavy를 선행 계산하는 비동기 시스템은 별도 실험입니다. A는 실제 이미지로 추론하는 강한 기준선이며, 성공률의 수학적 상한을 보장하지 않습니다.

동일 seed로 만든 도착 상태를 각 방식에서 다시 시작합니다. 기본 시나리오는 예상 상태 일치, 작은/큰 도착 오차, 움직이는 목표입니다. 성공률·위치/회전 오차·첫 액션 지연·재계획 지연·사용/폐기 action 수를 기록합니다. 실제 환경에서는 접촉 실패·충돌·완료 시간을 함께 반환할 수 있습니다.

지연은 CUDA 동기화 후 측정합니다. Reaching과의 중첩은 `max(0, 선행 계산 시간 - reaching 시간) + 도착 후 계산 시간`으로 계산합니다. **현재 실행기는 순차 측정과 중첩 시간 회계이며, GPU 동시 실행을 구현하거나 입증하지 않습니다.** `latency_mode=zero`는 지연을 제거한 정확도 평가, `wall`은 측정 지연 반영, `model`은 명시한 가상 지연 반영입니다.

## 실험 2: 생성 H / 실행 K / flow step N

- `H`: 실제 생성하는 미래 action timestep 수.
- `K`: 그중 실행한 뒤 다시 관측하는 timestep 수. `K <= H`만 실행합니다.
- `N`: flow ODE 적분 step 수. action timestep 수와 다릅니다.

`configs/horizon_sweep.json`은 H와 K를 독립적으로 바꾸고 N도 교차합니다. 동일 모델에서 H를 고정하고 K만 바꾸는 실험으로 먼저 피드백 빈도 효과를 확인한 뒤, 실제 action tensor 길이를 줄여 H의 영향을 측정하세요. Flow matching은 일반적으로 chunk 전체를 함께 갱신하므로, autoregressive token 조기 종료와 구분해야 합니다. 실제 모델의 학습 H와 다른 길이는 분포 밖 조건일 수 있습니다.

첫 실험에서는 N을 고정한 `configs/horizon_accuracy.json`과 `configs/horizon_realtime.json`을 사용합니다. 각각 추론 지연을 제거한 품질 평가와 측정 지연을 반영한 평가입니다. `horizon_comparisons.json`이 H 효과·K 효과·두 축의 결합 효과를 나누어 paired 비교하고, 실행 action당 비용도 기록합니다. **짧은 H/K의 정확도 이득은 PIM의 동기이며, PIM 우위는 실제 병목과 전송 비용을 포함한 별도 비교가 필요합니다.** 자세한 비교표와 판단 기준은 [docs/horizon_experiment.md](docs/horizon_experiment.md)에 있습니다.

실제 1차 평가 조합은 **LIBERO(MuJoCo/robosuite) + 공식 π0.5-LIBERO checkpoint + `physical-intelligence/libero` 데모**로 선정했습니다. Spatial부터 closed-loop로 평가하고 다른 suite로 확장합니다. 이 모델의 공식 H는 10이므로 실제 실험은 이에 맞추며, 아래 CPU 설정의 H=16과 구분합니다. 실제 adapter 연결·설치·GPU 검증은 아직 수행하지 않았습니다.

```powershell
python -m vla_eval run --config configs/horizon_accuracy.json
python -m vla_eval run --config configs/horizon_realtime.json
```

기본 toy Heavy는 해석적 제어기여서 N에 따른 학습 모델의 품질 변화가 없습니다. 이 sweep 결과는 실험 조합·로그·실행 루프를 검증합니다.

## Offline 데이터로 먼저 선별하기

일부 episode/sample만 로컬에서 평가할 수 있습니다. 실제 LIBERO episode 0에서 관측 16개를 추린 `data/libero_subset.npz`도 준비했습니다. 현재 실제 SmolVLA pilot은 별도의 6개 episode/24개 관측을 사용합니다. 데이터 일부 다운로드, CLI 제한 옵션, GPU에서의 native π0.5 진단 방법은 [docs/local_subset.md](docs/local_subset.md)를 참고하세요. π0.5 checkpoint 추론은 아직 실행하지 않았습니다.

```powershell
python scripts/make_fixture_dataset.py --output data/fixture.npz
python -m vla_eval train-small --config configs/offline_fixture.json --output checkpoints/fixture_small.npz
python -m vla_eval offline --config configs/offline_fixture.json
```

작은 CPU 검증은 `python -m vla_eval offline --config configs/offline_fixture.json --max-episodes 2 --max-samples-per-episode 1`로 실행할 수 있습니다.

마지막 명령은 앞서 학습한 `checkpoints/fixture_small.npz`를 Small로 사용합니다. Ridge는 관측 벡터와 이미지의 4×4 색상 격자를 입력으로 쓰는 저비용 출발점입니다. 언어를 처리하지 않으므로 여러 task의 의미를 구분해야 하는 실험에는 적합한 Small 모델을 연결해야 합니다. B와 C에 **같은 가중치**를 사용하도록 prior 유무를 함께 학습하며, 정식 Small VLA로 간주하지 않습니다.

데이터의 전체 episode를 train/val/test로 분리하고, test에서 prior나 Small을 학습하지 않습니다. Offline 실행은 실제 도착 관측을 읽기 전에 forecast 추론을 수행합니다. forecast 이미지가 없으면 실제 도착 이미지를 대신 쓰지 않고 오류를 냅니다. 데이터 형식은 [docs/data_and_adapters.md](docs/data_and_adapters.md)에 있습니다.

Offline 결과는 expert action MSE입니다. **실제 성공률·충돌·복구 능력은 closed-loop rollout에서 검증해야 합니다.** K가 다른 MSE는 비교 구간도 달라지므로 `first_action_mse`를 공통 구간 지표로 함께 보세요. 독립 episode 단위로 묶은 paired bootstrap을 제공합니다.

## GPU를 연결한 뒤

실제 모델·체크포인트·환경은 논문 검토와 기본 추론 재현 후 확정합니다. RTX 3090에서 batch=1로 시작하고, 모델 로드 → 기본 H 추론 → B–C 비교 → 같은 Heavy의 재사용 비교 → H/K/N sweep → 하드웨어 비교 순으로 진행합니다. 모델과 이미지 설정이 정해지기 전에는 메모리 적합성을 보장하지 않습니다.

```powershell
python -m vla_eval doctor
python -m vla_eval profile-gemm --horizons 1 2 4 8 16 32 --output results/gemm.json
```

`profile-gemm`은 GPU의 thin-GEMM 지연과 이론적 arithmetic intensity를 기록합니다. 반복 연산은 캐시 효과를 받으며, 이 값만으로 DRAM memory-bound나 PIM 가속을 주장할 수 없습니다. 실제 모델에는 다음 명령으로 shape와 연산별 profiler trace를 남길 수 있습니다.

```powershell
python -m vla_eval profile-policy --config configs/gr00t_offline.example.json --output results/policy_profile
```

GR00T N1.7 API를 위한 실험적 어댑터가 [vla_eval/gr00t.py](vla_eval/gr00t.py)에 있습니다. 공식 코드를 읽고 연결 지점을 작성했으며, **체크포인트 실행 및 공식 sampler와의 수치 일치 검증은 아직 하지 않았습니다.** 이 예시는 모델 선택을 의미하지 않습니다. 다른 GR00T 버전은 버전별 어댑터가 필요합니다. 기본 H만 허용하며, 짧은 H는 `allow_short_horizon=true`를 명시해야 합니다. decoder가 해당 길이를 지원하지 않으면 중단합니다.

## 결과 파일

실행마다 새로운 `results/…` 디렉터리를 만듭니다. 기존 결과를 덮어쓰지 않습니다.

| 파일 | 내용 |
|---|---|
| `manifest.json` | 설정, 코드 revision, 패키지 버전, 실행 당시 로드된 CUDA 정보 |
| `trials.jsonl` | trial별 원자료와 closed-loop call별 지연 |
| `trials.csv` | 표 형식 원자료 |
| `summary.json` | 성공률과 Wilson 95% 구간, 평균/p50/p95 |
| `paired_comparisons.json` | C−B / D−B, resume_only−A / D−A 등의 paired 차이와 95% 구간 |
| `horizon_comparisons.json` | `horizon_analysis=true`일 때 H·K 축별 paired 차이와 95% 구간 |
| `FAILED.txt` | 중단된 실행의 오류. 미완료 결과를 완성 결과로 취급하지 않음 |

연구 판단 기준과 남은 준비 사항은 [docs/experiment_protocol.md](docs/experiment_protocol.md)에 정리했습니다.
