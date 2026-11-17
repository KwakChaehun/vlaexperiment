# PIM 시뮬레이터 실행 결과와 다음 평가 범위

후속: [동일 projection에 대한 RTX 3090 직접 비교](../reports/pim_gpu_comparison_20261117/RESULTS.md)를 추가했습니다. 단일 벡터와 다중 벡터, GPU cache/launch 조건을 구분했으며 전체 action expert의 PIM 가속률은 여전히 미검증입니다.

2026-11-17 KST. RTX 3090은 기존 VLA 실행 장치입니다. 이번에는 CPU에서 Samsung의 공개 [PIMSimulator](https://github.com/SAITPublic/PIMSimulator)를 빌드하고 실제 사이클 시뮬레이션을 실행했습니다. 실제 PIM 보드에서 측정한 결과는 아닙니다.

## 실행한 것

- upstream commit: `3703d1f19c8f027360cc33a3243eb271e3bb6898`.
- 공식 기능 검사 `PIMKernelFixture.gemv`: 4096×1024 FP16 GEMV의 출력 비교 4096개 통과, 실패 0개.
- 공식 타이밍 예제 `PIMBenchFixture.gemv`: 4096×4096, 64채널, 채널당 8 PIM block, tCK=1ns. PIM 13,166 cycles = 13.166µs.
- GR00T N1.6 checkpoint의 첫 DiT block에서 행렬 shape만 읽어 Q/K·FFN 단일 벡터 연산 4종을 타이밍 시뮬레이션했습니다. 실제 모델 가중치나 activation은 실행하지 않았습니다.

| 연산 | 모델 shape [out,in] | 시뮬레이터에 넣은 패딩 shape | 사이클 | 시뮬레이션 시간 |
|---|---:|---:|---:|---:|
| Q projection | 1536×1536 | 4096×1536 | 5,147 | 5.147µs |
| K projection | 1536×2048 | 4096×2048 | 6,970 | 6.970µs |
| FFN up | 6144×1536 | 8192×1536 | 10,277 | 10.277µs |
| FFN down | 1536×6144 | 4096×6144 | 19,699 | 19.699µs |

출력 패딩 단위는 64 channels × 8 blocks × 8 GRF_B = 4096입니다. 입력은 두 bank parity의 256-element 단위에 맞췄습니다. 이 비효율을 숨기지 않았습니다. 채널 수를 바꾸거나 여러 토큰을 병렬 배치하면 다른 결과가 나올 수 있습니다.

**위 µs는 한 벡터의 커널 타이밍이며 action expert 전체 추론 시간이 아닙니다.** 가중치가 이미 PIM에 있다고 가정하고, dummy input을 사용합니다. 가중치 최초 적재, GPU↔PIM 연결, 명시적 host 결과 readback, attention softmax, 정규화·activation, 전체 토큰 시퀀스 및 flow 반복을 포함하지 않습니다. 원본 checkpoint는 F32 저장이고 GPU 평가는 BF16 실행이며, 해당 PIM 커널은 FP16 기반입니다. 모델 출력 정확도 일치도 아직 검사하지 않았습니다.

공식 예제 로그의 `Speed-up : 2.74054`는 **RTX 3090 대비 2.74배라는 뜻이 아닙니다.** 소스를 확인하면 non-PIM 비교기는 가중치·입력 read와 출력 write의 memory traffic만 계산하며 GPU 연산을 모델링하지 않습니다. 따라서 이 값을 기존 31.19ms action-expert 시간에 나눠 적용하면 안 됩니다. `shape_run.log`의 speed-up 값도 같은 제한을 가집니다.

[원자료](../reports/pim_simulator_20261117/shape_summary.json), [공식 예제 로그](../reports/pim_simulator_20261117/smoke.log), [shape별 로그](../reports/pim_simulator_20261117/shape_run.log), [자체 작성한 드라이버](../scripts/run_pim_shape_probe.py).

## 기존 실험과 연결한 판단

1. 접촉·수납 평가의 관측 성공률은 H50/K16=53/60, H50/K4=50/60, H16/K4=47/60입니다. H16/K4 우위는 확인되지 않았습니다. 통계적으로 일반적인 순위가 확정됐다는 뜻은 아닙니다.
2. 추가 20Hz 제어에서는 fresh H50/K16과 긴 미래 예측+25% MLP 보정/K16이 각각 8/12 성공했습니다. 보정은 50ms 대기 구간을 평균 5.9→0.1개로 줄였지만, 함께 성공한 8조건에서 준비 포함 완료는 3.364초 느렸습니다.
3. RTX 3090 단독 profile에서 전체 63.35ms 중 action expert는 31.19ms입니다. 이 부분을 실제로 얼마나 줄일 수 있는지가 하드웨어 질문입니다. 앞선 2배/4배 가속표는 가정에 따른 계산이고, 이번 커널 시뮬레이션으로 검증된 전체 모델 가속률이 아닙니다.

관련 전체 보고서: [접촉 H/K](../reports/contact_hk_20261117/RESULTS.html), [미래 feature](../reports/speculative_features_20261117/RESULTS.html), [K16·실시간·GPU breakdown](../reports/k_pim_followup_20261117/RESULTS.html).

## 전체 action expert 평가에 남은 작업

먼저 실제 forward에서 연산 shape·순서·dtype·의존성을 추출해야 합니다. H=50과 H=16 모두 대상으로 하고 N=1/4 flow를 구별해야 합니다. K는 생성한 action의 실행 개수이므로 동일 H의 한 번 inference shape를 줄이지 않습니다. K의 효과는 재호출 빈도와 제어 결과에 반영합니다.

다음으로 token 단위 GEMV 분해와 다중 토큰 재사용을 비교하고, bank/channel별 배치·padding·용량·activation 이동을 모델링합니다. GPU는 H개의 token에서 weight를 재사용하는 GEMM을 실행하므로 단일 GEMV가 빠르다는 이유만으로 전체 GPU보다 빠르다고 볼 수 없습니다.

PIM에서 지원하는 GEMV를 맡기고 attention softmax·LayerNorm·activation 등을 GPU에 남기는 분할부터 평가할 수 있습니다. 이때 매 layer 경계 전송과 동기화가 생기므로 앞서 사용한 '입력 한번, 출력 한번' 가정을 유지할 수 없습니다. 전체 expert를 PIM에서 돌리는 별도 설계는 비선형 연산 지원·수치 일치까지 구현해야 합니다.

끝으로 측정 GPU 시간과 시뮬레이션 PIM 시간, host link 시간으로 실제 20Hz 제어를 재평가합니다. GPU에서 동일 action을 계산하고 가상 PIM 지연을 주입하는 실험은 제어 지연 효과만 검사하며, PIM 수치 정확도나 실제 GPU 자원 절감을 입증하지 않습니다. FP16 등의 근사로 action이 달라지는 경우에는 기능 모델의 출력으로 접촉 성공률도 다시 평가해야 합니다.

## 재현

시뮬레이터는 `.runtime/PIMSimulator`, 로컬 빌드 도구는 `.runtime/pim-tools`, GoogleTest는 `.runtime/googletest`에 있습니다. 시스템 패키지 설치 없이 로컬에만 설치했습니다. 위 upstream pin과 GoogleTest v1.14.0 (`f8d7d77c06936315286eb55f8de22cd23c188571`)을 사용했습니다. 로컬 Python 환경에는 scons 4.11.1, cmake 4.4.5를 설치했습니다.

새 환경에서는 두 upstream을 해당 revision으로 checkout하고, GoogleTest를 다음처럼 빌드합니다. `PIM_ROOT`는 저장소의 절대 경로입니다.

```bash
PIM_ROOT="$PWD"
.venv/bin/python -m venv .runtime/pim-tools
.runtime/pim-tools/bin/pip install scons==4.11.1 cmake==4.4.5
.runtime/pim-tools/bin/cmake -S .runtime/googletest -B .runtime/googletest/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PIM_ROOT/.runtime/pim-deps"
.runtime/pim-tools/bin/cmake --build .runtime/googletest/build -j 4
.runtime/pim-tools/bin/cmake --install .runtime/googletest/build
```

PIMSimulator의 `Sconstruct`에서 `SetOption('num_jobs', 32)`를 4로 바꾸고, 각 target의 `CPPPATH`에 로컬 `pim-deps/include`, `LIBPATH`에 `pim-deps/lib64`와 `pim-deps/lib` 절대 경로를 추가했습니다. 타이밍·연산 소스는 변경하지 않았습니다. 라이브러리 생성은 생략합니다.

```bash
cd .runtime/PIMSimulator
../pim-tools/bin/scons NO_LIBRARY=1
./sim --gtest_filter='PIMKernelFixture.gemv:PIMBenchFixture.gemv'
cd ../..
.venv/bin/python scripts/run_pim_shape_probe.py --output results/pim_shape_new_run
```

드라이버는 checkpoint header에서 shape를 추출하고, 자체 작성한 `VLAShapeProbe.cpp`를 로컬 simulator에 생성한 뒤 재빌드·실행합니다. 결과가 이미 있는 출력 폴더에는 덮어쓰지 않습니다. Third-party 시뮬레이터 소스나 모델 가중치는 이 저장소에 포함하지 않았습니다.
