# 로컬 일부 데이터 평가

## 지금 준비된 것

공개 `physical-intelligence/libero`에서 **episode 0만** 받아 16개의 관측/action 표본을 만들었습니다. 실제 데이터 파일은 `data/libero_subset.npz`, revision·원본 파일 hash·표본화 방법은 `data/libero_subset.source.json`에 있습니다. 이미지 두 종류는 각각 `[16,1,256,256,3]`, state는 `[16,8]`, expert action은 `[16,10,7]`입니다.

이 파일은 **모델이 추론한 결과가 아닌 입력과 expert 정답**입니다. 공개 원본은 train split이므로 diagnostic 표식을 유지합니다. GPU 추론이나 실제 성공률 평가는 아직 수행하지 않았습니다.

## 데이터 일부만 받기

```powershell
python -m pip install -e ".[data]"
python scripts/download_libero_subset.py --episodes 0 --max-samples-per-episode 16 --horizon 10 --output data/libero_subset.npz
```

이미 생성된 경로에는 덮어쓰지 않습니다. 다른 subset은 output 이름과 episode 번호를 바꾸세요. 지정한 episode Parquet와 작은 metadata만 받으며, 전체 dataset이나 모델 weight는 다운로드하지 않습니다. 기본 source 파일 크기 한도는 합계 64 MiB입니다. 원본 episode 전체 파일을 받은 후, 끝까지 정답 chunk가 있는 위치에서 표본을 균등하게 추립니다. Temporal history와 action label을 임의로 보간하거나 이미지 방향을 바꾸지 않습니다.

데이터를 줄이면 전송량·추론 횟수는 줄어들지만 **VLA checkpoint 전체는 여전히 필요**합니다. Checkpoint를 작은 데이터만큼 잘라 로드할 수 있는 것은 아닙니다.

## 기존 offline 평가에서 episode/sample 수 제한

현재 들어 있는 CPU fixture는 다음 명령으로 즉시 일부만 실행할 수 있습니다.

```powershell
python -m vla_eval offline --config configs/offline_fixture.json --max-episodes 2 --max-samples-per-episode 1
```

이 명령은 실제 VLA 대신 toy controller와 학습된 Ridge를 사용하는 코드 검증입니다. 실제 LIBERO/π0.5 성능 결과로 해석하지 않습니다.

지원 옵션은 다음과 같습니다. CLI 값을 manifest에 보존하며, 모든 method/H/K/N 조건에서 동일한 표본 집합을 사용합니다.

| 옵션 | 역할 |
|---|---|
| `--max-episodes 3` | 선택 split에서 최대 3 episode |
| `--max-samples-per-episode 8` | 각 episode 안에서 최대 8 관측. 제한 시 균등한 위치로 선택 |
| `--sample-stride 5` | 각 episode의 저장된 표본에 먼저 stride 적용 |
| `--episode-ids ID1 ID2` | 특정 episode만 선택. 없는 ID나 다른 split의 ID는 오류 |

선택 로직은 NPZ를 처음부터 streaming하는 기능은 아닙니다. 큰 NPZ의 일부만 평가해도 schema 검증 등에서 전체 action/metadata를 읽을 수 있습니다. 대형 원본의 전송·이미지 메모리를 줄일 때는 먼저 위 importer로 작은 파일을 만드세요.

## 실제 π0.5 일부 관측 추론

공식 openpi 환경과 **로컬 `checkpoints/pi05_libero` checkpoint**를 준비한 뒤 실행합니다. 현재 이 세션에는 CPU PyTorch만 있고 openpi/checkpoint가 없으므로 아래 명령의 실제 VLA 실행은 아직 검증하지 않았습니다. Ubuntu/Linux의 공식 설치 경로를 우선합니다.

```powershell
python -m vla_eval offline --config configs/openpi_subset.example.json --max-episodes 1 --max-samples-per-episode 8
```

현재 예시는 H=10/N=10 고정, A-only의 native sampler 연결입니다. 최대 8개 관측 × K 다섯 조건으로 공통 prefix action MSE와 추론 시간을 기록합니다. 이 adapter는 H를 줄인 것처럼 출력만 잘라 반환하지 않으며, H 변경과 flow resume는 아직 지원하지 않습니다. `allow_train_diagnostics=true`와 `methods=["A"]`를 함께 지정해야 공개 train 관측의 진단을 실행할 수 있고, 결과에도 `diagnostic_only=true`를 기록합니다.

위 연결은 공식 source API에 맞춰 작성하고 단위 계약만 검증했습니다. 실제 checkpoint의 baseline 일치·CUDA 동작은 GPU에서 확인해야 합니다. K가 다른 expert prefix MSE는 feedback 효과나 성공률을 뜻하지 않습니다.

## 소수 task의 closed-loop pilot

GPU에서 첫 pilot은 **LIBERO-Spatial의 1–3 task × 각 5–10개 공식 초기 상태**로 제한합니다. 같은 initial-state/seed와 정책 checkpoint에서 native H를 고정하고 K만 먼저 바꿉니다. 생성 action을 시뮬레이터에 실행해 새로운 관측으로 재계획해야 성공률을 측정할 수 있습니다.

지금 추가한 subset 옵션은 offline snapshot 평가용입니다. LIBERO task 선택·시뮬레이터 rollout 연결을 구현했다는 의미는 아닙니다. Pilot은 실행 경로와 큰 품질 저하를 먼저 확인하기 위한 작은 실험이며, 최종 성능 결론에는 더 많은 task/초기 상태가 필요합니다.

참고: [공개 데이터](https://huggingface.co/datasets/physical-intelligence/libero), [공식 openpi policy 생성 API](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/policy_config.py), [LIBERO rollout 예시](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py).
