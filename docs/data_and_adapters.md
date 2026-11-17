# 데이터 형식과 모델·환경 연결

## Pickle-free NPZ 데이터

`python -m vla_eval offline`은 NumPy NPZ를 읽습니다. 경로는 실행 디렉터리 기준입니다. `M`은 도착 snapshot 수, `Hmax`는 저장된 expert action 길이, `D`는 실제 controller action 차원입니다.

| 키 | shape / dtype | 의미 |
|---|---|---|
| `episode_id` | `[M]` Unicode string | episode 단위 split 및 통계 단위 |
| `split` | `[M]` string | `train`, `val`, `test` |
| `state` | `[M,F]` float | 실제 도착 시 센서 feature. Small 입력 |
| `forecast_state` | `[M,F]` float | 도착 전에 예측한 feature |
| `actions` | `[M,Hmax,D]` float | 도착 후 expert 정답. 평가/훈련 target 전용 |
| `action_mask` | `[M,Hmax]` bool | 유효 action의 연속 prefix. Padding은 점수에서 제외 |
| `instruction` | `[M]` string, 선택 | task 언어 |
| `forecast_source` | `[M]` string | 아래 provenance 중 하나 |
| `forecast_generation_ms` | `[M]` float | 미래 feature/영상 생성 비용. 예측/렌더링은 필수 |
| `image__CAMERA` | `[M,T,height,width,3]` uint8, 선택 | 실제 관측 영상 |
| `forecast_image__CAMERA` | 위와 같음 | 도착 전에 사용 가능한 영상. 자동 대체 없음 |
| `native_state__KEY` | `[M,T,Dkey]` float, 선택 | 실제 GR00T modality별 proprioception |
| `forecast_native_state__KEY` | 위와 같음 | 예측한 modality별 proprioception |

단일 이미지도 T=1로 저장하세요. Native adapter는 checkpoint가 요구하는 카메라 이름·T·state key와 action key 순서를 사용합니다. `actions`의 D는 해당 순서로 이어 붙여야 합니다. 관절 위치, 관절 속도, EEF delta, 절대 EEF pose를 혼용하면 안 됩니다. 이미지 정규화는 모델의 공식 processor에 맡깁니다.

`forecast_source`는 `stale_observation`, `predicted_observation`, `rendered_prediction` 중 하나입니다. Loader는 이 표식을 검사하지만 수집 과정의 진실성까지 증명하지 않습니다. 예측/렌더링에 실제 미래 상태를 사용하면 정보 누출입니다. Stale image와 예측 proprioception을 결합했다면 수집 metadata에 각각의 출처와 시간 cutoff를 남기세요.

각 episode의 reaching 시점 관측과 목표 pre-grasp 상태로 forecast를 구성하고, 이후 실제 도착 관측과 정답 chunk를 기록하세요. 실제 도착 오차는 평가용 metadata로만 보존하고 forecast 생성에 넣지 않습니다. 데이터 생성에 사용한 모델·renderer·시간·task·action 단위·좌표계를 함께 보존해야 합니다.

현재 offline 실행은 각 snapshot의 첫 chunk를 비교합니다. 여기서 K는 점수를 계산하는 prefix 길이이며 실제 재관측/환경 전이는 없습니다. K가 성공률에 미치는 효과는 closed-loop 실행에서 측정합니다.

## GR00T 실험적 연결

`configs/gr00t_offline.example.json`은 N1.7 adapter 형식의 예시이며, 실제 모델 선정은 보류 중입니다. 선택한 upstream 환경에 GR00T 자체를 설치하고 checkpoint를 준비해야 합니다. 이 저장소는 이를 자동으로 설치·다운로드하지 않습니다.

1. `heavy.model_path`, `heavy.embodiment_tag`, `language_key`, `dataset`을 실제 환경에 맞춥니다.
2. 첫 검증은 `methods: ["A"]`, 기본 학습 H, 기본 N으로 수행합니다. 이후 공식 sampler와 동일 noise에서 출력이 맞는지 검증하세요. API source 기반 adapter는 checkpoint 검증을 마치지 않았습니다.
3. `train-small`로 train split에 대해 C prior를 만들고 joint B/C Ridge를 학습합니다. `small_checkpoint`는 이때 새로 저장한 경로로 지정합니다.
4. Val에서 설정을 정한 뒤 `offline`을 test에 적용합니다. `allow_short_horizon`은 기본 false입니다. True는 길이 변경 실험을 허용한다는 뜻이며 정확도나 공식 지원을 보증하지 않습니다.

```powershell
python -m vla_eval train-small --config configs/gr00t_offline.example.json --output checkpoints/small.npz
python -m vla_eval offline --config configs/gr00t_offline.example.json --output results/gr00t_offline_001
```

현재 adapter는 batch=1, 기본 Euler schedule, 기본 noise, RTC 미사용 조건입니다. 최신 이미지로 resume할 때 VLM과 state feature를 다시 만듭니다. 저장하는 latent는 normalized action 공간이며 최종 decoder는 실제 도착 state를 사용합니다. GR00T의 다른 버전이나 별도 guidance/RTC sampler와의 호환성을 추정하지 않습니다.

## 사용자 backend 연결

`backend: "my_adapter:make_bundle"`로 지정한 함수는 `(heavy, small, env_factory)`를 반환합니다. `env_factory(seed, env_config)`는 각 trial마다 같은 도착 상태를 재현하는 독립 환경을 만들어야 합니다. 구체 계약은 `vla_eval/contracts.py`를 확인하세요.

Heavy는 `predict`, `prefill`, `resume`, `synchronize`를 구현합니다. 모델에 독립적인 flow를 연결하려면 `EulerFlowPolicy(engine)`를 사용할 수 있습니다. Engine은 다음 기능을 제공합니다.

| 메서드 | 반환 |
|---|---|
| `condition(observation, horizon)` | 해당 관측에서 인코딩한 조건 |
| `noise(horizon, seed, condition)` | `[1,H,D]` latent. 실제 생성 길이를 줄임 |
| `velocity(latent, t, condition)` | 같은 shape의 flow velocity |
| `decode(latent, condition)` | controller 형식 `[H,D]` NumPy action |
| `synchronize()` | GPU 등 비동기 연산 완료 대기 |

Small은 `predict(obs,H,N,seed)`, `correct(obs,prior,forecast_obs)`, `synchronize()`를 구현합니다. Correction은 residual만 반환하지 말고 **보정이 끝난 `[H,D]` action**을 반환해야 합니다. Prior와 실제 action의 좌표계를 먼저 맞추세요.

환경의 `forecast()`는 실제 도착 관측을 읽지 않고 예상 관측을 반환해야 합니다. `observe()`는 최신 센서, `step(action)`은 controller 한 timestep 실행, `advance(seconds)`는 추론 대기 동안 외부 세계와 제어기의 동작을 모델링합니다. 실제 하드웨어에서는 이미 경과한 wall time을 중복 대기하지 않도록 구현하세요. Hold-last/hold-position 등 대기 중 제어 정책도 기록해야 합니다.

`metrics()`는 최소 `success`와 `environment_time_s`를 반환합니다. 위치 오차가 있으면 `position_error_mm`, 회전 오차가 있으면 `rotation_error_deg`를 사용하세요. 측정할 수 없는 지표에 임의의 0을 넣지 않습니다. `done()`은 성공·실패·시간/step 한도에서 종료되어야 하며, `close()`가 있으면 실행기가 호출합니다.

PIM 이전에는 GPU 모델의 `predict`를 그대로 baseline으로 남기고, 별도의 engine/backend를 구현해 동일한 환경과 action 형식에서 비교할 수 있습니다.
