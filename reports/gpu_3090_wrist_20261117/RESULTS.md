# RTX 3090 손목 갱신·H/K pilot

갱신: 2026-11-17T15:43:28+09:00. 원자료: `/home/kwak3006/vlaexperiment/results/wrist_gr00t_20261117`.

## 측정 범위

- RTX 3090 24GB, PyTorch 2.9/CUDA 12.8. GR00T N1.6 BF16와 SmolVLA BF16 autocast를 순차 실행했습니다.
- GR00T: 공개 커뮤니티 `0xAnkitSingh/GR00T-N1.6-LIBERO`, revision `d690a226ad06e81736786f56cf879d2ed1dd3f0f`. NVIDIA 공식 N1.6 코드 revision `9b37aa1ce69c73c6d165233fa88128283bba4508`. N1.7은 Cosmos backbone 접근 승인이 없어 실행하지 않았습니다.
- GR00T 공식 sampler 최대 절대 차이 약 3e-9, GPU allocated memory peak 6.34 GiB. SmolVLA 공식 sampler 차이 0.
- GR00T native는 **50개 생성 후 16개 decode**입니다. H=16/8/4 실험은 실제 action latent 길이를 바꿉니다. 재학습하지 않은 H 변경입니다.
- Offline: 공개 LIBERO episode 1274/1269/1285의 12관측. 시연과 checkpoint 학습의 overlap 가능성이 있습니다. NMSE는 동일 std로 정규화한 앞 4개 action 오차이며 성공률이 아닙니다.
- Closed loop: 기본 LIBERO Spatial task 0/1/2 × initial state 0/1, 설정당 6회, 최대 220 action. requested_hk_extra/small_n2_extra는 task 3/4/5입니다. 최신 관측 → H개 생성 → K개 실행 → 실제 시뮬레이터 이미지·state 재관측입니다. 미래 이미지를 예측하거나 oracle 미래 frame을 넣지 않았습니다.
- 시뮬레이터는 추론 중 멈춥니다. 지연은 전처리·vision·VLM·AE·decode를 포함하고 renderer/control은 제외합니다. 실제 지연을 반영한 제어 성공률 또는 sub-mm 정밀도 결과가 아닙니다.
- 여러 실행의 동일 fresh 대조군은 반복 확인이며 성공 횟수를 합쳐 독립 표본 수를 늘리지 않습니다. 6/6의 Wilson 95% 구간도 약 61–100%입니다.

## 요청한 세 조합: 서로 다른 12개 초기 조건

| model | H | K | N | success | p50_ms | p95_ms | amortized_ms |
|---|---|---|---|---|---|---|---|
| GR00T | 50 | 16 | 4 | 12/12 | 64.99 | 71.35 | 4.39 |
| GR00T | 50 | 4 | 4 | 12/12 | 64.66 | 65.2 | 16.38 |
| GR00T | 16 | 4 | 4 | 12/12 | 64.28 | 64.76 | 16.25 |
| SmolVLA | 16 | 4 | 2 | 9/12 | 59.61 | 62.1 | 15.2 |

첫 6개 조건의 천장 효과를 보고 task 3/4/5를 추가한 탐색 결과입니다. 같은 task/initial state를 중복 합산하지 않았습니다. GR00T 12/12의 Wilson 95% 구간은 약 76–100%, SmolVLA 9/12는 약 47–91%입니다. 관측된 성공률 차이는 일반적인 우월성이나 정밀도 개선의 증명이 아닙니다. 정책별 학습 데이터/학습 예산도 통제하지 않았습니다.

H=50→16의 호출 시간 감소는 약 0.6ms에 그쳤습니다. K=16은 재계획 빈도를 줄여 action당 추론 비용을 줄이지만, 이 실험에서는 접촉 정밀도·외란 복구·추론 중 움직이는 환경을 평가하지 않았습니다. 미래 feature 예측 또는 미리 계산한 action의 작은 보정은 이 GPU batch에 포함되지 않았습니다.

## H/K와 작은 모델 전체 비교

| run | model | H | K | N | success | p50_ms | p95_ms | amortized_ms |
|---|---|---|---|---|---|---|---|---|
| run | GR00T | 50 | 8 | 4 | 6/6 | 65.18 | 67.94 | 8.613 |
| run | GR00T | 16 | 4 | 4 | 6/6 | 64.62 | 65.3 | 16.43 |
| run | GR00T | 16 | 16 | 4 | 6/6 | 64.91 | 70.95 | 4.429 |
| run | GR00T | 16 | 8 | 4 | 6/6 | 64.59 | 66.37 | 8.4 |
| run | GR00T | 8 | 4 | 4 | 4/6 | 66.63 | 67.12 | 16.74 |
| run | GR00T | 4 | 4 | 4 | 2/6 | 66.59 | 67.14 | 16.72 |
| run | GR00T | 16 | 4 | 2 | 6/6 | 49.16 | 49.76 | 12.59 |
| small | SmolVLA | 16 | 4 | 10 | 3/6 | 165.8 | 171.6 | 42.06 |
| small | SmolVLA | 16 | 16 | 10 | 4/6 | 166.2 | 172.6 | 11.02 |
| small | SmolVLA | 16 | 8 | 10 | 2/6 | 165.7 | 171.8 | 21.41 |
| small_n2 | SmolVLA | 16 | 4 | 2 | 6/6 | 61.15 | 62.81 | 15.5 |
| small_n2 | SmolVLA | 16 | 16 | 2 | 2/6 | 61.88 | 63.77 | 4.009 |
| small_n2 | SmolVLA | 16 | 8 | 2 | 3/6 | 61.34 | 63.61 | 7.842 |
| small_n4 | SmolVLA | 16 | 4 | 4 | 3/6 | 87.57 | 89.92 | 21.95 |
| small_n4 | SmolVLA | 16 | 16 | 4 | 3/6 | 88.32 | 91.05 | 5.779 |
| small_n4 | SmolVLA | 16 | 8 | 4 | 2/6 | 88.18 | 90.36 | 11.26 |
| requested_hk | GR00T | 50 | 16 | 4 | 6/6 | 64.95 | 71.02 | 4.358 |
| requested_hk | GR00T | 50 | 4 | 4 | 6/6 | 64.6 | 65.07 | 16.42 |
| requested_hk | GR00T | 16 | 4 | 4 | 6/6 | 64.25 | 64.67 | 16.31 |
| requested_hk_extra | GR00T | 50 | 16 | 4 | 6/6 | 65.02 | 71.41 | 4.422 |
| requested_hk_extra | GR00T | 50 | 4 | 4 | 6/6 | 64.68 | 65.33 | 16.34 |
| requested_hk_extra | GR00T | 16 | 4 | 4 | 6/6 | 64.3 | 64.84 | 16.19 |
| small_n2_extra | SmolVLA | 16 | 4 | 2 | 3/6 | 59.5 | 60.8 | 15.03 |

`amortized_ms`는 실제 실행 action당 총 추론 시간입니다. N은 flow 횟수이며 모델 크기와 별개입니다.

## 외부/손목 갱신

| run | mode | H | K | N | success | p50_ms | p95_ms | amortized_ms |
|---|---|---|---|---|---|---|---|---|
| run | both_fresh | 16 | 4 | 4 | 6/6 | 64.62 | 65.3 | 16.43 |
| run | external_stale_cached | 16 | 4 | 4 | 4/6 | 60.21 | 70.98 | 15.79 |
| run | wrist_stale | 16 | 4 | 4 | 5/6 | 64.6 | 65.24 | 16.25 |
| run | both_stale | 16 | 4 | 4 | 3/6 | 64.61 | 65.07 | 16.24 |
| run | external_stale | 16 | 4 | 4 | 4/6 | 64.63 | 65.19 | 16.32 |
| run | external_stale_cached | 16 | 4 | 8 | 2/6 | 91.11 | 102 | 23.53 |
| run | both_fresh | 16 | 4 | 2 | 6/6 | 49.16 | 49.76 | 12.59 |
| refresh8 | both_fresh | 16 | 4 | 4 | 6/6 | 64.55 | 65.07 | 16.43 |
| refresh8 | external_stale_cached | 16 | 4 | 4 | 6/6 | 64.25 | 65.2 | 15.78 |
| refresh8 | external_stale | 16 | 4 | 4 | 6/6 | 64.57 | 65.05 | 16.43 |
| refresh8 | wrist_stale | 16 | 4 | 4 | 6/6 | 64.55 | 65.01 | 16.5 |
| adaptive | both_fresh | 16 | 4 | 4 | 6/6 | 64.47 | 65.01 | 16.36 |
| adaptive | external_stale_cached | 16 | 4 | 4 | 6/6 | 64.37 | 65.26 | 15.86 |
| adaptive | external_stale | 16 | 4 | 4 | 6/6 | 64.45 | 65.04 | 16.32 |
| adaptive | wrist_stale | 16 | 4 | 4 | 6/6 | 64.56 | 65.13 | 16.42 |
| requested_hk | both_fresh | 16 | 4 | 4 | 6/6 | 64.25 | 64.67 | 16.31 |
| requested_hk_extra | both_fresh | 16 | 4 | 4 | 6/6 | 64.3 | 64.84 | 16.19 |

- `run`: 외부/손목 stale 조건은 16 control steps마다 갱신. K=4에서 사용 영상 age는 0/4/8/12 steps, 20Hz 기준 최대 600ms입니다.
- `refresh8`: 8 steps마다 갱신, K=4에서 age 0/4 steps, 최대 200ms. 캐시 miss 때 두 카메라를 묶어 인코딩합니다.
- `adaptive`: 최대 16 steps, 외부 영상과 저장 영상의 평균 RGB 차이가 5/255를 초과하면 조기 갱신. 탐색 후 추가한 조건으로 독립 검증 결과는 아닙니다.
- 모든 재사용 조건에서 state는 최신입니다. 외부 cache는 vision encoder 출력만 재사용하며 VLM은 전체 재계산합니다.
- 단일 카메라 인코딩과 두 카메라 batch의 BF16 연산 차이 때문에 action 최대 차이 0.00549가 관찰됐습니다. 캐시 hit 자체의 반복 차이는 0이며, 전체 fresh 계산과 비트 단위 동등하다고 주장하지 않습니다.
- 최초 Small 실행 원자료의 `mean_camera_age_steps`는 사용하지 않은 과거 후보 snapshot의 age를 기록한 legacy 필드입니다. Small은 실제로 두 카메라 모두 최신 입력을 사용했습니다. 후속 실행에는 카메라별 실제 age가 별도로 기록됩니다.
- 최초 Small config의 `flow_steps=4`는 복사한 Heavy template의 잔여 필드입니다. 실제 N=10은 `closed_cells`, 각 trial의 N, timing과 manifest에 기록돼 있습니다. 후속 Small 설정은 이 필드도 실제 N에 맞춥니다.

## Offline: 같은 앞 4개 action 비교

| model | mode | H | N | lag | samples | expert_nmse | p50_ms |
|---|---|---|---|---|---|---|---|
| GR00T | both_fresh | 16 | 4 | 0 | 12 | 0.3178 | 63.93 |
| GR00T | both_fresh | 8 | 4 | 0 | 12 | 0.4374 | 65.89 |
| GR00T | external_stale_cached | 16 | 2 | 8 | 12 | 0.3023 | 43.79 |
| GR00T | wrist_stale | 16 | 4 | 8 | 12 | 0.389 | 63.83 |
| GR00T | both_stale | 16 | 4 | 8 | 12 | 0.4747 | 63.78 |
| GR00T | both_fresh | 4 | 4 | 0 | 12 | 0.7695 | 65.82 |
| GR00T | both_fresh | 50 | 4 | 0 | 12 | 0.296 | 64.29 |
| GR00T | external_stale_cached | 16 | 4 | 8 | 12 | 0.3191 | 59.04 |
| GR00T | external_stale_cached | 16 | 8 | 8 | 12 | 0.3406 | 89.39 |
| GR00T | external_stale | 16 | 4 | 8 | 12 | 0.318 | 63.71 |
| SmolVLA | both_fresh | 16 | 10 | 0 | 12 | 0.3363 | 164 |
| SmolVLA | both_fresh | 50 | 10 | 0 | 12 | 0.3365 | 162.9 |
| SmolVLA | both_fresh | 8 | 10 | 0 | 12 | 0.3365 | 161.8 |
| SmolVLA | both_fresh | 4 | 10 | 0 | 12 | 0.3366 | 159.9 |
| SmolVLA | both_fresh | 16 | 2 | 0 | 12 | 0.2686 | 59.16 |
| SmolVLA | both_fresh | 50 | 2 | 0 | 12 | 0.2687 | 58.97 |
| SmolVLA | both_fresh | 8 | 2 | 0 | 12 | 0.2687 | 58.83 |
| SmolVLA | both_fresh | 4 | 2 | 0 | 12 | 0.2687 | 58.52 |
| SmolVLA | both_fresh | 16 | 4 | 0 | 12 | 0.3022 | 85.25 |
| SmolVLA | both_fresh | 50 | 4 | 0 | 12 | 0.3023 | 84.76 |
| SmolVLA | both_fresh | 8 | 4 | 0 | 12 | 0.3024 | 84.52 |
| SmolVLA | both_fresh | 4 | 4 | 0 | 12 | 0.3025 | 83.95 |

## 상태와 재현

```json
{
  "run": {
    "status": "completed",
    "phase": "all",
    "updated_utc": "2026-11-17T06:28:14.628654+00:00",
    "offline_rows": 756,
    "rollouts": 72
  },
  "refresh8": {
    "status": "completed",
    "phase": "all",
    "updated_utc": "2026-11-17T06:31:29.467419+00:00",
    "offline_rows": 828,
    "rollouts": 24
  },
  "adaptive": {
    "status": "completed",
    "phase": "closed",
    "updated_utc": "2026-11-17T06:33:27.799394+00:00",
    "offline_rows": 0,
    "rollouts": 24
  },
  "small": {
    "status": "completed"
  },
  "small_n2": {
    "status": "completed"
  },
  "small_n4": {
    "status": "completed"
  },
  "requested_hk": {
    "status": "completed",
    "phase": "closed",
    "updated_utc": "2026-11-17T06:38:13.958586+00:00",
    "offline_rows": 0,
    "rollouts": 18
  },
  "requested_hk_extra": {
    "status": "completed",
    "phase": "closed",
    "updated_utc": "2026-11-17T06:39:41.330977+00:00",
    "offline_rows": 0,
    "rollouts": 18
  },
  "small_n2_extra": {
    "status": "completed"
  }
}
```

원자료·config·manifest·환경 목록을 같은 폴더에 보존했습니다. source_v1/source_v2는 GR00T batch 실행 당시 코드이며 각 manifest hash와 대응합니다. 계정 주간 사용량은 로컬 세션 한도 메타데이터에서 별도로 확인해 대화에 보고합니다.
