# ENVIRONMENT_STATUS — Isaac Sim / Isaac Lab / Bench2Dex 실행 환경

작성 2026-11-24. 이 문서의 수치는 모두 이 기계에서 직접 측정했다. 추정은 따로 표시한다.

## 1. 하드웨어와 공용 자원

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 24 GB 1장. 드라이버 620.70. 다른 사용자와 공유(다른 사용자의 vLLM 작업이 최대 약 19 GB를 점유한 시간대가 있었음) |
| CPU / RAM | 28 코어 / 62 GB |
| 권한 | root 없음. 드라이버 교체 불가 |
| 컨테이너 | apptainer 1.5.5 rootless(`/home/kwak3006/tools/apptainer`, 사용자 허락 후 설치). Ubuntu 22.04 sandbox `.runtime/bench2dex/ubuntu2204_isaac`(apt는 KAIST mirror). Vulkan ICD는 `library_path=libGLX_nvidia.so.0`로 바꿔 bind(`vk/nvidia_icd.json`) |

## 2. 설치 버전

| 환경 | 내용 | 용도 |
|---|---|---|
| `Bench2Dex/` | commit `c269f24b`. 원본이며 수정하지 않음 | 정책 서버, 오프라인 분석 |
| `IsaacLab/` | v2.3.2, commit `37ddf626` | Bench2Dex 공식 대상 |
| `IsaacLab3/` | v3.0.0, commit `e3f7d9ff` | Isaac Sim 6.2용 |
| `env_isaaclab` | py3.11, isaacsim 5.1.0, torch 2.7.0+cu126, isaaclab 0.54.2 | 공식 경로(실행 불가, §3) |
| `env_isaacsim6` | py3.12, isaacsim 6.2.0.0, isaaclab 28.0.0(v3.0.0), isaaclab-physx 8.0.0, torch 2.11.0+cu130, numpy 2.3.1 | 폐루프 시뮬레이터 |
| `env_gr00t` | py3.10, torch 2.5.1+cu124, transformers 4.51.3, flash-attn 2.7.4.post1 | GR00T N1.5 정책 서버와 오프라인 실험 |
| `Bench2Dex_is6/` | `c269f24b` 사본에 Isaac Lab 3.0 이식 변경을 더한 것. 변경 목록은 `Bench2Dex_is6/PORT_NOTES.md`(14항목, 코드에 `[is6 port]` 표시) | 폐루프 client |

자산은 HF `Bench2Dex/Assets`, `Bench2Dex/policy_ckpt`, `Bench2Dex/teleopdata`에서 필요한 것만 받았다.
- 로봇(xarm+leap, ur5+wuji, kuka+sharpa)
- 작업 물체, clutter 20개
- 텍스처 195개와 iTHOR 배경 장면 61개: 처음 받을 때 `--include`를 여러 번 넘겨 마지막 묶음만 적용됐다(텍스처 1개, 장면 0개). 11/24 00:00–00:24에 다시 받았다.
- 작업 09·26·27 체크포인트, replay-generalization 시연

## 3. Isaac Sim / Isaac Lab 실행 여부

| 경로 | 결과 | 근거 |
|---|---|---|
| Isaac Sim 5.1 + Isaac Lab 2.3.2(공식) | **실행 불가.** 시작 직후 `librtx.scenedb` segfault가 난다. RaytracedLighting 렌더러로 바꿔도, GPU 12 GB가 비어 있어도 같다 | NVIDIA 포럼에 알려진 문제다(드라이버 ≥595, 검증 드라이버 580.65.06). 드라이버를 바꿀 권한이 없다 |
| Isaac Sim 6.2 + Isaac Lab 3.0 | **실행 가능.** Bench2Dex를 이식했다(사용자 결정 "2번") | 아래 §4 |

이식 과정에서 고친 Isaac Lab 3.0 API 변화는 다음과 같다.
- 쿼터니언 순서가 WXYZ에서 XYZW로 바뀌었다.
- `ProxyArray`는 tensor 메서드를 전달하지 않는다.
- PhysX tensor view가 warp array를 돌려준다.
- `PhysxCfg`의 위치가 바뀌었다.
- `--headless` 옵션이 없어졌다(기본이 headless).
- TacMap 센서의 부모 클래스(MultiMeshRayCaster)가 다시 작성되어, 센서를 새로 구현했다.
- 에피소드 사이 장면 재구성 경로가 바뀌어, runner가 에피소드마다 client를 새로 띄운다. 프로세스의 첫 에피소드는 재구성 경로를 타지 않게 했다.

렌더링 충실도(카메라 영상을 시연 0번과 같은 프레임에서 비교, `closed_loop/camcheck*_gtreplay_is6`):

| 문제 | 원인 | 조치 |
|---|---|---|
| 테이블이 검게 나옴 | 텍스처 미다운로드(위) | 195개 다시 받음 |
| 영상 잡음이 심함 | apptainer `--nv`가 NGX(DLSS, DLSS-D 디노이저) 라이브러리를 bind하지 않음("Failed to create NGX context") | `run_in_container_ngx.sh`로 두 파일 bind |
| 손목 fisheye 시야가 좁음 | 6.2가 OpenCV fisheye 렌즈 스키마를 기본으로 올리지 않아 폐기된 KannalaBrandtK3로 대체 | client 시작 인자 `--enable omni.usd.schema.omni_lens_distortion` |

세 조치 후 스테레오와 손목 영상이 시연과 거의 같다. 남은 차이는 손 재질의 세부 표현과 물리 차이로 인한 물체 위치뿐이다. 이 조치 전의 폐루프 기록은 모두 분석에서 뺐다.

## 4. Bench2Dex 작업 실행 결과(작업 27, UR5 + Wuji hand, 측정)

**GT replay(시연 명령을 개루프로 재생, 6.2)**
- 기록: `closed_loop/gtreplay_is6_rec1`. 비교 스크립트: `scripts/ci23_compare_replay.py`.
- 명령: 시연과 완전히 같다(최대 차 0).
- 로봇 추종: 접촉 전 팔 관절 오차 중앙값 0.004 rad, p95 0.014–0.018 rad. 손 관절 중앙값 0.002 rad. 로봇 이식은 정상이다.
- 물체: 정지 물체는 0.02 mm 이내로 같은 자리에서 시작한다. 다만 미니 축구공은 시작 전 정착 단계에서 5.1에서는 약 5 cm 굴렀고 6.2에서는 0.2 cm만 움직였다. 접촉이 시작된 뒤에는 탁구공(173프레임), 골프공(300), 바구니가 1 cm 넘게 갈라진다.
- 결과: 실패, 단계 완료 0.25. 개루프 재생은 물리 차이를 보정하지 못한다. 따라서 6.2는 5.1과 **다른 물리 환경**으로 취급하고, 성공률을 5.1 리더보드와 일대일로 비교하지 않는다.

**TacMap 촉각 센서(6.2 이식본) 검증(측정)**
- 시연 0번을 client 안에서 재생하며 매 캡처의 깊이 지도를 저장했다(`closed_loop/tacval2_gtreplay_is6`). 5.1 시연에 기록된 `distance_along_normal_m`과 비교했다(`scripts/ci23_compare_tactile.py`).
- 비교 구간은 물체가 1 cm 넘게 갈라지기 전인 프레임 60–172이다.
  - 접촉이 있는 사이트-프레임 124개 중 99–100개가 양쪽 공통이고, 포팅에만 있는 것은 5–6개, 시연에만 있는 것은 18–19개다.
  - 둘 다 접촉한 경우의 마스크 IoU 중앙값은 0.66–0.72다(정렬 offset 0/1/2).
  - 깊이 합의 상관은 0.62–0.86이고, 총량 비는 1.12다.
  - 최대 깊이 중앙값은 양쪽 모두 15.0 mm다.
- 프레임 0–59에서는 양쪽 모두 접촉이 없다.
- **판정: 정상 촉각 관측이 생성된다.** 남은 차이는 알려진 물리 차이(로봇 mm 수준, 축구공 4.9 cm)와 맞는다.

검증 전 두 번의 실패와 그 원인은 NOTES.md 23:52 항목에 있다. GT_REPLAY 서버는 한 번 요청에 한 행을 내주는데, 촉각 client는 16스텝 chunk를 요청해서 시연이 금방 소진됐다.

**폐루프(정책) 실행**
- 에피소드마다 client를 새로 띄운다. 1190스텝(공식 예산) 한 에피소드가 약 4.5분이다. 두 arm을 병렬로 돌려 50 에피소드씩 약 2시간 50분 걸렸다(00:27–03:21).
- 두 arm 100 에피소드 모두 오류 없이 끝났다. 결과는 `CLOSED_LOOP_RESULTS.md`에 있다.
- 같은 장면과 seed로 시작한 짝도 첫 접촉 전에 궤적이 조금씩 갈라졌다. 첫 접촉 캡처의 접촉 픽셀 수가 같은 쌍은 50쌍 중 0개였다. GPU 물리나 렌더링의 비결정성으로 추정한다(`CLOSED_LOOP_RESULTS.md` §1).
- 분석에서 뺀 시도는 두 묶음이다. 둘 다 삭제하지 않고 옮겨 두었다.
  - `closed_loop/discarded_nobg_singleproc/`: 배경 자산 누락, 다중 에피소드 crash.
  - `closed_loop/discarded_broken_render/`: 텍스처·NGX·fisheye 문제.

## 5. 권한과 제약

- Isaac Sim EULA는 사용자 동의 후 `OMNI_KIT_ACCEPT_EULA=YES`만 설정했다. `PRIVACY_CONSENT`는 설정하지 않았다.
- Bench2Dex `setup_env.sh`와 `setup_policy_envs.sh`는 실행하지 않았다. 이 스크립트들은 `~/.condarc`와 전역 pip 설정을 덮어쓴다.
- 다른 사용자의 GPU 프로세스는 건드리지 않았다. GPU 메모리가 모자랄 때는 기다리거나 내 작업 순서를 바꿨다(NOTES 22:57, 23:17).
- 기존 결과 파일은 수정하거나 삭제하지 않았다. 중단한 폐루프 실행도 삭제하지 않고 따로 옮겼다.

## 6. 재현 명령

```bash
B=/home/kwak3006/vlaexperiment/.runtime/bench2dex
# 폐루프(정책 서버 env_gr00t + Isaac Sim 6.2 client, NGX bind), 작업 27, 에피소드 1-50
$B/run_in_container_ngx.sh env IS6_TACMAP_LOG=<out>/tacmap_log.jsonl IS6_TACTILE_ABLATION=none \
  bash $B/run_tactile_is6_v2.sh 27 27_ball_box_loading multi_ur5_wuji_with_flange 50 1190 <out> 9311 model 0
#   (zero arm: IS6_TACTILE_ABLATION=zero, port 9312)
# 카메라 점검: IS6_CAM_DUMP_DIR=<dir> IS6_CAM_DUMP_CAPS=0,80,150 IS6_GT_REPLAY_HDF5=<demo.hdf5> ... gtreplay
# 촉각 검증: IS6_GT_REPLAY_HDF5=<demo.hdf5> IS6_TACMAP_DUMP_DIR=<dir> ... run_tactile_is6.sh ... 300 <out> 9204 gtreplay
.venv/bin/python scripts/ci23_compare_tactile.py <dump> <demo.hdf5> <out.json> 60 172
# 오프라인 실험(env_gr00t, 컨테이너 안)
$B/run_in_container.sh bash -c "cd $B/Bench2Dex && CI23_TASK=27 PYTHONPATH=/home/kwak3006/vlaexperiment/scripts $B/env_gr00t/bin/python /home/kwak3006/vlaexperiment/scripts/<ci23_*.py>"
```
