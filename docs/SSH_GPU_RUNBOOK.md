# SSH GPU 환경에서 이어 실행하기

최신 연구 계획은 [카메라별 feature 재사용 메모](camera_feature_reuse.md)를 참고하세요. 아래 명령은 기존 H/K·prior 진단을 실행하며, 새로 제안한 외부/손목 카메라별 시차 실험은 별도 구현이 필요합니다.

이 문서는 로컬 SmolVLA pilot을 Linux GPU 서버에서 재현하는 경로입니다. 로컬에서 검증한 것은 Windows/CPU이며, CUDA 장치 선택·동기화와 Linux EGL 경로는 추가했지만 실제 RTX 3090 실행은 아직 검증하지 않았습니다. 처음에는 float32로 공식 sampler 일치를 확인하고, BF16/양자화는 별도 조건으로 평가하세요. GR00T/π0.5는 별도의 공식 환경이 필요합니다.

## 1. 저장소와 독립 환경

```bash
git clone https://github.com/KwakChaehun/vlaexperiment.git
cd vlaexperiment
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
nvidia-smi
```

다음은 로컬 Torch 2.7.1과 대응하는 CUDA 12.6 wheel 예시입니다. 서버 driver가 지원하는 wheel을 [PyTorch 공식 이전 버전 설치표](https://pytorch.org/get-started/previous-versions/#v271)에서 확인하세요. 시스템 CUDA toolkit이나 driver를 이 저장소의 실행기가 변경하지 않습니다.

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-smolvla.txt
python -m pip install lerobot==0.4.4 --no-deps
python -m pip install -e . --no-deps
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
```

`lerobot --no-deps`와 requirements 파일은 SmolVLA를 직접 호출하는 최소 환경입니다. 로컬에서도 사용하지 않는 robot/정책 extras를 설치하지 않았고 해당 registry eager import를 비활성화했습니다. 일반 LeRobot CLI 전체를 지원하는 환경으로 해석하지 않습니다. Linux에서 추가 의존성 문제가 나면 첫 traceback을 보존하고 이 환경을 조정해야 합니다.

## 2. Git에 포함하지 않은 assets 재생성

```bash
python scripts/prepare_remote_assets.py --patch-runtime
```

고정 revision의 SmolVLA weights(약 1.2 GB), VLM config/tokenizer(중복 VLM weight 제외), LIBERO source/assets, 공개 demo 6 episode(약 84 MiB)를 받습니다. 모델·데이터·가상환경 전체는 Git에 넣지 않습니다. `.runtime/assets_manifest.json`에 revision과 weight hash를 저장합니다.

`--patch-runtime`은 활성 venv 안에서만 허용합니다. LeRobot 0.4.4의 관련 없는 정책 eager import를 제거하고, 고정 LIBERO의 공식 initial-state 파일을 Torch 2.7에서 로드하도록 `weights_only=False`를 명시합니다. 원본은 `.vla_original` 파일로 보존합니다. 외부 임의 checkpoint를 이 방식으로 로드하는 도구는 아닙니다.

다운로드 script는 subset NPZ와 원본 Parquet cache를 함께 준비합니다. Prior 진단은 원본 episode에서 gripper 전환 구간을 읽으므로 **NPZ만 복사하면 충분하지 않습니다**. Local/remote 각각 준비하면 경로를 바꿀 필요가 없습니다.

## 3. Headless renderer와 실제 추론 확인

```bash
export MUJOCO_GL=egl
python scripts/libero_smoke.py
python scripts/check_smolvla_runtime.py --device cuda
python -m unittest discover -s tests -q
```

EGL은 서버의 NVIDIA/OpenGL runtime이 보여야 합니다. 컨테이너에서는 GPU와 graphics capability가 전달되어야 하며, renderer 생성에 실패하면 그 상태에서 성공률 batch를 진행하지 마세요. CPU smoke는 `--device cpu`로 실행합니다. Linux는 기본 EGL, Windows는 GLFW이며 환경 변수로 명시한 backend를 우선합니다.

추론 smoke는 공식 H=50/N=10 sampler와 custom sampler를 비교하고 H=4의 실제 latent shape `[1,4,32]`, action shape `[4,7]`, device, finite 값을 확인합니다. 로컬 CPU의 sampler 최대 오차는 0이었습니다. CUDA 실행은 자체 parity 검증을 통과해야 합니다. 현재 허용 오차는 `1e-4`입니다.

## 4. 새 GPU 결과 디렉터리에서 H/K 평가

```bash
python scripts/run_hk_matrix.py \
  --config configs/smolvla_hk_matrix_cuda.json \
  --output results/hk_matrix_cuda \
  --hours 12
```

`device=cuda`를 명시한 별도 설정입니다. CPU 결과 디렉터리에 GPU 결과를 섞지 않습니다. CUDA가 없으면 CPU로 조용히 대체하지 않고 종료합니다. Native-H CPU noise를 만든 뒤 device로 옮겨 H별 공통 prefix를 유지하며, CUDA stage 경계에서 synchronize 후 시간을 측정합니다.

먼저 prior 진단 12관측을 수행하고, 유효 H/K 13조합 × 6회 = 78 rollout을 실행합니다. 진단을 제외하려면 config를 새 이름으로 복사해 `prior_diagnostics.enabled=false`로 바꾸고 새로운 output을 사용하세요. 동일 output에서 config를 변경하면 중단합니다. 중단 후 같은 config/output으로 재실행하면 완료 trial을 이어 사용합니다. 같은 output에 두 worker를 동시에 실행하지 않습니다.

SSH 연결이 끊겨도 계속 실행하려면 서버에서 제공하는 job scheduler나 tmux를 사용하세요. 예:

```bash
tmux new -s vla-hk
source .venv/bin/activate
export MUJOCO_GL=egl
python scripts/run_hk_matrix.py --config configs/smolvla_hk_matrix_cuda.json --output results/hk_matrix_cuda --hours 12
# detach: Ctrl-b, d / reconnect: tmux attach -t vla-hk
```

## 5. 결과 확인과 해석

- `results/hk_matrix_cuda/RESULTS.html`: 자동 갱신 보고서. 파일을 로컬로 복사하거나 SSH port forwarding으로 볼 수 있습니다.
- `status.json`, `manifest.json`: 진행 상태, device, sampler parity, config와 code hash.
- `matrix_summary.json`, `closed_loop.jsonl`, `replays/`: H/K 성공률·원자료·GIF.
- `prior_diagnostics_summary.json`, `prior_diagnostic_comparisons.json`, `prior_diagnostics_by_phase.json`: prior 조건·refinement·작업 구간 비교.
- `errors.jsonl`: 실행 오류. 오류/censored는 task 실패와 구분합니다.

`cpu_call_*` 필드명은 첫 CPU 버전과 호환되는 legacy 이름입니다. 새 CUDA 실행에서는 실제 장치를 manifest에서 확인하세요. 이 값은 preprocessing/backbone/AE stage 합이며 카메라·controller·전체 통신 지연을 포함한 deadline 측정은 아닙니다. Simulator는 추론 중 정지하므로 task 품질 실험이고, GPU real-time/PIM speedup의 증거와 구분합니다.

## 6. 실제 목표 모델로 확장

SmolVLA causal mask 결과를 GR00T에 일반화하지 않습니다. GR00T adapter는 실험적이며 실제 checkpoint/공식 sampler 일치를 검증하지 않았습니다. π0.5 adapter도 native H=10 offline만 연결되어 있습니다. 버전과 task checkpoint를 확정하고 공식 native 성공률을 먼저 재현한 뒤 H/K/N·reuse를 확대합니다. Full training이 RTX 3090에 들어간다고 보장하지 않습니다.
