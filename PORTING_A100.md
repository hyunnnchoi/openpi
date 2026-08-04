# DGX Spark(GB10) → A100 80GB ×2 이식 가이드

이 리포지토리는 **DGX Spark(GB10, `sm_121`, aarch64)** 에서 pi0.5 정책 서빙을 돌리고
프로파일링하기 위해 손본 openpi 포크입니다. A100으로 옮길 때 **되돌려야 할 것**과
**그대로 가져가야 할 것**이 섞여 있어서, 그 경계를 명시하는 것이 이 문서의 목적입니다.

핵심 요약 한 줄: **GB10 우회의 대부분은 A100에서 불필요하지만(되돌리세요), LIBERO
클라이언트 우회 2개는 아키텍처와 무관한 업스트림 버그라 그대로 필요합니다.**

관련 문서:
- `RUN_DGX_SPARK.md` — GB10에서 무엇을 왜 바꿨는지 + 측정된 기준선 전체
- `profiling/` — 다중 클라이언트 부하 + Nsight 하네스
- `profiling/baselines/gb10/` — 비교 대상이 될 GB10 측정 원본(요약 JSON/CSV)

---

## 0. 옮겨간 박스에서 가장 먼저 확인할 것

```bash
uname -m                      # x86_64 기대. aarch64면 1장의 판단이 달라집니다
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
nvcc --version | tail -2      # 또는: nvidia-smi | head -4 의 CUDA Version
python3 -c "import sys; print(sys.version)"
nsys --version && ncu --version
```

기대값: `x86_64`, `NVIDIA A100-SXM4-80GB` ×2, compute capability `8.0`.

`compute_cap`이 `8.0`이라는 게 이 이식의 전제 전부입니다. GB10에서 겪은 문제는
전부 "`sm_121`은 너무 새로워서 빌드된 커널이 없다"였고, `sm_80`은 정반대로
PyTorch/JAX가 **가장 오래 지원해온 아키텍처**입니다.

---

## 1. 의존성 — GB10 우회를 되돌린다

`pyproject.toml.orig`가 손대지 않은 업스트림 파일입니다. A100에서는 이게 그대로 맞습니다.

```bash
cp pyproject.toml.orig pyproject.toml
rm -f uv.lock                 # GB10용 락(aarch64 휠로 고정)은 버리고 다시 풉니다
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

되돌려지는 것:

| 항목 | GB10에서 | A100에서 |
| --- | --- | --- |
| `torch` | `2.11.0+cu130` (cu130 인덱스) | `2.7.1` (업스트림 핀) |
| `torchvision` | `0.26.0+cu130` | 업스트림 핀 |
| `jax` | `jax==0.5.3` (**CPU 전용**) | `jax[cuda12]==0.5.3` 가능 |
| 커스텀 `[[tool.uv.index]]` | pytorch-cu130 | 불필요 |

### ⚠️ 함정: JAX가 GPU 메모리를 선점합니다

GB10에서는 JAX가 CPU 전용이라 신경 쓸 일이 없었지만, A100에서 `jax[cuda12]`를 깔면
**XLA가 기본적으로 GPU 메모리의 75%를 미리 잡습니다**(`XLA_PYTHON_CLIENT_PREALLOCATE`
기본값이 true). openpi의 PyTorch 서빙 경로에서 JAX는 transform과 orbax 체크포인트
로딩에만 쓰이므로, 그대로 두면 torch가 할당에 실패합니다.

둘 중 하나를 선택하세요:

```bash
# (a) 권장 — PyTorch 서빙만 할 거라면 JAX는 계속 CPU로 두는 게 가장 간단합니다
#     pyproject.toml에서 jax[cuda12] → jax 로 두면 GB10과 동일하게 동작

# (b) JAX GPU가 필요하다면(학습/JAX 추론) 반드시 선점을 끕니다
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.10
```

**(a)를 권장합니다.** 이 리포지토리의 벤치마크·서빙·프로파일링 경로는 전부 PyTorch이고,
JAX를 GPU에 올려서 얻는 것이 없습니다.

### 메모리 모델이 달라집니다

GB10은 CPU/GPU가 **128 GB 통합 풀**을 공유해서, vLLM 같은 게 떠 있으면 openpi가
아무것도 할당하지 못하는 문제가 있었습니다(`RUN_DGX_SPARK.md` 마지막 절). A100은
**80 GB 전용 HBM**이라 이 문제 자체가 없어집니다. 가중치 ~9 GB + 컴파일 후 여유가
넉넉하므로, GB10에서 못 해본 큰 배치 실험이 가능합니다.

---

## 2. 체크포인트 변환

아키텍처 무관하게 동일합니다. 변환된 PyTorch 체크포인트는 **박스 간 그대로 복사해도
됩니다**(bf16 텐서일 뿐 아키텍처 종속성 없음). 7.2 GB이므로 rsync가 재변환보다 대개 빠릅니다.

```bash
# 방법 A — GB10에서 복사 (권장)
rsync -avP ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/ \
      <a100-host>:~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/

# 방법 B — A100에서 새로 변환
.venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --config_name pi05_libero \
  --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --precision bfloat16
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets \
      ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/
```

**`assets/` 복사를 빠뜨리지 마세요.** `create_trained_policy`가 정규화 통계를
`<checkpoint_dir>/assets`에서 읽는데 변환 스크립트가 복사해주지 않습니다. 방법 A는
이미 복사된 상태를 통째로 가져오므로 해당 없음.

---

## 3. LIBERO 클라이언트 venv — 이건 **그대로 필요합니다**

아래 두 우회는 GB10 때문이 아니라 업스트림 버그라, A100에서도 똑같이 필요합니다.
다만 **경로가 하드코딩돼 있어서 새 박스에서 반드시 다시 만들어야 합니다.**

```bash
uv venv --python 3.11 examples/libero/.venv
examples/libero/.venv/bin/pip install -e packages/openpi-client
examples/libero/.venv/bin/pip install robosuite==1.4.1 bddl easydict \
    "numpy==1.26.4" torch torchvision imageio[ffmpeg] tyro

SP=$(examples/libero/.venv/bin/python -c "import site;print(site.getsitepackages()[0])")

# (1) third_party/libero/libero/ 에 __init__.py 가 없어서 find_packages()가 아무것도
#     찾지 못합니다. editable 설치가 무의미하므로 경로를 직접 넣어줍니다.
#     ★ 절대경로이므로 새 박스에서 반드시 재생성해야 합니다 ★
echo "$PWD/third_party/libero" > "$SP/_libero_root.pth"

# (2) LIBERO(f78abd6)는 PyTorch 2.6 이전 코드라, weights_only 기본값이 True로
#     바뀐 뒤 번들된 .pruned_init 상태 파일을 못 읽습니다.
cat > "$SP/sitecustomize.py" <<'PY'
try:
    import torch as _torch
except ImportError:
    pass
else:
    _orig_load = _torch.load
    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)
    _torch.load = _load
PY
```

> 업스트림은 Python 3.8 + `numpy==1.22.4` + `llvmlite==0.36`을 지시하지만, 그건
> aarch64에서 빌드가 안 돼서 3.11로 올린 것이었습니다. x86_64라면 업스트림 조합도
> 빌드는 되겠지만, **3.11 조합이 이미 검증됐으므로 그대로 쓰는 것을 권합니다.**

### EGL 렌더링

GB10에서는 glvnd가 Mesa를 골라버리고 이 계정이 `render` 그룹에 없어서 강제 지정이
필요했습니다. A100 서버는 상황이 다를 수 있으니 순서대로 시도하세요.

```bash
export MUJOCO_GL=egl
examples/libero/.venv/bin/python -c "import mujoco; mujoco.MjrContext"   # 되면 끝

# 실패하면 — NVIDIA 벤더를 강제 (GB10에서 쓴 방법)
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# 그래도 실패하면 (헤드리스 노드에 GPU 렌더 경로가 없을 때) — CPU 렌더로 폴백
export MUJOCO_GL=osmesa      # 느립니다. 시뮬 스텝 시간이 기준선과 비교 불가해집니다
```

`osmesa`로 떨어지면 `env.step` 시간이 GB10 기준선(9.8 ms)과 비교 불가능해진다는
점을 기록해두세요. **정책 추론 지연에는 영향이 없으므로 프로파일링 결과 자체는 유효합니다.**

---

## 4. 벤치마크 재현

셋업이 끝났으면 GB10에서 돌린 것과 **완전히 동일한 명령**으로 재현합니다.

```bash
# (1) 단일 추론 지연 — 시뮬 불필요, 가장 빠른 sanity check
.venv/bin/python bench_infer.py

# (2) 배치 스케일링 — "GPU 한 장에 로봇 몇 대"
.venv/bin/python bench_batch.py

# (3) 다중 클라이언트 + Nsight (이번 작업의 본체)
profiling/run_bench.sh -c "1 2 4 8" -n 30
profiling/run_bench.sh -c 4 -m saturate --no-nsys

# 결과 재출력 (재실행 없이)
.venv/bin/python profiling/report.py data/profiling
```

`profiling/run_bench.sh`는 클라이언트 수마다 서버를 새로 띄우고, **소켓을 열기 전에
워밍업**해서 `torch.compile` 비용이 첫 클라이언트에게 청구되지 않게 합니다. 캡처는
`cudaProfilerStart/Stop` 구간에만 걸립니다.

---

## 5. 비교 기준선 (GB10에서 측정된 값)

`profiling/baselines/gb10/`에 원본 요약이 있습니다. 표로는:

**다중 클라이언트 (250 ms 데드라인 = `replan_steps=5` @ 20 Hz, 클라이언트당 30요청)**

| 클라이언트 | 처리량 | e2e 평균 | 큐 대기 p50 | `policy.infer` p50 | 서버 busy | 데드라인 미스 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4.04 req/s | 152.7 ms | 0.5 ms | 151.9 ms | 61.4% | 0% |
| 2 | 6.66 req/s | 294.1 ms | 148.9 ms | 149.2 ms | 99.8% | 97% |
| 4 | 6.68 req/s | 589.4 ms | 448.0 ms | 149.2 ms | 99.8% | 99% |
| 8 | 6.69 req/s | 1175.2 ms | 1044.6 ms | 149.1 ms | 99.8% | 100% |

**배치 스케일링 (컴파일 경로, 10 Euler steps)**

| 배치 | 지연 | 로봇당 | 직렬 대비 |
| --- | --- | --- | --- |
| 1 | 148.2 ms | 148.2 ms | 1.00x |
| 2 | 236.7 ms | 118.4 ms | 1.25x |
| 4 | 421.1 ms | 105.3 ms | 1.41x |
| 8 | 811.3 ms | 101.4 ms | 1.46x |

**커널 분해 (4 클라이언트, `--cuda-graph-trace node`)** — 요청당 GPU 142.2 ms / `sample_actions` 148.1 ms = **96% GPU-busy**

| 분류 | GPU ms | % |
| --- | --- | --- |
| GEMM `cutlass_80_*tensorop` | 3530 | 31.0 |
| GEMM `cutlass_80_*wmma*` | 2980 | 26.2 |
| Triton (Inductor fused) | 2754 | 24.2 |
| GEMM `nvjet_sm121_*` | 2010 | 17.7 |

기타(reduction/attention/elementwise)는 합쳐서 1% 미만.

---

## 6. A100에서 달라질 것 — 검증 가능한 예측

**이건 예측이지 결과가 아닙니다.** 이식의 가치는 이 4개를 실제로 확인하는 데 있습니다.

### 예측 1: `cutlass_80_*` 발견이 자동으로 해소된다 (가장 확실)

GB10에서 GPU 시간의 **57%가 `cutlass_80_*`**, 즉 Ampere 세대 CUTLASS 커널에 들어갔고,
정작 이 칩 이름을 단 `nvjet_sm121_*`는 18%뿐이었습니다. **A100은 그 `sm_80`이
네이티브 아키텍처입니다.** 즉 A100에서는 같은 커널이 "레거시 경로"가 아니라 "제대로
튜닝된 경로"가 됩니다.

> 확인 방법: `profiling/baselines/gb10/c4_node_cuda_gpu_kern_sum.csv`와 A100의 같은
> 파일을 비교. A100 쪽에 `nvjet`/`sm121` 계열이 사라지고 `cutlass_80` 비중이 올라가면
> 예측대로입니다. **이 경우 GB10에서 제기한 "커널 선택이 잘못됐을 수 있다"는 가설은
> A100에서는 애초에 성립하지 않습니다** — GB10 고유 이슈였다는 뜻이 됩니다.

### 예측 2: 배치 스케일링이 GB10보다 크게 좋아진다

GB10에서 배치가 1.46배밖에 못 번 이유는 Inductor가 남긴
`Not enough SMs to use max_autotune_gemm mode` 한 줄이 설명합니다 — 배치 1에서 이미
compute-saturated였습니다. **A100은 108 SM**이라 이 메시지가 사라질 가능성이 높고,
그러면 prefill이 "로봇당 ~110 ms 고정"에서 벗어날 수 있습니다.

> 확인 방법: `bench_batch.py` 실행 후 prefill의 **로봇당** 시간이 배치 크기에 따라
> 줄어드는지 확인. GB10에서는 1/2/4/8/16 배치 전부 105–116 ms로 평평했습니다.
> 서버 로그에서 `Not enough SMs` 문구도 grep 해보세요.

### 예측 3: 절대 지연은 방향이 자명하지 않다 — 반드시 재세요

메모리 대역폭은 A100 80GB(HBM2e, 약 2.0 TB/s)가 GB10(LPDDR5X 통합, 약 273 GB/s)보다
**한 자릿수 가깝게 큽니다**. denoising 루프처럼 메모리 바운드인 구간은 크게 빨라질
수 있습니다. 반면 순수 연산 처리량은 세대가 달라 단순 비교가 안 됩니다.

> 즉 **"A100이니까 당연히 빠르다"고 가정하지 마세요.** `bench_infer.py` 한 번이면
> 149 ms 대비 어느 쪽인지 바로 나옵니다. 그 값에 따라 아래 8절의 배치 결론이 달라집니다.

### 예측 4: 서버 직렬화는 **전혀 달라지지 않는다**

이건 하드웨어 이슈가 아닙니다. `Policy.infer`가 배치 1로 하드코딩돼 있고
(`inputs[None, ...]` / `x[0, ...]`), `WebsocketPolicyServer`가 asyncio 핸들러 **안에서
동기적으로** 호출합니다. A100 2장을 꽂아도 서버 하나는 여전히 FIFO 큐입니다.

> 확인 방법: `profiling/report.py`의 `server busy`가 다시 99.8%, `peak_inflight`가
> 1로 나오면 동일. 추론이 빨라진 만큼 처리량 상한(1/지연)만 올라갑니다.

---

## 7. A100 2장 활용

openpi 추론에는 텐서 병렬이 없으므로, **데이터 병렬(서버 2개)이 유일하게 의미 있는
수평 확장**입니다. NVLink는 쓰이지 않습니다.

```bash
# GPU 0 — 포트 8000
CUDA_VISIBLE_DEVICES=0 .venv/bin/python profiling/serve_profiled.py \
    --port 8000 --tag gpu0 --out data/profiling &

# GPU 1 — 포트 8001
CUDA_VISIBLE_DEVICES=1 .venv/bin/python profiling/serve_profiled.py \
    --port 8001 --tag gpu1 --out data/profiling &
```

이렇게 하면 로봇 수용량이 정확히 2배가 됩니다(각 서버는 여전히 FIFO). 로봇을 두
포트에 나눠 붙이면 되고, 앞단 라우터는 필요 없습니다.

**더 나아가려면** 서버에 동적 배칭을 붙이는 것이 남은 큰 작업입니다. 근거는 이미
확보돼 있습니다: `PI0Pytorch.sample_actions`는 전 구간 batch-first라 모델은 준비돼
있고, 막고 있는 것은 `Policy.infer`의 배치-1 하드코딩과 동기 핸들러뿐입니다.
`torch.compile` 재컴파일 비용은 걱정하지 않아도 됩니다 — GB10 측정 기준 서로 다른
배치 크기 2개를 겪고 나면 Dynamo가 배치 차원을 dynamic으로 표시해서 이후는 1초 미만입니다.

---

## 8. Nsight — A100에서 더 되는 것들

| | GB10(이 박스) | A100 기대 |
| --- | --- | --- |
| CUDA + NVTX 트레이싱 | ✅ 됨 | ✅ 됨 |
| CPU 샘플링 | ❌ `perf_event_paranoid=4` | root 있으면 ✅ |
| nsys GPU 메트릭(SM occupancy) | ❌ `ERR_NVGPUCTRPERM` | 권한 풀면 ✅ |
| Nsight Compute (`ncu`) 커널 카운터 | ❌ `ERR_NVGPUCTRPERM` | 권한 풀면 ✅ |

A100 인스턴스는 root를 쓸 수 있는 경우가 많으니, 그렇다면 GB10에서 **못 했던 것**을
할 수 있습니다:

```bash
sudo sysctl -w kernel.perf_event_paranoid=2
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nsight.conf
sudo update-initramfs -u && sudo reboot

# 재부팅 후 — GB10에서 막혀 있던 것들
nsys profile --gpu-metrics-devices=0 ...        # SM occupancy 타임라인
ncu --set full --launch-count 20 ...            # 커널별 roofline
```

권한이 풀리면 `run_bench.sh`에 `--gpu-metrics-devices=0`을 추가해서, 지금 NVML을
10 Hz로 샘플링해 얻고 있는 SM 사용률(93%)을 **실제 카운터로** 대체하세요.

### ⚠️ `--cuda-graph-trace node`는 A100에서도 그대로 필수입니다

이건 아키텍처와 무관합니다. `max-autotune`이 추론을 CUDA 그래프로 컴파일하기 때문에,
nsys 기본값(`graph`)에서는 커널 리포트에 **36초 실행에 GPU 시간 5 ms**가 찍힙니다.
GPU가 놀고 있는 것처럼 읽히지만 완전히 틀린 값입니다. `run_bench.sh`의 기본값으로
이미 넣어뒀으니 건드리지 마세요.

---

## 9. 체크리스트

- [ ] `uname -m` = `x86_64`, compute capability `8.0` ×2 확인
- [ ] `cp pyproject.toml.orig pyproject.toml`, `rm uv.lock`, `uv sync`
- [ ] JAX를 GPU에 올렸다면 `XLA_PYTHON_CLIENT_PREALLOCATE=false` (안 올렸으면 해당 없음)
- [ ] 체크포인트 복사 + `assets/` 포함 확인
- [ ] LIBERO venv 재생성 — **`_libero_root.pth`의 절대경로 갱신**, `sitecustomize.py` 재작성
- [ ] `MUJOCO_GL` 경로 확정 (egl → 벤더 강제 → osmesa 순)
- [ ] `bench_infer.py` — 149 ms 기준선 대비 어디인지 (예측 3)
- [ ] `bench_batch.py` — prefill이 배치에 따라 줄어드는지 + `Not enough SMs` 사라졌는지 (예측 2)
- [ ] `profiling/run_bench.sh -c "1 2 4 8" -n 30` — 직렬화가 그대로인지 (예측 4)
- [ ] 커널 CSV 비교 — `nvjet_sm121` 소멸 확인 (예측 1)
- [ ] root 있으면 프로파일링 권한 해제 후 `ncu` 재실행

---

## 부록: 이 리포지토리에 추가된 파일

| 경로 | 내용 |
| --- | --- |
| `RUN_DGX_SPARK.md` | GB10 셋업 근거 + 측정된 기준선 전체 |
| `PORTING_A100.md` | 이 문서 |
| `profiling/serve_profiled.py` | NVTX + 요청별 메트릭 + `cudaProfilerApi` 제어 서버 |
| `profiling/load_clients.py` | 로봇 N대를 프로세스 N개로 (rate / saturate) |
| `profiling/report.py` | 클라이언트·서버 타임스탬프 조인 → 단계별 분해 |
| `profiling/run_bench.sh` | 클라이언트 수 스윕 + nsys 캡처 |
| `profiling/baselines/gb10/` | GB10 측정 원본 (비교용) |
| `bench_infer.py` | 단일 추론 지연 |
| `bench_batch.py` | 배치 스케일링 + prefill/denoise 분리 |
| `examples/libero/main_timed.py` | 타이밍 계측된 LIBERO 롤아웃 |
| `webapp/` | 단일 프로세스 웹 콘솔 (정책 + 시뮬 + 디노이징 트레이스) |
| `pyproject.toml.orig` | 손대지 않은 업스트림 파일 — **1장의 복원 원본** |
