# A100 80GB PCIe ×2 이식 결과 — GB10 대비

`PORTING_A100.md`의 절차를 A100 박스에서 실제로 수행하고, 같은 벤치마크를 돌린 결과입니다.
GB10 기준선은 `profiling/baselines/gb10/`, 이번 측정 원본은 `profiling/baselines/a100/`.

한 줄 요약: **4개 예측 중 3개가 맞았고, "지연 방향이 자명하지 않다"던 예측 3은
A100이 2.5배 빠른 쪽으로 명확히 갈렸습니다. 다만 그 이득은 전부 서버 처리량 상한으로만
가고, 서버 직렬화라는 진짜 병목은 예측대로 전혀 달라지지 않았습니다.**

| | 박스 |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe ×2, `sm_80`, SM 108개 |
| 드라이버 / CUDA | 575.51.03 / 12.9 |
| torch | 2.7.1+cu126 (업스트림 핀 복원) |
| JAX | 0.5.3 CPU 전용 (`PORTING_A100.md` §1 옵션 (a)) |
| nsys / ncu | 2025.3.1 / 2025.2.0 |

---

## 1. 예측 검증

### 예측 1 — `cutlass_80_*` 발견이 자동 해소된다 → **맞음 (그리고 그 이상)**

4 클라이언트 동일 워크로드, `--cuda-graph-trace node` 기준 커널 분류:

| 분류 | GB10 ms | GB10 % | A100 ms | A100 % |
| --- | ---: | ---: | ---: | ---: |
| Triton (Inductor fused) | 2780 | 24.4% | 3228 | 54.8% |
| GEMM `cutlass_80_*tensorop` | 3454 | 30.4% | 219 | 3.7% |
| GEMM `cutlass_80_*wmma*` | 2980 | 26.2% | 0 | **0%** |
| GEMM `ampere_*gemm` (cuBLAS) | 0 | 0% | 2142 | **36.4%** |
| GEMM `nvjet_sm121_*` | 2010 | 17.7% | 0 | **0%** |
| 기타 | 152 | 1.3% | 302 | 5.1% |
| **합계** | **11377** | | **5890** | |

`nvjet_sm121_*`는 완전히 사라졌고, GPU 시간 총량은 1.93배 줄었습니다.

다만 예측이 예상한 "A100에서는 `cutlass_80` 비중이 **올라간다**"는 빗나갔습니다. A100의
네이티브 GEMM 경로는 CUTLASS가 아니라 **cuBLAS `ampere_bf16_s16816gemm_*`** 였고,
`cutlass_80`은 오히려 30.4% → 3.7%로 줄었습니다. 즉 GB10에서 본 `cutlass_80` 지배는
"A100에서 잘 튜닝된 경로를 미리 쓰고 있던 것"이 아니라, **sm_121용 경로가 부실해 세대가
다른 라이브러리로 폴백한 것**이었습니다. 결론적으로 GB10에서 제기한 커널 선택 가설은
A100에서는 성립하지 않으며, GB10 고유 이슈였다는 판정이 맞습니다.

재현: `.venv/bin/python profiling/compare_kernels.py profiling/baselines/gb10/c4_node_cuda_gpu_kern_sum.csv profiling/baselines/a100/c4_node_cuda_gpu_kern_sum.csv`

### 예측 2 — 배치 스케일링이 크게 좋아진다 → **방향은 맞고, 크기는 빗나감**

`Not enough SMs to use max_autotune_gemm mode` 경고는 A100(108 SM)에서 **사라졌습니다.**
그런데 배치 효율 개선은 1.46x → 1.56x로 소폭입니다.

컴파일 경로(실제 서빙 경로, 10 Euler steps):

| 배치 | 컴파일 s | 지연 ms | 로봇당 ms | 직렬 대비 | GB10 로봇당 | GB10 직렬 대비 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.4 | 54.7 | 54.7 | 1.00x | 148.2 | 1.00x |
| 2 | 67.9 | 98.2 | 49.1 | 1.11x | 118.4 | 1.25x |
| 4 | 0.3 | 162.9 | 40.7 | 1.34x | 105.3 | 1.41x |
| 8 | 0.4 | 279.8 | 35.0 | **1.56x** | 101.4 | 1.46x |

이유는 uncompiled 분해에서 드러납니다 — **denoise 단계가 배치 1~16 내내 평평합니다**:

| 배치 | prefill ms | ms/denoise step | prefill 비중 |
| ---: | ---: | ---: | ---: |
| 1 | 101.0 | 41.8 | 19.5% |
| 2 | 125.4 | 43.4 | 22.4% |
| 4 | 175.3 | 43.7 | 28.6% |
| 8 | 279.6 | 44.0 | 38.9% |
| 16 | 499.2 | 43.4 | 53.5% |

배치로 버는 것은 전부 prefill 쪽(로봇당 101 → 31 ms)이고, denoise 루프는 SM을 더 줘도
줄지 않습니다. GB10에서 "SM이 부족해서 배치 이득이 없다"고 본 진단은 절반만 맞았습니다.
SM을 5배 늘려도 denoise는 그대로였으므로, **denoise는 SM 부족이 아니라 다른 것에 묶여
있습니다.** 배치 16까지도 43 ms/step로 고정이라는 점이 그 근거입니다.

### 예측 3 — 절대 지연의 방향은 자명하지 않다 → **A100이 2.5배 빠름으로 판정**

| | A100 | GB10 |
| --- | ---: | ---: |
| `bench_infer.py` 평균 | **59.4 ms** | 148.2 ms |
| p50 / p99 | 59.5 / 60.5 ms | — / — |
| 지속 제어 주파수 (청크마다 재계획) | 168.3 Hz | 67.5 Hz |
| 가중치 GPU 메모리 | 7.00 GiB | 7.00 GiB |

메모리 대역폭 논리(HBM2e ~2.0 TB/s vs LPDDR5X ~273 GB/s)가 이겼습니다. denoise가
배치에 둔감하면서(예측 2) 대역폭에는 민감하다는 것이 서로 일관됩니다.

> ⚠️ **이 값을 재현할 때 주의**: 첫 측정에서는 146.6 ms가 나왔습니다. 아래 §3의 GPU 공유
> 문제 때문이며, 오염된 측정은 정확히 GB10 값처럼 보입니다.

### 예측 4 — 서버 직렬화는 전혀 달라지지 않는다 → **완전히 맞음**

250 ms 데드라인(`replan_steps=5` @ 20 Hz), 클라이언트당 30요청:

| 클라이언트 | 처리량 | e2e 평균 | 큐 대기 p50 | `policy.infer` p50 | 서버 busy | 데드라인 미스 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.07 req/s | 73.2 ms | 0.8 ms | 71.9 ms | 29.5% | 0% |
| 2 | 8.03 req/s | 109.5 ms | 70.5 ms | 71.8 ms | 58.2% | **0%** |
| 4 | 13.88 req/s | 276.1 ms | 213.9 ms | 70.8 ms | 99.2% | 91.7% |
| 8 | 13.79 req/s | 567.7 ms | 504.6 ms | 71.4 ms | 99.2% | 98.8% |

GB10과 동일한 양상입니다: `policy.infer`는 어떤 부하에서도 70.8–71.9 ms로 평평하고, 늘어난
지연은 전부 큐 대기이며, `peak_inflight`는 saturate 모드에서도 **1**입니다. 처리량 상한만
6.7 → 13.9 req/s로 정확히 2배 올랐습니다(= 1/72 ms).

실질적 차이는 **한 서버가 감당하는 로봇이 1대 → 2대**가 됐다는 것입니다. GB10은 2대에서
이미 97% 미스였지만 A100은 2대에서 0% 미스, 4대에서 92.5% 미스입니다.

saturate 모드(4 클라이언트, 백투백): 15.78 req/s, e2e 248.3 ms, busy 99.1%, `peak_inflight` 1.

위 표는 GB10 기준선과 조건을 맞추려고 **nsys를 켠 상태**로 측정했습니다. 같은 c4를 nsys
없이 재면 13.88 → 15.12 req/s, `infer` p50 70.8 → 65.0 ms로, **nsys 트레이싱 오버헤드가
약 9%** 있습니다(`profiling/baselines/a100/c4_rate_no_nsys.json`). 절대 지연을 인용할
때는 §1 예측 3의 59.4 ms(nsys 없음)를 쓰세요.

---

## 2. 절차상 가이드와 달랐던 점

`PORTING_A100.md`대로 하면 막히는 지점들입니다.

| 단계 | 가이드 | 실제 A100 박스 |
| --- | --- | --- |
| §1 의존성 | `cp pyproject.toml.orig pyproject.toml` 후 `uv sync` | 그대로 동작. 추가로 **`cp -r src/openpi/models_pytorch/transformers_replace/* .venv/.../transformers/` 필요** — 세 문서 어디에도 없는데 없으면 체크포인트 변환이 즉시 실패 |
| §2 체크포인트 | `maybe_download`가 알아서 받음 | **동작 안 함.** 두 가지가 겹침: (a) gcsfs가 fsspec 콜백 객체를 aiohttp 쿼리 파라미터로 넘겨 `yarl`이 `int()`에서 죽음, (b) `storage.googleapis.com`이 AAAA만 응답하는데 박스에 IPv6 경로가 없어 `requests`/`curl`은 아예 불통(gcsfs의 aiohttp만 IPv4 폴백). 우회 스크립트로 11.58 GiB를 받아야 했고, 토크나이저(`gs://big_vision/paligemma_tokenizer.model`)도 같은 이유로 수동 배치 필요 |
| §3 LIBERO venv | 나열된 패키지 설치 | 목록에 **`matplotlib`, `cloudpickle`, `gym` 누락**. 그리고 **`mujoco==2.3.7` 핀 필요** — robosuite 1.4.1이 `mujoco>=2.3.0`만 요구해서 3.11이 깔리는데, `MjData.qM`이 `M`으로 개명돼 환경 생성이 실패. 첫 `import libero`는 데이터셋 경로를 대화형으로 물으므로 자동화 시 입력을 먹여야 함 |
| §3 EGL | egl → 벤더 강제 → osmesa 순 | **`/usr/share/glvnd/egl_vendor.d/`에 `50_mesa.json`만 있음.** `MUJOCO_GL=egl`이 실패하지 않고 조용히 Mesa 소프트웨어 렌더로 떨어져 `env.step`이 407 ms(GB10의 40배). `libEGL_nvidia.so.0`는 설치돼 있으므로 벤더 JSON을 직접 써서 `__EGL_VENDOR_LIBRARY_FILENAMES`로 지정하면 **51 ms**로 회복 |
| §4 `run_bench.sh` | 그대로 실행 | **nsys의 `--trace`에서 `cudnn`을 빼야 함.** 넣으면 Triton의 드라이버 핸들이 초기화되지 않아 모든 inductor 컴파일이 `Triton Error [CUDA]: initialization error`로 죽고 서버가 뜨지 못함. `cuda,nvtx,cublas`는 정상 (`run_bench.sh`에 반영함) |
| §8 프로파일링 권한 | "A100은 root 있는 경우가 많으니 풀 수 있다" | **불가.** 이 박스는 도커 컨테이너이고 `CAP_SYS_ADMIN`이 없어 `sudo`로도 `ERR_NVGPUCTRPERM`, `/proc/sys`도 read-only라 `perf_event_paranoid`도 못 바꿈. 결과적으로 GB10과 **동일하게** CUDA+NVTX 트레이싱만 가능하고 SM 사용률은 NVML 폴백 — 비교 조건이 같아진다는 점에서는 오히려 다행 |
| §8 `--cuda-graph-trace node` | A100에서도 필수 | 그대로 필수. 확인함 |

`--recurse-submodules`, `assets/` 복사, `_libero_root.pth` 절대경로 재생성,
`sitecustomize.py` weights_only 우회는 모두 가이드대로 필요했고 그대로 동작했습니다.

---

## 3. ⚠️ 이 박스 고유의 측정 함정 — GPU를 다른 워크로드가 공유합니다

컨테이너 안에서는 `nvidia-smi`가 **다른 프로세스를 보여주지 않습니다**(`--query-compute-apps`가
빈 결과). 반면 `torch.cuda.mem_get_info()`는 장치 전체 사용량을 정확히 봅니다. 실제로
측정 중 외부 워크로드가 65 GiB를 점유한 구간이 있었고, 그때의 결과는:

| | 오염된 측정 | 깨끗한 측정 |
| --- | ---: | ---: |
| `bench_infer` 평균 | 146.6 ms | **59.4 ms** |
| `bench_batch` B=2 컴파일 | 589.9 s | **67.9 s** |
| `bench_batch` 컴파일 구간 | 조용히 프로세스 사망 | 정상 완료 |

오염된 값이 하필 GB10 기준선(148 ms)과 거의 같아서, **그대로 믿으면 "A100이나 GB10이나
똑같다"는 정반대 결론**이 납니다.

측정 전후로 반드시 확인하세요:

```bash
.venv/bin/python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'free {f/2**30:.1f}/{t/2**30:.1f} GiB')"
# 79.3 GiB 중 78 GiB 이상 free여야 깨끗한 상태
```

`bench_batch.py`와 `serve_profiled.py`는 이 값을 자체적으로 찍습니다
(`GPU free at start`, `gpu_used_gib`). 리포트를 읽을 때 먼저 그 줄을 보세요.

---

## 4. A100 2장 활용에 대한 함의

가이드 §7의 결론(데이터 병렬 = 서버 2개)은 그대로 유효하고, 이번 측정으로 숫자가 붙습니다.

- 서버 1개 = 로봇 2대 (250 ms 데드라인, 미스 0%)
- A100 2장 = **로봇 4대**, 라우터 없이 포트만 나눠서
- 동적 배칭을 붙이면 B=8에서 로봇당 35 ms이므로 이론상 서버당 더 올릴 수 있지만,
  denoise가 배치에 평평하다는 §1 예측 2 결과 때문에 **기대치는 낮춰 잡아야 합니다.**
  B=8의 총지연 279.8 ms는 250 ms 데드라인을 이미 넘습니다. 데드라인을 지키면서 쓸 수
  있는 최대 배치는 B=4(162.9 ms)이고, 이때 서버당 로봇 4대 → 2장이면 8대가 상한선입니다.

---

## 5. 재현 명령

```bash
# 깨끗한 GPU 확인 후
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_infer.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_batch.py
CUDA_VISIBLE_DEVICES=0 profiling/run_bench.sh -c "1 2 4 8" -n 30
CUDA_VISIBLE_DEVICES=0 profiling/run_bench.sh -c 4 -m saturate --no-nsys -n 30
.venv/bin/python profiling/compare_kernels.py \
    profiling/baselines/gb10/c4_node_cuda_gpu_kern_sum.csv \
    profiling/baselines/a100/c4_node_cuda_gpu_kern_sum.csv

# LIBERO 클라이언트를 쓸 때 (NVIDIA EGL 강제)
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=$PWD/profiling/egl_nvidia.json
```

로그 원본은 `logs/2026-08-04_*`에 있습니다.
