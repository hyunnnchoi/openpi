# A100 80GB PCIe ×2 이식 결과 — GB10 대비

`PORTING_A100.md`의 절차를 A100 박스에서 실제로 수행하고, 같은 벤치마크를 돌린 결과입니다.
GB10 기준선은 `profiling/baselines/gb10/`, 이번 측정 원본은 `profiling/baselines/a100/`.

> **측정 출처.** 이 문서의 모든 A100 수치는 **GPU 단독 점유가 확인된 상태에서 다시 측정한
> 값**입니다(§3 참조). 각 런의 `resources.jsonl`에서 `gpu_used_gib`가 전 구간 7.76 GiB
> 고정(= 자기 가중치뿐)임을 확인했고, `bench_batch`는 `GPU free at start 78.8/79.3 GiB`로
> 시작했습니다. 초기에 GPU를 공유한 상태로 나온 측정치는 `archive_contaminated/`에 격리해
> 두었고 이 문서에는 반영하지 않았습니다.

한 줄 요약: **4개 예측 중 2개가 맞고, 1개는 반증됐으며, 1개는 방향이 갈렸습니다.
A100이 2.3배 빠르지만 그 이득은 전부 서버 처리량 상한으로만 가고, 서버 직렬화라는 진짜
병목은 예측대로 전혀 달라지지 않았습니다. 예측에 없던 발견으로, 이 박스에서는 GPU가
추론 시간의 1/4을 호스트 대기로 놉니다.**

| | 박스 |
| --- | --- |
| GPU | NVIDIA A100 80GB PCIe ×2, `sm_80`, SM 108개 (측정은 `CUDA_VISIBLE_DEVICES=0` 1장) |
| CPU | Intel Xeon Gold 6230 @ 2.10GHz, 16 코어 |
| 드라이버 / CUDA | 575.51.03 / 12.9 |
| torch | 2.7.1+cu126 (업스트림 핀 복원) |
| JAX | 0.5.3 CPU 전용 (`PORTING_A100.md` §1 옵션 (a)) |
| nsys / ncu | 2025.3.1 / 2025.2.0 |

---

## 1. 예측 검증

### 예측 1 — `cutlass_80_*` 발견이 자동 해소된다 → **맞음, 단 세부는 빗나감**

4 클라이언트 동일 워크로드, `--cuda-graph-trace node` 기준 커널 분류:

| 분류 | GB10 ms | GB10 % | A100 ms | A100 % |
| --- | ---: | ---: | ---: | ---: |
| Triton (Inductor fused) | 2780 | 24.4% | 3233 | **54.8%** |
| GEMM `cutlass_80_*tensorop` | 3454 | 30.4% | 219 | 3.7% |
| GEMM `cutlass_80_*wmma*` | 2980 | 26.2% | 0 | **0%** |
| GEMM `ampere_*gemm` (cuBLAS) | 0 | 0% | 2146 | **36.4%** |
| GEMM `nvjet_sm121_*` | 2010 | 17.7% | 0 | **0%** |
| 기타 | 152 | 1.3% | 302 | 5.1% |
| **합계** | **11377** | | **5900** | |

`nvjet_sm121_*`는 완전히 사라졌고, GPU 시간 총량은 1.93배 줄었습니다.

다만 예측이 예상한 "A100에서는 `cutlass_80` 비중이 **올라간다**"는 빗나갔습니다. A100의
네이티브 GEMM 경로는 CUTLASS가 아니라 **cuBLAS `ampere_bf16_s16816gemm_*`** 였고,
`cutlass_80`은 오히려 30.4% → 3.7%로 줄었습니다. 즉 GB10에서 본 `cutlass_80` 지배는
"A100에서 잘 튜닝된 경로를 미리 쓰고 있던 것"이 아니라, **sm_121용 경로가 부실해 세대가
다른 라이브러리로 폴백한 것**이었습니다. 결론적으로 GB10에서 제기한 커널 선택 가설은
A100에서는 성립하지 않으며, GB10 고유 이슈였다는 판정이 맞습니다.

재현: `.venv/bin/python profiling/compare_kernels.py profiling/baselines/gb10/c4_node_cuda_gpu_kern_sum.csv profiling/baselines/a100/c4_node_cuda_gpu_kern_sum.csv`

### 예측 2 — 배치 스케일링이 크게 좋아진다 → **반증에 가까움**

`Not enough SMs to use max_autotune_gemm mode` 경고는 A100(108 SM)에서 **사라졌습니다**
(`bench_batch` 로그 전체에서 0회). 그런데 배치 효율 개선은 1.46x → 1.61x로 소폭입니다.

컴파일 경로(실제 서빙 경로, 10 Euler steps):

| 배치 | 컴파일 s | 지연 ms | 로봇당 ms | 직렬 대비 | GB10 로봇당 | GB10 직렬 대비 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 34.8 | 56.5 | 56.5 | 1.00x | 148.2 | 1.00x |
| 2 | 55.9 | 98.8 | 49.4 | 1.14x | 118.4 | 1.25x |
| 4 | 0.3 | 162.6 | 40.6 | 1.39x | 105.3 | 1.41x |
| 8 | 0.4 | 280.4 | 35.1 | **1.61x** | 101.4 | 1.46x |

이유는 uncompiled 분해에서 드러납니다 — **denoise 단계가 배치 1~16 내내 평평합니다**:

| 배치 | prefill ms | ms/denoise step | prefill 비중 |
| ---: | ---: | ---: | ---: |
| 1 | 103.1 | 41.8 | 19.8% |
| 2 | 124.0 | 43.9 | 22.0% |
| 4 | 175.7 | 43.8 | 28.6% |
| 8 | 282.6 | 43.2 | 39.6% |
| 16 | 497.1 | 44.8 | 52.6% |

배치로 버는 것은 전부 prefill 쪽(로봇당 103 → 31 ms)이고, denoise 루프는 SM을 더 줘도
줄지 않습니다. **GB10에서 "SM이 부족해서 배치 이득이 없다"고 본 진단은 A100에서 반증됐습니다.**
SM을 5배 늘려도 denoise는 배치 16까지 43~45 ms/step로 고정이므로, denoise를 묶고 있는 것은
SM 수가 아닙니다.

### 예측 3 — 절대 지연의 방향은 자명하지 않다 → **A100이 2.3배 빠름으로 판정**

| | A100 | GB10 |
| --- | ---: | ---: |
| `bench_infer.py` 평균 | **64.61 ms** | 148.2 ms |
| p50 / p99 | 64.64 / 65.83 ms | — / — |
| min / max | 63.37 / 65.83 ms | — / — |
| 지속 제어 주파수 (청크마다 재계획) | 154.8 Hz | 67.5 Hz |
| 가중치 GPU 메모리 | 7.00 GiB | 7.00 GiB |

메모리 대역폭 논리(HBM2e ~2.0 TB/s vs LPDDR5X ~273 GB/s)가 이겼습니다. denoise가
배치에 둔감하면서(예측 2) 대역폭에는 민감하다는 것이 서로 일관됩니다.

> p99가 평균보다 1.2 ms밖에 높지 않다는 점이 이 측정이 단독 점유였다는 증거입니다.
> GPU를 공유한 상태의 같은 측정은 평균 75.85 / p99 102.56 ms로 꼬리가 길게 늘어졌습니다.

### 예측 4 — 서버 직렬화는 전혀 달라지지 않는다 → **완전히 맞음**

250 ms 데드라인(`replan_steps=5` @ 20 Hz), 클라이언트당 30요청, **nsys 켠 상태로 통일**:

| 클라이언트 | 처리량 | e2e 평균 | e2e p99 | 큐 대기 p50 | `policy.infer` p50 | 서버 busy | 데드라인 미스 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.07 req/s | 75.4 ms | 81.6 ms | 0.8 ms | 73.1 ms | 30.4% | 0% |
| 2 | 8.04 req/s | 115.3 ms | 164.9 ms | 71.5 ms | 76.5 ms | 61.3% | **0%** |
| 4 | 13.70 req/s | 281.8 ms | 295.7 ms | 216.4 ms | 71.5 ms | 99.2% | 93.3% |
| 8 | 13.29 req/s | 589.8 ms | 648.7 ms | 508.4 ms | 72.3 ms | 99.2% | 98.8% |

GB10과 동일한 양상입니다: `policy.infer`는 어떤 부하에서도 71.5–76.5 ms로 평평하고, 늘어난
지연은 전부 큐 대기이며, `peak_inflight`는 saturate 모드에서도 **1**입니다. 처리량 상한만
6.7 → 13.3 req/s로 정확히 2배 올랐습니다(= 1/75 ms).

실질적 차이는 **한 서버가 감당하는 로봇이 1대 → 2대**가 됐다는 것입니다. GB10은 2대에서
이미 97% 미스였지만 A100은 2대에서 0% 미스, 4대에서 93% 미스입니다.

saturate 모드(4 클라이언트, 백투백): **16.19 req/s, e2e 240.0 ms** (GB10 6.80 req/s /
575.7 ms) — 2.38배.

---

## 2. 예측에 없던 발견 — 이 박스에서는 GPU가 호스트를 기다립니다

GB10 실험은 "요청당 커널 런치가 5,102개인데도 96% GPU-busy이므로 런치 바운드가 아니다"로
결론지었습니다. **A100에서는 이 결론이 성립하지 않습니다.**

| | GB10 | A100 |
| --- | ---: | ---: |
| 요청당 GPU 시간 | 142.2 ms | **49.2 ms** |
| 요청당 `sample_actions` | 148.1 ms | **65.4 ms** |
| **GPU-busy** | **96%** | **75.2%** |
| NVML SM 사용률 (c4 구간) | 93% | **64%** |

즉 추론 시간의 약 1/4 동안 GPU가 놉니다. 근거는 비컴파일(eager) 경로에서 더 선명합니다:

| | GB10 | A100 |
| --- | ---: | ---: |
| eager B=1, 10 steps | 214.0 ms | **520.9 ms** (2.4배 느림) |
| eager denoise / step | 9.8 ms | **41.8 ms** (4.3배 느림) |
| `torch.compile` 이득 (eager → 컴파일) | 1.45배 | **9.2배** |

A100 쪽 GPU가 GB10보다 모든 면에서 빠른데도 eager 경로만 크게 느립니다. 커널 런치를
발행하는 것은 호스트이고, 이 박스 CPU는 **Xeon Gold 6230 @ 2.10GHz(2019년형)** 로 GB10의
Grace 코어보다 단일 스레드가 느립니다. 그래서 CUDA 그래프로 런치를 묶어주는
`torch.compile`의 효과가 A100에서 훨씬 큽니다.

**실무적 함의:** 이 박스에서 pi0.5 서빙 성능을 더 밀어붙이려면 GPU를 바꾸는 것보다
호스트 오버헤드를 줄이는 쪽(그래프 캡처 범위 확대, 런치 수 감소)이 남은 여지입니다.

### nsys 계측 오버헤드가 무시할 수 없습니다

같은 c4를 nsys 유/무로 재면:

| c4 (4 클라이언트) | nsys 있음 | nsys 없음 |
| --- | ---: | ---: |
| 처리량 | 13.70 req/s | **15.83 req/s** |
| `policy.infer` p50 | 71.5 ms | **60.9 ms** |
| e2e 평균 | 281.8 ms | **157.9 ms** |
| 데드라인 미스 | 93.3% | **9.2%** |

`--cuda-graph-trace node`는 CUDA 그래프를 노드 단위로 풀어 기록하므로 호스트 비용이 붙고,
위에서 본 대로 이 박스는 호스트가 병목이라 그 비용이 증폭됩니다(**약 17%**). GB10에서는
이 차이가 거의 없었습니다. **절대 지연을 인용할 때는 §1 예측 3의 64.61 ms(nsys 없음)를
쓰고, GB10 기준선과 비교할 때만 위의 nsys 표를 쓰세요.**
원본: `profiling/baselines/a100/c4_rate_nonsys.json`.

---

## 3. ⚠️ 측정 위생 — 이 인스턴스에서 GPU를 공유하게 되는 두 경로

초기 측정이 오염됐고, 원인이 둘이었습니다.

**(1) 다른 학습/검증 잡.** `training_validation.py`가 두 장 모두 65.8 GiB / 99% util을
점유한 구간이 있었습니다. 이 경우 `bench_infer`가 CUDA OOM으로 죽습니다(80 GB 카드에서
7 GiB 로드 실패).

**(2) 포크된 헤드리스 에이전트 세션.** 같은 대화에서 갈라져 나온 Claude Code 세션이
`bg-pty-host` 아래에서 **동일한 벤치마크를 독립적으로 실행**하고 있었습니다.
프로세스 계보로 확인:

```bash
pgrep -af "claude.exe --session-id"     # --fork-session --resume ... 이 보이면 그것
```

이쪽이 더 위험합니다. 서로 같은 `data/profiling/`에 쓰기 때문에 파일이 덮어써지고,
GPU 경합으로 `Triton Error [CUDA]: initialization error`나 컴파일 중 프로세스 사망이
간헐적으로 발생합니다.

**오염된 값과 깨끗한 값의 차이:**

| | 오염된 측정 | 깨끗한 측정 |
| --- | ---: | ---: |
| `bench_infer` 평균 / p99 | 75.85 / 102.56 ms | **64.61 / 65.83 ms** |
| `bench_batch` 시작 시 여유 | 70.8 / 79.3 GiB | **78.8 / 79.3 GiB** |
| 스윕 | Triton init 에러로 중단 | 정상 완료 |

**측정 전 확인:**

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
.venv/bin/python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'free {f/2**30:.1f}/{t/2**30:.1f} GiB')"
# 79.3 GiB 중 78 GiB 이상 free여야 깨끗한 상태
pgrep -af "claude.exe --session-id"   # 포크 세션이 돌고 있지 않은지
```

**측정 후 확인:** `bench_batch.py`와 `serve_profiled.py`는 이 값을 자체적으로 찍습니다
(`GPU free at start`, `resources.jsonl`의 `gpu_used_gib`). 리포트를 읽기 전에
`gpu_used_gib`가 전 구간 7.76 GiB 고정인지 먼저 보세요 — 이번 스윕 4개 런
(샘플 91/97/115/206개)이 모두 그 조건을 만족합니다.

---

## 4. 절차상 가이드와 달랐던 점

`PORTING_A100.md`대로 하면 막히는 지점들입니다.

| 단계 | 가이드 | 실제 A100 박스 |
| --- | --- | --- |
| §1 의존성 | `cp pyproject.toml.orig pyproject.toml` 후 `uv sync` | 그대로 동작. 추가로 **`cp -r src/openpi/models_pytorch/transformers_replace/* .venv/.../transformers/` 필요** — 세 문서 어디에도 없는데 없으면 체크포인트 변환이 즉시 실패 |
| §2 체크포인트 | `maybe_download`가 알아서 받음 | **동작 안 함.** 두 가지가 겹침: (a) gcsfs가 fsspec 콜백 객체를 aiohttp 쿼리 파라미터로 넘겨 `yarl`이 `int()`에서 죽음, (b) `storage.googleapis.com`이 AAAA만 응답하는데 박스에 IPv6 경로가 없어 `requests`/`curl`은 아예 불통(gcsfs의 aiohttp만 IPv4 폴백). 우회 스크립트로 11.58 GiB를 받아야 했고, 토크나이저(`gs://big_vision/paligemma_tokenizer.model`)도 같은 이유로 수동 배치 필요 |
| §3 LIBERO venv | 나열된 패키지 설치 | 목록에 **`matplotlib`, `cloudpickle`, `gym` 누락**. 그리고 **`mujoco==2.3.7` 핀 필요** — robosuite 1.4.1이 `mujoco>=2.3.0`만 요구해서 3.11이 깔리는데, `MjData.qM`이 `M`으로 개명돼 환경 생성이 실패. 첫 `import libero`는 데이터셋 경로를 대화형으로 물으므로 자동화 시 입력을 먹여야 함. `uv venv`에는 `pip`이 없으므로 `uv pip install -p <venv>/bin/python` 사용 |
| §3 EGL | egl → 벤더 강제 → osmesa 순 | **`/usr/share/glvnd/egl_vendor.d/`에 `50_mesa.json`만 있음.** `MUJOCO_GL=egl`이 실패하지 않고 조용히 Mesa 소프트웨어 렌더로 떨어져 `env.step`이 407 ms(GB10의 40배). `libEGL_nvidia.so.0`는 설치돼 있으므로 벤더 JSON을 직접 써서 `__EGL_VENDOR_LIBRARY_FILENAMES`로 지정하면 **51 ms**로 회복 |
| §4 `run_bench.sh` | 그대로 실행 | **nsys의 `--trace`에서 `cudnn`을 빼야 함.** 넣으면 Triton의 드라이버 핸들이 초기화되지 않아 모든 inductor 컴파일이 `Triton Error [CUDA]: initialization error`로 죽고 서버가 뜨지 못함. `cuda,nvtx,cublas`는 정상 (`run_bench.sh`에 반영함) |
| §4 c8 실행 | 그대로 실행 | **제어 연결이 keepalive 타임아웃으로 죽음.** 서버가 asyncio 핸들러 안에서 `policy.infer`를 동기 호출하므로, 8 클라이언트에서는 큐가 빠지는 동안 이벤트 루프가 ping에 응답하지 못하고 기본 20초 타임아웃에 걸립니다. 그 블로킹 자체가 측정 대상이므로 `load_clients.py`의 연결에 `ping_interval=None`을 넣었습니다 |
| §8 프로파일링 권한 | "A100은 root 있는 경우가 많으니 풀 수 있다" | **불가.** 이 박스는 도커 컨테이너이고 `CAP_SYS_ADMIN`이 없어 `sudo`로도 `ERR_NVGPUCTRPERM`(`RmProfilingAdminOnly: 1`), `/proc/sys`도 read-only라 `perf_event_paranoid`도 못 바꿈. 결과적으로 GB10과 **동일하게** CUDA+NVTX 트레이싱만 가능하고 SM 사용률은 NVML 폴백 — 비교 조건이 같아진다는 점에서는 오히려 다행 |
| §8 `--cuda-graph-trace node` | A100에서도 필수 | 그대로 필수. 확인함. 다만 §2에서 본 대로 호스트 오버헤드 17%를 감수하는 것임 |

`--recurse-submodules`, `assets/` 복사, `_libero_root.pth` 절대경로 재생성,
`sitecustomize.py` weights_only 우회는 모두 가이드대로 필요했고 그대로 동작했습니다.

---

## 5. A100 2장 활용에 대한 함의

가이드 §7의 결론(데이터 병렬 = 서버 2개)은 그대로 유효하고, 이번 측정으로 숫자가 붙습니다.

- 서버 1개 = 로봇 2대 (250 ms 데드라인, 미스 0%)
- A100 2장 = **로봇 4대**, 라우터 없이 포트만 나눠서
- 동적 배칭을 붙이면 B=8에서 로봇당 35.1 ms이므로 이론상 서버당 더 올릴 수 있지만,
  denoise가 배치에 평평하다는 예측 2 결과 때문에 **기대치는 낮춰 잡아야 합니다.**
  B=8의 총지연 280.4 ms는 250 ms 데드라인을 이미 넘습니다. 데드라인을 지키면서 쓸 수
  있는 최대 배치는 **B=4(162.6 ms)** 이고, 이때 서버당 로봇 4대 → 2장이면 8대가 상한선입니다.
- 그 위로 더 가려면 §2의 호스트 병목(GPU-busy 75%)을 먼저 걷어내야 합니다.

---

## 6. 재현 명령

```bash
# 깨끗한 GPU 확인 후 (§3)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_infer.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_batch.py
CUDA_VISIBLE_DEVICES=0 profiling/run_bench.sh -c "1 2 4 8" -n 30
CUDA_VISIBLE_DEVICES=0 profiling/run_bench.sh -c 4 -m saturate --no-nsys -n 30
CUDA_VISIBLE_DEVICES=0 profiling/run_bench.sh -c 4 -n 30 --no-nsys   # nsys 오버헤드 측정용
.venv/bin/python profiling/compare_kernels.py \
    profiling/baselines/gb10/c4_node_cuda_gpu_kern_sum.csv \
    profiling/baselines/a100/c4_node_cuda_gpu_kern_sum.csv

# LIBERO 클라이언트를 쓸 때 (NVIDIA EGL 강제)
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=$PWD/profiling/egl_nvidia.json
```

`profiling/baselines/a100/` 파일 대응:

| 파일 | 내용 |
| --- | --- |
| `bench_infer.log` | 예측 3 (nsys 없음, 단독 점유) |
| `bench_batch.json` | 예측 2 (배치 1–16, 컴파일/비컴파일) |
| `report.json` | 예측 4 스윕 전체 + NVTX/커널/NVML |
| `c1_rate.json` `c2_rate.json` `c4_rate.json` `c8_rate.json` | 스윕 각 런 (nsys 켬) |
| `c4_rate_nonsys.json` | §2의 nsys 오버헤드 비교군 |
| `c4_saturate.json` | saturate 모드 |
| `c4_node_cuda_gpu_kern_sum.csv` | 예측 1 커널 분해 |

로그 원본은 `logs/2026-08-04_15-10_*` 이후 파일들입니다(그 이전은 오염 구간).
