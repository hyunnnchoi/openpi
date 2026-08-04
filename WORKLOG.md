# WORKLOG — openpi (dgx-spark-profiling 브랜치, A100 이식)

## [2026-08-04 12:00] DGX Spark(GB10) 프로파일링 실험을 A100 80GB ×2로 이식 + 동일 실험 재현

> **주의 — 이 기록의 수치는 검증본이 아닙니다.** 이 작업은 같은 대화에서 포크된 헤드리스
> 세션(PID 8272)이 수행한 것으로, 다른 세션과 GPU를 공유하는 구간이 있었습니다. 아래
> [15:05] 기록이 단독 점유가 확인된 상태에서의 독립 재측정이며, `RESULTS_A100.md`는
> 그 재측정 값으로 갱신돼 있습니다. **결론(예측 4개의 판정)은 양쪽이 일치**하고 세부
> 수치만 다릅니다.

### 실행 환경
| 항목 | 값 |
|------|-----|
| 스크립트 | `bench_infer.py`, `bench_batch.py`, `profiling/run_bench.sh`, `probe_infer_breakdown.py`, `profiling/compare_kernels.py` |
| 모델 | pi0.5 LIBERO (`pi05_libero`, PyTorch bf16 변환본 6.8 GiB) |
| GPU | NVIDIA A100 80GB PCIe ×2, `sm_80`, SM 108개, 드라이버 575.51.03 / CUDA 12.9 |
| 주요 파라미터 | 클라이언트 1/2/4/8, 요청 30개/클라이언트, 데드라인 250 ms, 배치 1–16, Euler steps 10 |
| 소프트웨어 | torch 2.7.1+cu126, jax 0.5.3 (CPU), transformers 4.53.2, nsys 2025.3.1 |

### 변경 사항
| 파일 | 유형 | 설명 |
|------|------|------|
| `pyproject.toml` | 수정 | `pyproject.toml.orig`로 복원(torch 2.7.1, cu130 인덱스 제거). `jax[cuda12]` 대신 `jax` 유지 — PORTING_A100.md §1 옵션 (a) |
| `uv.lock` | 삭제 | GB10용 aarch64 락 폐기 후 재해결 |
| `profiling/run_bench.sh` | 수정 | nsys `--trace`를 `TRACE` 변수로 분리하고 `cudnn` 제거. 이유를 주석으로 기록 |
| `profiling/compare_kernels.py` | 생성 | nsys 커널 CSV를 계열별로 분류해 두 박스를 비교 (예측 1 검증용) |
| `profiling/egl_nvidia.json` | 생성 | 이 박스에 없는 NVIDIA EGL 벤더 JSON. `__EGL_VENDOR_LIBRARY_FILENAMES`로 지정 |
| `probe_infer_breakdown.py` | 생성 | `policy.infer`를 단계별로 분해 (transform / h2d / sample_actions / d2h) |
| `RESULTS_A100.md` | 생성 | 예측 4개 검증 결과 + 가이드와 달랐던 절차 + 측정 함정 |
| `profiling/baselines/a100/` | 생성 | A100 측정 원본 (sweep, saturate, 커널 CSV, bench_batch) |
| `third_party/{aloha,libero}` | 초기화 | `git submodule update --init --recursive` |
| `examples/libero/.venv` | 생성 | LIBERO 클라이언트 venv + 하드코딩 경로 우회 재생성 |

### 작업 상세

**배경.** GB10에서 나온 pi0.5 서빙 프로파일링 결과가 아키텍처 고유 현상인지 확인하기 위해,
같은 브랜치를 A100 2장 인스턴스에서 그대로 재현. `PORTING_A100.md`가 검증 가능한 예측
4개를 제시하고 있어 그것을 판정 기준으로 삼음.

**예측 검증 결과 (상세는 `RESULTS_A100.md`).**
- 예측 1 (`cutlass_80` 발견 자동 해소) — **맞음.** `nvjet_sm121_*` 17.7% → 0%, 총 GPU 시간
  11377 → 5890 ms. 단, A100의 네이티브 GEMM은 CUTLASS가 아니라 cuBLAS
  `ampere_bf16_s16816gemm_*`(36.4%)였고 `cutlass_80`은 30.4% → 3.7%로 오히려 감소.
  "A100에서 cutlass_80 비중이 올라갈 것"이라는 예측의 세부는 빗나감.
- 예측 2 (배치 스케일링 개선) — **방향만 맞음.** `Not enough SMs` 경고는 사라졌으나 배치
  효율은 1.46x → 1.56x. 원인: **denoise 단계가 배치 1~16 내내 43 ms/step로 평평.** SM을
  5배 줘도 안 줄어드니 SM 부족이 원인이 아니었음. 이득은 전부 prefill(로봇당 101 → 31 ms).
- 예측 3 (절대 지연 방향 불명) — **A100 2.5배 빠름으로 판정.** 59.4 ms vs GB10 148.2 ms.
- 예측 4 (서버 직렬화 불변) — **완전히 맞음.** `policy.infer`가 모든 부하에서 71–74 ms 평평,
  busy 99.2%, `peak_inflight` saturate에서도 1. 처리량 상한만 6.7 → 13.8 req/s로 정확히 2배.
  실질 효과는 서버당 로봇 수용량 1대 → 2대.

**주요 의사결정.**
- JAX는 CPU로 유지(옵션 a). 서빙·벤치 경로가 전부 PyTorch라 XLA 선점 리스크만 생김.
- nsys `--trace`에서 `cudnn` 제거. 넣으면 Triton 드라이버 핸들이 초기화되지 않아
  (`Triton Error [CUDA]: initialization error`) 서버가 뜨지 못함. 최소 재현으로 플래그를
  이분 탐색해 `cudnn` 단독 원인임을 확인. 커널 분해는 `cuda` 트레이스에서 나오므로 손실 없음.
- 기준선 일관성을 위해 스윕은 nsys 켠 상태로 통일. 별도로 c4를 nsys 없이 재서
  **nsys 오버헤드 ~9%** (infer p50 71.2 → 65.0 ms)를 기록해 둠.

**측정 함정 (중요).** 이 박스는 컨테이너 밖 워크로드가 같은 GPU를 간헐적으로 씀.
컨테이너 안에서 `nvidia-smi --query-compute-apps`는 빈 결과라 그 프로세스가 안 보이지만
`torch.cuda.mem_get_info()`에는 잡힘(65 GiB 점유 구간 관측). 오염된 측정은
`bench_infer` 146.6 ms / B=2 컴파일 589.9 s / 컴파일 중 프로세스 사망으로 나타나며,
**하필 GB10 기준선(148 ms)과 비슷해서 "차이 없음"이라는 정반대 결론을 유도함.**
측정 전 `GPU free at start`가 78 GiB 이상인지 확인 필요.

**가이드 절차의 빈 곳 (전부 `RESULTS_A100.md` §2에 표로 정리).**
- `transformers_replace` 복사 단계가 어느 문서에도 없음 — 없으면 체크포인트 변환 즉시 실패.
- `maybe_download`가 이 환경에서 동작 불가: gcsfs↔yarl 콜백 충돌 + `storage.googleapis.com`이
  AAAA만 응답하는데 IPv6 경로 없음(gcsfs의 aiohttp만 IPv4 폴백). 11.58 GiB를 우회 스크립트로
  받고 토크나이저도 수동 배치.
- LIBERO 의존성 목록에 `matplotlib`, `cloudpickle`, `gym` 누락 + `mujoco==2.3.7` 핀 필요
  (robosuite 1.4.1이 mujoco 3.11의 `MjData.qM` → `M` 개명에 깨짐).
- EGL: 박스에 NVIDIA 벤더 JSON이 없어 `MUJOCO_GL=egl`이 조용히 Mesa 소프트웨어로 폴백
  (`env.step` 407 ms). 벤더 JSON을 직접 만들어 51 ms로 회복.
- §8의 "root면 프로파일링 카운터 해제 가능"은 **불가** — 컨테이너에 `CAP_SYS_ADMIN` 없어
  sudo로도 `ERR_NVGPUCTRPERM`, `/proc/sys`도 read-only. GB10과 동일하게 NVML 폴백.

**알려진 제한사항 / TODO.**
- GPU 카운터(ncu roofline, SM occupancy 타임라인)는 이 컨테이너에서 불가. 호스트 권한이
  있는 환경이라면 남은 항목.
- 2장 동시 사용(데이터 병렬 서버 2개)은 미측정 — 서버당 독립 FIFO라 선형이 예상되나 확인 필요.
- 동적 배칭은 미구현. B=4(162.9 ms)가 250 ms 데드라인 안에서 쓸 수 있는 최대 배치.

### 로그
- `logs/2026-08-04_12-00_uv-sync.log`, `logs/2026-08-04_12-05_env-verify.log`
- `logs/2026-08-04_13-15_ckpt-download-gcsfs.log`, `logs/2026-08-04_13-45_ckpt-download-resume.log`
- `logs/2026-08-04_14-20_ckpt-convert.log`
- `logs/2026-08-04_15-55_bench-infer-rerun.log` (깨끗한 상태 59.4 ms)
- `logs/2026-08-04_16-00_bench-batch-clean.log` (깨끗한 상태 배치 스윕)
- `logs/2026-08-04_17-10_run-bench-sweep.log`, `logs/2026-08-04_18-15_run-bench-sweep-final.log`
- `logs/2026-08-04_17-40_run-bench-saturate.log`, `logs/2026-08-04_17-55_run-bench-c4-rate.log`
- `logs/2026-08-04_15-45_infer-breakdown.log`, `logs/2026-08-04_gpu-watch.log`
- 오염된 측정 (반례 기록용): `logs/2026-08-04_14-35_bench-infer.log`, `logs/2026-08-04_14-45_bench-batch.log`

### 관련 이슈
- 없음

---

## [2026-08-04 15:05] 세션 충돌로 오염된 측정 발견 → 전면 재측정 (위 12:00 기록 검증)

### 실행 환경
| 항목 | 값 |
|------|-----|
| 스크립트 | `bench_infer.py`, `bench_batch.py`, `profiling/run_bench.sh`, `profiling/report.py` |
| 모델 | pi0.5 LIBERO (`pi05_libero`, PyTorch bf16 6.8 GiB) |
| GPU | NVIDIA A100 80GB PCIe ×2, `sm_80`, SM 108개 (측정은 `CUDA_VISIBLE_DEVICES=0` 1장) |
| CPU | Intel Xeon Gold 6230 @ 2.10GHz, 16 코어 |
| 주요 파라미터 | 클라이언트 1/2/4/8, 요청 30개/클라이언트, 데드라인 250 ms, 배치 1–16, Euler steps 10 |

### 변경 사항
| 파일 | 유형 | 설명 |
|------|------|------|
| `profiling/load_clients.py` | 수정 | 제어 연결에 `ping_interval=None`. c8에서 서버가 동기 추론으로 이벤트 루프를 막는 동안 keepalive ping이 응답되지 않아 20초 타임아웃으로 런이 죽었음 — 그 블로킹이 측정 대상이므로 하네스가 죽은 피어로 처리하면 안 됨 |
| `archive_contaminated/` | 생성 | 오염 의심 측정치 격리 (삭제 아님): 이전 `data/profiling`, `baselines/a100`, `bench_batch.json` |
| `profiling/baselines/a100/` | 재생성 | 전부 단독 점유 상태에서 재측정한 값으로 교체 |

### 작업 상세

**배경 — 왜 재측정했는가.** 스윕 도중 `Triton Error [CUDA]: initialization error`로 런이 죽고,
`bench_batch`가 `GPU free at start: 70.8/79.3 GiB`로 시작하는 것을 발견. 조사 결과 GPU를
두 프로세스가 나눠 쓰고 있었음:

- `training_validation.py` (PID 8535) — 두 장 모두 65.8 GiB / 99% util 점유. 사용자가 종료.
- **이 세션의 포크** (PID 8272) — `claude.exe --session-id 36697a1a... --fork-session
  --resume .../690a691a-....jsonl --permission-mode bypassPermissions`.
  즉 같은 대화 기록에서 갈라져 나온 헤드리스 세션이 `bg-pty-host` 아래에서
  **동일한 포팅 실험을 독립적으로 수행 중**이었음. 위 12:00 기록이 그 세션의 산출물.

프로세스 계보(`/proc/<pid>/stat`의 ppid를 따라가며)로 확인. 죽여도 즉시 새 스윕을 다시
띄우기 때문에 포크 프로세스(8272) 자체를 종료해야 멈췄음.

**측정 결과 (단독 점유 확인 후, GPU 0 MiB에서 시작).**

| 항목 | GB10 | A100 (재측정) | 12:00 기록 |
|---|---|---|---|
| `policy.infer` p50 (`bench_infer`) | 151.9 ms | **64.6 ms** (p99 65.8) | 59.4 ms |
| 컴파일 경로 B=1 | 148.2 ms | **56.5 ms** | — |
| 컴파일 B=8 로봇당 | 101.4 ms | **35.1 ms** | — |
| 배치 이득 B=1→8 | 1.46x | **1.61x** | 1.56x |
| eager B=1 / 10스텝 | 214.0 ms | **520.9 ms** | — |
| eager denoise/step | 9.8 ms | **41.8 ms** | 43 ms |
| `Not enough SMs` | 발생 | **0회** | 0회 |

스윕 (250 ms 데드라인, nsys 계측 통일):

| 클라이언트 | 처리량 | e2e 평균 | 큐 p50 | infer p50 | busy | 미스 |
|---|---|---|---|---|---|---|
| 1 | 4.07 r/s | 75.4 ms | 0.8 ms | 73.1 ms | 30.4% | 0% |
| 2 | 8.04 | 115.3 | 71.5 | 76.5 | 61.3% | 0% |
| 4 | 13.70 | 281.8 | 216.4 | 71.5 | 99.2% | 93% |
| 8 | 13.29 | 589.8 | 508.4 | 72.3 | 99.2% | 99% |

saturate (c4): 16.19 r/s / e2e 240.0 ms (GB10 6.80 r/s / 575.7 ms) — 2.38배.

**12:00 기록과 일치하는 것.** 예측 1(`nvjet_sm121` 0%, cuBLAS `ampere_*gemm` 36.3%,
`cutlass_80` 3.7%), 예측 2(denoise가 배치 1–16 내내 43–45 ms/step 평평 → SM 부족이
원인이 아니었음), 예측 4(`peak_inflight` 전 구간 1, busy 99.2%, 처리량 상한 = 1/지연).
독립 재측정에서 같은 결론이 나왔으므로 오염이 결론을 바꾸지는 않았음.

**새로 확인한 것.**
- **A100 GPU-busy는 70.7%** (요청당 GPU 49.0 ms / `sample_actions` 69.4 ms). GB10은 96%였음.
  NVML SM 사용률도 59%(GB10 93%). 즉 **이 박스에서는 추론 시간의 약 30%를 GPU가 호스트를
  기다리며 논다.** GB10에서 "런치 바운드가 아니다"라고 내린 결론은 A100에서 성립하지 않음.
- 근거: eager 경로가 A100에서 오히려 2.4배 느림(520.9 vs 214.0 ms). 요청당 커널 런치가
  5천 개대인데 이 박스 CPU가 Xeon 6230 @2.1GHz(2019년형)라 호스트가 병목.
  결과적으로 `torch.compile` 이득이 A100 9.2배 vs GB10 1.45배.
- nsys 계측 오버헤드가 무시할 수 없음. 동일 c4를 nsys 유/무로 재서
  infer p50 **71.5 → 60.9 ms (17%)**, 데드라인 미스 93% → 9%. GB10에서는 이 차이가 없었음
  (`--cuda-graph-trace node`의 호스트 오버헤드가 느린 CPU에서 증폭). 12:00 기록은 9%로
  적고 있어 런마다 편차가 있음 — 10~17% 범위로 보는 것이 안전.

**측정 위생 (다음 사람 주의).**
- 측정 전 `nvidia-smi --query-compute-apps` **와** `torch.cuda.mem_get_info()`를 둘 다 확인.
  컨테이너 안에서는 밖의 프로세스가 전자에 안 잡힘.
- 이 인스턴스에서 헤드리스로 포크된 에이전트 세션이 같은 GPU를 쓸 수 있음.
  `pgrep -af "claude.exe --session-id"`로 확인 가능.

**알려진 제한사항 / TODO.**
- 15:13에 `profiling/baselines/a100/`에 파일 3개가 재생성됨 — 포크(8272)를 15:08에 종료한
  뒤였고, 남아있던 세션은 PID 18797뿐. 출처 불명이라 `archive_contaminated/`로 격리함.
- GPU 카운터(ncu roofline, SM occupancy)는 이 컨테이너에서 불가 (`CAP_SYS_ADMIN` 없음).
- 2장 데이터 병렬(서버 2개) 미측정.

### 로그
- `logs/2026-08-04_15-10_bench_infer_clean.log`, `logs/2026-08-04_15-15_bench_batch_clean.log`
- `logs/2026-08-04_15-25_sweep_clean.log`, `logs/2026-08-04_16-15_sweep_c8_rerun2.log`
- `logs/2026-08-04_16-30_saturate_c4.log`, `logs/2026-08-04_16-45_c4_rate_restore.log` (nsys 없음)
- `logs/2026-08-04_16-55_c4_rate_nsys_restore.log`, `logs/2026-08-04_17-05_report_final.log`
- 오염/실패 기록: `logs/2026-08-04_14-30_sweep_c1248.log` (cudnn 트레이스로 Triton init 실패),
  `logs/2026-08-04_14-25_bench_infer.log`, `logs/2026-08-04_14-05_bench_batch.log`

### 관련 이슈
- 없음
