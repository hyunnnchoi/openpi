# A100 80GB PCIe ×2 이식 결과 — GB10 대비

`PORTING_A100.md`의 절차를 A100 박스에서 실제로 수행하고, 같은 벤치마크를 돌린 결과입니다.
GB10 기준선은 `profiling/baselines/gb10/`, 이번 측정 원본은 `profiling/baselines/a100/`.

> **측정 출처.** 이 문서의 모든 A100 수치는 **GPU 단독 점유가 확인된 상태에서 다시 측정한
> 값**입니다(§4 참조). 각 런의 `resources.jsonl`에서 `gpu_used_gib`가 전 구간 7.76 GiB
> 고정(= 자기 가중치뿐)임을 확인했고, `bench_batch`는 `GPU free at start 78.8/79.3 GiB`로
> 시작했습니다. 초기에 GPU를 공유한 상태로 나온 측정치는 `archive_contaminated/`에 격리해
> 두었고 이 문서에는 반영하지 않았습니다.

한 줄 요약: **4개 예측 중 2개가 맞고, 1개는 반증됐으며, 1개는 방향이 갈렸습니다.
A100이 2.3배 빠르지만 그 이득은 전부 서버 처리량 상한으로만 가고, 서버 직렬화라는 진짜
병목은 예측대로 전혀 달라지지 않았습니다. 예측에 없던 발견으로, 이 박스에서는 GPU가
추론 시간의 1/4을 호스트 대기로 놉니다. 그 여파로 prefill/denoise 비중이 eager에서 본
것과 정반대(§3)이며, 동적 배칭을 붙이면 GPU당 로봇 수용량이 2대 → 6대가 됩니다.**

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

## 3. prefill / denoise 분해 — eager로 본 그림은 틀렸습니다

§1 예측 2의 분해표는 **비컴파일 경로**에서 잰 것입니다(그쪽은 `num_steps`를 공짜로 바꿀 수
있어서). 그 표는 "denoise가 전체의 80%이고 배치 1~16 내내 총량이 평평하다"고 말하고,
그대로 읽으면 **denoise는 묶어 돌리는 게 사실상 공짜**라는 결론이 나옵니다.

§2에서 본 대로 이 박스의 eager 경로는 호스트 바운드라, 그 "평평함"이 실제 성질인지
"어차피 GPU가 놀고 있어서"인지 구분되지 않습니다. 그래서 **컴파일 경로에서 같은 분해를
다시 측정**했습니다(`bench_split_compiled.py`). 배치별로 `num_steps` 1/5/10을 각각 컴파일해
`지연 = prefill + 스텝수 × per_denoise`를 최소자승 피팅합니다. 2점이 아니라 3점을 쓰는
이유는 선형성 가정 자체를 검증하기 위해서이고, 잔차는 전 배치에서 **0.08~1.07 ms**로
분해가 유효합니다.

### 결과 — 주역이 뒤바뀝니다

| 로봇 1대 | eager (§1 예측 2) | **컴파일 (실제 서빙 경로)** |
| --- | ---: | ---: |
| prefill | 103.1 ms (19.8%) | **33.1 ms (60.2%)** |
| denoise 10스텝 | 417.8 ms (80.2%) | **21.8 ms (39.8%)** |
| denoise 1스텝 | 41.8 ms | **2.2 ms** |

denoise 한 스텝이 **41.8 → 2.2 ms (19배)**, prefill은 103.1 → 33.1 ms (3.1배)입니다.
denoise 루프는 작은 커널을 10번 반복하는 구조라 런치 오버헤드가 집중돼 있었고, CUDA
그래프가 그걸 걷어내자 실제로는 싼 단계였음이 드러납니다. **§2의 호스트 바운드 진단을
가장 선명하게 보여주는 수치이며, 동시에 eager 분해로 스케줄링을 설계하면 안 된다는 뜻입니다.**

### 배치 경제성 — prefill이 병목이고, 묶어도 안 싸집니다

| 배치 | prefill ms | ms/denoise | denoise 총 | prefill/로봇 | denoise/로봇 | 전체 지연 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.1 | 2.2 | 21.8 | 33.1 | 21.8 | 55.0 |
| 2 | 64.6 | 3.4 | 33.9 | 32.3 | 16.9 | 98.6 |
| 4 | 118.7 | 4.4 | 43.9 | 29.7 | 11.0 | 162.5 |
| 5 | 128.7 | 5.8 | 58.4 | 25.7 | 11.7 | 187.1 |
| 6 | 163.2 | 5.3 | 52.8 | 27.2 | 8.8 | 216.5 |
| 7 | 189.0 | 6.0 | 59.6 | 27.0 | 8.5 | 248.2 |
| 8 | 215.6 | 6.5 | 65.2 | 27.0 | 8.1 | 280.7 |
| 16 | 411.0 | 11.1 | 110.9 | 25.7 | 6.9 | 521.9 |

- **prefill/로봇은 33.1 → 25.7 ms, 1.3배밖에 안 좋아집니다.** 16배 일을 시키면 시간도
  거의 16배 듭니다. 배치가 거의 안 먹는 단계입니다.
- denoise/로봇은 21.8 → 6.9 ms로 3.2배 좋아지지만, eager가 시사한 "15배 공짜"와는 거리가 멉니다.
- 그래서 배치를 키울수록 prefill 비중이 60% → 79%로 올라가고, **전체 지연은 prefill이
  결정합니다.**

### 250 ms 데드라인에서의 수용량

로봇 N대가 각각 250 ms마다 한 번 요청하면, 한 주기에 `prefill(N) + 10 × denoise(N)`가
들어가야 합니다. 위 표의 "전체 지연"이 그 값입니다:

| 로봇 수 | 필요 시간 (평균) | p99 | 250 ms 대비 |
| ---: | ---: | ---: | --- |
| 4 | 162.5 ms | 164.1 | 65% ✅ |
| 5 | 187.1 ms | 188.5 | 75% ✅ |
| **6** | **216.5 ms** | 218.8 | **87% ✅ 실용 상한** |
| 7 | 248.2 ms | **250.4** | 99% ⚠️ 평균은 들어오지만 p99가 데드라인 초과 |
| 8 | 280.7 ms | 281.9 | 112% ❌ |

**GPU 한 장당 6대가 실용 상한**입니다(2장이면 12대). 요청당 transform+H2D/D2H가 6.6 ms
더 붙으므로(§1 예측 4의 분해) 6대에서 실제 여유는 **약 27 ms**입니다.

### 스케줄링에 주는 함의

- **동적 배칭은 확실한 이득입니다.** 지금의 순차 서버(2대)에서 6대로, 3배입니다.
  이득의 대부분이 여기 있습니다.
- **prefill/denoise를 분리해도 처리량 상한은 안 올라갑니다.** 총 GPU 일감이 위 표로
  고정돼 있고, 모두 한 배치로 묶어 돌리면 이미 같은 값에 도달합니다.
- **시퀀스 축 chunked prefill은 이 모델에서 불가능합니다.** `embed_prefix`가 prefix
  전체에 `att_masks = 0`을 주므로(`pi0_pytorch.py:211,226`, 마스크 규약은 `:59-66`)
  이미지·언어 토큰이 **양방향으로 서로를 봅니다.** LLM의 chunked prefill은 causal
  마스크에 기대는 기법이라 그대로 옮겨올 수 없습니다.
- **배치 축으로 prefill을 쪼개는 것은 가능하지만 손해입니다.** prefill이 배치를 거의
  타지 않으므로(위 1.3배) 6대를 2대씩 3번에 나누면 163.2 → 193.8 ms로 **+30.6 ms**가
  붙고, 이는 6대에서 남는 여유 27 ms를 통째로 먹습니다.
- **남는 것은 지연 안정성입니다.** 6대를 한 배치로 묶으면 모든 로봇이 216.5 ms를 통째로
  기다리므로, 배치가 뜬 직후 도착한 로봇은 다음 주기까지 밀립니다. 진행 중인 로봇들을
  denoise 링으로 굴리면 대기 단위가 배치 하나(216 ms)에서 denoise 한 스텝(5~6 ms)으로
  줄어듭니다. 모델은 이미 이걸 받을 수 있습니다 — `sample_actions`가 timestep을
  `time.expand(bsize)`로 **per-sample 텐서**로 넘기고(`pi0_pytorch.py:406-415`),
  `embed_suffix`가 그대로 per-sample로 임베딩하므로(`:265-278`) **서로 다른 denoise
  스텝에 있는 로봇을 한 배치에 섞는 데 모델 수정이 필요 없습니다.**

### 레이어 축 chunking — 시퀀스 축이 막힌 자리를 대신합니다

prefix가 양방향이라 시퀀스 축으로는 못 쪼개지만, **레이어 축으로는 쪼갤 수 있습니다.**
각 레이어는 여전히 전체 시퀀스를 다 보므로 결과가 비트 단위로 동일하고, 레이어 사이에서만
멈추는 것이기 때문입니다. `gemma_pytorch.py:239-253`의 루프가
`for layer_idx in range(num_layers): inputs_embeds = compute_layer_complete(...)` 형태라
구간을 잘라내는 것 자체는 간단하고(VLM·expert 모두 18층), 청크 사이에 넘길 상태는
hidden states `[B, prefix_len, 2048]`(배치 6에서 약 20 MB)와 그때까지 쌓인 KV뿐입니다.

비용은 청크 경계에서 늘어나는 CUDA 그래프 런치입니다. c4 API 트레이스에 실측값이 있습니다:

| | 값 |
| --- | ---: |
| `cudaGraphLaunch` 호출 수 | 1440회 / 120요청 = **요청당 12회** |
| 호출당 호스트 시간 | 평균 0.80 ms, 중앙값 0.63 ms |

요청당 12회는 `prefill 1 + denoise 10 + 1`과 맞아떨어집니다. 즉 **컴파일된
`sample_actions`는 이미 하나의 거대 그래프가 아니라 스텝별로 쪼개진 12개 그래프**이고,
여기서 두 가지가 따라옵니다:

1. **denoise 링(continuous batching)은 그래프 경계를 새로 만들지 않습니다.** 스텝 경계가
   이미 존재하므로 구조적 추가 비용이 없습니다.
2. **레이어 chunking은 청크당 약 0.8 ms**(nsys 켠 값이라 상한)만 더 듭니다.

6대(prefill 163.2 ms) 기준으로 K개 청크로 나눴을 때:

| K | 청크당 | 진행 중 로봇의 최대 stall | 추가 비용(상한) | 주기 총합 |
| ---: | ---: | ---: | ---: | ---: |
| 1 (현재) | 163.2 ms | 163.2 ms | — | 216.5 ms |
| 3 | 54.4 | 54.4 | +1.6 | 218.1 |
| 6 | 27.2 | **27.2** | **+4.0** | 220.5 |
| 9 | 18.1 | 18.1 | +6.4 | 222.9 |
| 18 (매 레이어) | 9.1 | 9.1 | +13.6 | 230.1 |

**배치 축 chunking과 직접 비교하면 차이가 분명합니다.** 6대를 2대씩 3그룹으로 나누는
방식은 stall이 64.6 ms까지밖에 안 줄면서 **+30.6 ms**가 듭니다. 레이어 축 K=6은
stall을 27.2 ms로 더 줄이면서 **+4.0 ms**만 씁니다 — 같은 목적에 **비용이 1/8**입니다.
prefill이 배치를 거의 안 타는 반면(로봇당 1.3배) 레이어는 그냥 순차 구간이라 쪼개도
연산량이 그대로이기 때문입니다.

K는 stall과 오버헤드를 맞바꾸는 손잡이이고, 6대 기준으로 여유 27 ms 안에서 K=6~9가
합리적입니다. **다만 위 오버헤드는 그래프 런치 실측에서 유도한 추정이지 측정값이
아닙니다** — 레이어 구간 API를 붙여 단발 prefill과 K분할 prefill을 직접 재는 것이
구현 전 마지막 검증 단계입니다.

막고 있는 것은 모델이 아니라 서빙 계층입니다: `Policy.infer`의 배치-1 하드코딩
(`inputs[None, ...]` / `x[0, ...]`)과 `WebsocketPolicyServer`의 동기 핸들러 —
§1 예측 4에서 `peak_inflight`가 끝까지 1이었던 바로 그 원인입니다.

### 권장 순서

| 단계 | 처리량 | 지연 안정성 | 비용 |
| --- | --- | --- | --- |
| 1. 동적 배칭 | **2 → 6대** | — | 서빙 계층 2곳 |
| 2. continuous batching (denoise 링) | 변화 없음 | 대기 단위 216 → 5.3 ms | 스케줄러, 그래프 경계 추가 없음 |
| 3. 레이어 축 chunked prefill | 변화 없음 (−4 ms) | prefill stall 163 → 27 ms | 레이어 구간 API + 컴파일 형상 증가 |

이득의 크기는 1단계가 압도적이고, 2·3단계는 그 6대가 위상이 어긋난 채로도 데드라인을
지키게 만드는 작업입니다. 3단계의 실질 리스크는 연산이 아니라 **컴파일 형상 수**입니다 —
이번 측정에서 새 형상의 첫 컴파일이 264~455초 걸렸고(이후 동일 형상은 0.2~0.6초),
청크 구성 × 배치 크기만큼 형상이 늘어나면 워밍업이 그만큼 길어집니다.

원본: `profiling/baselines/a100/bench_split_compiled.json`,
로그 `logs/2026-08-04_17-30_split_compiled.log`, `logs/2026-08-04_17-50_split_b567.log`.

## 4. ⚠️ 측정 위생 — 이 인스턴스에서 GPU를 공유하게 되는 두 경로

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

## 5. 절차상 가이드와 달랐던 점

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

## 6. A100 2장 활용에 대한 함의

가이드 §7의 결론(데이터 병렬 = 서버 2개)은 그대로 유효하고, 이번 측정으로 숫자가 붙습니다.

| 구성 | 로봇 / GPU | 로봇 / 박스(2장) | 근거 |
| --- | ---: | ---: | --- |
| 지금 (스톡 서버, 순차 처리) | 2대 | 4대 | §1 예측 4 — 2대 미스 0%, 4대 미스 93% |
| + 동적 배칭 | **6대** | **12대** | §3 — 6대 216.5 ms < 250 ms |
| + prefill/denoise 분리 | 6대 | 12대 | 상한은 그대로, 지연 안정성만 개선 |

- 서버 1개 = 로봇 2대가 현재값이고, 포트만 나눠 2장을 쓰면 4대입니다(라우터 불필요).
- **동적 배칭이 3배를 만듭니다.** 데드라인을 지키면서 쓸 수 있는 최대 배치는 **B=6
  (216.5 ms)** 이고, B=7은 평균 248.2 ms로 턱걸이지만 p99가 250.4 ms로 넘습니다.
  §1 예측 2의 배치 이득(1.61배)만 보면 과소평가하게 되는데, 실제로는 순차 처리 대비
  로봇 수가 3배가 됩니다.
- **분리 스케줄링은 상한을 못 올립니다**(§3). 6대에서 남는 여유가 27 ms뿐이라는 점이
  분리의 진짜 명분입니다 — 대기 단위를 배치 하나에서 denoise 한 스텝으로 줄이는 것.
- 그 위로 더 가려면 §2의 호스트 병목(GPU-busy 75%)을 먼저 걷어내야 합니다. GPU를 바꾸는
  것보다 런치 수를 줄이는 쪽이 남은 여지입니다.

---

## 7. 재현 명령

```bash
# 깨끗한 GPU 확인 후 (§4)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_infer.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_batch.py
CUDA_VISIBLE_DEVICES=0 profiling/run_bench.sh -c "1 2 4 8" -n 30
CUDA_VISIBLE_DEVICES=0 profiling/run_bench.sh -c 4 -m saturate --no-nsys -n 30
CUDA_VISIBLE_DEVICES=0 profiling/run_bench.sh -c 4 -n 30 --no-nsys   # nsys 오버헤드 측정용
CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_split_compiled.py      # 3장 prefill/denoise 분해
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
| `bench_split_compiled.json` | §3 컴파일 경로 prefill/denoise 분해 + 수용량 |

로그 원본은 `logs/2026-08-04_15-10_*` 이후 파일들입니다(그 이전은 오염 구간).
