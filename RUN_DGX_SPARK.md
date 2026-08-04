# Running openpi on this DGX Spark (GB10)

Everything below is already set up. This file records *what had to change* and the
exact commands to re-run things, because none of it matches the upstream README.

> Moving this to an A100 box? See **[`PORTING_A100.md`](PORTING_A100.md)** — it marks
> which of the changes below are GB10-specific (revert them) and which are upstream bugs
> that travel with you (keep them), plus the GB10 baselines to compare against.

## Why the upstream instructions don't work here

| Upstream assumption | Reality on this box | Fix applied |
| --- | --- | --- |
| `jax[cuda12]==0.5.3` | GB10 is `sm_121` / aarch64; no JAX CUDA plugin exists | `jax==0.5.3` (CPU). JAX is only used for transforms + orbax checkpoint loading; all compute runs in PyTorch |
| `torch==2.7.1` | 2.7.1 has no `sm_121` kernels | `torch==2.11.0+cu130` + `torchvision==0.26.0+cu130` from the cu130 index |
| LIBERO client on Python 3.8 with `numpy==1.22.4`, `llvmlite==0.36` | Those don't build on aarch64 | Separate Python 3.11 venv with modern equivalents |
| `pip install -e third_party/libero` | `third_party/libero/libero/` has no `__init__.py`, so `find_packages()` returns nothing and the editable install maps nothing | `_libero_root.pth` adds the repo root to the venv path |
| `torch.load` of LIBERO init states | PyTorch >= 2.6 defaults to `weights_only=True` | `sitecustomize.py` shim in the LIBERO venv |
| EGL "just works" | glvnd picks Mesa (`50_mesa.json`), which needs `/dev/dri/renderD128`; this account is not in the `render` group | Force the NVIDIA vendor: `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json` |

`pyproject.toml.orig` is the unmodified upstream file.

## Layout

- `.venv` — server side (policy inference). Python 3.11, torch cu130, JAX CPU.
- `examples/libero/.venv` — client side (MuJoCo sim). Python 3.11, torch CPU.
- `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero` — original JAX checkpoint (11.6 GB fp32)
- `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch` — converted (7.2 GB bf16) + copied `assets/` for norm stats

## Commands

Inference latency benchmark (no sim needed):

```bash
.venv/bin/python bench_infer.py
```

Terminal 1 — policy server:

```bash
.venv/bin/python scripts/serve_policy.py --env LIBERO \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

Terminal 2 — LIBERO client:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl
examples/libero/.venv/bin/python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1
```

Videos land in `data/libero/videos/`.

Timing-instrumented variant (same rollout, records where the wall clock goes):

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl
export PYTHONPATH=$PWD/examples/libero
examples/libero/.venv/bin/python examples/libero/main_timed.py --num-tasks 2 --num-trials-per-task 1
```

Raw per-call timings are written to `data/libero/timing.json`.

## Measured baseline (pi05_libero, bf16, libero_spatial)

| Quantity | Value |
| --- | --- |
| Policy inference, warm | 148.8 ms mean, p99 150.2 ms (n=35) — remarkably deterministic |
| Policy inference, first call | 18,027 ms — 121x, `torch.compile(mode="max-autotune")` |
| Model load | 39 s |
| `env.step` (MuJoCo sim + render) | 9.8 ms |
| Client-side image preprocess | 1.0 ms |
| Inference share of loop time | 93% |
| GPU memory, weights | 15.3 GiB (from a 7.2 GB bf16 file) |
| GPU memory, whole process | 50.4 GiB |
| Success (2 tasks x 1 trial) | 2/2 |

With `replan_steps=5`, one chunk covers 250 ms of robot time at 20 Hz while
inference blocks for 149 ms — 60% of the window. Real-time is possible but only
because the robot commits to 5 open-loop actions per query.

## Web console

`webapp/` runs the policy, the MuJoCo sim, and the HTTP/WebSocket server in **one
process**, which is what makes it possible to instrument the denoising loop directly
instead of going over the wire. This required installing the sim deps into the main
`.venv` (they coexist — both venvs were already on numpy 1.26.4).

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl
.venv/bin/python webapp/server.py --host 127.0.0.1 --port 8080
```

From a laptop: `ssh -N -L 8080:localhost:8080 <this-host>` then open
<http://localhost:8080>. It binds to loopback on purpose; pass `--host 0.0.0.0` to
expose it on the network instead.

All five LIBERO suites are selectable (`libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`, `libero_90`) — the `pi05_libero` checkpoint was trained across them, and the
norm stats are shared, so no reload is needed to switch. The episode cap follows the
suite (220 / 280 / 300 / 520 / 400 steps, matching `examples/libero/main.py`); switching
suites tears down the MuJoCo env and builds the new scene on the next action.

What it shows:

- **로봇 뷰** — live agentview + wrist camera as the episode runs.
- **모델 입력** — the actual model-facing tensors: both 224x224 resize+pad images, the
  zero-filled and masked `right_wrist_0_rgb` slot LIBERO does not fill, the 8-D
  proprioceptive state, and (via 입력 상세) the tokenized prompt.
- **지표** — inference / sim-step / streaming latency with mean-p50-p99, realtime
  factor, and what fraction of wall time the loop spends blocked on inference,
  against a dashed line at the `replan_steps x 50 ms` control budget.
- **플로우 매칭 디노이징** — every Euler step of the ODE integration, unnormalized
  into real action units, plus a convergence curve.

Two honesty notes wired into the UI:

- The diffusion trace deliberately calls the **uncompiled** `sample_actions`
  (`type(model).sample_actions`), because the fast path is `torch.compile`d as a whole
  graph and patching `denoise_step` inside it would force recompiles. Its latency is
  therefore not comparable to rollout latency.
- Streaming cost (JPEG encode + serialize) is measured separately and displayed. It
  came out at 0.8 ms/step, so it does not meaningfully distort the other numbers.

### Do not compare step counts without fixing the noise

The console seeds the initial Gaussian when "노이즈 고정" is checked. This matters a
lot — measured on task 0, comparing final action chunks:

| Euler steps | L2 vs 10-step (same noise) |
| --- | --- |
| 1 | 0.123 |
| 2 | 0.091 |
| 5 | 0.047 |
| 8 | 0.017 |
| 10 | 0 (reference) |

versus **0.152** between two 10-step runs that differ only in the noise draw. In
other words, on this observation, dropping from 10 Euler steps to 1 perturbs the
output *less than resampling the noise does*. That is a hypothesis about wasted
compute, not a result — it is one observation on one task, and it says nothing yet
about task success rate. The obvious follow-up is a success-rate sweep over
`num_steps`.

## Batch scaling — how many robots fit on this GPU

```bash
.venv/bin/python bench_batch.py     # writes data/bench_batch.json
```

`Policy.infer` is hardcoded to batch 1 (`inputs[None, ...]` in, `x[0, ...]` out) and
`WebsocketPolicyServer` calls it *synchronously inside the asyncio handler*, so the
stock server serializes concurrent robots — there is no batching and no scheduler.
The model itself is fine: `PI0Pytorch.sample_actions` is batch-first throughout.
`bench_batch.py` therefore calls `sample_actions` directly with a replicated
observation.

Compiled path (`max-autotune`, i.e. what serving actually runs), 10 Euler steps:

| Batch | Latency | Per robot | Speedup vs serial |
| --- | --- | --- | --- |
| 1 | 148.2 ms | 148.2 ms | 1.00x |
| 2 | 236.7 ms | 118.4 ms | 1.25x |
| 4 | 421.1 ms | 105.3 ms | 1.41x |
| 8 | 811.3 ms | 101.4 ms | 1.46x |

**Batching buys ~1.46x at 8x the work.** The reason shows up when prefill is split
from denoising (uncompiled path, comparing `num_steps=1` against `num_steps=10`):

| Batch | Prefill | Prefill/robot | ms per denoise step | Prefill share |
| --- | --- | --- | --- | --- |
| 1 | 115.8 ms | 115.8 ms | 9.8 | 54.1% |
| 2 | 212.6 ms | 106.3 ms | 14.4 | 59.5% |
| 4 | 438.9 ms | 109.7 ms | 18.6 | 70.2% |
| 8 | 913.0 ms | 114.1 ms | 29.4 | 75.6% |
| 16 | 1772.4 ms | 110.8 ms | 51.6 | 77.4% |

The two halves behave in opposite ways:

- **Prefill** (one VLM forward over the image + language tokens, producing the KV
  cache) is flat at ~110 ms *per robot* at every batch size — it batches not at all.
  Inductor says why: `Not enough SMs to use max_autotune_gemm mode`. GB10 is already
  compute-saturated at batch 1.
- **Denoising** (Euler steps through the action expert against the cached prefix)
  batches well — 16x the work for 3.2x the time, a 5x efficiency gain.

And prefill is the majority of the latency, rising to 77% at batch 16.

Consequence for the 250 ms control budget (`replan_steps=5` at 20 Hz): batch 2 fits
at 236.7 ms (95% of budget), batch 4 does not (421 ms). **One GB10 serves two
LIBERO robots, and only if you batch them.** With the stock server the second robot
queues behind the first at 296 ms and already misses.

Memory is not the constraint at this scale — peak allocation goes 7.12 GiB (batch 1)
to 9.24 GiB (batch 16). Weights alone are 8.56 GiB measured at load time; the 15.3 GiB
in the table above was measured after `torch.compile` had run.

One `torch.compile` note: batch size is part of the traced shape, so the first call at
a new batch pays a recompile (10.5 s at batch 1, 135.4 s at batch 2). After two
distinct sizes Dynamo marks the batch dim dynamic and further sizes cost < 1 s — so a
dynamic-batching server pays this twice at startup, not on every batch-size change.

## Multi-client load + Nsight profiling

`bench_batch.py` above answers "what if you batched N robots". This answers the
different question of **what the stock server actually does when N robots connect at
once**, with an nsys capture of each run.

```bash
profiling/run_bench.sh -c "1 2 4 8" -n 30          # full sweep, one .nsys-rep per run
profiling/run_bench.sh -c 4 -m saturate --no-nsys  # throughput only, no profiler
.venv/bin/python profiling/report.py data/profiling # re-print without re-running
```

Each client-count gets a fresh server, warmed before the socket opens (so the ~10 s
`torch.compile` first call is not charged to a client), profiled only between
`cudaProfilerStart`/`Stop`. Open a timeline with `nsys-ui data/profiling/c4.nsys-rep`.

### The measurement that needs two clocks

Queue wait is not observable from inside the server. `await websocket.recv()` returns
only once the blocking `policy.infer` has released the event loop, so the server's
"receive" timestamp is really "when I got around to you". `profiling/load_clients.py`
therefore runs each robot as its own process and logs its own send timestamp; the
server logs its own receive timestamp; `report.py` joins them on `(client_id, seq)`.
Both are on this host, so CLOCK_MONOTONIC is shared and the difference is real.

### Result: latency scales linearly, throughput does not move

30 requests per client, 250 ms deadline per robot (`replan_steps=5` at 20 Hz):

| clients | throughput | e2e mean | e2e p99 | queue p50 | `policy.infer` p50 | server busy | missed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4.04 req/s | 152.7 ms | 155.7 ms | 0.5 ms | 151.9 ms | 61.4% | 0% |
| 2 | 6.66 req/s | 294.1 ms | 303.8 ms | 148.9 ms | 149.2 ms | 99.8% | 97% |
| 4 | 6.68 req/s | 589.4 ms | 600.7 ms | 448.0 ms | 149.2 ms | 99.8% | 99% |
| 8 | 6.69 req/s | 1175.2 ms | 1490.7 ms | 1044.6 ms | 149.1 ms | 99.8% | 100% |

Inference is flat at ~149 ms under every load. Every added millisecond of latency is
queue wait, throughput saturates at 6.7 req/s = 1/149 ms, and the server's `inflight`
counter never exceeded 1 in any run. `-m saturate` at 4 clients (no deadline, send as
fast as responses arrive) reaches 6.80 req/s — the same ceiling, so nothing is being
lost to the 250 ms pacing. This is a pure FIFO queue of depth N with no
concurrency — which is what the code says it is, now measured rather than asserted.

**One GB10 serves one LIBERO robot on the stock server.** The second robot lands at
294 ms against a 250 ms budget, which matches the 296 ms predicted from the batch table
above. Everything off the critical path is negligible: request 301 KB / response 680 B,
msgpack unpack 0.02 ms, pack 0.01 ms, transforms + H2D + D2H 2.3 ms.

### Result: the GPU is busy, on partly the wrong kernels

At 4 clients with `--cuda-graph-trace node` (80 requests, 408,160 kernel launches,
77 distinct kernels, 11,377 ms of GPU time):

| category | GPU ms | % | launches/req | µs/call |
| --- | --- | --- | --- | --- |
| GEMM `cutlass_80_*tensorop` | 3530 | 31.0 | 577 | 76.5 |
| GEMM `cutlass_80_*wmma*` | 2980 | 26.2 | 1620 | 23.0 |
| Triton (Inductor fused) | 2754 | 24.2 | 2008 | 17.1 |
| GEMM `nvjet_sm121_*` | 2010 | 17.7 | 168 | 149.5 |
| reduction / norm | 38 | 0.3 | 370 | 1.3 |
| attention | 26 | 0.2 | 243 | 1.3 |
| elementwise / copy | 18 | 0.2 | 96 | 2.3 |

142.2 ms of GPU time per 148.1 ms `sample_actions` range — **96% GPU-busy, so this is
not launch-bound or host-bound**, despite 5,102 kernel launches per request (they are
inside CUDA graphs, ~12 graph launches per request).

The lead worth chasing: **57% of GPU time is in `cutlass_80_*` kernels** — the SM80
(Ampere) generation CUTLASS path — while only 18% runs on `nvjet_sm121_*`, the kernels
actually named for this chip. The `wmma` variants in particular are the older
tensor-core path, and they are being called 1,620 times per request at 23 µs each.
That is an observation about which kernels get selected, not proof that better ones
exist for these shapes; the test would be to force a newer cuBLAS/CUTLASS path and
re-measure. It does line up with the earlier Inductor complaint that there are
`Not enough SMs to use max_autotune_gemm mode`.

### nsys settings that matter here

`--cuda-graph-trace node` is not optional for this model. `max-autotune` compiles
inference into CUDA graphs, so under the nsys default (`graph`) each launch is one
opaque blob and the kernel report shows only the few eager kernels outside the graphs —
**5 ms of "GPU time" for a 36 s run**, which looks like an idle GPU and is not. Use the
default `graph` mode only when you care about the concurrency timeline rather than
kernels.

### What this box will not let you profile

| | status | fix (needs root + reboot) |
| --- | --- | --- |
| CUDA + NVTX tracing | works | — |
| CPU sampling / context switches | unavailable | `sudo sysctl -w kernel.perf_event_paranoid=2` |
| nsys GPU metric counters (SM occupancy) | `ERR_NVGPUCTRPERM` | `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/`, `update-initramfs -u`, reboot |
| Nsight Compute (`ncu`) per-kernel counters | `ERR_NVGPUCTRPERM` | same as above |

Until then SM utilization comes from NVML sampled at 10 Hz by the server itself
(93% mean during every loaded window, 57–66 W — well under this part's power budget).

## Re-converting the checkpoint

```bash
.venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --config_name pi05_libero \
  --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --precision bfloat16
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets \
      ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/
```

The `assets/` copy is required — `create_trained_policy` loads normalization
stats from `<checkpoint_dir>/assets`, and the converter does not copy them.

## Memory note

CPU and GPU share one 128 GB pool on this machine. A vLLM server left running with
the default `--gpu-memory-utilization` will take ~110 GB and openpi will fail to
allocate anything. Check with:

```bash
.venv/bin/python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'{f/2**30:.1f} / {t/2**30:.1f} GiB free')"
```
