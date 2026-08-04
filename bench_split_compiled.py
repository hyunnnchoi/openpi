"""Prefill vs denoise split on the COMPILED path.

`bench_batch.py` derives this split from the uncompiled path only, because
varying num_steps there is free. That split is what motivated the idea of
scheduling prefill and denoise separately -- denoise total came out flat across
batch 1..16, i.e. denoise looked free to batch.

But the uncompiled path on this box is host-bound (GPU-busy 75%, SM 64%, see
RESULTS_A100.md section 2), so "flat" there may just mean "the GPU was idle
waiting on kernel launches either way". Under torch.compile the launches are
folded into CUDA graphs, and the question is whether denoise still batches for
free once the host is out of the way.

Method: for each batch size, compile sample_actions at several num_steps and fit
latency = prefill + num_steps * per_denoise_step. Three points instead of the
two bench_batch uses, so the linearity assumption itself gets checked -- if the
denoise loop is not linear in num_steps the whole derivation is void.

Every (batch, num_steps) pair is a distinct shape and pays its own max-autotune
compile; that cost is recorded but excluded from the timings.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_split_compiled.py
"""

import argparse
import json
import pathlib
import statistics
import time

import jax
import numpy as np
import torch

from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

CKPT = pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
OUT = pathlib.Path("data/bench_split_compiled.json")


def gib(x):
    return x / 2**30


def make_batch(inputs, batch_size, device):
    def rep(x):
        a = np.asarray(x)
        t = torch.from_numpy(a).to(device)
        return t.unsqueeze(0).expand(batch_size, *a.shape).contiguous()

    return jax.tree.map(rep, inputs)


def timeit(fn, n_warmup, n_iters):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    lat = []
    for _ in range(n_iters):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1e3)
    lat.sort()
    return {
        "mean": statistics.mean(lat),
        "p50": lat[len(lat) // 2],
        "p99": lat[min(int(len(lat) * 0.99), len(lat) - 1)],
        "n": len(lat),
    }


def linfit(xs, ys):
    """Least-squares fit y = a + b*x, plus max residual as a linearity check."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys, strict=True)]
    return a, b, max(abs(r) for r in resid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    train_config = _config.get_config("pi05_libero")
    free_before, total = torch.cuda.mem_get_info()
    print(f"GPU free at start    : {gib(free_before):.1f} / {gib(total):.1f} GiB", flush=True)
    if gib(free_before) < gib(total) - 2:
        print("!! GPU is not idle -- see RESULTS_A100.md section 4 before trusting this", flush=True)

    t0 = time.perf_counter()
    policy = _policy_config.create_trained_policy(train_config, CKPT)
    print(f"model load           : {time.perf_counter() - t0:.1f} s", flush=True)
    free_after, _ = torch.cuda.mem_get_info()
    print(f"weights              : {gib(free_before - free_after):.2f} GiB", flush=True)

    model = policy._model  # noqa: SLF001
    device = policy._pytorch_device  # noqa: SLF001
    inputs = policy._input_transform(libero_policy.make_libero_example())  # noqa: SLF001

    raw = {"raw": {}, "fit": {}, "meta": {}}

    print("\n=== compiled latency at each (batch, num_steps) ===", flush=True)
    print(f"{'B':>3} {'steps':>6} {'compile s':>10} {'mean ms':>9} {'p99 ms':>8}", flush=True)
    for b in args.batches:
        obs = _model.Observation.from_dict(make_batch(inputs, b, device))
        for s in args.steps:
            try:
                t = time.perf_counter()
                with torch.no_grad():
                    model.sample_actions(device, obs, num_steps=s)
                torch.cuda.synchronize()
                compile_s = time.perf_counter() - t
                with torch.no_grad():
                    st = timeit(
                        lambda s=s: model.sample_actions(device, obs, num_steps=s),
                        args.warmup,
                        args.iters,
                    )
            except torch.OutOfMemoryError:
                print(f"{b:>3} {s:>6}  OOM", flush=True)
                torch.cuda.empty_cache()
                break
            st["compile_s"] = compile_s
            raw["raw"][f"b{b}_s{s}"] = st
            print(f"{b:>3} {s:>6} {compile_s:>10.1f} {st['mean']:>9.1f} {st['p99']:>8.1f}", flush=True)
        del obs
        torch.cuda.empty_cache()

    print("\n=== fitted split: latency = prefill + steps x per_denoise ===", flush=True)
    print(
        f"{'B':>3} {'prefill ms':>11} {'ms/denoise':>11} {'denoise tot':>12} "
        f"{'prefill/robot':>14} {'denoise/robot':>14} {'fit err ms':>11}",
        flush=True,
    )
    for b in args.batches:
        pts = [(s, raw["raw"][f"b{b}_s{s}"]["mean"]) for s in args.steps if f"b{b}_s{s}" in raw["raw"]]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        prefill, per_step, err = linfit(xs, ys)
        n_steps = max(args.steps)
        denoise_tot = per_step * n_steps
        raw["fit"][f"b{b}"] = {
            "prefill_ms": prefill,
            "per_denoise_ms": per_step,
            "denoise_total_ms": denoise_tot,
            "prefill_per_robot_ms": prefill / b,
            "denoise_per_robot_ms": denoise_tot / b,
            "max_resid_ms": err,
        }
        print(
            f"{b:>3} {prefill:>11.1f} {per_step:>11.1f} {denoise_tot:>12.1f} "
            f"{prefill / b:>14.1f} {denoise_tot / b:>14.1f} {err:>11.2f}",
            flush=True,
        )

    free_end, _ = torch.cuda.mem_get_info()
    raw["meta"] = {
        "batches": args.batches,
        "steps": args.steps,
        "iters": args.iters,
        "weights_gib": gib(free_before - free_after),
        "gpu_free_at_start_gib": gib(free_before),
        "gpu_total_gib": gib(total),
        "gpu_used_end_gib": gib(total - free_end),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(raw, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
