"""Batch-scaling benchmark for pi0.5 inference on one GPU.

The question this answers: if N robots share one GPU, does batching them together
cost anything? If latency(B=8) ~= latency(B=1), many robots per GPU is free and
the scheduling problem is trivial. If it scales linearly, one GPU saturates at
N=1-2 and "many robots per GPU" is hardware-bound on this box.

Two paths are measured separately:
  * uncompiled -- type(model).sample_actions, bypasses the whole-graph
    torch.compile. Cheap to sweep, and num_steps can be varied for free, which
    lets us split prefill (VLM forward -> KV cache) from denoise (Euler steps).
  * compiled   -- the real serving path. Batch size is part of the input shape,
    so every distinct B pays a fresh max-autotune compile. That cost is recorded.

Run: .venv/bin/python bench_batch.py
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
OUT = pathlib.Path("data/bench_batch.json")


def gib(x):
    return x / 2**30


def make_batch(inputs, batch_size, device):
    """Replicate a single transformed observation into a batch of `batch_size`."""

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
        "min": lat[0],
        "max": lat[-1],
        "n": len(lat),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--compiled-batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=10)
    args = ap.parse_args()

    train_config = _config.get_config("pi05_libero")
    free_before, total = torch.cuda.mem_get_info()
    print(f"GPU free at start    : {gib(free_before):.1f} / {gib(total):.1f} GiB", flush=True)

    t0 = time.perf_counter()
    policy = _policy_config.create_trained_policy(train_config, CKPT)
    print(f"model load           : {time.perf_counter() - t0:.1f} s", flush=True)
    free_after, _ = torch.cuda.mem_get_info()
    print(f"weights              : {gib(free_before - free_after):.2f} GiB", flush=True)

    model = policy._model  # noqa: SLF001
    device = policy._pytorch_device  # noqa: SLF001
    inputs = policy._input_transform(libero_policy.make_libero_example())  # noqa: SLF001

    # The compiled entry point is an attribute on the instance; the raw method
    # still lives on the class.
    raw_sample = type(model).sample_actions

    results = {"uncompiled": {}, "compiled": {}, "meta": {}}

    # ---- uncompiled sweep: batch scaling + prefill/denoise split -------------
    print("\n=== uncompiled (type(model).sample_actions) ===", flush=True)
    print(f"{'B':>3} {'steps':>6} {'mean ms':>9} {'p99 ms':>8} {'ms/robot':>9} {'peak GiB':>9}", flush=True)
    for b in args.batches:
        obs = _model.Observation.from_dict(make_batch(inputs, b, device))
        for steps in sorted({1, args.num_steps}):
            torch.cuda.reset_peak_memory_stats()
            try:
                with torch.no_grad():
                    st = timeit(
                        lambda: raw_sample(model, device, obs, num_steps=steps),
                        args.warmup,
                        args.iters,
                    )
            except torch.OutOfMemoryError:
                print(f"{b:>3} {steps:>6}  OOM", flush=True)
                torch.cuda.empty_cache()
                break
            peak = gib(torch.cuda.max_memory_allocated())
            st["peak_gib"] = peak
            st["per_robot_ms"] = st["mean"] / b
            results["uncompiled"][f"b{b}_s{steps}"] = st
            print(
                f"{b:>3} {steps:>6} {st['mean']:>9.1f} {st['p99']:>8.1f} "
                f"{st['per_robot_ms']:>9.1f} {peak:>9.2f}",
                flush=True,
            )
        del obs
        torch.cuda.empty_cache()

    # ---- derived prefill / denoise split ------------------------------------
    print("\n=== derived split (from steps=1 vs steps=%d) ===" % args.num_steps, flush=True)
    print(f"{'B':>3} {'prefill ms':>11} {'ms/denoise':>11} {'denoise tot':>12} {'prefill %':>10}", flush=True)
    for b in args.batches:
        k1, kn = f"b{b}_s1", f"b{b}_s{args.num_steps}"
        if k1 not in results["uncompiled"] or kn not in results["uncompiled"]:
            continue
        t1 = results["uncompiled"][k1]["mean"]
        tn = results["uncompiled"][kn]["mean"]
        per_step = (tn - t1) / (args.num_steps - 1)
        prefill = t1 - per_step
        denoise_tot = per_step * args.num_steps
        results["uncompiled"][kn]["prefill_ms"] = prefill
        results["uncompiled"][kn]["per_denoise_ms"] = per_step
        print(
            f"{b:>3} {prefill:>11.1f} {per_step:>11.1f} {denoise_tot:>12.1f} "
            f"{100 * prefill / tn:>9.1f}%",
            flush=True,
        )

    # ---- compiled sweep: real serving path + per-shape compile cost ---------
    print("\n=== compiled (max-autotune, the real serving path) ===", flush=True)
    print(f"{'B':>3} {'compile s':>10} {'mean ms':>9} {'p99 ms':>8} {'ms/robot':>9}", flush=True)
    for b in args.compiled_batches:
        obs = _model.Observation.from_dict(make_batch(inputs, b, device))
        try:
            t = time.perf_counter()
            with torch.no_grad():
                model.sample_actions(device, obs, num_steps=args.num_steps)
            torch.cuda.synchronize()
            compile_s = time.perf_counter() - t
            with torch.no_grad():
                st = timeit(
                    lambda: model.sample_actions(device, obs, num_steps=args.num_steps),
                    args.warmup,
                    args.iters,
                )
        except torch.OutOfMemoryError:
            print(f"{b:>3}  OOM", flush=True)
            torch.cuda.empty_cache()
            break
        st["first_call_s"] = compile_s
        st["per_robot_ms"] = st["mean"] / b
        results["compiled"][f"b{b}"] = st
        print(
            f"{b:>3} {compile_s:>10.1f} {st['mean']:>9.1f} {st['p99']:>8.1f} "
            f"{st['per_robot_ms']:>9.1f}",
            flush=True,
        )
        del obs
        torch.cuda.empty_cache()

    free_end, _ = torch.cuda.mem_get_info()
    results["meta"] = {
        "num_steps": args.num_steps,
        "iters": args.iters,
        "gpu_total_gib": gib(total),
        "gpu_used_end_gib": gib(total - free_end),
        "weights_gib": gib(free_before - free_after),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}", flush=True)
    print(f"GPU in use at end    : {gib(total - free_end):.1f} / {gib(total):.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
