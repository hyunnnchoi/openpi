"""Standalone latency benchmark for a pi0.5 PyTorch policy on this machine.

Measures per-inference wall time for one action chunk, plus GPU memory footprint.
Run: .venv/bin/python bench_infer.py
"""

import pathlib
import statistics
import time

import torch

from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

CKPT = pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
N_WARMUP = 3
N_ITERS = 20


def gib(x):
    return x / 2**30


def main():
    train_config = _config.get_config("pi05_libero")

    free_before, total = torch.cuda.mem_get_info()
    t0 = time.perf_counter()
    policy = _policy_config.create_trained_policy(train_config, CKPT)
    load_s = time.perf_counter() - t0
    free_after, _ = torch.cuda.mem_get_info()

    print(f"model load           : {load_s:7.2f} s")
    print(f"GPU mem for weights  : {gib(free_before - free_after):7.2f} GiB")
    print(f"action_horizon       : {train_config.model.action_horizon}")
    print(f"action_dim           : {train_config.model.action_dim}")

    example = libero_policy.make_libero_example()

    for _ in range(N_WARMUP):
        policy.infer(example)
    torch.cuda.synchronize()

    lat = []
    for _ in range(N_ITERS):
        t = time.perf_counter()
        out = policy.infer(example)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1e3)

    lat.sort()
    horizon = out["actions"].shape[0]
    mean = statistics.mean(lat)
    print()
    print(f"actions shape        : {out['actions'].shape}")
    print(f"latency mean         : {mean:7.2f} ms")
    print(f"latency p50          : {lat[len(lat) // 2]:7.2f} ms")
    print(f"latency p99          : {lat[int(len(lat) * 0.99)]:7.2f} ms")
    print(f"latency min / max    : {lat[0]:7.2f} / {lat[-1]:7.2f} ms")
    print()
    print(f"-> one chunk = {horizon} actions, so {mean / horizon:.1f} ms of compute per action step")
    print(f"-> sustainable control rate if replanning every chunk: {1000 * horizon / mean:.1f} Hz")

    free_end, _ = torch.cuda.mem_get_info()
    print(f"GPU mem in use total : {gib(total - free_end):7.2f} / {gib(total):.1f} GiB")


if __name__ == "__main__":
    main()
