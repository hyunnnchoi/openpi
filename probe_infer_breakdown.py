"""Where does policy.infer's wall time go on this box?

bench_infer reports ~147 ms for policy.infer while bench_batch reports ~57 ms for
the compiled sample_actions at batch 1. On GB10 those two numbers were 152 and 148,
i.e. the model was essentially all of it. This splits infer into its stages to find
what the rest is here.

Stages are timed with the same calls Policy.infer makes (policy.py:68-106), plus
the infer_ms the policy reports for itself. Note that Policy's own infer_ms stops
before the device sync in the output conversion, so it under-counts GPU time; the
`sample_actions + sync` row below is the honest model number.

Run: .venv/bin/python probe_infer_breakdown.py
"""

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
N_WARMUP = 3
N_ITERS = 20


def stats(name, xs):
    xs = sorted(xs)
    print(f"{name:<32} mean {statistics.mean(xs):7.2f} ms   p50 {xs[len(xs) // 2]:7.2f}   max {xs[-1]:7.2f}")


def main():
    train_config = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(train_config, CKPT)
    device = policy._pytorch_device  # noqa: SLF001
    example = libero_policy.make_libero_example()

    for _ in range(N_WARMUP):
        policy.infer(example)
    torch.cuda.synchronize()

    t_total, t_reported, t_in, t_todev, t_sample, t_out = [], [], [], [], [], []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        res = policy.infer(example)
        torch.cuda.synchronize()
        t_total.append((time.perf_counter() - t0) * 1e3)
        t_reported.append(res["policy_timing"]["infer_ms"])

        # Re-run the same stages individually.
        t0 = time.perf_counter()
        inputs = policy._input_transform(jax.tree.map(lambda x: x, example))  # noqa: SLF001
        t_in.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        obs = _model.Observation.from_dict(inputs)
        torch.cuda.synchronize()
        t_todev.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        actions = policy._sample_actions(device, obs, **policy._sample_kwargs)  # noqa: SLF001
        torch.cuda.synchronize()
        t_sample.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        outputs = jax.tree.map(
            lambda x: np.asarray(x[0, ...].detach().cpu()),
            {"state": inputs["state"], "actions": actions},
        )
        policy._output_transform(outputs)  # noqa: SLF001
        t_out.append((time.perf_counter() - t0) * 1e3)

    print()
    stats("policy.infer wall (sync'd)", t_total)
    stats("  infer_ms as policy reports", t_reported)
    stats("  input transform (CPU)", t_in)
    stats("  host->device + wrap", t_todev)
    stats("  sample_actions + sync (GPU)", t_sample)
    stats("  device->host + out transform", t_out)
    print()
    print(f"sum of stages: {sum(map(statistics.mean, [t_in, t_todev, t_sample, t_out])):.2f} ms")


if __name__ == "__main__":
    main()
