"""Replay a phase-offset arrival trace against real pi0.5 inference.

RESULTS_A100.md section 3 puts the deadline capacity at 6 robots per GPU, but
that number assumes every robot's request shows up at the same instant and gets
batched together. Real arms are not synchronised: each runs its own 250 ms
control loop, started whenever it was started. A robot arriving just after a
batch departs waits for the whole batch before its own can even start.

This measures what that costs, and what the scheduling policies do about it,
on the real model and real GPU rather than a cost model.

Policies:
  serial      -- what the stock server does: one request at a time, FIFO.
  batch       -- dynamic batching: take everything waiting, run it as one batch.
  continuous  -- denoise ring: prefill arrivals, then advance every in-flight
                 request by one denoise step per tick, each at its own timestep.
                 Late arrivals join at the next step instead of the next batch.

Time is real, not simulated: the loop executes actual GPU work and reads the
clock, so queueing and compute interleave the way they would in the server. A
request is a miss if it completes more than --deadline ms after it arrived.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_scheduler.py
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
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

CKPT = pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
OUT = pathlib.Path("data/bench_scheduler.json")

# Phase offsets within the first control period, in ms. Each robot then repeats
# every --period ms. Deliberately uneven -- evenly spaced arrivals would flatter
# the batching policies.
DEFAULT_PHASES = [0.0, 40.0, 85.0, 120.0, 190.0, 230.0]


class Request:
    __slots__ = ("robot", "cycle", "arrival_ms", "start_ms", "done_ms", "steps_done", "kv", "mask", "state", "x_t")

    def __init__(self, robot, cycle, arrival_ms):
        self.robot = robot
        self.cycle = cycle
        self.arrival_ms = arrival_ms
        self.start_ms = None
        self.done_ms = None
        self.steps_done = 0
        self.kv = self.mask = self.state = self.x_t = None

    @property
    def e2e_ms(self):
        return self.done_ms - self.arrival_ms


def build_trace(phases, period_ms, cycles):
    reqs = []
    for robot, phase in enumerate(phases):
        for c in range(cycles):
            reqs.append(Request(robot, c, phase + c * period_ms))
    reqs.sort(key=lambda r: r.arrival_ms)
    return reqs


def summarize(reqs, deadline_ms, wall_ms, label):
    lat = sorted(r.e2e_ms for r in reqs if r.done_ms is not None)
    done = len(lat)
    miss = sum(1 for x in lat if x > deadline_ms)
    out = {
        "policy": label,
        "completed": done,
        "total": len(reqs),
        "wall_ms": wall_ms,
        "throughput_rps": done / (wall_ms / 1e3) if wall_ms else 0.0,
        "e2e_mean_ms": statistics.mean(lat) if lat else None,
        "e2e_p50_ms": lat[len(lat) // 2] if lat else None,
        "e2e_p99_ms": lat[min(int(len(lat) * 0.99), len(lat) - 1)] if lat else None,
        "e2e_max_ms": lat[-1] if lat else None,
        "misses": miss,
        "miss_pct": 100.0 * miss / done if done else None,
    }
    return out


def print_row(s):
    print(
        f"{s['policy']:<14} {s['completed']:>4}/{s['total']:<4} "
        f"{s['throughput_rps']:>8.2f} {s['e2e_mean_ms']:>10.1f} {s['e2e_p50_ms']:>9.1f} "
        f"{s['e2e_p99_ms']:>9.1f} {s['e2e_max_ms']:>9.1f} {s['miss_pct']:>7.1f}%",
        flush=True,
    )


# --------------------------------------------------------------------------
# model plumbing
# --------------------------------------------------------------------------


def make_inputs(policy, seed, prompt):
    """One robot's transformed observation, kept as a pytree so it can be stacked.

    Contents differ per robot so nothing is accidentally shared between them.
    """
    ex = libero_policy.make_libero_example()
    rng = np.random.default_rng(seed)
    for k, v in ex.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            ex[k] = rng.integers(0, 255, size=v.shape, dtype=v.dtype)
        elif isinstance(v, np.ndarray):
            ex[k] = (rng.random(v.shape) * 2 - 1).astype(v.dtype)
    ex["prompt"] = prompt
    return policy._input_transform(ex)  # noqa: SLF001


def stack_observations(inputs_list, device):
    """Stack per-robot input pytrees into one batched Observation.

    jax.tree.map, not a flat dict walk: the transformed inputs nest.
    """

    def rep(*xs):
        return torch.from_numpy(np.stack([np.asarray(x) for x in xs], axis=0)).to(device)

    return _model.Observation.from_dict(jax.tree.map(rep, *inputs_list))


def prefill(model, observation):
    """The prefix half of sample_actions, returned instead of consumed."""
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(  # noqa: SLF001
        observation, train=False
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    masks_4d = model._prepare_attention_masks_4d(prefix_att_2d)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=masks_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return past_key_values, prefix_pad_masks, state


def cache_rows(kv):
    """Split a batched KV cache into one single-row cache per batch element."""
    import copy

    n = kv.key_cache[0].shape[0]
    rows = []
    for i in range(n):
        c = copy.copy(kv)
        c.key_cache = [k[i : i + 1] for k in kv.key_cache]
        c.value_cache = [v[i : i + 1] for v in kv.value_cache]
        rows.append(c)
    return rows


def own_cache(kv):
    """Copy a KV cache out of the CUDA-graph pool so it survives later calls.

    max-autotune runs compiled regions as CUDA graph trees, and every output lives
    in a pool that the *next* invocation of that graph overwrites. A denoise ring
    holds each request's prefill KV across ten subsequent denoise calls, so the
    cache has to be copied out or it is silently clobbered (torch raises here
    rather than corrupting, which is how this was caught).

    A production ring would avoid the copy with a preallocated KV pool; here it is
    ~18 MB per request and is charged to the policy that needs it.
    """
    import copy

    c = copy.copy(kv)
    c.key_cache = [k.clone() for k in kv.key_cache]
    c.value_cache = [v.clone() for v in kv.value_cache]
    return c


def merge_caches(caches):
    """Concatenate per-request KV caches along the batch axis."""
    import copy

    merged = copy.copy(caches[0])
    n_layers = len(caches[0].key_cache)
    merged.key_cache = [torch.cat([c.key_cache[i] for c in caches], dim=0) for i in range(n_layers)]
    merged.value_cache = [torch.cat([c.value_cache[i] for c in caches], dim=0) for i in range(n_layers)]
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", type=float, nargs="+", default=DEFAULT_PHASES)
    ap.add_argument("--period", type=float, default=250.0, help="control period per robot, ms")
    ap.add_argument("--deadline", type=float, default=250.0)
    ap.add_argument("--cycles", type=int, default=8, help="control cycles per robot")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--dry-runs", type=int, default=1, help="untimed passes per policy, to absorb compilation")
    ap.add_argument(
        "--policies", nargs="+", default=["serial", "batch", "continuous"], help="serial | batch | continuous"
    )
    args = ap.parse_args()

    free_before, total = torch.cuda.mem_get_info()
    print(f"GPU free at start    : {free_before / 2**30:.1f} / {total / 2**30:.1f} GiB", flush=True)
    if free_before < total - 2 * 2**30:
        print("!! GPU is not idle -- see RESULTS_A100.md section 4", flush=True)

    cfg = _config.get_config("pi05_libero")
    t0 = time.perf_counter()
    policy = _policy_config.create_trained_policy(cfg, CKPT)
    print(f"model load           : {time.perf_counter() - t0:.1f} s", flush=True)
    model = policy._model  # noqa: SLF001
    device = policy._pytorch_device  # noqa: SLF001

    n_robots = len(args.phases)
    prompts = [
        "pick up the black bowl and place it on the plate",
        "put the plate in the drawer",
        "open the top drawer",
        "pick up the mug",
        "move the bowl to the stove",
        "close the drawer",
    ]
    robot_inputs = [make_inputs(policy, 100 + i, prompts[i % len(prompts)]) for i in range(n_robots)]
    print(f"robots               : {n_robots}, phases {args.phases}", flush=True)

    # PI0Pytorch's constructor already wraps sample_actions in
    # torch.compile(mode="max-autotune"), but denoise_step and the prefill half are
    # left eager. Comparing a compiled whole-request path against an eager
    # step-at-a-time path would decide this experiment before it runs -- a denoise
    # step is 2.2 ms compiled and 41.8 ms eager (RESULTS_A100.md section 3). Compile
    # the pieces the same way so the difference measured is the scheduling, not the
    # compilation.
    compile_mode = cfg.model.pytorch_compile_mode
    sample_actions = model.sample_actions
    denoise_step = model.denoise_step
    prefill_fn = prefill
    if compile_mode:
        denoise_step = torch.compile(denoise_step, mode=compile_mode)
        prefill_fn = torch.compile(prefill, mode=compile_mode)
    print(f"compile mode         : {compile_mode}", flush=True)

    # No separate shape warmup: matching every shape a policy will produce turned out
    # to be guesswork (the denoise ring's KV is assembled from single-row caches, which
    # guards differently from a directly-prefilled batch), and a missed shape compiles
    # *inside* the measured window -- a 240 s "latency" in the first attempt. Instead
    # each policy runs its whole trace once and that result is thrown away, so the
    # measured pass exercises only code that is already compiled.

    results = []

    # ---- policy: serial -------------------------------------------------
    def run_serial():
        reqs = build_trace(args.phases, args.period, args.cycles)
        pending = list(reqs)
        base = time.perf_counter()
        now_ms = lambda: (time.perf_counter() - base) * 1e3  # noqa: E731
        queue = []
        i = 0
        with torch.no_grad():
            while i < len(pending) or queue:
                t = now_ms()
                while i < len(pending) and pending[i].arrival_ms <= t:
                    queue.append(pending[i])
                    i += 1
                if not queue:
                    time.sleep(0.0005)
                    continue
                r = queue.pop(0)
                r.start_ms = now_ms()
                torch.compiler.cudagraph_mark_step_begin()
                sample_actions(device, stack_observations([robot_inputs[r.robot]], device), num_steps=args.num_steps)
                torch.cuda.synchronize()
                r.done_ms = now_ms()
        return reqs, now_ms()

    # ---- policy: naive dynamic batching ---------------------------------
    def run_batch():
        reqs = build_trace(args.phases, args.period, args.cycles)
        pending = list(reqs)
        base = time.perf_counter()
        now_ms = lambda: (time.perf_counter() - base) * 1e3  # noqa: E731
        queue = []
        i = 0
        with torch.no_grad():
            while i < len(pending) or queue:
                t = now_ms()
                while i < len(pending) and pending[i].arrival_ms <= t:
                    queue.append(pending[i])
                    i += 1
                if not queue:
                    time.sleep(0.0005)
                    continue
                group = queue[: args.max_batch]
                del queue[: len(group)]
                for r in group:
                    r.start_ms = now_ms()
                obs = stack_observations([robot_inputs[r.robot] for r in group], device)
                torch.compiler.cudagraph_mark_step_begin()
                sample_actions(device, obs, num_steps=args.num_steps)
                torch.cuda.synchronize()
                done = now_ms()
                for r in group:
                    r.done_ms = done
        return reqs, now_ms()

    # ---- policy: continuous batching (denoise ring) ---------------------
    def run_continuous():
        reqs = build_trace(args.phases, args.period, args.cycles)
        pending = list(reqs)
        base = time.perf_counter()
        now_ms = lambda: (time.perf_counter() - base) * 1e3  # noqa: E731
        waiting, inflight = [], []
        i = 0
        with torch.no_grad():
            while i < len(pending) or waiting or inflight:
                t = now_ms()
                while i < len(pending) and pending[i].arrival_ms <= t:
                    waiting.append(pending[i])
                    i += 1

                # Admit whatever is waiting, prefilled as one batch.
                if waiting and len(inflight) < args.max_batch:
                    group = waiting[: args.max_batch - len(inflight)]
                    del waiting[: len(group)]
                    obs = stack_observations([robot_inputs[r.robot] for r in group], device)
                    torch.compiler.cudagraph_mark_step_begin()
                    kv, mask, state = prefill_fn(model, obs)
                    kv, mask, state = own_cache(kv), mask.clone(), state.clone()
                    rows = cache_rows(kv)
                    for j, r in enumerate(group):
                        r.start_ms = now_ms()
                        r.kv = rows[j]
                        r.mask = mask[j : j + 1]
                        r.state = state[j : j + 1]
                        r.x_t = model.sample_noise(
                            (1, model.config.action_horizon, model.config.action_dim), device
                        )
                        r.steps_done = 0
                        inflight.append(r)

                if not inflight:
                    time.sleep(0.0005)
                    continue

                # One Euler step for every in-flight request, each at its own time.
                dt = -1.0 / args.num_steps
                kv = merge_caches([r.kv for r in inflight])
                mask = torch.cat([r.mask for r in inflight], dim=0)
                state = torch.cat([r.state for r in inflight], dim=0)
                x_t = torch.cat([r.x_t for r in inflight], dim=0)
                times = torch.tensor(
                    [1.0 + dt * r.steps_done for r in inflight], dtype=torch.float32, device=device
                )
                torch.compiler.cudagraph_mark_step_begin()
                v_t = denoise_step(state, mask, kv, x_t, times)
                x_next = x_t + dt * v_t
                torch.cuda.synchronize()
                t_done = now_ms()

                still = []
                for j, r in enumerate(inflight):
                    r.x_t = x_next[j : j + 1]
                    r.steps_done += 1
                    if r.steps_done >= args.num_steps:
                        r.done_ms = t_done
                        r.kv = None
                    else:
                        still.append(r)
                inflight = still
        return reqs, now_ms()

    runners = {"serial": run_serial, "batch": run_batch, "continuous": run_continuous}

    print(
        f"\n{'policy':<14} {'done':>9} {'thru r/s':>8} {'e2e mean':>10} {'e2e p50':>9} "
        f"{'e2e p99':>9} {'e2e max':>9} {'miss':>8}",
        flush=True,
    )
    for name in args.policies:
        for _ in range(args.dry_runs):
            t_dry = time.perf_counter()
            runners[name]()
            torch.cuda.synchronize()
            print(f"  {name}: dry run (compiling) {time.perf_counter() - t_dry:.1f} s", flush=True)
        reqs, wall = runners[name]()
        s = summarize(reqs, args.deadline, wall, name)
        s["per_request"] = [
            {"robot": r.robot, "cycle": r.cycle, "arrival_ms": r.arrival_ms, "e2e_ms": r.e2e_ms}
            for r in reqs
            if r.done_ms is not None
        ]
        results.append(s)
        print_row(s)
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "meta": {
                    "phases": args.phases,
                    "period_ms": args.period,
                    "deadline_ms": args.deadline,
                    "cycles": args.cycles,
                    "num_steps": args.num_steps,
                    "max_batch": args.max_batch,
                    "robots": n_robots,
                    "gpu_free_at_start_gib": free_before / 2**30,
                },
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
