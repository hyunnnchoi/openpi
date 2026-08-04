"""Can prefill and denoise run at the same time on two CUDA streams?

Everything measured so far says a request should not be cut into more pieces on
this box: the denoise ring lost in both load regimes (RESULTS_A100.md section 4)
because it turned one graph launch per request into eleven, and this box is
host-bound at 75% GPU-busy (section 2). Overlapping on streams is the opposite
move -- both halves stay whole, they just run beside each other.

The case for expecting a win:
  * prefill barely batches (33.1 -> 25.7 ms per robot over a 16x batch, section
    3), which is what a compute-bound stage looks like.
  * a denoise step is 2.2 ms at B=1, and bandwidth explains almost none of it:
    the KV read is 17.8 MB (~0.01 ms at HBM speed) and the expert's weights are
    ~600 MB (~0.32 ms). The other ~1.9 ms is small kernels at low occupancy.
    Latency-bound, not bandwidth-bound -- so it leaves SMs idle for prefill.
  * prefill runs PaliGemma and denoise runs the 300m expert, so the two are not
    contending for the same weights.

If that holds, wall time for the pair lands near max(prefill, denoise) instead
of their sum. This measures where between the two it actually lands.

The catch is torch.compile: max-autotune runs compiled regions as CUDA graph
trees, which manage their own stream and may serialise the two anyway. So both
compile modes are measured -- with cudagraphs to see what the real serving path
would get, and without to see whether the hardware overlap is there at all.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_stream_overlap.py
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
OUT = pathlib.Path("data/bench_stream_overlap.json")

# cudagraph trees record a shape on its second call, not its first; one warm call
# leaves the recording to land inside the measurement (RESULTS_A100.md section 4).
WARM_CALLS = 3


def gib(x):
    return x / 2**30


def make_inputs(policy, seed, prompt):
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
    def rep(*xs):
        return torch.from_numpy(np.stack([np.asarray(x) for x in xs], axis=0)).to(device)

    return _model.Observation.from_dict(jax.tree.map(rep, *inputs_list))


def prefill(model, observation):
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


def own_cache(kv):
    import copy

    c = copy.copy(kv)
    c.key_cache = [k.clone() for k in kv.key_cache]
    c.value_cache = [v.clone() for v in kv.value_cache]
    return c


def timed(fn, n_iters):
    lat = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1e3)
    lat.sort()
    return statistics.mean(lat), lat[len(lat) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill-batch", type=int, default=2)
    ap.add_argument("--denoise-batch", type=int, default=4)
    ap.add_argument("--denoise-steps", type=int, default=4, help="denoise steps fired per trial")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--modes", nargs="+", default=["max-autotune", "max-autotune-no-cudagraphs"])
    args = ap.parse_args()

    free_before, total = torch.cuda.mem_get_info()
    print(f"GPU free at start    : {gib(free_before):.1f} / {gib(total):.1f} GiB", flush=True)
    if free_before < total - 2 * 2**30:
        print("!! GPU is not idle -- see RESULTS_A100.md section 5", flush=True)

    cfg = _config.get_config("pi05_libero")
    t0 = time.perf_counter()
    policy = _policy_config.create_trained_policy(cfg, CKPT)
    print(f"model load           : {time.perf_counter() - t0:.1f} s", flush=True)
    model = policy._model  # noqa: SLF001
    device = policy._pytorch_device  # noqa: SLF001

    prompts = ["pick up the black bowl", "put the plate in the drawer", "open the drawer", "pick up the mug"]
    n = max(args.prefill_batch, args.denoise_batch)
    robot_inputs = [make_inputs(policy, 200 + i, prompts[i % len(prompts)]) for i in range(n)]

    obs_p = stack_observations(robot_inputs[: args.prefill_batch], device)
    obs_d = stack_observations(robot_inputs[: args.denoise_batch], device)

    results = {"meta": vars(args) | {"gpu_free_at_start_gib": gib(free_before)}, "modes": {}}

    for mode in args.modes:
        print(f"\n=== compile mode: {mode} ===", flush=True)
        torch.compiler.reset()
        pf = torch.compile(prefill, mode=mode)
        dn = torch.compile(model.denoise_step, mode=mode)

        # Build the denoise inputs once, outside the graph pool.
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            kv_d, mask_d, state_d = prefill(model, obs_d)
            kv_d, mask_d, state_d = own_cache(kv_d), mask_d.clone(), state_d.clone()
        x_d = model.sample_noise(
            (args.denoise_batch, model.config.action_horizon, model.config.action_dim), device
        )
        t_d = torch.tensor(
            [1.0 - 0.1 * i for i in range(args.denoise_batch)], dtype=torch.float32, device=device
        )

        def run_prefill():
            torch.compiler.cudagraph_mark_step_begin()
            pf(model, obs_p)

        def run_denoise():
            for _ in range(args.denoise_steps):
                torch.compiler.cudagraph_mark_step_begin()
                dn(state_d, mask_d, kv_d, x_d, t_d)

        print("warming...", flush=True)
        with torch.no_grad():
            for _ in range(WARM_CALLS):
                run_prefill()
                run_denoise()
            torch.cuda.synchronize()

        with torch.no_grad():
            p_mean, p_p50 = timed(run_prefill, args.iters)
            d_mean, d_p50 = timed(run_denoise, args.iters)

            def run_serial():
                run_prefill()
                run_denoise()

            s_mean, s_p50 = timed(run_serial, args.iters)

            stream = torch.cuda.Stream()

            def run_overlap():
                # Fire denoise on a side stream so it can fill whatever prefill leaves.
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run_denoise()
                run_prefill()
                torch.cuda.current_stream().wait_stream(stream)

            o_mean, o_p50 = timed(run_overlap, args.iters)

        lower = max(p_mean, d_mean)
        saved = s_mean - o_mean
        # 1.0 means the shorter side hid completely; 0.0 means no overlap at all.
        efficiency = saved / min(p_mean, d_mean) if min(p_mean, d_mean) else 0.0
        print(
            f"  prefill B={args.prefill_batch:<2}          : {p_mean:7.2f} ms\n"
            f"  denoise B={args.denoise_batch:<2} x{args.denoise_steps:<2}       : {d_mean:7.2f} ms\n"
            f"  serial (sum)          : {s_mean:7.2f} ms\n"
            f"  two streams           : {o_mean:7.2f} ms\n"
            f"  perfect-overlap floor : {lower:7.2f} ms\n"
            f"  -> saved {saved:6.2f} ms, overlap efficiency {efficiency * 100:5.1f}%",
            flush=True,
        )
        results["modes"][mode] = {
            "prefill_ms": p_mean,
            "denoise_ms": d_mean,
            "serial_ms": s_mean,
            "overlap_ms": o_mean,
            "floor_ms": lower,
            "saved_ms": saved,
            "overlap_efficiency": efficiency,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
