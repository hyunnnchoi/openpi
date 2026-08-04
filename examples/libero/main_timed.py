"""LIBERO rollout with a wall-clock breakdown of the control loop.

Same rollout logic as main.py, but records where the time actually goes:
policy inference (a blocking round-trip to the server), simulator stepping,
and client-side image preprocessing. Reuses main.py's helpers so the two stay
in sync.

Run with the same env as main.py:
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  MUJOCO_GL=egl examples/libero/.venv/bin/python examples/libero/main_timed.py
"""

import collections
import dataclasses
import json
import pathlib
import statistics
import time

import imageio
from libero.libero import benchmark
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

from main import LIBERO_DUMMY_ACTION
from main import LIBERO_ENV_RESOLUTION
from main import _get_libero_env
from main import _quat2axisangle

# LIBERO's OSC controller runs at 20 Hz, so one env.step is 50 ms of robot time.
CONTROL_PERIOD_S = 0.05


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_tasks: int = 2
    num_trials_per_task: int = 1
    max_steps: int = 220
    seed: int = 7
    video_out_path: str = "data/libero/videos_timed"
    stats_out_path: str = "data/libero/timing.json"


def _pct(x, total):
    return 100.0 * x / total if total else 0.0


def main(args: Args) -> None:
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.stats_out_path).parent.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    infer_ms, step_ms, prep_ms = [], [], []
    episodes = []

    for task_id in range(min(args.num_tasks, task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for episode_idx in range(args.num_trials_per_task):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            replay_images = []
            done = False
            t = 0
            ep_start = time.perf_counter()

            while t < args.max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs = env.step(LIBERO_DUMMY_ACTION)[0]
                    t += 1
                    continue

                t0 = time.perf_counter()
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                prep_ms.append((time.perf_counter() - t0) * 1e3)
                replay_images.append(img)

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }
                    t0 = time.perf_counter()
                    action_chunk = client.infer(element)["actions"]
                    infer_ms.append((time.perf_counter() - t0) * 1e3)
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                t0 = time.perf_counter()
                obs, reward, done, info = env.step(action.tolist())
                step_ms.append((time.perf_counter() - t0) * 1e3)
                if done:
                    break
                t += 1

            wall_s = time.perf_counter() - ep_start
            robot_s = t * CONTROL_PERIOD_S
            episodes.append(
                {
                    "task": task_description,
                    "success": bool(done),
                    "steps": t,
                    "wall_s": wall_s,
                    "robot_s": robot_s,
                    "realtime_factor": robot_s / wall_s if wall_s else 0.0,
                }
            )
            print(
                f"[task {task_id} ep {episode_idx}] success={done} steps={t} "
                f"wall={wall_s:.1f}s robot={robot_s:.1f}s rtf={robot_s / wall_s:.2f}x"
            )
            suffix = "success" if done else "failure"
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_task{task_id}_ep{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

        env.close()

    total_ms = sum(infer_ms) + sum(step_ms) + sum(prep_ms)
    successes = sum(e["success"] for e in episodes)

    def summarize(name, xs):
        if not xs:
            return
        xs_sorted = sorted(xs)
        print(
            f"  {name:<12} n={len(xs):<5} mean={statistics.mean(xs):7.2f} ms  "
            f"p50={xs_sorted[len(xs) // 2]:7.2f}  p99={xs_sorted[int(len(xs) * 0.99)]:7.2f}  "
            f"share={_pct(sum(xs), total_ms):5.1f}%"
        )

    print()
    print(f"episodes: {len(episodes)}, success {successes}/{len(episodes)}")
    print("per-call latency:")
    summarize("policy.infer", infer_ms)
    summarize("env.step", step_ms)
    summarize("preprocess", prep_ms)

    if infer_ms and step_ms:
        # One replan covers `replan_steps` control steps; how much of that budget is the stall?
        budget_ms = args.replan_steps * CONTROL_PERIOD_S * 1e3
        print()
        print(f"control budget per replan window ({args.replan_steps} steps @ 20 Hz): {budget_ms:.0f} ms")
        print(f"blocking inference stall per window          : {statistics.mean(infer_ms):.0f} ms")
        print(f"-> inference consumes {_pct(statistics.mean(infer_ms), budget_ms):.0f}% of the window")
        rtf = statistics.mean(e["realtime_factor"] for e in episodes)
        print(f"-> mean realtime factor across episodes      : {rtf:.2f}x")

    stats = {
        "episodes": episodes,
        "infer_ms": infer_ms,
        "step_ms": step_ms,
        "prep_ms": prep_ms,
        "replan_steps": args.replan_steps,
    }
    pathlib.Path(args.stats_out_path).write_text(json.dumps(stats))
    print(f"\nraw timings -> {args.stats_out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
