"""One-off: does an out-of-distribution 'throw' prompt actually change the robot's
physical motion, or does the model just fall back to its trained pick-and-place
behavior regardless of the words? Measures the target bowl's peak speed and peak
height above the table for two prompts on the same initial state, so it's a fair
comparison (same physics, same starting pose, only the language differs).

Standalone (not through the web server) so it can log MuJoCo body kinematics
directly. Reuses the same checkpoint and preprocessing as webapp/server.py.
"""

import collections
import os
import pathlib

import numpy as np
import torch

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

CKPT = pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
RESIZE = 224
NUM_WAIT = 10
MAX_STEPS = 220
REPLAN = 5
DUMMY = [0.0] * 6 + [-1.0]
BOWL_BODY = "akita_black_bowl_2_main"  # the one "between the plate and the ramekin" for task 0


def quat2axisangle(q):
    q = np.asarray(q, dtype=np.float64)
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if np.isclose(den, 0.0) else (q[:3] * 2.0 * np.arccos(q[3])) / den


def views(obs):
    return (
        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
    )


def element(obs, prompt):
    agent, wrist = views(obs)
    return {
        "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(agent, RESIZE, RESIZE)),
        "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE)),
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": prompt,
    }


def run(policy, env, task_suite, task_id, prompt, tag):
    task = task_suite.get_task(task_id)
    init = task_suite.get_task_init_states(task_id)[0]
    env.reset()
    obs = env.set_init_state(init)

    body_id = env.sim.model.body_name2id(BOWL_BODY)
    table_z = 0.8  # LIBERO table height; heights below are reported relative to this

    plan = collections.deque()
    speeds, heights, grip = [], [], []
    t = 0
    while t < MAX_STEPS + NUM_WAIT:
        if t < NUM_WAIT:
            obs = env.step(DUMMY)[0]
            t += 1
            continue

        if not plan:
            chunk = policy.infer(element(obs, prompt))["actions"]
            plan.extend(chunk[:REPLAN])

        action = plan.popleft()
        obs, _, done, _ = env.step(np.asarray(action).tolist())

        v = env.sim.data.get_body_xvelp(BOWL_BODY)
        z = env.sim.data.body_xpos[body_id][2]
        speeds.append(float(np.linalg.norm(v)))
        heights.append(float(z - table_z))
        grip.append(float(obs["robot0_gripper_qpos"][0]))
        t += 1
        if done:
            break

    peak_speed = max(speeds) if speeds else 0.0
    peak_height = max(heights) if heights else 0.0
    peak_t = speeds.index(peak_speed) if speeds else -1
    print(
        f"[{tag:>22}] success={done!s:5} steps={t:3d}  "
        f"peak bowl speed={peak_speed:.3f} m/s (@step {peak_t})  "
        f"peak height={peak_height*100:.1f} cm above table"
    )
    return {"speeds": speeds, "heights": heights, "grip": grip, "success": bool(done), "steps": t}


def main():
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    os.environ.setdefault("MUJOCO_GL", "egl")

    train_config = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(train_config, CKPT)
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = task_suite.get_task(0)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(7)

    # warm up torch.compile once so the timed-looking prints below aren't polluted
    env.reset()
    warm_obs = env.set_init_state(task_suite.get_task_init_states(0)[0])
    policy.infer(element(warm_obs, "warmup"))

    baseline = run(policy, env, task_suite, 0, task.language, "trained instruction")
    thrown = run(policy, env, task_suite, 0, "throw the black bowl on the floor", "'throw' prompt")

    print()
    print(f"peak speed ratio (throw / baseline): {max(thrown['speeds'])/max(baseline['speeds'] or [1e-6]):.2f}x")
    print(f"peak height ratio (throw / baseline): {max(thrown['heights'])/max(baseline['heights'] or [1e-6]):.2f}x")
    env.close()


if __name__ == "__main__":
    main()
