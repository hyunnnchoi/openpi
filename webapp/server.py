"""Web console for poking at a pi0.5 VLA policy on LIBERO.

Single process: policy (GPU) + MuJoCo sim + HTTP/WebSocket server, so the
denoising loop can be instrumented directly instead of going over the wire.

Run:
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    MUJOCO_GL=egl .venv/bin/python webapp/server.py

Then from your laptop:
    ssh -N -L 8080:localhost:8080 <this-host>
    open http://localhost:8080
"""

import argparse
import asyncio
import base64
import collections
import concurrent.futures
import contextlib
import dataclasses
import io
import logging
import pathlib
import queue
import threading
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vla-console")

# LIBERO ships init states as numpy pickles; PyTorch >= 2.6 defaults to weights_only=True.
_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

from fastapi import FastAPI  # noqa: E402
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from PIL import Image  # noqa: E402
import uvicorn  # noqa: E402

from openpi.policies import libero_policy  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

CKPT = pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
CONFIG_NAME = "pi05_libero"
DEFAULT_SUITE = "libero_spatial"
ENV_RESOLUTION = 256
RESIZE = 224
CONTROL_HZ = 20.0
CONTROL_PERIOD_S = 1.0 / CONTROL_HZ
DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_WAIT_STEPS = 10

# Episode caps copied from examples/libero/main.py — each suite's longest training
# demo differs, so a shared cap would either truncate long tasks or waste time.
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
SUITES = list(MAX_STEPS_BY_SUITE)
STATIC_DIR = pathlib.Path(__file__).parent / "static"

# LIBERO proprioception: end-effector pose (axis-angle) + both gripper finger joints.
STATE_LABELS = ["eef x", "eef y", "eef z", "axis rx", "axis ry", "axis rz", "grip L", "grip R"]


# --------------------------------------------------------------------------- helpers


def jpeg_b64(arr: np.ndarray, quality: int = 80) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def gpu_used_gib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30


def quat2axisangle(quat):
    """Same convention as examples/libero/main.py."""
    quat = np.asarray(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


class Ring:
    """Fixed-size window with cheap summary stats."""

    def __init__(self, n=200):
        self.buf = collections.deque(maxlen=n)

    def add(self, x):
        self.buf.append(x)

    def stats(self):
        if not self.buf:
            return {"n": 0, "last": None, "mean": None, "p50": None, "p99": None}
        s = sorted(self.buf)
        return {
            "n": len(s),
            "last": self.buf[-1],
            "mean": sum(s) / len(s),
            "p50": s[len(s) // 2],
            "p99": s[min(len(s) - 1, int(len(s) * 0.99))],
        }

    def recent(self, k=60):
        return list(self.buf)[-k:]


# --------------------------------------------------------------------------- engine


@dataclasses.dataclass
class RunConfig:
    suite: str = DEFAULT_SUITE
    task_id: int = 0
    episode_idx: int = 0
    replan_steps: int = 5
    prompt: str = ""
    pace_realtime: bool = False


class Engine:
    """Owns the policy, the sim, and the rollout thread."""

    def __init__(self):
        self.policy = None
        self._suites = {}  # name -> instantiated benchmark, built lazily
        self._env = None
        self._env_key = None  # (suite, task_id)
        self._obs = None
        # Per-connection event queues: every websocket gets its own copy of each
        # event. A single shared queue splits frames between tabs (one tab steals
        # half the frames of the other), which looks like stutter.
        self._subs: list[queue.Queue] = []
        self._subs_lock = threading.Lock()
        # ALL policy inference runs on this one thread. The torch.compile'd graph
        # uses CUDA graph trees whose bookkeeping lives in thread-local storage;
        # calling the compiled model from any other thread dies with
        # `assert torch._C._is_key_in_tls(attr_name)`.
        self._infer_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._last_frame_emit = 0.0
        self.running = False
        self.cfg = RunConfig()

        self.infer_ms = Ring()
        self.step_ms = Ring()
        self.stream_ms = Ring()
        self.state = {
            "loaded": False,
            "phase": "시작 중",
            "running": False,
            "t": 0,
            "chunks": 0,
            "success": None,
            "task": "",
            "wall_s": 0.0,
            "robot_s": 0.0,
        }

    # -- lifecycle ---------------------------------------------------------

    def load(self):
        """Runs on the inference thread (submitted to _infer_pool at startup) so that
        the torch.compile warmup below initializes CUDA-graph TLS on the same thread
        every later inference uses."""
        t0 = time.perf_counter()
        logger.info("loading policy from %s", CKPT)
        self.state["phase"] = "가중치 로딩 중 (~40s)"
        train_config = _config.get_config(CONFIG_NAME)
        self.policy = _policy_config.create_trained_policy(train_config, CKPT)
        self.suite(DEFAULT_SUITE)
        self.state["load_s"] = time.perf_counter() - t0
        self.state["action_horizon"] = train_config.model.action_horizon

        # Pay the torch.compile bill now, not on the user's first click. Two passes:
        # the first triggers compilation (~18s), the second flushes the remaining
        # lazy-init stragglers (~0.3s) so the first real inference is steady-state.
        self.state["phase"] = "torch.compile 워밍업 중 (~20s)"
        t0 = time.perf_counter()
        example = libero_policy.make_libero_example()
        self.policy.infer(example)
        self.policy.infer(example)
        self.state["warmup_s"] = time.perf_counter() - t0

        self.state["phase"] = "준비 완료"
        self.state["loaded"] = True
        logger.info("policy ready: load %.1fs, warmup %.1fs", self.state["load_s"], self.state["warmup_s"])

    def _infer(self, element, **kw):
        """Funnel every inference through the single GPU thread (see _infer_pool)."""
        return self._infer_pool.submit(self.policy.infer, element, **kw).result()

    def suite(self, name: str):
        """Benchmark suites are instantiated on demand and cached; building one costs
        a directory walk over its bddl files, so we do not want it per request."""
        if name not in MAX_STEPS_BY_SUITE:
            raise ValueError(f"unknown suite {name!r}; expected one of {SUITES}")
        if name not in self._suites:
            self._suites[name] = benchmark.get_benchmark_dict()[name]()
        return self._suites[name]

    def tasks(self, suite: str = DEFAULT_SUITE):
        ts = self.suite(suite)
        return [{"id": i, "language": ts.get_task(i).language} for i in range(ts.n_tasks)]

    def _get_env(self, suite: str, task_id: int):
        key = (suite, task_id)
        if self._env is not None and self._env_key == key:
            return self._env
        if self._env is not None:
            with contextlib.suppress(Exception):
                self._env.close()
            self._obs = None  # stale: belongs to the previous scene
        task = self.suite(suite).get_task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            **{"bddl_file_name": str(bddl), "camera_heights": ENV_RESOLUTION, "camera_widths": ENV_RESOLUTION}
        )
        env.seed(7)
        self._env, self._env_key = env, key
        return env

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=64)
        with self._subs_lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._subs_lock:
            with contextlib.suppress(ValueError):
                self._subs.remove(q)

    def emit(self, event: dict):
        with self._subs_lock:
            subs = list(self._subs)
        for q in subs:
            with contextlib.suppress(queue.Full):  # slow client: drop, never block the rollout
                q.put_nowait(event)

    # -- observation plumbing ---------------------------------------------

    def _views(self, obs):
        """Model-facing views (rotated 180 deg to match training preprocessing)."""
        agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        return agent, wrist

    def _element(self, obs, prompt):
        agent, wrist = self._views(obs)
        return {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(agent, RESIZE, RESIZE)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist, RESIZE, RESIZE)
            ),
            "observation/state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
            "prompt": prompt,
        }

    def _model_input(self, element: dict) -> dict:
        """What the VLM actually consumes, for display.

        The third camera slot exists in the model but LIBERO has no right wrist
        camera, so LiberoInputs feeds it zeros with image_mask=False. Showing that
        explicitly is more informative than hiding it.
        """
        state = np.asarray(element["observation/state"], dtype=float)
        return {
            "base_rgb": jpeg_b64(element["observation/image"], quality=72),
            "wrist_rgb": jpeg_b64(element["observation/wrist_image"], quality=72),
            "state": [round(float(v), 4) for v in state],
            "state_labels": STATE_LABELS,
            "prompt": element["prompt"],
            "resolution": list(element["observation/image"].shape),
        }

    def inspect_input(self, suite: str, task_id: int, prompt: str) -> dict:
        """Full input-side dump: images, state, and the tokenized prompt."""
        ts = self.suite(suite)
        env = self._get_env(suite, task_id)   # resets self._obs if the scene changed
        task = ts.get_task(task_id)
        prompt = prompt.strip() or task.language
        if self._obs is None:
            env.reset()
            self._obs = env.set_init_state(ts.get_task_init_states(task_id)[0])
            for _ in range(NUM_WAIT_STEPS):
                self._obs = env.step(DUMMY_ACTION)[0]

        element = self._element(self._obs, prompt)
        transformed = self.policy._input_transform(dict(element))  # noqa: SLF001

        tokens = transformed.get("tokenized_prompt")
        mask = transformed.get("tokenized_prompt_mask")
        n_real = int(np.asarray(mask).sum()) if mask is not None else None
        token_ids = np.asarray(tokens).reshape(-1).tolist() if tokens is not None else []

        masks = {k: bool(np.asarray(v)) for k, v in (transformed.get("image_mask") or {}).items()}
        shapes = {k: list(np.asarray(v).shape) for k, v in (transformed.get("image") or {}).items()}

        return {
            **self._model_input(element),
            "token_ids": token_ids[: n_real or 32],
            "token_count": n_real,
            "token_budget": len(token_ids),
            "image_masks": masks,
            "image_shapes": shapes,
            "state_dim_model": int(np.asarray(transformed["state"]).shape[-1]),
        }

    # -- rollout -----------------------------------------------------------

    def start(self, cfg: RunConfig):
        with self._lock:
            if self.running:
                return False
            self.cfg = cfg
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self.running = True
            self.state["running"] = True
            self._thread.start()
            return True

    def stop(self):
        self._stop.set()

    def _run(self):
        cfg = self.cfg
        try:
            ts = self.suite(cfg.suite)
            env = self._get_env(cfg.suite, cfg.task_id)
            task = ts.get_task(cfg.task_id)
            prompt = cfg.prompt.strip() or task.language
            init_states = ts.get_task_init_states(cfg.task_id)
            max_steps = MAX_STEPS_BY_SUITE[cfg.suite]

            env.reset()
            obs = env.set_init_state(init_states[cfg.episode_idx % len(init_states)])
            self.infer_ms = Ring()
            self.step_ms = Ring()
            self.stream_ms = Ring()
            self.state.update(
                {"t": 0, "chunks": 0, "success": None, "task": prompt, "wall_s": 0.0, "robot_s": 0.0}
            )
            self.emit({"type": "status", "msg": f"rollout started: {prompt}", "state": dict(self.state)})

            plan = collections.deque()
            t = 0
            done = False
            infer_wall = 0.0
            ep_start = time.perf_counter()
            next_deadline = ep_start

            while t < max_steps + NUM_WAIT_STEPS and not self._stop.is_set():
                if t < NUM_WAIT_STEPS:
                    obs = env.step(DUMMY_ACTION)[0]
                    t += 1
                    continue

                # main.py preprocesses every control step regardless of whether it
                # replans, so building this each iteration matches upstream behaviour.
                element = self._element(obs, prompt)

                if not plan:
                    t0 = time.perf_counter()
                    chunk = self._infer(element)["actions"]
                    dt_ms = (time.perf_counter() - t0) * 1e3
                    infer_wall += dt_ms / 1e3
                    self.infer_ms.add(dt_ms)
                    self.state["chunks"] += 1
                    plan.extend(chunk[: cfg.replan_steps])
                    self.emit(
                        {
                            "type": "infer",
                            "ms": dt_ms,
                            "chunk": np.asarray(chunk).round(4).tolist(),
                            "used": cfg.replan_steps,
                        }
                    )

                action = plan.popleft()
                t0 = time.perf_counter()
                obs, _reward, done, _info = env.step(np.asarray(action).tolist())
                self.step_ms.add((time.perf_counter() - t0) * 1e3)
                t += 1

                wall = time.perf_counter() - ep_start
                self.state.update({"t": t, "wall_s": wall, "robot_s": t * CONTROL_PERIOD_S})

                # Streaming is pure observer overhead: it is not part of the control loop
                # a real robot would run, so measure it, show it separately, and cap it
                # at ~20 fps — un-paced episodes step far faster than a browser can paint.
                now = time.perf_counter()
                if done or (now - self._last_frame_emit) >= 0.05:
                    self._last_frame_emit = now
                    t0 = time.perf_counter()
                    agent, _ = self._views(obs)
                    frame = {
                        "type": "frame",
                        "agent": jpeg_b64(agent),
                        "model_input": self._model_input(element),
                        "t": t,
                        "plan_left": len(plan),
                        "replan": cfg.replan_steps,
                    }
                    self.stream_ms.add((time.perf_counter() - t0) * 1e3)
                    frame["metrics"] = self._metrics(infer_wall, wall)
                    self.emit(frame)

                if done:
                    break

                if cfg.pace_realtime:
                    next_deadline += CONTROL_PERIOD_S
                    slack = next_deadline - time.perf_counter()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        next_deadline = time.perf_counter()

            self.state["success"] = bool(done)
            self.emit(
                {
                    "type": "episode_end",
                    "success": bool(done),
                    "steps": t,
                    "wall_s": time.perf_counter() - ep_start,
                    "stopped": self._stop.is_set(),
                    "state": dict(self.state),
                }
            )
        except Exception as exc:  # noqa: BLE001 - surface any sim/policy failure to the UI
            logger.exception("rollout failed")
            self.emit({"type": "error", "msg": f"{type(exc).__name__}: {exc}"})
        finally:
            self.running = False
            self.state["running"] = False

    def _metrics(self, infer_wall: float, wall: float) -> dict:
        robot_s = self.state["robot_s"]
        return {
            "infer": self.infer_ms.stats(),
            "step": self.step_ms.stats(),
            "stream": self.stream_ms.stats(),
            "infer_series": [round(x, 2) for x in self.infer_ms.recent()],
            "rtf": (robot_s / wall) if wall > 0 else 0.0,
            "stall_pct": (100.0 * infer_wall / wall) if wall > 0 else 0.0,
            "gpu_gib": gpu_used_gib(),
            "chunks": self.state["chunks"],
            "t": self.state["t"],
        }

    # -- one-shot inference with the denoising trace -----------------------

    def trace_inference(
        self, suite: str, task_id: int, prompt: str, num_steps: int = 10, seed: int | None = None
    ) -> dict:
        """Run one inference on the current (or freshly reset) observation, recording
        every Euler step of the flow-matching integration.

        The fast path used during rollouts is `torch.compile`d as a whole graph, so we
        deliberately call the *uncompiled* class method here. Timings from this path are
        therefore not comparable to rollout latency.
        """
        if self.running:
            raise RuntimeError("stop the rollout first")

        ts = self.suite(suite)
        env = self._get_env(suite, task_id)   # resets self._obs if the scene changed
        task = ts.get_task(task_id)
        prompt = prompt.strip() or task.language
        if self._obs is None:
            env.reset()
            self._obs = env.set_init_state(ts.get_task_init_states(task_id)[0])
            for _ in range(NUM_WAIT_STEPS):
                self._obs = env.step(DUMMY_ACTION)[0]
        obs = self._obs

        model = self.policy._model  # noqa: SLF001 - intentional: we are instrumenting internals
        trace = []
        orig_denoise = model.denoise_step

        def recording_denoise(state, prefix_pad_masks, past_key_values, x_t, timestep):
            trace.append((float(timestep[0].item()), x_t.detach().float().cpu().numpy()[0].copy()))
            return orig_denoise(state, prefix_pad_masks, past_key_values, x_t, timestep)

        uncompiled = type(model).sample_actions

        def traced_sample(device, observation, **kw):
            kw.setdefault("num_steps", num_steps)
            return uncompiled(model, device, observation, **kw)

        element = self._element(obs, prompt)

        # Fixing the noise draw is what makes step-count comparisons meaningful: without
        # it, run-to-run variation from a fresh Gaussian swamps the effect of num_steps.
        noise = None
        if seed is not None:
            cfg = self.policy._model.config  # noqa: SLF001
            noise = np.random.default_rng(seed).standard_normal(
                (cfg.action_horizon, cfg.action_dim)
            ).astype(np.float32)

        saved = self.policy._sample_actions  # noqa: SLF001
        model.denoise_step = recording_denoise
        self.policy._sample_actions = traced_sample  # noqa: SLF001
        try:
            t0 = time.perf_counter()
            out = self._infer(element, noise=noise)
            total_ms = (time.perf_counter() - t0) * 1e3
        finally:
            # Remove the instance-attribute shadow instead of assigning orig_denoise
            # back: a leftover entry in the instance __dict__ invalidates dynamo's
            # guards and costs a ~4s partial recompile on the next compiled rollout.
            with contextlib.suppress(AttributeError):
                del model.denoise_step
            self.policy._sample_actions = saved  # noqa: SLF001

        final = np.asarray(out["actions"])

        # Intermediate x_t live in the model's normalized action space. Push them through
        # the same output transform as the final actions so the axes mean something.
        # Unnormalize also touches "state", so hand it the same normalized state the
        # model saw rather than a placeholder.
        norm_state = np.asarray(self.policy._input_transform(dict(element))["state"])  # noqa: SLF001
        unnorm = []
        for tau, x in trace:
            y = self.policy._output_transform({"state": norm_state, "actions": x})["actions"]  # noqa: SLF001
            unnorm.append({"tau": tau, "actions": np.asarray(y).round(4).tolist()})
        unnorm.append({"tau": 0.0, "actions": final.round(4).tolist()})

        agent, _ = self._views(obs)
        return {
            "type": "diffusion",
            "prompt": prompt,
            "num_steps": len(trace),
            "total_ms": total_ms,
            "steps": unnorm,
            "final": final.round(4).tolist(),
            "agent": jpeg_b64(agent),
            "model_input": self._model_input(element),
        }

    def reset_obs(self, suite: str, task_id: int):
        ts = self.suite(suite)
        env = self._get_env(suite, task_id)
        task = ts.get_task(task_id)
        env.reset()
        self._obs = env.set_init_state(ts.get_task_init_states(task_id)[0])
        for _ in range(NUM_WAIT_STEPS):
            self._obs = env.step(DUMMY_ACTION)[0]
        agent, _ = self._views(self._obs)
        return {
            "agent": jpeg_b64(agent),
            "model_input": self._model_input(self._element(self._obs, task.language)),
        }


engine = Engine()
app = FastAPI(title="VLA console")


@app.on_event("startup")
def _startup():
    # Load (and warm up) on the inference thread itself — see Engine.load docstring.
    engine._infer_pool.submit(engine.load)  # noqa: SLF001


@app.get("/api/state")
def api_state(suite: str = DEFAULT_SUITE):
    loaded = engine.state["loaded"]
    return {
        "state": engine.state,
        "suites": SUITES,
        "suite": suite,
        "tasks": engine.tasks(suite) if loaded else [],
        "max_steps": MAX_STEPS_BY_SUITE.get(suite),
        "gpu_gib": gpu_used_gib(),
        "config": dataclasses.asdict(engine.cfg),
    }


# NOTE: these handlers are deliberately plain `def`, not `async def`. FastAPI runs
# sync handlers in a worker threadpool; an `async def` that blocks (env.reset takes
# seconds, a diffusion trace ~1s) freezes the event loop, which stalls /api/state
# polling and the websocket for every connected browser — the page visibly hangs.


@app.post("/api/start")
def api_start(body: dict):
    if not engine.state["loaded"]:
        return {"ok": False, "error": "policy still loading"}
    suite = str(body.get("suite", DEFAULT_SUITE))
    if suite not in MAX_STEPS_BY_SUITE:
        return {"ok": False, "error": f"unknown suite {suite!r}"}
    cfg = RunConfig(
        suite=suite,
        task_id=int(body.get("task_id", 0)),
        episode_idx=int(body.get("episode_idx", 0)),
        replan_steps=max(1, min(10, int(body.get("replan_steps", 5)))),
        prompt=str(body.get("prompt", "")),
        pace_realtime=bool(body.get("pace_realtime", False)),
    )
    return {"ok": engine.start(cfg)}


@app.post("/api/stop")
def api_stop():
    engine.stop()
    return {"ok": True}


@app.post("/api/reset")
def api_reset(body: dict):
    if not engine.state["loaded"]:
        return {"ok": False, "error": "policy still loading"}
    if engine.running:
        return {"ok": False, "error": "rollout is running"}
    try:
        return {"ok": True, **engine.reset_obs(str(body.get("suite", DEFAULT_SUITE)), int(body.get("task_id", 0)))}
    except Exception as exc:  # noqa: BLE001 - report to UI rather than 500
        logger.exception("reset failed")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/inspect")
def api_inspect(body: dict):
    if not engine.state["loaded"]:
        return {"ok": False, "error": "policy still loading"}
    if engine.running:
        return {"ok": False, "error": "rollout is running — stop it first"}
    try:
        return {
            "ok": True,
            **engine.inspect_input(
                str(body.get("suite", DEFAULT_SUITE)), int(body.get("task_id", 0)), str(body.get("prompt", ""))
            ),
        }
    except Exception as exc:  # noqa: BLE001 - report to UI rather than 500
        logger.exception("inspect failed")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/diffusion")
def api_diffusion(body: dict):
    if not engine.state["loaded"]:
        return {"ok": False, "error": "policy still loading"}
    try:
        seed = body.get("seed")
        result = engine.trace_inference(
            str(body.get("suite", DEFAULT_SUITE)),
            int(body.get("task_id", 0)),
            str(body.get("prompt", "")),
            num_steps=int(body.get("num_steps", 10)),
            seed=None if seed is None else int(seed),
        )
    except Exception as exc:  # noqa: BLE001 - report to UI rather than 500
        logger.exception("diffusion trace failed")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, **result}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    q = engine.subscribe()
    try:
        while True:
            try:
                ev = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            await sock.send_json(ev)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - a dead socket should not kill the server
        logger.debug("websocket closed", exc_info=True)
    finally:
        engine.unsubscribe(q)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
