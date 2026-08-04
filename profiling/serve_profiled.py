"""Policy server with NVTX ranges, per-request server-side metrics, and nsys capture control.

This is a drop-in replacement for `scripts/serve_policy.py` for benchmarking. It differs
in four ways, all of them to make a multi-client run measurable:

  1. **Warmup before serving.** `torch.compile(mode="max-autotune")` costs ~10 s on the
     first call (measured here; it is ~18 s cold, before the Inductor cache is warm).
     The stock server pays that inside the first client's request, which poisons every
     latency number. Here it happens before the socket opens.
  2. **NVTX ranges** around unpack / input transform / sample_actions / output transform /
     pack+send, tagged with the client id. In the nsys timeline this is what shows
     whether concurrent clients overlap on the GPU or queue behind each other.
  3. **Per-request JSONL** with monotonic timestamps for every stage. Clients log their
     own send/recv timestamps; `report.py` joins the two on (client_id, seq). Both sides
     run on this host, so CLOCK_MONOTONIC is a shared clock and the join yields the true
     queue wait -- which is *not* observable from inside the server alone, because
     `await websocket.recv()` only returns once the blocking inference has released
     the event loop.
  4. **cudaProfilerApi control.** The load generator sends `__ctl__` messages that call
     `torch.cuda.profiler.start()/stop()`, so nsys (run with
     `--capture-range=cudaProfilerApi`) records only the measured window, not model
     loading or autotuning.

The serving path itself is untouched: same `Policy`, same `WebsocketPolicyServer`
protocol, same one-request-at-a-time asyncio handler. What is measured is the stock
behaviour.
"""

import argparse
import asyncio
import contextlib
import dataclasses
import http
import json
import logging
import pathlib
import threading
import time
import traceback

from openpi_client import msgpack_numpy
import torch
import websockets.asyncio.server as _server
import websockets.frames

from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger("serve_profiled")

DEFAULT_CKPT = pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"


# --------------------------------------------------------------------------------------
# NVTX
# --------------------------------------------------------------------------------------


class _Nvtx:
    """NVTX range helper that compiles away to nothing when disabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    @contextlib.contextmanager
    def range(self, name: str):
        if not self.enabled:
            yield
            return
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def mark(self, name: str) -> None:
        if self.enabled:
            torch.cuda.nvtx.mark(name)


def instrument_policy(policy, nvtx: _Nvtx) -> None:
    """Wrap the policy's internal stages with NVTX ranges.

    Done by wrapping attributes rather than editing `Policy.infer`, so the upstream
    serving path stays byte-identical. The wrappers sit *outside* the torch.compile'd
    region, so they cannot trigger a recompile.
    """
    if not nvtx.enabled:
        return

    in_tf, sample, out_tf = policy._input_transform, policy._sample_actions, policy._output_transform  # noqa: SLF001

    def wrapped_in(x):
        with nvtx.range("input_transform"):
            return in_tf(x)

    def wrapped_sample(*a, **kw):
        with nvtx.range("sample_actions"):
            r = sample(*a, **kw)
            # sample_actions is async w.r.t. the GPU; close the range on the real end of
            # compute, otherwise the NVTX range is meaningless on the timeline.
            torch.cuda.synchronize()
            return r

    def wrapped_out(x):
        with nvtx.range("output_transform"):
            return out_tf(x)

    policy._input_transform = wrapped_in  # noqa: SLF001
    policy._sample_actions = wrapped_sample  # noqa: SLF001
    policy._output_transform = wrapped_out  # noqa: SLF001


# --------------------------------------------------------------------------------------
# resource sampler
# --------------------------------------------------------------------------------------


class ResourceSampler(threading.Thread):
    """Background sampler for GPU memory and utilization.

    nsys cannot collect GPU metric counters on this box without root
    (ERR_NVGPUCTRPERM), so SM utilization comes from NVML instead. Coarse (it is a
    sampled percentage, not a counter) but enough to distinguish "GPU idle waiting on
    the client" from "GPU saturated".
    """

    def __init__(self, path: pathlib.Path, period_s: float = 0.1):
        super().__init__(daemon=True)
        self.path, self.period_s = path, period_s
        self._stop = threading.Event()
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:  # noqa: BLE001
            logger.warning("NVML unavailable, sampling memory only: %s", e)
            self._pynvml = None

    def run(self):
        with self.path.open("w") as f:
            while not self._stop.wait(self.period_s):
                free, total = torch.cuda.mem_get_info()
                rec = {
                    "t": time.monotonic(),
                    "gpu_used_gib": (total - free) / 2**30,
                    "torch_alloc_gib": torch.cuda.memory_allocated() / 2**30,
                    "torch_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                }
                if self._handle is not None:
                    try:
                        u = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                        rec["sm_util_pct"] = u.gpu
                        rec["mem_util_pct"] = u.memory
                    except Exception:  # noqa: BLE001, S110
                        pass
                    try:
                        rec["power_w"] = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                    except Exception:  # noqa: BLE001, S110
                        pass
                f.write(json.dumps(rec) + "\n")
                f.flush()

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class ServerState:
    profiling: bool = False
    requests: int = 0
    inflight: int = 0
    peak_inflight: int = 0


class ProfiledServer:
    """Same protocol as WebsocketPolicyServer, plus metrics, NVTX and profiler control."""

    def __init__(self, policy, host: str, port: int, metrics_path: pathlib.Path, nvtx: _Nvtx, ready_file: str = ""):
        self._policy = policy
        self._host, self._port = host, port
        self._nvtx = nvtx
        self._ready_file = ready_file
        self._metrics_path = metrics_path
        self._metrics_file = metrics_path.open("w")
        self._state = ServerState()
        self._shutdown = asyncio.Event()
        logging.getLogger("websockets.server").setLevel(logging.WARNING)

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ):
            logger.info("serving on %s:%d", self._host, self._port)
            if self._ready_file:
                pathlib.Path(self._ready_file).write_text(f"{time.monotonic()}\n")
            await self._shutdown.wait()
        self._metrics_file.close()

    def _control(self, cmd: str) -> dict:
        st = self._state
        if cmd == "profile_start":
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            self._nvtx.mark("PROFILE_START")
            st.profiling = True
            logger.info("cudaProfilerStart (requests so far: %d)", st.requests)
        elif cmd == "profile_stop":
            torch.cuda.synchronize()
            self._nvtx.mark("PROFILE_STOP")
            torch.cuda.profiler.stop()
            st.profiling = False
            logger.info("cudaProfilerStop (requests so far: %d)", st.requests)
        elif cmd == "reset_peak":
            torch.cuda.reset_peak_memory_stats()
            st.peak_inflight = 0
        elif cmd == "shutdown":
            self._shutdown.set()
        elif cmd != "ping":
            return {"ok": False, "error": f"unknown control command: {cmd}"}
        free, total = torch.cuda.mem_get_info()
        return {
            "ok": True,
            "cmd": cmd,
            "profiling": st.profiling,
            "requests": st.requests,
            "peak_inflight": st.peak_inflight,
            "gpu_used_gib": (total - free) / 2**30,
            "torch_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "t_server": time.monotonic(),
        }

    async def _handler(self, websocket: _server.ServerConnection):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({"server": "profiled", "t_server": time.monotonic()}))
        st = self._state

        while True:
            try:
                raw = await websocket.recv()
                t_recv = time.monotonic()

                with self._nvtx.range("unpack"):
                    obs = msgpack_numpy.unpackb(raw)
                t_unpacked = time.monotonic()

                if isinstance(obs, dict) and "__ctl__" in obs:
                    await websocket.send(packer.pack(self._control(obs["__ctl__"])))
                    continue

                meta = obs.pop("__meta__", {}) or {}
                cid, seq = meta.get("cid", -1), meta.get("seq", -1)

                # inflight > 1 would mean the server is genuinely concurrent. It cannot
                # be here (infer is a blocking call inside the coroutine); recording it
                # makes that claim falsifiable rather than assumed.
                st.inflight += 1
                st.peak_inflight = max(st.peak_inflight, st.inflight)

                # Name the range per client, not per request: `nsys stats --report
                # nvtx_sum` aggregates by name, so a per-request name yields one row per
                # request instead of a usable per-client total. seq stays in the JSONL.
                with self._nvtx.range(f"infer c{cid}"):
                    t_infer0 = time.monotonic()
                    action = self._policy.infer(obs)
                    t_infer1 = time.monotonic()
                st.inflight -= 1

                model_ms = float(action.get("policy_timing", {}).get("infer_ms", float("nan")))
                action["server_timing"] = {"infer_ms": (t_infer1 - t_infer0) * 1000}

                with self._nvtx.range("pack"):
                    payload = packer.pack(action)
                t_packed = time.monotonic()
                await websocket.send(payload)
                t_sent = time.monotonic()

                st.requests += 1
                self._metrics_file.write(
                    json.dumps(
                        {
                            "cid": cid,
                            "seq": seq,
                            "t_recv": t_recv,
                            "t_unpacked": t_unpacked,
                            "t_infer0": t_infer0,
                            "t_infer1": t_infer1,
                            "t_packed": t_packed,
                            "t_sent": t_sent,
                            "unpack_ms": (t_unpacked - t_recv) * 1e3,
                            "infer_ms": (t_infer1 - t_infer0) * 1e3,
                            "model_ms": model_ms,
                            "pack_ms": (t_packed - t_infer1) * 1e3,
                            "send_ms": (t_sent - t_packed) * 1e3,
                            "req_bytes": len(raw),
                            "resp_bytes": len(payload),
                            "profiling": st.profiling,
                        }
                    )
                    + "\n"
                )

            except websockets.ConnectionClosed:
                break
            except Exception:  # noqa: BLE001
                logger.exception("handler error")
                with contextlib.suppress(Exception):
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error.",
                    )
                break
        self._metrics_file.flush()


def _health_check(connection: _server.ServerConnection, request: _server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--dir", default=str(DEFAULT_CKPT))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="data/profiling", help="directory for metrics files")
    ap.add_argument("--tag", default="run", help="prefix for this run's metrics files")
    ap.add_argument("--warmup", type=int, default=5, help="pre-serving inferences (absorbs torch.compile)")
    ap.add_argument("--no-nvtx", action="store_true")
    ap.add_argument("--ready-file", default="", help="touched once warm and listening")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nvtx = _Nvtx(not args.no_nvtx)

    free_before, total = torch.cuda.mem_get_info()
    logger.info("GPU free at start: %.1f / %.1f GiB", free_before / 2**30, total / 2**30)

    t0 = time.perf_counter()
    policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.dir)
    load_s = time.perf_counter() - t0
    free_after, _ = torch.cuda.mem_get_info()
    logger.info("model load: %.1f s, weights %.2f GiB", load_s, (free_before - free_after) / 2**30)

    # Warm up before instrumenting, so compile artifacts are not attributed to NVTX ranges.
    example = libero_policy.make_libero_example()
    warm = []
    for i in range(args.warmup):
        t = time.perf_counter()
        policy.infer(example)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) * 1e3
        warm.append(dt)
        logger.info("warmup %d/%d: %.1f ms", i + 1, args.warmup, dt)

    instrument_policy(policy, nvtx)
    torch.cuda.reset_peak_memory_stats()

    sampler = ResourceSampler(out / f"{args.tag}.resources.jsonl")
    sampler.start()

    (out / f"{args.tag}.meta.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "checkpoint": args.dir,
                "load_s": load_s,
                "weights_gib": (free_before - free_after) / 2**30,
                "warmup_ms": warm,
                "nvtx": nvtx.enabled,
                "gpu_total_gib": total / 2**30,
                "torch": torch.__version__,
            },
            indent=2,
        )
    )

    server = ProfiledServer(
        policy, args.host, args.port, out / f"{args.tag}.server.jsonl", nvtx, ready_file=args.ready_file
    )
    try:
        asyncio.run(server.run())
    finally:
        sampler.stop()
        logger.info("shutdown complete")


if __name__ == "__main__":
    main()
