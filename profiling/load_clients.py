"""Multi-client load generator for the openpi policy server.

Each simulated robot is a **separate OS process** holding its own websocket connection,
so client-side CPU work (msgpack of two 224x224x3 images per request) never contends on
one GIL and distorts the latency it is trying to measure.

Two request patterns:

  * `rate` (default) -- what a real robot does. Fixed control period; with
    `replan_steps=5` at 20 Hz a robot needs a fresh action chunk every 250 ms. A client
    that receives its response late has *missed its deadline*: the robot ran out of
    queued actions. This is the metric that decides how many robots fit on one GPU.
  * `saturate` -- send the next request the instant the previous response lands. No
    deadline, measures peak achievable throughput.

Every request carries `__meta__ = {cid, seq}`. The client logs its own send/recv
timestamps and the server logs its own; `report.py` joins them on (cid, seq). Since both
sides are on this host, CLOCK_MONOTONIC is shared and `t_recv_server - t_send` is the
real queue wait -- the time a request spent sitting in the socket while the single-
threaded server was blocked inside another client's inference. That number is invisible
from either side alone.
"""

import argparse
import json
import multiprocessing as mp
import pathlib
import statistics
import time

import numpy as np
from openpi_client import msgpack_numpy
import websockets.sync.client

PROMPTS = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the alphabet soup and place it in the basket",
    "open the middle drawer of the cabinet",
    "put the bowl on the stove",
]


def make_obs_pool(rng: np.random.Generator, n: int, prompt: str) -> list[dict]:
    """Pre-generate observations so per-request numpy work does not enter the timing."""
    return [
        {
            "observation/state": rng.random(8),
            "observation/image": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
            "prompt": prompt,
        }
        for _ in range(n)
    ]


def connect(uri: str, timeout_s: float = 120.0):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            conn = websockets.sync.client.connect(uri, compression=None, max_size=None, open_timeout=30)
            msgpack_numpy.unpackb(conn.recv())  # server metadata handshake
            return conn
        except (ConnectionRefusedError, OSError):
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)


def control(conn, cmd: str) -> dict:
    conn.send(msgpack_numpy.Packer().pack({"__ctl__": cmd}))
    return msgpack_numpy.unpackb(conn.recv())


def client_proc(cid: int, args, ready_barrier, go_barrier, out_path: str):
    """One simulated robot."""
    rng = np.random.default_rng(1000 + cid)
    pool = make_obs_pool(rng, args.pool, PROMPTS[cid % len(PROMPTS)])
    packer = msgpack_numpy.Packer()
    conn = connect(args.uri)

    records = []
    period = args.period_ms / 1e3

    # Warmup requests: excluded from the report, and they also make sure every
    # connection is established and every client's first-request cost is paid before
    # the measured window opens.
    for i in range(args.warmup):
        payload = packer.pack({**pool[i % args.pool], "__meta__": {"cid": cid, "seq": -1 - i}})
        conn.send(payload)
        conn.recv()

    ready_barrier.wait()  # warm and connected; the parent now arms the profiler
    go_barrier.wait()  # all clients start the measured window together
    t_start = time.monotonic()

    for seq in range(args.requests):
        if args.mode == "rate":
            deadline = t_start + seq * period
            now = time.monotonic()
            if now < deadline:
                time.sleep(deadline - now)
        else:
            deadline = None

        obs = pool[seq % args.pool]
        t_pack0 = time.monotonic()
        payload = packer.pack({**obs, "__meta__": {"cid": cid, "seq": seq}})
        t_send = time.monotonic()
        conn.send(payload)
        t_sent = time.monotonic()
        resp_raw = conn.recv()
        t_recv = time.monotonic()
        if isinstance(resp_raw, str):
            raise RuntimeError(f"server error:\n{resp_raw}")
        resp = msgpack_numpy.unpackb(resp_raw)
        t_unpacked = time.monotonic()

        rec = {
            "cid": cid,
            "seq": seq,
            "t_pack0": t_pack0,
            "t_send": t_send,
            "t_sent": t_sent,
            "t_recv": t_recv,
            "t_unpacked": t_unpacked,
            "pack_ms": (t_send - t_pack0) * 1e3,
            "send_ms": (t_sent - t_send) * 1e3,
            "e2e_ms": (t_recv - t_send) * 1e3,
            "total_ms": (t_unpacked - t_pack0) * 1e3,
            "unpack_ms": (t_unpacked - t_recv) * 1e3,
            "server_infer_ms": resp.get("server_timing", {}).get("infer_ms"),
            "model_infer_ms": resp.get("policy_timing", {}).get("infer_ms"),
            "req_bytes": len(payload),
            "resp_bytes": len(resp_raw),
            "actions_shape": list(np.asarray(resp["actions"]).shape),
        }
        if deadline is not None:
            # Late by how much relative to when this chunk was supposed to be in hand.
            rec["deadline"] = deadline
            rec["lateness_ms"] = (t_recv - (deadline + period)) * 1e3
            rec["missed"] = rec["lateness_ms"] > 0
        records.append(rec)

    t_end = time.monotonic()
    conn.close()

    lat = sorted(r["e2e_ms"] for r in records)
    summary = {
        "cid": cid,
        "n": len(records),
        "wall_s": t_end - t_start,
        "achieved_hz": len(records) / (t_end - t_start),
        "e2e_mean": statistics.mean(lat),
        "e2e_p50": lat[len(lat) // 2],
        "e2e_p99": lat[min(int(len(lat) * 0.99), len(lat) - 1)],
        "e2e_max": lat[-1],
    }
    if args.mode == "rate":
        summary["missed"] = sum(1 for r in records if r.get("missed"))
    pathlib.Path(out_path).write_text(json.dumps({"summary": summary, "records": records}))
    print(
        f"  client {cid}: n={summary['n']} e2e mean {summary['e2e_mean']:7.1f} ms "
        f"p99 {summary['e2e_p99']:7.1f} ms  {summary['achieved_hz']:5.2f} Hz"
        + (f"  missed {summary['missed']}/{summary['n']}" if args.mode == "rate" else ""),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:8000")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--requests", type=int, default=20, help="measured requests per client")
    ap.add_argument("--warmup", type=int, default=2, help="unmeasured requests per client")
    ap.add_argument("--mode", choices=["rate", "saturate"], default="rate")
    ap.add_argument(
        "--period-ms",
        type=float,
        default=250.0,
        help="control period per robot in rate mode (replan_steps=5 at 20 Hz -> 250 ms)",
    )
    ap.add_argument("--pool", type=int, default=8, help="pre-generated observations per client")
    ap.add_argument("--out", default="data/profiling")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--no-profiler-control", action="store_true", help="do not toggle cudaProfilerStart/Stop")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.clients} clients, {args.requests} req each, mode={args.mode}", flush=True)
    if args.mode == "rate":
        print(f"[load] control period {args.period_ms:.0f} ms/robot -> offered {args.clients / (args.period_ms / 1e3):.1f} req/s", flush=True)

    ctl = connect(args.uri)
    print("[load] ctl reset_peak:", control(ctl, "reset_peak"), flush=True)

    ctx = mp.get_context("spawn")
    ready_barrier, go_barrier = ctx.Barrier(args.clients + 1), ctx.Barrier(args.clients + 1)
    procs = []
    for cid in range(args.clients):
        p = ctx.Process(
            target=client_proc,
            args=(cid, args, ready_barrier, go_barrier, str(out / f"{args.tag}.client{cid}.json")),
        )
        p.start()
        procs.append(p)

    # Arm the profiler while the server is idle (all clients warm, none sending), so the
    # capture window is not delayed behind an in-flight inference.
    ready_barrier.wait()
    if not args.no_profiler_control:
        print("[load] ctl profile_start:", control(ctl, "profile_start"), flush=True)
    t0 = time.monotonic()
    go_barrier.wait()

    for p in procs:
        p.join()
    wall = time.monotonic() - t0

    if not args.no_profiler_control:
        print("[load] ctl profile_stop:", control(ctl, "profile_stop"), flush=True)
    ctl.close()

    failed = [p.exitcode for p in procs if p.exitcode != 0]
    if failed:
        raise SystemExit(f"client processes failed with exit codes {failed}")

    summaries = [json.loads((out / f"{args.tag}.client{cid}.json").read_text())["summary"] for cid in range(args.clients)]
    total = sum(s["n"] for s in summaries)
    agg = {
        "tag": args.tag,
        "clients": args.clients,
        "mode": args.mode,
        "period_ms": args.period_ms,
        "requests_per_client": args.requests,
        "wall_s": wall,
        "total_requests": total,
        "throughput_rps": total / wall,
        "e2e_mean_ms": statistics.mean(s["e2e_mean"] for s in summaries),
        "e2e_p99_ms": max(s["e2e_p99"] for s in summaries),
        "per_client": summaries,
    }
    if args.mode == "rate":
        agg["missed_total"] = sum(s.get("missed", 0) for s in summaries)
        agg["missed_pct"] = 100.0 * agg["missed_total"] / total
    (out / f"{args.tag}.load.json").write_text(json.dumps(agg, indent=2))

    print(
        f"[load] {total} requests in {wall:.1f} s -> {agg['throughput_rps']:.2f} req/s, "
        f"e2e mean {agg['e2e_mean_ms']:.1f} ms, p99 {agg['e2e_p99_ms']:.1f} ms"
        + (f", deadline misses {agg['missed_total']}/{total} ({agg['missed_pct']:.0f}%)" if args.mode == "rate" else ""),
        flush=True,
    )


if __name__ == "__main__":
    main()
