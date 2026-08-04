"""Join client-side, server-side, NVML and nsys data into one report.

The interesting numbers only exist after the join. A request's end-to-end latency splits
into:

    e2e = queue_wait + unpack + input_transform + sample_actions + output_transform + pack + send

where `queue_wait = t_recv_server - t_send` is measurable only by pairing the client's
send timestamp with the server's receive timestamp. Under N concurrent clients on the
stock server this term is the whole story: inference itself is ~constant, and everything
above ~150 ms is time spent waiting for the single request-at-a-time handler.

Usage:
    .venv/bin/python profiling/report.py data/profiling --tag c4
    .venv/bin/python profiling/report.py data/profiling            # sweep over all tags
"""

import argparse
import csv
import json
import pathlib
import statistics


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(int(len(xs) * p / 100), len(xs) - 1)]


def stats(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "p50": pct(xs, 50),
        "p90": pct(xs, 90),
        "p99": pct(xs, 99),
        "min": min(xs),
        "max": max(xs),
    }


def fmt_row(name, s, width=26):
    if not s.get("n"):
        return f"{name:<{width}}      (no data)"
    return (
        f"{name:<{width}} {s['mean']:>9.1f} {s['p50']:>9.1f} {s['p90']:>9.1f} "
        f"{s['p99']:>9.1f} {s['min']:>9.1f} {s['max']:>9.1f}"
    )


def load_run(d: pathlib.Path, tag: str):
    load = json.loads((d / f"{tag}.load.json").read_text())
    clients = []
    for cid in range(load["clients"]):
        p = d / f"{tag}.client{cid}.json"
        if p.exists():
            clients.extend(json.loads(p.read_text())["records"])

    server = {}
    sp = d / f"{tag}.server.jsonl"
    if sp.exists():
        for line in sp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["seq"] >= 0:
                server[(r["cid"], r["seq"])] = r

    resources = []
    rp = d / f"{tag}.resources.jsonl"
    if rp.exists():
        resources = [json.loads(x) for x in rp.read_text().splitlines() if x.strip()]

    meta = {}
    mp_ = d / f"{tag}.meta.json"
    if mp_.exists():
        meta = json.loads(mp_.read_text())

    return load, clients, server, resources, meta


def analyze(load, clients, server):
    """Per-stage breakdown for the measured window."""
    joined = []
    for c in clients:
        s = server.get((c["cid"], c["seq"]))
        if s is None:
            continue
        joined.append(
            {
                **c,
                "queue_ms": (s["t_recv"] - c["t_send"]) * 1e3,
                "srv_unpack_ms": s["unpack_ms"],
                "srv_infer_ms": s["infer_ms"],
                "srv_model_ms": s["model_ms"],
                "srv_pack_ms": s["pack_ms"],
                "srv_send_ms": s["send_ms"],
                "return_ms": (c["t_recv"] - s["t_sent"]) * 1e3,
            }
        )

    out = {
        "joined": len(joined),
        "e2e": stats([r["e2e_ms"] for r in joined]),
        "queue_wait": stats([r["queue_ms"] for r in joined]),
        "server_unpack": stats([r["srv_unpack_ms"] for r in joined]),
        "policy_infer": stats([r["srv_infer_ms"] for r in joined]),
        "model_sample": stats([r["srv_model_ms"] for r in joined if r["srv_model_ms"] == r["srv_model_ms"]]),
        "server_pack": stats([r["srv_pack_ms"] for r in joined]),
        "server_send": stats([r["srv_send_ms"] for r in joined]),
        "return_to_client": stats([r["return_ms"] for r in joined]),
        "client_pack": stats([r["pack_ms"] for r in clients]),
        "client_unpack": stats([r["unpack_ms"] for r in clients]),
    }
    if joined:
        out["transform_overhead"] = stats(
            [r["srv_infer_ms"] - r["srv_model_ms"] for r in joined if r["srv_model_ms"] == r["srv_model_ms"]]
        )
        # Server busy fraction: how much of the measured window the single handler spent
        # inside inference. > ~95% means the GPU pipeline, not the network, is the wall.
        window = max(r["t_recv"] for r in joined) - min(r["t_send"] for r in joined)
        busy = sum(r["srv_infer_ms"] for r in joined) / 1e3
        out["window_s"] = window
        out["server_busy_pct"] = 100 * busy / window if window > 0 else float("nan")
    return out, joined


def nsys_summary(d: pathlib.Path, tag: str):
    """Read whatever `nsys stats` produced for this run, if anything."""
    res = {}
    kern = d / f"{tag}_cuda_gpu_kern_sum.csv"
    if kern.exists():
        rows = list(csv.DictReader(kern.open()))
        for r in rows:
            r["_time"] = float(r.get("Total Time (ns)") or r.get("Total Time") or 0)
            r["_inst"] = int(float(r.get("Instances") or 0))
        rows.sort(key=lambda r: -r["_time"])
        total = sum(r["_time"] for r in rows)
        res["kernels"] = {
            "total_gpu_ms": total / 1e6,
            "distinct": len(rows),
            "launches": sum(r["_inst"] for r in rows),
            "top": [
                {
                    "name": (r.get("Name") or "")[:70],
                    "total_ms": r["_time"] / 1e6,
                    "pct": 100 * r["_time"] / total if total else 0,
                    "instances": r["_inst"],
                }
                for r in rows[:10]
            ],
        }
    nvtx = d / f"{tag}_nvtx_sum.csv"
    if nvtx.exists():
        rows = list(csv.DictReader(nvtx.open()))
        res["nvtx"] = [
            {
                "range": (r.get("Range") or r.get("Name") or "")[:40],
                "instances": r.get("Instances"),
                "total_ms": float(r.get("Total Time (ns)") or 0) / 1e6,
                "avg_ms": float(r.get("Avg (ns)") or 0) / 1e6,
            }
            for r in rows[:15]
        ]
    api = d / f"{tag}_cuda_api_sum.csv"
    if api.exists():
        rows = list(csv.DictReader(api.open()))
        for r in rows:
            r["_time"] = float(r.get("Total Time (ns)") or 0)
        rows.sort(key=lambda r: -r["_time"])
        res["cuda_api_top"] = [
            {"name": r.get("Name"), "total_ms": r["_time"] / 1e6, "calls": r.get("Num Calls")} for r in rows[:8]
        ]
    return res


def print_run(d: pathlib.Path, tag: str):
    load, clients, server, resources, meta = load_run(d, tag)
    a, joined = analyze(load, clients, server)

    print(f"\n{'=' * 92}")
    print(f"  {tag}   {load['clients']} client(s), mode={load['mode']}", end="")
    if load["mode"] == "rate":
        print(f", period {load['period_ms']:.0f} ms", end="")
    print(f", {load['total_requests']} requests in {load['wall_s']:.1f} s")
    print("=" * 92)

    print(f"\nthroughput            : {load['throughput_rps']:.2f} req/s")
    if load["mode"] == "rate":
        offered = load["clients"] / (load["period_ms"] / 1e3)
        print(f"offered load          : {offered:.2f} req/s")
        print(f"deadline misses       : {load['missed_total']}/{load['total_requests']} ({load['missed_pct']:.0f}%)")
    if "server_busy_pct" in a:
        print(f"server busy (infer)   : {a['server_busy_pct']:.1f}% of the {a['window_s']:.1f} s window")

    print(f"\n{'stage (ms)':<26} {'mean':>9} {'p50':>9} {'p90':>9} {'p99':>9} {'min':>9} {'max':>9}")
    print("-" * 92)
    print(fmt_row("end-to-end (client)", a["e2e"]))
    print(fmt_row("  queue wait", a["queue_wait"]))
    print(fmt_row("  server unpack", a["server_unpack"]))
    print(fmt_row("  policy.infer", a["policy_infer"]))
    print(fmt_row("    sample_actions", a["model_sample"]))
    print(fmt_row("    transforms+h2d+d2h", a.get("transform_overhead", {})))
    print(fmt_row("  server pack", a["server_pack"]))
    print(fmt_row("  server send", a["server_send"]))
    print(fmt_row("  return to client", a["return_to_client"]))
    print(fmt_row("client pack (untimed)", a["client_pack"]))
    print(fmt_row("client unpack", a["client_unpack"]))

    print("\nper-client:")
    print(f"  {'cid':>4} {'n':>5} {'mean ms':>9} {'p99 ms':>9} {'max ms':>9} {'Hz':>7} {'missed':>7}")
    for s in load["per_client"]:
        print(
            f"  {s['cid']:>4} {s['n']:>5} {s['e2e_mean']:>9.1f} {s['e2e_p99']:>9.1f} "
            f"{s['e2e_max']:>9.1f} {s['achieved_hz']:>7.2f} {s.get('missed', '-'):>7}"
        )

    if resources:
        window = [r for r in resources if joined and min(c["t_send"] for c in joined) <= r["t"] <= max(c["t_recv"] for c in joined)]
        window = window or resources
        sm = [r["sm_util_pct"] for r in window if "sm_util_pct" in r]
        pw = [r["power_w"] for r in window if "power_w" in r]
        mem = [r["gpu_used_gib"] for r in window]
        print("\nGPU during window (NVML sampled at 10 Hz):")
        if sm:
            print(f"  SM utilization      : {statistics.mean(sm):.0f}% mean, {max(sm)}% peak")
        if pw:
            print(f"  power               : {statistics.mean(pw):.0f} W mean, {max(pw):.0f} W peak")
        print(f"  memory in use       : {statistics.mean(mem):.1f} GiB mean, {max(mem):.1f} GiB peak")

    if meta.get("warmup_ms"):
        w = meta["warmup_ms"]
        print(f"\nmodel load {meta.get('load_s', 0):.1f} s, weights {meta.get('weights_gib', 0):.2f} GiB")
        print(f"first inference (torch.compile) {w[0]:.0f} ms, warm {statistics.mean(w[1:]) if len(w) > 1 else w[0]:.1f} ms")

    ns = nsys_summary(d, tag)
    if ns.get("nvtx"):
        print(f"\nnsys NVTX ranges (GPU-side, projected):")
        print(f"  {'range':<40} {'inst':>6} {'total ms':>11} {'avg ms':>9}")
        for r in ns["nvtx"]:
            print(f"  {r['range']:<40} {str(r['instances']):>6} {r['total_ms']:>11.1f} {r['avg_ms']:>9.2f}")
    if ns.get("kernels"):
        k = ns["kernels"]
        print(f"\nnsys CUDA kernels: {k['launches']} launches, {k['distinct']} distinct, {k['total_gpu_ms']:.0f} ms GPU time")
        print(f"  {'kernel':<70} {'ms':>9} {'%':>6} {'calls':>7}")
        for r in k["top"]:
            print(f"  {r['name']:<70} {r['total_ms']:>9.1f} {r['pct']:>6.1f} {r['instances']:>7}")
    if ns.get("cuda_api_top"):
        print(f"\nnsys CUDA API (host time):")
        for r in ns["cuda_api_top"]:
            print(f"  {str(r['name']):<40} {r['total_ms']:>10.1f} ms  {r['calls']:>8} calls")

    return {"tag": tag, "load": load, "analysis": a, "nsys": ns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="data/profiling")
    ap.add_argument("--tag", default=None, help="single run; omit to report every run in the directory")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    tags = [args.tag] if args.tag else sorted(p.name[: -len(".load.json")] for p in d.glob("*.load.json"))
    if not tags:
        raise SystemExit(f"no runs found in {d}")

    results = [print_run(d, t) for t in tags]

    if len(results) > 1:
        print(f"\n{'=' * 92}\n  SWEEP\n{'=' * 92}")
        hdr = f"{'run':<12} {'clients':>8} {'thru r/s':>9} {'e2e mean':>9} {'e2e p99':>9} {'queue p50':>10} {'infer p50':>10} {'busy %':>7} {'miss %':>7}"
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            a, l = r["analysis"], r["load"]
            print(
                f"{r['tag']:<12} {l['clients']:>8} {l['throughput_rps']:>9.2f} {a['e2e']['mean']:>9.1f} "
                f"{a['e2e']['p99']:>9.1f} {a['queue_wait']['p50']:>10.1f} {a['policy_infer']['p50']:>10.1f} "
                f"{a.get('server_busy_pct', float('nan')):>7.1f} "
                f"{l.get('missed_pct', float('nan')):>7.1f}"
            )

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
