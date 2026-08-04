"""Bucket an nsys cuda_gpu_kern_sum CSV by kernel family and diff two boxes.

Answers PORTING_A100.md prediction 1: on GB10, 57% of GPU time went to
cutlass_80_* (Ampere-generation) kernels while only 18% went to the chip's own
nvjet_sm121_*. If that was a GB10 kernel-selection artifact, the A100 run should
show no sm121 kernels at all.

Run: .venv/bin/python profiling/compare_kernels.py A.csv B.csv
"""

import csv
import sys


def bucket(name):
    if "nvjet" in name:
        return "GEMM nvjet_sm121_* (Blackwell-gen)"
    if "cutlass_80" in name and "wmma" in name:
        return "GEMM cutlass_80_*wmma*"
    if "cutlass_80" in name:
        return "GEMM cutlass_80_*tensorop"
    if "ampere_" in name and "gemm" in name:
        return "GEMM ampere_*gemm (cuBLAS)"
    if "cutlass" in name or "gemm" in name.lower():
        return "GEMM other"
    if name.startswith("triton_"):
        return "Triton (Inductor fused)"
    return "other (reduction/attention/elementwise)"


def load(path):
    tot = {}
    grand = 0.0
    with open(path) as f:
        for row in csv.DictReader(f):
            ns = float(row["Total Time (ns)"])
            grand += ns
            tot[bucket(row["Name"])] = tot.get(bucket(row["Name"]), 0.0) + ns
    return tot, grand


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, a_tot = load(a_path)
    b, b_tot = load(b_path)

    print(f"A = {a_path}   total {a_tot / 1e6:.0f} ms")
    print(f"B = {b_path}   total {b_tot / 1e6:.0f} ms")
    print()
    print(f"{'bucket':<40} {'A ms':>9} {'A %':>7} {'B ms':>9} {'B %':>7}")
    for k in sorted(set(a) | set(b), key=lambda k: -(a.get(k, 0) + b.get(k, 0))):
        av, bv = a.get(k, 0.0), b.get(k, 0.0)
        print(
            f"{k:<40} {av / 1e6:>9.0f} {100 * av / a_tot:>6.1f}% "
            f"{bv / 1e6:>9.0f} {100 * bv / b_tot:>6.1f}%"
        )


if __name__ == "__main__":
    main()
