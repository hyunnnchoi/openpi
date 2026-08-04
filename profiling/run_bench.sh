#!/usr/bin/env bash
# Multi-client benchmark + Nsight Systems capture for the openpi policy server.
#
#   profiling/run_bench.sh [-c "1 2 4"] [-n 20] [-m rate|saturate] [-p 250] [--no-nsys]
#
# One nsys capture per client-count, because a single .nsys-rep spanning all of them
# would be huge and hard to read. Each server process is started fresh, warmed (so
# torch.compile's ~18 s first call is outside the measurement), profiled only between
# cudaProfilerStart/Stop, and shut down cleanly so nsys can finalize its report.
#
# `--cuda-graph-trace node` is the default here and it matters: pi0.5 is compiled with
# mode="max-autotune", so inference runs as ~12 CUDA graph launches per request. Under
# nsys's default (`graph`), each launch is one opaque blob and the kernel report shows
# only the handful of eager kernels outside the graphs -- 5 ms of "GPU time" for a 36 s
# run. `node` unrolls the graphs into their constituent kernels. It costs some overhead,
# so use `--graph-trace graph` when only the concurrency timeline matters.
#
# Note on this box: nsys cannot collect CPU samples (kernel.perf_event_paranoid=4) or
# GPU metric counters (ERR_NVGPUCTRPERM) without root. CUDA + NVTX tracing works, and
# SM utilization is sampled from NVML by the server instead. To unlock the rest:
#   sudo sysctl -w kernel.perf_event_paranoid=2
#   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nsight.conf && sudo update-initramfs -u && reboot

set -euo pipefail
cd "$(dirname "$0")/.."

CLIENTS="1 2 4"
REQUESTS=20
MODE=rate
PERIOD=250
PORT=8000
OUTDIR=data/profiling
USE_NSYS=1
GRAPH_TRACE=node
# cudnn tracing is deliberately absent. On the A100 box (nsys 2025.3.1, torch 2.7.1,
# triton 3.3.1) adding cudnn makes Triton's own driver handle come up uninitialized,
# and every inductor compile dies with "Triton Error [CUDA]: initialization error"
# before the server ever listens. cuda,nvtx,cublas reproduces on its own machine-wide,
# and the kernel breakdown comes from the cuda trace regardless.
TRACE=cuda,nvtx,cublas
CKPT="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--clients)  CLIENTS="$2"; shift 2 ;;
    -n|--requests) REQUESTS="$2"; shift 2 ;;
    -m|--mode)     MODE="$2"; shift 2 ;;
    -p|--period)   PERIOD="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    --out)         OUTDIR="$2"; shift 2 ;;
    --ckpt)        CKPT="$2"; shift 2 ;;
    --no-nsys)     USE_NSYS=0; shift ;;
    --graph-trace) GRAPH_TRACE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUTDIR"
PY=.venv/bin/python

for N in $CLIENTS; do
  TAG="c${N}"
  READY="$OUTDIR/$TAG.ready"
  rm -f "$READY" "$OUTDIR/$TAG".*.jsonl "$OUTDIR/$TAG".*.json

  echo "############################################################"
  echo "# $TAG : $N client(s), $REQUESTS requests each, mode=$MODE"
  echo "############################################################"

  SERVER_CMD=("$PY" profiling/serve_profiled.py
      --config pi05_libero --dir "$CKPT"
      --host 127.0.0.1 --port "$PORT"
      --out "$OUTDIR" --tag "$TAG" --ready-file "$READY")

  if [[ "$USE_NSYS" == "1" ]]; then
    nsys profile \
      --output "$OUTDIR/$TAG" --force-overwrite true \
      --trace "$TRACE" \
      --cuda-graph-trace "$GRAPH_TRACE" \
      --capture-range cudaProfilerApi --capture-range-end stop \
      --cuda-memory-usage true \
      --sample none --cpuctxsw none \
      "${SERVER_CMD[@]}" > "$OUTDIR/$TAG.server.log" 2>&1 &
  else
    "${SERVER_CMD[@]}" > "$OUTDIR/$TAG.server.log" 2>&1 &
  fi
  SERVER_PID=$!

  # Wait for warmup + listen (model load ~40 s, compile ~18 s, plus nsys overhead).
  for _ in $(seq 300); do
    [[ -f "$READY" ]] && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server died during startup; tail of $OUTDIR/$TAG.server.log:" >&2
      tail -30 "$OUTDIR/$TAG.server.log" >&2
      exit 1
    fi
    sleep 2
  done
  [[ -f "$READY" ]] || { echo "server never became ready" >&2; kill "$SERVER_PID"; exit 1; }
  echo "[bench] server ready"

  "$PY" profiling/load_clients.py \
      --uri "ws://127.0.0.1:$PORT" --clients "$N" --requests "$REQUESTS" \
      --mode "$MODE" --period-ms "$PERIOD" --out "$OUTDIR" --tag "$TAG"

  # Clean shutdown -> nsys finalizes the .nsys-rep.
  "$PY" - "$PORT" <<'EOF'
import sys
from openpi_client import msgpack_numpy
import websockets.sync.client
c = websockets.sync.client.connect(f"ws://127.0.0.1:{sys.argv[1]}", compression=None, max_size=None)
msgpack_numpy.unpackb(c.recv())
c.send(msgpack_numpy.Packer().pack({"__ctl__": "shutdown"}))
try:
    print("[bench] shutdown ack:", msgpack_numpy.unpackb(c.recv()))
except Exception:
    pass
c.close()
EOF
  wait "$SERVER_PID" || true
  echo "[bench] server exited"

  if [[ "$USE_NSYS" == "1" && -f "$OUTDIR/$TAG.nsys-rep" ]]; then
    echo "[bench] nsys stats..."
    nsys stats --force-export true --format csv --force-overwrite true \
      --report cuda_gpu_kern_sum --report cuda_api_sum --report nvtx_sum \
      --output "$OUTDIR/$TAG" "$OUTDIR/$TAG.nsys-rep" > "$OUTDIR/$TAG.stats.log" 2>&1 \
      || echo "[bench] nsys stats failed, see $OUTDIR/$TAG.stats.log" >&2
  fi
  sleep 3   # let GPU memory settle before the next server starts
done

echo
"$PY" profiling/report.py "$OUTDIR" --json-out "$OUTDIR/report.json"
