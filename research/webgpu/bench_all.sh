#!/bin/bash
# Run bench.html cases one after another in headless Chrome (server: serve.mjs on $PORT), one JSON line each to
# $W/results/bench.jsonl. Wrap the whole script in the heavy-job lock.   bench_all.sh '<query>' ['<query>' ...]
W=${W:-/Users/divyansh/work/mac-testing/webgpu}; PORT=${PORT:-8791}; HERE=$(cd "$(dirname "$0")" && pwd)
for q in "$@"; do
  echo "== $q" >&2
  NODE_PATH=$W/node_modules node "$HERE/run.mjs" "$q" --port $PORT 2>>$W/results/bench_console.log | tee -a $W/results/bench.jsonl |
    python3 -c 'import json,sys; r=json.loads(sys.stdin.read()); print({k: r.get(k) for k in ("tag","model_mb","session_ms","run_ms","median_ms","s_per_audio_s_chunk","mask_snr_db","audio_snr_db","stft_ms","istft_ms","peak_footprint_mb","error")})'
done
