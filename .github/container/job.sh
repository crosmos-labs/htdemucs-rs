#!/usr/bin/env bash
# One separation through the container's HTTP interface, in a fresh container (so the cgroup's
# memory.peak is this job's):
#
#   job.sh IMAGE LABEL QUERY INPUT OUTDIR AUDIO_SECONDS
#
# Starts IMAGE, waits for /health, POSTs INPUT to /job?QUERY, polls /job, fetches /job/result
# into OUTDIR, and appends one JSON line to results.jsonl: the job's status (wall and CPU seconds,
# peak RSS of the separator from wait4), the container cgroup's memory.peak (RSS + page cache of
# everything in the container) and the time from `docker run` to a healthy server.
set -euo pipefail
img=$1 label=$2 query=$3 input=$4 out=$5 audio=$6
port=18080
t0=$(date +%s.%N)
cid=$(docker run -d -p 127.0.0.1:$port:8080 "$img")
cleanup() { docker logs "$cid" > "$out.log" 2>&1 || true; docker rm -f "$cid" > /dev/null || true; }
trap cleanup EXIT
for _ in $(seq 300); do curl -sf localhost:$port/health > /dev/null && break; sleep 0.05; done
ready=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }')
cg=$(find /sys/fs/cgroup -maxdepth 3 -type d -name "*$cid*" | head -1)
curl -sf -X POST "localhost:$port/job?$query" -H 'Content-Type: application/octet-stream' --data-binary @"$input" > /dev/null
while :; do
  s=$(curl -sf localhost:$port/job)
  state=$(jq -r .state <<< "$s")
  [ "$state" = running ] || [ "$state" = receiving ] || break
  sleep 2
done
peak=$(cat "$cg/memory.peak" 2> /dev/null || echo 0)
if [ "$state" != done ]; then
  echo "$label: $state: $(jq -r .error <<< "$s")" >&2
  exit 1
fi
mkdir -p "$out"
curl -sf localhost:$port/job/result -o "$out.tar"
tar -xf "$out.tar" -C "$out"
jq -c --arg label "$label" --arg img "$img" --argjson peak "$peak" --argjson ready "$ready" --argjson audio "$audio" --argjson tar "$(stat -c %s "$out.tar")" \
  '{label: $label, image: $img, audio_s: $audio, ready_s: $ready, cgroup_peak_mb: ($peak / 1048576 | round), tar_bytes: $tar} + . + {cpu_per_audio_s: (.cpu_s / $audio)}' <<< "$s" | tee -a results.jsonl
