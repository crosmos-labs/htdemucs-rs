#!/bin/bash
# CPU cost of today's two splits (PyTorch Demucs 4.0.1, torch 2.2.2, exact pipeline arguments but -d cpu)
# and of Kim's Mel-RoFormer on a 60 s excerpt, all at 4 threads. Logs load before/after each run.
# Usage: timing.sh <tag> [song id, default system-of-a-down-aerials (243.6 s)]
set -uo pipefail
TAG=$1; id=${2:-system-of-a-down-aerials}
R=/Users/divyansh/htdemucs-rs/research/round1; GT=/Users/divyansh/guitar-tabs
O=$R/data/timing/$TAG; mkdir -p "$O"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
load() { echo "$1 load: $(sysctl -n vm.loadavg) | top: $(top -l 2 -s 1 -o cpu -n 4 -stats command,cpu | tail -4 | tr -s ' ' | tr '\n' ';')"; }
cd "$GT"
ffmpeg -v error -y -i ".cache/audio/$id.m4a" -c:a pcm_f32le "$O/$id.wav"
DEM=(uv run --quiet --python 3.11.17 --with-requirements scripts/requirements-demucs.txt python scripts/demucs_seeded.py 0)
load "before 4s"
/usr/bin/time -l "${DEM[@]}" -n htdemucs -d cpu --shifts 2 -o "$O/out" --mp3 --mp3-bitrate 160 ".cache/audio/$id.m4a" > "$O/4s.log" 2> "$O/4s.time"
load "after 4s"
/usr/bin/time -l "${DEM[@]}" -n htdemucs_6s -d cpu --shifts 2 --flac --clip-mode clamp -o "$O/out" "$O/$id.wav" > "$O/6s.log" 2> "$O/6s.time"
load "after 6s"
cd "$R"
/usr/bin/time -l uv run --quiet --python 3.11.17 --with-requirements kim_env.txt python -I kim_sep.py data/ext/kim data/models/MelBandRoformer.ckpt "$GT/.cache/audio/$id.m4a" "$O/kim60.wav" --device cpu --threads 4 --start 60 --dur 60 > "$O/kim.json" 2> "$O/kim.time"
load "after kim"
rm -f "$O/$id.wav"
for f in 4s 6s kim; do echo "== $f"; grep -E " real |maximum resident" "$O/$f.time"; done; tail -1 "$O/kim.json"
