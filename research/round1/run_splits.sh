#!/bin/bash
# Re-split songs with Fretboard's exact Demucs invocations (server/audio.ts lines ~590 and ~694),
# with another seed, into research/round1/data/stems/seed<S>/<model>/<id>/.
# Usage: run_splits.sh <device: mps|cpu> <seed> <id>...
set -euo pipefail
DEV=$1; SEED=$2; shift 2
GT=/Users/divyansh/guitar-tabs
R=/Users/divyansh/htdemucs-rs/research/round1
OUT=$R/data/stems/${TAG:-seed$SEED}
TMP=$R/data/tmp; mkdir -p "$OUT" "$TMP"
DEMUCS=(uv run --quiet --python 3.11.17 --with-requirements scripts/requirements-demucs.txt python scripts/demucs_seeded.py "$SEED")
cd "$GT"
for id in "$@"; do
  if [ ! -f "$OUT/htdemucs/$id/vocals.mp3" ]; then
    t0=$(date +%s)
    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 "$R/fretlock.py" "${DEMUCS[@]}" -n htdemucs -d "$DEV" --shifts 2 -o "$OUT" --mp3 --mp3-bitrate 160 ".cache/audio/$id.m4a" > "$TMP/$id.4s.log" 2>&1
    echo "$id htdemucs seed$SEED $(( $(date +%s) - t0 ))s"
  fi
  if [ ! -f "$OUT/htdemucs_6s/$id/vocals.flac" ]; then
    ffmpeg -v error -y -i ".cache/audio/$id.m4a" -c:a pcm_f32le "$TMP/$id.wav"
    t0=$(date +%s)
    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 "$R/fretlock.py" "${DEMUCS[@]}" -n htdemucs_6s -d "$DEV" --shifts 2 --flac --clip-mode clamp -o "$OUT" "$TMP/$id.wav" > "$TMP/$id.6s.log" 2>&1
    rm -f "$TMP/$id.wav"
    echo "$id htdemucs_6s seed$SEED $(( $(date +%s) - t0 ))s"
  fi
done
