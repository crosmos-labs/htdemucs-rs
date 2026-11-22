#!/bin/sh
# Full-song references for the CLI parity test: Fretboard's four test songs, decoded once with
# ffmpeg to float WAV (the same file is the Rust input), separated by PyTorch Demucs on the CPU
# with shifts 2 (seed 0) and shifts 0, both models. Into ref/songs/ (gitignored, ~7 GB).
set -e
cd "$(dirname "$0")/.."
FRETBOARD=${FRETBOARD:-/Users/divyansh/guitar-tabs}
PY="uv run --quiet --python 3.11.17 --with-requirements $FRETBOARD/scripts/requirements-demucs.txt python"
SONGS=${SONGS:-"system-of-a-down-aerials nirvana-the-man-who-sold-the-world three-days-grace-never-too-late chaar-diwaari-iss-tarah"}
mkdir -p ref/songs
for s in $SONGS; do
  [ -f ref/songs/$s.wav ] || ffmpeg -v error -y -i "$FRETBOARD/.cache/audio/$s.m4a" -c:a pcm_f32le ref/songs/$s.wav
done
for shifts in 2 0; do
  for m in htdemucs htdemucs_6s; do
    for s in $SONGS; do
      out=ref/songs/$s.$m.s$shifts.safetensors
      [ -f "$out" ] || $PY tools/ref_separate.py --model $m --shifts $shifts --seed 0 ref/songs/$s.wav "$out"
    done
  done
done
