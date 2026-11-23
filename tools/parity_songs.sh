#!/bin/sh
# Full-song parity: run `htdemucs-rs separate` (float32 WAV, no clipping) on each decoded test song
# for which tools/ref_songs.sh made a PyTorch reference, and compare stem by stem.
#   tools/parity_songs.sh [extra cargo features, e.g. accelerate]
set -e
cd "$(dirname "$0")/.."
FEAT=${1:+--features $1}
cargo build --release -q $FEAT --bin htdemucs-rs --bin compare
OUT=${TMPDIR:-/tmp}/htdemucs-parity-${1:-gemm}
LOG=ref/songs/parity-${1:-gemm}.txt                # results; references already in it are skipped
touch "$LOG"
for ref in ref/songs/*.s[0-9].safetensors; do
  base=$(basename "$ref" .safetensors)          # song.model.sN
  grep -q "^$base " "$LOG" && continue
  song=${base%%.*}; rest=${base#*.}; model=${rest%%.*}; shifts=${rest#*.s}
  rm -rf "$OUT"
  ./target/release/htdemucs-rs separate -n "$model" --shifts "$shifts" --seed 0 --float32 --clip-mode none -o "$OUT" "ref/songs/$song.wav"
  echo "$base $(./target/release/compare "$ref" "$OUT/$model/$song")" | tee -a "$LOG"
done
rm -rf "$OUT"
