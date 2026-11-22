#!/bin/sh
# Full-song parity: run `htdemucs-rs separate` (float32 WAV, no clipping) on each decoded test song
# for which tools/ref_songs.sh made a PyTorch reference, and compare stem by stem.
#   tools/parity_songs.sh [extra cargo features, e.g. accelerate]
set -e
cd "$(dirname "$0")/.."
FEAT=${1:+--features $1}
cargo build --release -q $FEAT --bin htdemucs-rs --bin compare
OUT=${TMPDIR:-/tmp}/htdemucs-parity
for ref in ref/songs/*.s[0-9].safetensors; do
  base=$(basename "$ref" .safetensors)          # song.model.sN
  song=${base%%.*}; rest=${base#*.}; model=${rest%%.*}; shifts=${rest#*.s}
  rm -rf "$OUT"
  ./target/release/htdemucs-rs separate -n "$model" --shifts "$shifts" --seed 0 --float32 --clip-mode none -o "$OUT" "ref/songs/$song.wav"
  printf '%s s%s ' "$model" "$shifts"
  ./target/release/compare "$ref" "$OUT/$model/$song"
done
rm -rf "$OUT"
