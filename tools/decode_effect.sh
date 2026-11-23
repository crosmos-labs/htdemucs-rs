#!/bin/sh
# Effect of the decoder on the stems: run `htdemucs-rs separate` on the original m4a (symphonia
# decode) and compare with the PyTorch reference made from ffmpeg's decode of the same file.
#   tools/decode_effect.sh [model] [shifts] [features]
set -e
cd "$(dirname "$0")/.."
FRETBOARD=${FRETBOARD:-/Users/divyansh/guitar-tabs}
MODEL=${1:-htdemucs}; SHIFTS=${2:-2}
FEAT=${3:+--features $3}
cargo build --release -q $FEAT --bin htdemucs-rs --bin compare --bin decodecmp
OUT=${TMPDIR:-/tmp}/htdemucs-decode
for ref in ref/songs/*.$MODEL.s$SHIFTS.safetensors; do
  song=$(basename "$ref" .$MODEL.s$SHIFTS.safetensors)
  ./target/release/decodecmp "$FRETBOARD/.cache/audio/$song.m4a" "ref/songs/$song.wav"
  rm -rf "$OUT"
  ./target/release/htdemucs-rs separate -n "$MODEL" --shifts "$SHIFTS" --seed 0 --float32 --clip-mode none -o "$OUT" "$FRETBOARD/.cache/audio/$song.m4a"
  echo "m4a-input $song.$MODEL.s$SHIFTS $(./target/release/compare "$ref" "$OUT/$MODEL/$song")"
done
rm -rf "$OUT"
