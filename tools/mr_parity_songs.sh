#!/bin/sh
# Full-song Mel-RoFormer parity: for each input WAV, the PyTorch reference (upstream demix_track,
# tools/melroformer.py separate), then `htdemucs-rs separate -n melroformer`, compared per stem
# (src/bin/compare.rs); CPU time of both. References are kept in ref/mr/ (~40 MB per minute of
# audio; DELETE=1 removes each after use); REFS_ONLY=1 only makes them.
#   [THREADS=4] [KIM=<upstream repo>] [CKPT=<MelBandRoformer.ckpt>] tools/mr_parity_songs.sh <htdemucs-rs> <compare> <log> <wav>...
set -e
cd "$(dirname "$0")/.."
BIN=$1; CMP=$2; LOG=$3; shift 3
THREADS=${THREADS:-4}
KIM=${KIM:-research/round1/data/ext/kim}
CKPT=${CKPT:-research/round1/data/models/MelBandRoformer.ckpt}
PY=${PY:-"uv run --quiet --python 3.11.17 --with-requirements research/round1/kim_env.txt python"}
OUT=${TMPDIR:-/tmp}/mr-parity
touch "$LOG"
for wav in "$@"; do
  base=$(basename "$(dirname "$wav")")__$(basename "$wav" .wav)
  grep -q "^$base " "$LOG" && continue
  mkdir -p ref/mr
  ref=ref/mr/$(echo "$base" | tr -c 'A-Za-z0-9_.\n-' '_').safetensors
  if [ ! -f "$ref" ]; then
    OMP_NUM_THREADS=$THREADS $PY tools/melroformer.py separate "$KIM" "$CKPT" "$wav" "$ref.tmp" --threads "$THREADS" 2>/dev/null | tail -1 > "$ref.json"
    mv "$ref.tmp" "$ref"
  fi
  [ -n "$REFS_ONLY" ] && continue
  torch=$(cat "$ref.json")
  rm -rf "$OUT"; mkdir -p "$OUT"
  cp "$wav" "$OUT/track.wav"
  rust=$( { /usr/bin/time -p "$BIN" separate -q --threads "$THREADS" -n melroformer --float32 --clip-mode none -o "$OUT" "$OUT/track.wav" ; } 2>&1 | awk '/^user/{u=$2} /^sys/{s=$2} /^real/{r=$2} END{printf "rust_wall=%s rust_cpu=%.1f", r, u+s}')
  echo "$base $($CMP "$ref" "$OUT/melroformer/track" | sed 's|^[^:]*: ||') | $rust | torch $torch" | tee -a "$LOG"
  [ -n "$DELETE" ] && rm -f "$ref"
done
rm -rf "$OUT"
