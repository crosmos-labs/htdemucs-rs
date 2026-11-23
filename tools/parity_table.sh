#!/bin/sh
# The 16-row full-song parity table (4 songs x 2 models x shifts 2/0) for one build, generating
# each PyTorch reference, comparing, then deleting it (references are 0.3-0.7 GB each).
#   [THREADS=n] tools/parity_table.sh <htdemucs-rs binary> <compare binary> <log>
set -e
cd "$(dirname "$0")/.."
BIN=$1; CMP=$2; LOG=$3
FRETBOARD=${FRETBOARD:-/Users/divyansh/guitar-tabs}
PY="uv run --quiet --python 3.11.17 --with-requirements $FRETBOARD/scripts/requirements-demucs.txt python"
OUT=${TMPDIR:-/tmp}/htdemucs-table
T=${THREADS:+--threads $THREADS}
touch "$LOG"
for shifts in 2 0; do for m in htdemucs htdemucs_6s; do
  for s in system-of-a-down-aerials nirvana-the-man-who-sold-the-world three-days-grace-never-too-late chaar-diwaari-iss-tarah; do
    base=$s.$m.s$shifts
    grep -q "^$base " "$LOG" && continue
    ref=ref/songs/$base.safetensors
    [ -f "$ref" ] || $PY tools/ref_separate.py $T --model $m --shifts $shifts --seed 0 ref/songs/$s.wav "$ref" >&2
    rm -rf "$OUT"
    $BIN separate -q $T -n $m --shifts $shifts --seed 0 --float32 --clip-mode none -o "$OUT" ref/songs/$s.wav
    echo "$base $($CMP "$ref" "$OUT/$m/$s" | sed 's|^[^:]*: ||')" | tee -a "$LOG"
    [ "$base" = system-of-a-down-aerials.htdemucs.s2 ] || rm -f "$ref"
  done
done; done
rm -rf "$OUT"
