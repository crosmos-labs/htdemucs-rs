#!/bin/sh
# Decode the test songs and dump the reference segments used by the parity test (into ref/,
# gitignored). Needs ffmpeg, uv, Fretboard's demucs requirements and its audio cache.
set -e
cd "$(dirname "$0")/.."
FRETBOARD=${FRETBOARD:-/Users/divyansh/guitar-tabs}
PY="uv run --quiet --python 3.11.17 --with-requirements $FRETBOARD/scripts/requirements-demucs.txt python"
mkdir -p ref
for s in system-of-a-down-aerials oasis-wonderwall three-days-grace-never-too-late; do
  [ -f ref/$s.wav ] || ffmpeg -v error -y -i "$FRETBOARD/.cache/audio/$s.m4a" -c:a pcm_f32le ref/$s.wav
done
dump() { # song tag model chunk [shift]
  out=ref/$2-$3-$(echo $4 | sed 's/^-1$/last/').safetensors
  [ -f "$out" ] && return
  if [ -n "$5" ]; then extra="--shift-offset $5"; else extra=""; fi
  $PY tools/harness.py --wav ref/$1.wav --model $3 --chunk $4 $extra --out "$out"
}
for m in htdemucs htdemucs_6s; do
  dump system-of-a-down-aerials aerials $m 10            # loud rock, mid-song chunk
  dump oasis-wonderwall wonderwall $m 5                   # acoustic + vocals
  dump three-days-grace-never-too-late ntl $m -1 12345    # last (padded) chunk of a shifted pass
done
