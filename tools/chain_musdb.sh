#!/bin/sh
# MUSDB18-HQ uSDR of `htdemucs-rs separate -n <model>` (chain or melroformer), scored with
# research/round2/r2_score.py (--no-museval), plus the process CPU time per track.
#   [THREADS=4] [MELROFORMER_WEIGHTS=melroformer-f16] tools/chain_musdb.sh <htdemucs-rs> <model> <out dir> <musdb test dir> <track>...
# Writes <out dir>/scores/<track>.json (round 2's format, + cpu_s); the stems are deleted after
# scoring unless KEEP=1 (then kept in <out dir>/keep/<track>/{htdemucs,htdemucs_6s,...}).
set -e
cd "$(dirname "$0")/.."
BIN=$1; MODEL=$2; OUT=$3; M=$4; shift 4
THREADS=${THREADS:-4}
PY=${PY:-"uv run --quiet --python 3.11.17 --with-requirements research/round1/kim_env.txt python"}
mkdir -p "$OUT/in" "$OUT/scores"
for tr in "$@"; do
  [ -f "$OUT/scores/$tr.json" ] && continue
  ln -sf "$M/$tr/mixture.wav" "$OUT/in/$tr.wav"
  cpu=$( { /usr/bin/time -p "$BIN" separate -q --threads "$THREADS" -n "$MODEL" --float32 --clip-mode none -o "$OUT/sep" "$OUT/in/$tr.wav" ; } 2>&1 | awk '/^user/{u=$2} /^sys/{s=$2} END{printf "%.1f", u+s}')
  case $MODEL in chain) v=htdemucs ;; *) v=$MODEL ;; esac
  $PY research/round2/r2_score.py "$M" "$OUT/sep/$v" "$tr" "$OUT/scores/$tr.json" --no-museval > /dev/null
  python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['cpu_s']=float(sys.argv[2]); json.dump(d,open(p,'w')); print(json.dumps(d))" "$OUT/scores/$tr.json" "$cpu"
  if [ -n "$KEEP" ]; then mkdir -p "$OUT/keep/$tr" && for d in "$OUT"/sep/*; do mv "$d/$tr" "$OUT/keep/$tr/$(basename "$d")"; done; fi
  rm -rf "$OUT/sep"
done
