#!/bin/bash
# Drums + percussion follow-up (r5_drums.py): Mega 53's eight drum-family heads on Kim's instrumental of every item,
# one item per lock acquisition, then the scores.   env: as r5_queue.sh (~/work/round5/env.sh on the box)
#   r5_drums.sh [out.json]   (default scores/drums.json)
set -u
R=$(cd "$(dirname "$0")/../.." && pwd); DEV=${DEV:-cuda}; THREADS=${THREADS:-2}
HEADS=drums,percussion,tambourine,congas,kick,snare,toms,hh
for s in cmtx cmt8; do for d in "$D/$s"/*/; do
  it=$(basename "$d"); o=$P/$s/$it
  [ -f "$o/mega_drums.npy" ] && continue
  echo "$(date +%T) $s/$it"
  $LOCKCMD "$PY" -I "$R/research/round5/r5_mega.py" "$MSST" $M/mega53.ckpt $M/mega53.yaml "$d/mixture.wav" "$o/mega_drums.npy" \
    --stems $HEADS --minus "$o/chain/htdemucs_6s/mixture/vocals.wav" --device $DEV --threads $THREADS \
    >> "$o/mega_drums.json" 2>> "$o/mega.err" || echo "FAIL $it"
done; done
nice "$PY" -I "$R/research/round5/r5_drums.py" "$D" "$P" "$R/research/round5/${1:-scores/drums.json}"
