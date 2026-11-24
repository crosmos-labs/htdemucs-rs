#!/bin/bash
# Round 5 separations, one heavy job at a time under the shared lock (one acquisition per item).
#   r5_queue.sh <set> [<set> ...]   env: D (items root), P (pass root), MSST (repo at 1523206), PY (venv python), M (Mega
#   ckpt dir), W (htdemucs-rs weights), LOCKCMD (default: lockf -k on the Mac's lock; the box: "flock ~/work/heavy.lock"),
#   DEV / THREADS for Mega (default cpu / 8; the box runs cuda, as round 4)
# Per item: hyb6 = htdemucs-rs -n ensemble (shifts 0), chain = htdemucs-rs -n chain (Kim f16 weights),
# mega_mix = r5_mega.py heads on the mix, mega_inst = r5_mega.py heads on mix - chain's (Kim's) vocals.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
D=${D:-/Users/divyansh/datasets/r4}; P=${P:-/Users/divyansh/work/mac-testing/r5pass}
MSST=${MSST:-/Users/divyansh/work/mac-testing/Music-Source-Separation-Training}; PY=${PY:-/Users/divyansh/work/mac-testing/r5env/bin/python}
LOCKCMD=${LOCKCMD:-lockf -k /private/tmp/claude-503/heavy.lock}; BIN=$R/target/release/htdemucs-rs; M=${M:-$R/research/round4/data/models}
DEV=${DEV:-cpu}; THREADS=${THREADS:-8}
export HTDEMUCS_WEIGHTS=${W:-$R/weights} MELROFORMER_WEIGHTS=melroformer-f16
MIX_HEADS=vocal,lead-vocal,back-vocal,drums,bass,guitar,piano,keys,digital-piano,organ,synth
INST_HEADS=drums,bass,guitar,piano,keys,digital-piano,organ,synth
item() {  # all four passes of one item; run under ONE lock acquisition (the lock is shared with another agent's long jobs,
          # so one acquisition per item instead of four cuts the waiting)
  local s=$1 d=$2 it o in; it=$(basename "$d"); o=$P/$s/$it; mkdir -p "$o"; in=$d/mixture.wav
  for n in hyb6:ensemble chain:chain; do
    tag=${n%%:*}; model=${n##*:}
    if [ ! -f "$o/$tag/htdemucs_6s/mixture/other.wav" ]; then
      echo "$(date +%T) $s/$it $tag"
      "$BIN" separate -n "$model" --float32 --clip-mode none -q -v -o "$o/$tag" "$in" 2>> "$o/$tag.log" || echo "FAIL $tag $it"
    fi
  done
  if [ ! -f "$o/mega_mix.npy" ]; then
    echo "$(date +%T) $s/$it mega_mix"
    "$PY" -I "$R/research/round5/r5_mega.py" "$MSST" $M/mega53.ckpt $M/mega53.yaml "$in" "$o/mega_mix.npy" \
      --stems $MIX_HEADS --device $DEV --threads $THREADS >> "$o/mega_mix.json" 2>> "$o/mega.err" || echo "FAIL mega_mix $it"
  fi
  if [ ! -f "$o/mega_inst.npy" ]; then
    echo "$(date +%T) $s/$it mega_inst"
    "$PY" -I "$R/research/round5/r5_mega.py" "$MSST" $M/mega53.ckpt $M/mega53.yaml "$in" "$o/mega_inst.npy" \
      --stems $INST_HEADS --minus "$o/chain/htdemucs_6s/mixture/vocals.wav" --device $DEV --threads $THREADS \
      >> "$o/mega_inst.json" 2>> "$o/mega.err" || echo "FAIL mega_inst $it"
  fi
}
if [ "${1:-}" = __item ]; then item "$2" "$3"; exit; fi
for s in "$@"; do for d in "$D/$s"/*/; do
  [ -f "$P/$s/stop" ] && exit 0
  [ -f "$P/$s/$(basename "$d")/mega_inst.npy" ] && continue
  $LOCKCMD "$0" __item "$s" "$d"
done; done
echo "$(date +%T) done $*"
