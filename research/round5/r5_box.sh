#!/bin/bash
# Round 5 on the Linux box (i5-12450H, RTX 4060 8 GB): separations (Mega on CUDA fp32, as round 4), scoring, cost.
# Every heavy job takes ~/work/heavy.lock (shared with other agents). Env: ~/work/round5/env.sh.
#   r5_box.sh [sep] [score] [cost]   (default: all three)
. ~/work/round5/env.sh
cd "$R5/research/round5" || exit 1
steps=${*:-sep score cost}
for s in $steps; do case $s in
  sep)   DEV=cuda THREADS=2 ./r5_queue.sh cmtx cmt8;;
  score) for set in cmtx cmt8; do nice $PY -I r5_score.py "$D/$set" "$P/$set" "scores/$set"; done
         $PY -I r5_summary.py > tables.md;;
  cost)  FULL="$HOME/work/datasets/musdb18hq/test/AM Contra - Heart Peripheral/mixture.wav" HOST=box-i5-12450H \
           ./r5_cost.sh gpu rust8 mega8 rust1 mega1 rust1full mega1full;;
esac; done
echo "BOX $(date +%T) done: $steps"
