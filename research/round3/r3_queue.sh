#!/bin/bash
# Round 3 Demucs queue: per track, htdemucs then htdemucs_6s, passes id, sw, po, swpo, inst (inst waits for Kim's vocals).
# On MPS (outputs match the CPU's to 4 decimals of uSDR); CPU cost per pass is taken from round 2's CPU runs.
cd /Users/divyansh/htdemucs-rs/research/round3
M=/Users/divyansh/datasets/musdb18hq/test; DEV=${DEV:-mps}
S() { PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=1 uv run --quiet --python 3.11.17 --with-requirements ../round2/r2_env.txt python -I r3_sep.py "$@" --device $DEV --threads 1 --wait 2>&1 | grep -v -i warn; }
ls "$M" | while read -r tr; do
  S htdemucs hd $M data/pass "$tr"
  S htdemucs_6s 6s $M data/pass "$tr"
done
echo QUEUEDONE
