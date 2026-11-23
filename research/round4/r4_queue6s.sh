#!/bin/bash
# htdemucs_6s guitar passes (s0, s2, inst) for every round 4 item, on MPS by default (DEV=cpu for the CPU); waits for Kim.
R4=/Users/divyansh/htdemucs-rs/research/round4; D=/Users/divyansh/datasets/r4; DEV=${DEV:-mps}
cd $R4
for set in ${@:-gtechs gset egdb cmtx cmt8}; do
  for it in $(ls "$D/$set" 2>/dev/null); do
    [ -f "$D/$set/$it/mixture.wav" ] || continue
    while [ ! -f "data/kim/$set/$it/done" ]; do sleep 20; done
    PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=1 uv run --quiet --python 3.11.17 --with-requirements ../round2/r2_env.txt python -I r4_sep.py "$D/$set/$it" "data/pass/$set/$it" --kim "data/kim/$set/$it/vocals.wav" --device $DEV --threads 1 2>&1 | grep -v -i warn
    echo "6s $set $it $(ls data/pass/$set/$it | tr '\n' ' ')"
  done
done
echo Q6DONE
