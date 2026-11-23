#!/bin/bash
# htdemucs_6s guitar passes (s0, s2, inst) for every round 4 item, on MPS by default (DEV=cpu for the CPU); waits for Kim.
R4=${R4:-/Users/divyansh/htdemucs-rs/research/round4}; D=${D:-/Users/divyansh/datasets/r4}; DEV=${DEV:-cuda}
cd $R4
for set in ${@:-gtechs gset egdb cmtx cmt8}; do
  for it in $(ls "$D/$set" 2>/dev/null); do
    [ -f "$D/$set/$it/mixture.wav" ] || continue
    while [ ! -f "data/kim/$set/$it/done" ]; do sleep 20; done
    PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2 flock ${GPULOCK:-/tmp/r4.gpu.lock} uv run --quiet --python ${PY:-3.11.17} --with-requirements ${R2ENV:-../round2/r2_env.txt} python -I r4_sep.py "$D/$set/$it" "data/pass/$set/$it" --kim "data/kim/$set/$it/vocals.wav" --device $DEV --threads 2 2>&1 | grep -v -i warn
    echo "6s $set $it $(ls data/pass/$set/$it | tr '\n' ' ')"
  done
done
echo Q6DONE
