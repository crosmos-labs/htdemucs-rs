#!/bin/bash
# Kim Mel-RoFormer vocals for every round 4 item (sets given as args, default all). MODE=cuda (default on the Linux box):
# r4_kim_sep.py on CUDA in fp32 under `flock $GPULOCK`; MODE=mps: round 3's kim_sep_mps.py (transformer on MPS).
# Writes $R4/data/kim/<set>/<item>/vocals.wav, then a "done" JSON.
R4=${R4:-/Users/divyansh/htdemucs-rs/research/round4}; D=${D:-/Users/divyansh/datasets/r4}
KREPO=${KREPO:-/Users/divyansh/htdemucs-rs/research/round1/data/ext/kim}; KCKPT=${KCKPT:-/Users/divyansh/htdemucs-rs/research/round1/data/models/MelBandRoformer.ckpt}
KENV=${KENV:-/Users/divyansh/htdemucs-rs/research/round1/kim_env.txt}; MODE=${MODE:-cuda}; GPULOCK=${GPULOCK:-/tmp/r4.gpu.lock}
cd $R4
for set in ${@:-gtechs gset egdb cmtx cmt8}; do
  for it in $(ls "$D/$set" 2>/dev/null); do
    o="$R4/data/kim/$set/$it"; [ -f "$o/done" ] && continue; [ -f "$D/$set/$it/mixture.wav" ] || continue; mkdir -p "$o"
    if [ $MODE = mps ]; then
      OMP_NUM_THREADS=1 uv run --quiet --python ${PY:-3.11.17} --with-requirements $KENV python -I $R4/../round3/kim_sep_mps.py $KREPO $KCKPT "$D/$set/$it/mixture.wav" "$o/vocals.wav" 2>/dev/null | tail -1 > "$o/kim.json"
    else
      OMP_NUM_THREADS=2 flock $GPULOCK uv run --quiet --python ${PY:-3.11.17} --with-requirements $KENV python -I r4_kim_sep.py $KREPO $KCKPT "$D/$set/$it/mixture.wav" "$o/vocals.wav" --device cuda --threads 2 --fp32 2>/dev/null | tail -1 > "$o/kim.json"
    fi
    if [ -s "$o/vocals.wav" ] && [ -s "$o/kim.json" ]; then mv "$o/kim.json" "$o/done"; echo "kim $set $it $(cat $o/done)"; else echo "kim $set $it FAILED"; fi
  done
done
echo KIMDONE
