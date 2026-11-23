#!/bin/bash
# Mega 53 guitar heads (guitar, electric-guitar, acoustic-guitar) on the mix and on Kim's instrumental, for every round 4 item.
# Several workers can run at once (DEV=mps / DEV=cpu THREADS=n); a mkdir claim per pass stops two doing the same one.
R4=${R4:-/Users/divyansh/htdemucs-rs/research/round4}; D=${D:-/Users/divyansh/datasets/r4}; DEV=${DEV:-cuda}; THREADS=${THREADS:-1}
cd $R4
for set in ${SETS:-cmt8 cmtx gtechs gset egdb}; do
  for it in $(ls "$D/$set" 2>/dev/null); do
    for p in mix inst; do
      o="data/pass/$set/$it"; mkdir -p "$o"; [ -f "$o/mega_$p.npy" ] && continue
      mkdir "$o/claim_mega_$p" 2>/dev/null || continue
      M=""; if [ $p = inst ]; then while [ ! -f "data/kim/$set/$it/done" ]; do sleep 20; done; M="--minus data/kim/$set/$it/vocals.wav"; fi
      PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=$THREADS flock ${GPULOCK:-/tmp/r4.gpu.lock} uv run --quiet --python ${PY:-3.11.17} --with-requirements r4_msst_env.txt python -I r4_mega.py data/ext/msst data/models/mega53.ckpt data/models/mega53.yaml "$D/$set/$it/mixture.wav" "$o/mega_$p.tmp.npy" --device $DEV --threads $THREADS $M 2>&1 | grep '^{' > "$o/mega_$p.json"
      if [ -s "$o/mega_$p.json" ] && [ -f "$o/mega_$p.tmp.npy" ]; then mv "$o/mega_$p.tmp.npy" "$o/mega_$p.npy"; echo "mega $DEV $set $it $p $(cat $o/mega_$p.json)"; else rmdir "$o/claim_mega_$p"; echo "mega $DEV $set $it $p FAILED"; fi
    done
  done
done
echo MEGADONE $DEV
