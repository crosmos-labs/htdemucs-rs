#!/bin/bash
# Kim vocals for round 3. MODE=mps: transformer on MPS, STFT/mask/iSTFT on CPU (kim_sep_mps.py), tracks in order.
# MODE=cpu: round 1's kim_sep.py on the CPU, THREADS threads, tracks in reverse order. A mkdir claim stops both doing one track.
R1=/Users/divyansh/htdemucs-rs/research/round1; R3=/Users/divyansh/htdemucs-rs/research/round3; M=/Users/divyansh/datasets/musdb18hq/test
MODE=${MODE:-mps}; THREADS=${THREADS:-1}
cd $R1
if [ $MODE = mps ]; then L=$(ls "$M"); else L=$(ls "$M" | sort -r); fi
echo "$L" | while read -r tr; do
  d="$R3/data/kim/$tr"; mkdir -p "$d"; mkdir "$d/claim" 2>/dev/null || continue
  [ -f "$d/done" ] && continue
  if [ $MODE = mps ]; then
    OMP_NUM_THREADS=1 uv run --quiet --python 3.11.17 --with-requirements kim_env.txt python -I $R3/kim_sep_mps.py data/ext/kim data/models/MelBandRoformer.ckpt "$M/$tr/mixture.wav" "$d/vocals.wav" 2>/dev/null | tail -1 > "$d/kim.json"
  else
    OMP_NUM_THREADS=$THREADS uv run --quiet --python 3.11.17 --with-requirements kim_env.txt python -I kim_sep.py data/ext/kim data/models/MelBandRoformer.ckpt "$M/$tr/mixture.wav" "$d/vocals.wav" --device cpu --threads $THREADS 2>/dev/null | tail -1 > "$d/kim.json"
  fi
  if [ -s "$d/vocals.wav" ] && [ -s "$d/kim.json" ]; then mv "$d/kim.json" "$d/done"; echo "kim $MODE $tr done"; else echo "kim $MODE $tr FAILED"; fi
done
