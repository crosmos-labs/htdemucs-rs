#!/bin/bash
# Kim Mel-RoFormer vocals for every round 4 item (sets given as args, default all), with the transformer on MPS and the
# STFT / mask / iSTFT on the CPU (round 3's kim_sep_mps.py). Writes data/kim/<set>/<item>/vocals.wav, then a "done" JSON.
R1=/Users/divyansh/htdemucs-rs/research/round1; R3=/Users/divyansh/htdemucs-rs/research/round3; R4=/Users/divyansh/htdemucs-rs/research/round4
D=/Users/divyansh/datasets/r4
cd $R1
for set in ${@:-gtechs gset egdb cmtx cmt8}; do
  for it in $(ls "$D/$set" 2>/dev/null); do
    o="$R4/data/kim/$set/$it"; [ -f "$o/done" ] && continue; [ -f "$D/$set/$it/mixture.wav" ] || continue; mkdir -p "$o"
    OMP_NUM_THREADS=1 uv run --quiet --python 3.11.17 --with-requirements kim_env.txt python -I $R3/kim_sep_mps.py data/ext/kim data/models/MelBandRoformer.ckpt "$D/$set/$it/mixture.wav" "$o/vocals.wav" 2>/dev/null | tail -1 > "$o/kim.json"
    if [ -s "$o/vocals.wav" ] && [ -s "$o/kim.json" ]; then mv "$o/kim.json" "$o/done"; echo "kim $set $it $(cat $o/done)"; else echo "kim $set $it FAILED"; fi
  done
done
echo KIMDONE
