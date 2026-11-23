#!/bin/bash
# Kim Mel-RoFormer vocals for each MUSDB test track (CPU, f32, overlap 2; 1 thread so the total stays within 4).
R1=/Users/divyansh/htdemucs-rs/research/round1; R2=/Users/divyansh/htdemucs-rs/research/round2; M=/Users/divyansh/datasets/musdb18hq/test
THREADS=${THREADS:-1}
cd $R1
ls "$M" | while read -r tr; do
  d="$R2/data/sep/kim/$tr"; [ -f "$d/done" ] && continue; mkdir -p "$d"
  OMP_NUM_THREADS=$THREADS uv run --quiet --python 3.11.17 --with-requirements kim_env.txt python -I kim_sep.py data/ext/kim data/models/MelBandRoformer.ckpt "$M/$tr/mixture.wav" "$d/vocals.wav" --device cpu --threads $THREADS 2>/dev/null | tail -1 > "$d/kim.json"
  if [ -s "$d/vocals.wav" ]; then python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d['track']=sys.argv[2]; open(sys.argv[3],'a').write(json.dumps(d)+'\n')" "$d/kim.json" "$tr" "$R2/data/sep/kim_cpu.jsonl"; cp "$d/kim.json" "$d/done"; echo "kim $tr done"; else echo "kim $tr FAILED"; fi
done
