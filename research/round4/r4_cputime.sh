#!/bin/bash
# CPU-s per audio-s of every round 4 pass type on the Linux box's CPU (i5-12450H), 1 thread, on the first 30 s of one item,
# under `flock $HEAVYLOCK`. Appends JSON lines to data/cputime.jsonl.
#   6s     r4_sep.py --device cpu --threads 1 (pass s0; s2 = 2x, inst = 1x)
#   kim    r4_kim_sep.py --device cpu --threads 1 --dur 30 (fp32 anyway on the CPU)
#   mega   r4_mega.py --device cpu --threads 1 with 1, 3 and all 53 heads; 'rate_long' = median CPU per 20 s chunk x 2 / 20
R4=${R4:-$HOME/work/round4/research/round4}; D=${D:-$HOME/work/round4/items}; IT=${IT:-cmtx/Woodfire_GetFooled}
KREPO=$R4/../round1/data/ext/kim; KCKPT=$R4/../round1/data/models/MelBandRoformer.ckpt; L=${HEAVYLOCK:-$HOME/work/.heavy.lock}
cd $R4; mkdir -p data/cpu
uv run --quiet --python ${PY:-3.11.17} --with soundfile==0.12.1 --with numpy==1.26.4 python -I -c "
import soundfile as sf; x,sr=sf.read('$D/$IT/mixture.wav',dtype='float32'); sf.write('data/cpu/mixture.wav',x[:30*sr],sr,subtype='FLOAT')"
OMP_NUM_THREADS=1 flock $L uv run --quiet --python ${PY:-3.11.17} --with-requirements ../round2/r2_env.txt python -I r4_sep.py data/cpu data/cpu/6s --passes s0 --device cpu --threads 1 2>/dev/null
echo "{\"model\": \"6s\", $(tail -1 data/cpu/6s/cpu.jsonl | cut -c2-)" >> data/cputime.jsonl
echo "{\"model\": \"kim\", $(OMP_NUM_THREADS=1 flock $L uv run --quiet --python ${PY:-3.11.17} --with-requirements ../round1/kim_env.txt python -I r4_kim_sep.py $KREPO $KCKPT data/cpu/mixture.wav data/cpu/kim.wav --device cpu --threads 1 2>/dev/null | tail -1 | cut -c2-)" >> data/cputime.jsonl
ALL=$(python3 -I -c "
import re; s=open('data/models/mega53.yaml').read(); i=s.index('instruments: ['); print(','.join(re.findall(r\"'([^']+)'\", s[i:s.index(']',i)])))")
for st in guitar guitar,electric-guitar,acoustic-guitar $ALL; do
  echo "{\"model\": \"mega\", $(OMP_NUM_THREADS=1 flock $L uv run --quiet --python ${PY:-3.11.17} --with-requirements r4_msst_env.txt python -I r4_mega.py data/ext/msst data/models/mega53.ckpt data/models/mega53.yaml data/cpu/mixture.wav data/cpu/mega.npy --stems $st --device cpu --threads 1 2>/dev/null | grep '^{' | cut -c2- | sed 's/"stems": \[[^]]*\]/"n_stems": '$(echo $st | tr ',' '\n' | wc -l)'/')" >> data/cputime.jsonl
done
echo CPUTIMEDONE
