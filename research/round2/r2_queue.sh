#!/bin/bash
# Round 2 Demucs queue, in priority order, 2 threads (Kim runs on 1 more, scoring on 1 more: 4 in all).
cd /Users/divyansh/htdemucs-rs/research/round2
M=/Users/divyansh/datasets/musdb18hq/test; T=${THREADS:-2}
S() { OMP_NUM_THREADS=$T uv run --quiet --python 3.11.17 --with-requirements r2_env.txt python -I r2_sep.py "$@" --threads $T 2>&1 | grep -v -i warn; }
S htdemucs hd $M data/sep --passes s0,A,B
S htdemucs_6s 6s $M data/sep --passes s0,A,B
S htdemucs hdint8 $M data/sep --passes s0 --weights int8
S htdemucs hdbf16 $M data/sep --passes s0 --weights bf16
S htdemucs hdo10 $M data/sep --passes s0 --overlap 0.1
S htdemucs hdo50 $M data/sep --passes s0 --overlap 0.5
S htdemucs_ft_vocals ftv $M data/sep --passes s0
S htdemucs hdo10 $M data/sep --passes A,B --overlap 0.1
S htdemucs hdo50 $M data/sep --passes A,B --overlap 0.5
echo QUEUEDONE
