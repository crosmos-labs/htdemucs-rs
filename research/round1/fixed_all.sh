#!/bin/bash
# Fixed-placement sung-pitch match for every variant of a set, on the songs whose vocal line places well
# (First Love's pinned voice part; Nirvana's and Aerials' Songsterr vocal parts). Reference placement: seed0 V0L.
# Usage: fixed_all.sh <set>
SET=$1; R=/Users/divyansh/htdemucs-rs/research/round1; cd $R; mkdir -p data/fixed/$SET
for spec in mitski-first-love-late-spring:tab nirvana-the-man-who-sold-the-world:vocalPart system-of-a-down-aerials:vocalPart; do
  id=${spec%%:*}; which=${spec#*:}; V=data/variants/$SET/$id
  [ -d $V ] || continue; [ -f data/fixed/$SET/$id.json ] && continue
  P=(); for v in V0/vocals.mp3 V0L/vocals.wav V1a/vocals.wav V1b/vocals.wav K/vocals.wav; do [ -e $V/$v ] && P+=($V/$v); done
  echo "$id $which ${P[*]}"
done | xargs -P 3 -L 1 sh -c 'id=$0; which=$1; shift 1; uv run --quiet --python 3.11.17 --with-requirements /Users/divyansh/guitar-tabs/scripts/requirements-audio.txt python -I fixed_pitch.py data/requests.json $id $which data/results/seed0/$id/V0L.json "$@" > data/fixed/'$SET'/$id.json 2>>data/fixed_errors.log; echo "$id fixed done"'
