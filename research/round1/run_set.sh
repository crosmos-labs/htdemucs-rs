#!/bin/bash
# Build variants and score them for one stem set. Usage: run_set.sh <set: seed0|seed1|seed2> <variants…>
# Variant WAVs are ~100 MB each (≈3.5 GB per set of 12 songs): delete data/variants/**/*.wav after scoring.
# seed0 = the pipeline's own cached splits (guitar-tabs/.cache/stems); seedN = research/round1/data/stems/seedN.
set -uo pipefail
SET=$1; shift; VARS=("$@")
R=/Users/divyansh/htdemucs-rs/research/round1; GT=/Users/divyansh/guitar-tabs
IDS=(mitski-first-love-late-spring chris-isaak-wicked-game alvvays-archie-marry-me arctic-monkeys-do-i-wanna-know arijit-singh-ilahi billie-eilish-wildflower frank-ocean-american-wedding jeff-buckley-hallelujah nirvana-the-man-who-sold-the-world oasis-wonderwall system-of-a-down-aerials three-days-grace-never-too-late)
cd "$R"
for id in "${IDS[@]}"; do
  if [ "$SET" != seed0 ] && [ ! -d data/stems/$SET/htdemucs_6s/$id ]; then continue; fi
  if [ "$SET" = seed0 ]; then d4=$GT/.cache/stems/htdemucs/$id; d6=$GT/.cache/stems/htdemucs_6s/$id; else d4=data/stems/$SET/htdemucs/$id; d6=data/stems/$SET/htdemucs_6s/$id; fi
  KIM=(); [ -f data/kim/$id.vocals.wav ] && KIM=(--kim data/kim/$id.vocals.wav)
  [ -f data/variants/$SET/$id/V1a/vocals.wav ] && [ ${#KIM[@]} -eq 0 -o -f data/variants/$SET/$id/K/vocals.wav ] || \
    uv run --quiet --python 3.11.17 --with-requirements $GT/scripts/requirements-audio.txt python -I make_variants.py "$SET" "$d4" "$d6" "$GT/.cache/audio/$id.m4a" "data/variants/$SET/$id" ${KIM[@]+"${KIM[@]}"} >&2
  for v in "${VARS[@]}"; do
    [ -d data/variants/$SET/$id/$v ] && [ ! -f data/results/$SET/$id/$v.json ] && echo "data/variants/$SET/$id/$v data/results/$SET/$id/$v.json"
  done
done | xargs -P 3 -L 1 sh -c 'python3 -I score.py data/requests.json "$0" "$1" > /dev/null 2>>data/score_errors.log; echo "$0 done"'
