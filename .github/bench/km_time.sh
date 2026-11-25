#!/bin/bash
# Whole-track CPU time and peak RSS of htdemucs-rs recipes on synthetic clips (bench-kimmega.yml).
#   km_time.sh <threads> "<clip seconds ...>" "<recipes ...>"     (env PY: python with numpy)
# Appends to time-<threads>.txt and the step summary: RESULT recipe= secs= threads= wall= cpu= cpu_per_audio_s= rss_mb=
set -eu
t=$1; PY=${PY:-python}
for secs in $2; do
  [ -f clip$secs.wav ] || $PY .github/bench/synth.py clip$secs.wav $secs
  for r in $3; do
    /usr/bin/time -f "%e %U %S %M" -o tm.txt ./target/release/htdemucs-rs separate -q -v -n $r --threads $t --flac --clip-mode clamp -o out clip$secs.wav 2> err.txt || { cat err.txt; exit 1; }
    read wall user sys rss < tm.txt
    line=$(awk -v r=$r -v s=$secs -v t=$t -v w=$wall -v u=$user -v y=$sys -v m=$rss 'BEGIN { c = u + y; printf "RESULT recipe=%s secs=%s threads=%s wall=%.1f cpu=%.1f cpu_per_audio_s=%.3f rss_mb=%.0f", r, s, t, w, c, c / s, m / 1024 }')
    echo "$line" | tee -a time-$t.txt
    grep -E "separated in|peak RSS" err.txt | sed 's/^/  /' | tee -a time-$t.txt
    echo "- \`$line\`" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
    rm -rf out
  done
done
