#!/bin/sh
# The chain measurements on a shared Linux box (one heavy job at a time: each step takes
# $LOCK with flock; at most $T threads). Steps, each skipped when its output exists:
#   parity   Rust chain vs tools/chain_ref.py (PyTorch CPU float32) on two MUSDB tracks, per stem
#   sdr      Rust chain with float32 and float16 Kim weights on four MUSDB tracks, r2_score.py uSDR
#   memory   peak RSS of the chain on a 4-minute and a 22-minute track (/usr/bin/time -v)
#   speed    CPU-s per audio-s of the chain, Rust and PyTorch CPU, at 1, 4 and 8 threads
#   tools/chain_box.sh <work dir> <htdemucs-rs> <weights dir> <musdb test dir> <songs dir> [steps...]
# <work dir> holds py (a python with torch / demucs / Kim's deps), kim (upstream repo), ckpt/.
set -e
W=$1; BIN=$2; WEIGHTS=$3; M=$4; SONGS=$5; shift 5
STEPS=${*:-parity sdr memory speed}
T=${T:-8}
LOCK=${LOCK:-$HOME/work/.heavy.lock}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$W/py
KIM=$W/kim; CKPT=$W/ckpt/MelBandRoformer.ckpt
OUT=$W/out; mkdir -p "$OUT"
heavy() { flock "$LOCK" "$@"; }
rust() { # rust <name> <threads> <in.wav> <out dir> [env...]: chain, float32 WAV, clip none; prints CPU s, peak RSS KB
  heavy /usr/bin/time -f "%U %S %M %e" -o "$4.time" env HTDEMUCS_WEIGHTS="$WEIGHTS" $5 "$BIN" separate -q -v --threads "$2" -n chain --float32 --clip-mode none -o "$4" "$3" 2> "$4.log"
  awk '{printf "cpu_s=%.1f peak_rss_mb=%.0f wall_s=%.1f", $1 + $2, $3 / 1024, $4}' "$4.time"
}
P1="Signe Jakobsen - What Have You Done To Me"; P2="Angels In Amplifiers - I'm Alright"
P3="Enda Reilly - Cur An Long Ag Seol"; P4="Al James - Schoolboy Facination"
key() { echo "$1" | tr -c 'A-Za-z0-9_.\n-' '_'; }
for step in $STEPS; do case $step in
parity)
  for tr in "$P1" "$P2"; do
    k=$(key "$tr"); d=$OUT/parity/$k; mkdir -p "$d"; [ -f "$d/result.txt" ] && continue
    ln -sf "$M/$tr/mixture.wav" "$d/$k.wav"
    [ -f "$d/ref.4.safetensors" ] || heavy "$PY" "$REPO/tools/chain_ref.py" "$KIM" "$CKPT" "$d/$k.wav" "$d/ref" --threads "$T" 2>/dev/null | tail -1 > "$d/torch.json"
    r=$(rust chain "$T" "$d/$k.wav" "$d/sep")
    { echo "$tr | rust $r | torch $(cat "$d/torch.json")"
      "$REPO/target/release/compare" "$d/ref.4.safetensors" "$d/sep/htdemucs/$k"
      "$REPO/target/release/compare" "$d/ref.6.safetensors" "$d/sep/htdemucs_6s/$k"
      grep "stems sum" "$d/sep.log"; } > "$d/result.txt"
    cat "$d/result.txt"
  done ;;
sdr)
  for v in f32 f16; do
    wf=melroformer; [ $v = f16 ] && wf=melroformer-f16
    for tr in "$P1" "$P2" "$P3" "$P4"; do
      k=$(key "$tr"); d=$OUT/sdr/$v; mkdir -p "$d/in" "$d/scores"; [ -f "$d/scores/$tr.json" ] && continue
      ln -sf "$M/$tr/mixture.wav" "$d/in/$tr.wav"
      r=$(rust chain "$T" "$d/in/$tr.wav" "$d/sep" MELROFORMER_WEIGHTS=$wf)
      "$PY" "$REPO/research/round2/r2_score.py" "$M" "$d/sep/htdemucs" "$tr" "$d/scores/$tr.json" --no-museval > /dev/null
      echo "$v | $tr | $r | $(cat "$d/scores/$tr.json")"
      rm -rf "$d/sep"
    done
  done ;;
memory)
  d=$OUT/memory; mkdir -p "$d"
  [ -f "$SONGS/long22.wav" ] && ln -sf "$SONGS/long22.wav" "$d/long22.wav"  # the Demucs path's 22-min track
  [ -f "$d/long22.wav" ] || ffmpeg -v error -stream_loop 5 -i "$SONGS/system-of-a-down-aerials.wav" -t 1320 -c:a pcm_f32le "$d/long22.wav"
  for f in "$SONGS/system-of-a-down-aerials.wav" "$d/long22.wav"; do
    n=$(basename "$f" .wav); [ -f "$d/$n.txt" ] && continue
    echo "$n: $(rust chain "$T" "$f" "$d/sep") | $(grep 'stems sum' "$d/sep.log")" | tee "$d/$n.txt"
    rm -rf "$d/sep"
  done ;;
speed)
  d=$OUT/speed; mkdir -p "$d"
  [ -f "$d/aerials120.wav" ] || ffmpeg -v error -ss 60 -t 120 -i "$SONGS/system-of-a-down-aerials.wav" -c:a pcm_f32le "$d/aerials120.wav"
  for t in 8 4 1; do
    [ -f "$d/rust-$t.txt" ] || { echo "rust threads $t: $(rust chain "$t" "$d/aerials120.wav" "$d/sep")" | tee "$d/rust-$t.txt"; rm -rf "$d/sep"; }
    [ -f "$d/torch-$t.txt" ] || { heavy /usr/bin/time -f "%U %S %M %e" -o "$d/torch-$t.time" env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$t "$PY" "$REPO/tools/chain_ref.py" "$KIM" "$CKPT" "$d/aerials120.wav" "$d/torch" --threads "$t" 2>/dev/null | tail -1 > "$d/torch-$t.json"
      echo "torch threads $t: $(awk '{printf "cpu_s=%.1f peak_rss_mb=%.0f wall_s=%.1f", $1 + $2, $3 / 1024, $4}' "$d/torch-$t.time") $(cat "$d/torch-$t.json")" | tee "$d/torch-$t.txt"; }
  done ;;
esac; done
