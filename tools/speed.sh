#!/bin/sh
# Wall time / CPU time / peak RSS of a full-song separation, Rust (gemm crate and Accelerate) vs
# PyTorch CPU, at 4 and 8 threads, with the load average before and after each run.
#   tools/speed.sh <song.wav> [model] [shifts] [configs...]
# configs: gemm8 gemm4 acc8 acc4 torch8 torch4 (default: all)
cd "$(dirname "$0")/.."
FRETBOARD=${FRETBOARD:-/Users/divyansh/guitar-tabs}
IN=$1; MODEL=${2:-htdemucs}; SHIFTS=${3:-2}; shift 3 2>/dev/null
CONFIGS=${*:-"gemm8 gemm4 acc8 acc4 torch8 torch4"}
OUT=${TMPDIR:-/tmp}/htdemucs-speed
mkdir -p target/speed
cargo build --release -q --bin htdemucs-rs && cp target/release/htdemucs-rs target/speed/gemm
cargo build --release -q --features accelerate --bin htdemucs-rs && cp target/release/htdemucs-rs target/speed/acc
load() { uptime | sed 's/.*load averages*: //'; }
for c in $CONFIGS; do
  t=${c#gemm}; t=${t#acc}; t=${t#torch}
  case $c in
    gemm*) cmd="target/speed/gemm separate -n $MODEL --shifts $SHIFTS --seed 0 --threads $t --flac -o $OUT $IN" ;;
    acc*) cmd="env VECLIB_MAXIMUM_THREADS=$t target/speed/acc separate -n $MODEL --shifts $SHIFTS --seed 0 --threads $t --flac -o $OUT $IN" ;;
    torch*) cmd="uv run --quiet --python 3.11.17 --with-requirements $FRETBOARD/scripts/requirements-demucs.txt python tools/ref_separate.py --model $MODEL --shifts $SHIFTS --seed 0 --threads $t $IN $OUT.safetensors" ;;
  esac
  l0=$(load)
  /usr/bin/time -l $cmd 2> $OUT.time >/dev/null
  l1=$(load)
  wall=$(awk '/real/ {print $1}' $OUT.time); user=$(awk '/real/ {print $3}' $OUT.time); sys=$(awk '/real/ {print $5}' $OUT.time)
  rss=$(awk '/maximum resident set size/ {printf "%.0f", $1/1048576}' $OUT.time)
  echo "$(basename $IN .wav) $MODEL s$SHIFTS $c: wall ${wall}s cpu $(echo "$user + $sys" | bc)s peak RSS ${rss} MB | load before [$l0] after [$l1]"
  rm -rf $OUT $OUT.safetensors
done
