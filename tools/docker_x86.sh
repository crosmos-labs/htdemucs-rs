#!/bin/sh
# Build and check the Linux x86-64 release in a linux/amd64 container (Docker Desktop on Apple
# silicon emulates it, so timings are meaningless here: this is a correctness check of the x86
# code paths). Runs the per-layer segment parity on the kept reference dumps, then optionally one
# full song: tools/docker_x86.sh [song model shifts]
set -e
cd "$(dirname "$0")/.."
SONG=$1; MODEL=${2:-htdemucs}; SHIFTS=${3:-0}
docker run --rm --platform linux/amd64 -v "$PWD":/src -v htdemucs-cargo:/usr/local/cargo/registry -w /src \
  -e CARGO_TARGET_DIR=/src/target/linux-x86 -e CARGO_PROFILE_RELEASE_LTO=thin -e CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
  rust:slim-bookworm sh -c "
    set -e; uname -m; grep -o -w -E 'avx2|fma|avx512f' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
    cargo build --release -q --bins
    ls -l target/linux-x86/release/htdemucs-rs
    for r in ref/*.safetensors; do ./target/linux-x86/release/parity weights \$r | grep -E '^## |final output'; done
    if [ -n '$SONG' ]; then
      ./target/linux-x86/release/htdemucs-rs separate -v -n $MODEL --shifts $SHIFTS --seed 0 --float32 --clip-mode none -o /tmp/out ref/songs/$SONG.wav
      ./target/linux-x86/release/compare ref/songs/$SONG.$MODEL.s$SHIFTS.safetensors /tmp/out/$MODEL/$SONG
    fi"
