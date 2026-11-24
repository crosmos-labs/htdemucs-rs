#!/bin/bash
# Round 5 cost: CPU seconds per audio second of every pass, one job at a time under the shared heavy lock. Each pass runs
# under r5_time.py (user + sys CPU of the child, wall, peak RSS) and appends a JSON line to scores/cost.jsonl.
#   r5_cost.sh <tag> [<tag> ...]   tags: rust8 rust1 rust1full mega8 mega1 mega1full gpu
# env: as r5_queue.sh (D, PY, MSST, M, W, LOCKCMD) plus FULL (the full song) and HOST (label written into each line).
# Inputs: the 30 s excerpt = the first 30 s of cmt8/AngelsInAmplifiers_ImAlright (its 60-90 s of the song); the full
# song = MUSDB18-HQ test "AM Contra - Heart Peripheral" (210.1 s).
set -u
R=$(cd "$(dirname "$0")/../.." && pwd); OUT=$R/research/round5/scores/cost.jsonl; mkdir -p "$(dirname "$OUT")"
WK=${WK:-$HOME/work/round5/cost}; mkdir -p "$WK"
D=${D:-/Users/divyansh/datasets/r4}; PY=${PY:-python3}; MSST=${MSST:?}; M=${M:?}
LOCKCMD=${LOCKCMD:-lockf -k /private/tmp/claude-503/heavy.lock}; BIN=$R/target/release/htdemucs-rs; HOST=${HOST:-$(uname -n)}
export HTDEMUCS_WEIGHTS=${W:-$R/weights} MELROFORMER_WEIGHTS=melroformer-f16
EX=$WK/ex30.wav; FULL=${FULL:?}
[ -f "$EX" ] || "$PY" -I -c "import soundfile as sf; x,sr=sf.read('$D/cmt8/AngelsInAmplifiers_ImAlright/mixture.wav',dtype='float32'); sf.write('$EX',x[:30*sr],sr,subtype='FLOAT')"

run() {  # run <pass> <threads> <input name> <input> <device> <cmd...>
  local pass=$1 th=$2 iname=$3 in=$4 dev=$5; shift 5
  $LOCKCMD "$PY" -I "$R/research/round5/r5_time.py" "$pass" "$th" "$iname" "$in" "$dev" "$HOST" -- "$@" >> "$OUT" || echo "FAIL $pass $th $iname"
  tail -1 "$OUT"
}
rust() {  # <pass> <model> <threads> <iname> <input>
  rm -rf "$WK/o"; run "$1" "$3" "$4" "$5" cpu "$BIN" separate -n "$2" --threads "$3" --float32 --clip-mode none -q -o "$WK/o" "$5"
}
mega() {  # <pass> <heads> <threads> <iname> <input> <device> [--minus vocals.wav]
  local p=$1 h=$2 th=$3 iname=$4 in=$5 dev=$6; shift 6
  run "$p" "$th" "$iname" "$in" "$dev" "$PY" -I "$R/research/round5/r5_mega.py" "$MSST" $M/mega53.ckpt $M/mega53.yaml "$in" "$WK/m.npy" \
    --stems "$h" --device "$dev" --threads "$th" "$@"
}
kimvoc() { [ -f "$WK/kim/melroformer/ex30/vocals.wav" ] || $LOCKCMD "$BIN" separate -n melroformer --float32 --clip-mode none -q -o "$WK/kim" "$EX"; }
H5=vocal,drums,bass,guitar,piano; H4=drums,bass,guitar,piano
for tag in "$@"; do case $tag in
  rust8) for x in "ex30:$EX" "full:$FULL"; do for p in hyb6:ensemble chain:chain kim:melroformer; do
           rust "${p%%:*}" "${p#*:}" 8 "${x%%:*}" "${x#*:}"; done; done;;
  rust1) for p in hyb6:ensemble chain:chain kim:melroformer; do rust "${p%%:*}" "${p#*:}" 1 ex30 "$EX"; done;;
  rust1full) rust hyb6 ensemble 1 full "$FULL"; rust chain chain 1 full "$FULL";;
  mega8) kimvoc
    mega mega5 $H5 8 ex30 "$EX" cpu; mega mega4_inst $H4 8 ex30 "$EX" cpu --minus "$WK/kim/melroformer/ex30/vocals.wav"
    mega mega1 guitar 8 ex30 "$EX" cpu; mega mega5 $H5 8 full "$FULL" cpu;;
  mega1) mega mega5 $H5 1 ex30 "$EX" cpu; mega mega1 guitar 1 ex30 "$EX" cpu;;
  mega1full) mega mega5 $H5 1 full "$FULL" cpu;;
  gpu) kimvoc
    for x in "ex30:$EX" "full:$FULL"; do mega mega5 $H5 2 "${x%%:*}" "${x#*:}" cuda; mega mega1 guitar 2 "${x%%:*}" "${x#*:}" cuda; done
    mega mega4_inst $H4 2 ex30 "$EX" cuda --minus "$WK/kim/melroformer/ex30/vocals.wav";;
esac; done
