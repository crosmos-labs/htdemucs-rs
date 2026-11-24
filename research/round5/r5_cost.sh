#!/bin/bash
# Round 5 CPU cost on the Mac (Apple M1 Pro: 6 performance + 2 efficiency cores, 16 GB), one job at a time under the
# shared heavy lock. Every pass is timed by /usr/bin/time -l (user + sys CPU seconds, wall, peak RSS) and appended as a
# JSON line to scores/cost.jsonl.   r5_cost.sh <tag> [<tag> ...]   tags: rust | mega | mega1 (1-thread Mega, slow)
# Inputs: the 30 s excerpt = the first 30 s of cmt8/AngelsInAmplifiers_ImAlright (its 60-90 s); the full song =
# MUSDB18-HQ test "AM Contra - Heart Peripheral" (210.1 s).
set -u
R=$(cd "$(dirname "$0")/../.." && pwd); OUT=$R/research/round5/scores/cost.jsonl; mkdir -p "$(dirname "$OUT")"
W=${W:-/Users/divyansh/work/mac-testing/r5cost}; mkdir -p "$W"
PY=${PY:-/Users/divyansh/work/mac-testing/r5env/bin/python}; MSST=${MSST:-/Users/divyansh/work/mac-testing/Music-Source-Separation-Training}
LOCK=/private/tmp/claude-503/heavy.lock; BIN=$R/target/release/htdemucs-rs; M=$R/research/round4/data/models
export HTDEMUCS_WEIGHTS=$R/weights MELROFORMER_WEIGHTS=melroformer-f16
EX=$W/ex30.wav; FULL="/Users/divyansh/datasets/musdb18hq/test/AM Contra - Heart Peripheral/mixture.wav"
[ -f "$EX" ] || "$PY" -I -c "import soundfile as sf; x,sr=sf.read('/Users/divyansh/datasets/r4/cmt8/AngelsInAmplifiers_ImAlright/mixture.wav',dtype='float32'); sf.write('$EX',x[:30*sr],sr,subtype='FLOAT')"
dur() { "$PY" -I -c "import soundfile as sf,sys; print(sf.info(sys.argv[1]).duration)" "$1"; }

run() {  # run <pass> <threads> <input name> <input> <cmd...>
  local pass=$1 th=$2 iname=$3 in=$4; shift 4
  local t=$W/time.txt log=$W/run.log
  lockf -k $LOCK /usr/bin/time -l "$@" > "$log" 2> "$t" || { echo "FAIL $pass $th $iname"; tail -3 "$t"; return; }
  "$PY" -I - "$pass" "$th" "$iname" "$(dur "$in")" "$t" "$log" >> "$OUT" <<'EOF'
import json, re, sys
p, th, iname, d, t, log = sys.argv[1:]; s = open(t).read(); d = float(d)
real, user, sys_ = (float(x) for x in re.search(r'([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys', s).groups())
rss = int(re.search(r'(\d+)\s+maximum resident set size', s).group(1))
r = {'pass': p, 'threads': int(th), 'input': iname, 'audio_s': round(d, 2), 'cpu_s': round(user + sys_, 2), 'wall_s': real,
     'cpu_per_audio_s': round((user + sys_) / d, 3), 'wall_per_audio_s': round(real / d, 3), 'peak_rss_gb': round(rss / 2**30, 2)}
for line in open(log):   # r5_mega.py's own JSON (per-chunk CPU → long-track rate)
    if line.startswith('{'): j = json.loads(line); r['mega_rate_long'] = j['rate_long']; r['mega_chunks'] = j['chunks']
print(json.dumps(r)); sys.stderr.write(json.dumps(r) + '\n')
EOF
}

rust() {  # <pass> <model> <threads> <iname> <input>
  rm -rf "$W/o"; run "$1" "$3" "$4" "$5" "$BIN" separate -n "$2" --threads "$3" --float32 --clip-mode none -q -o "$W/o" "$5"
}
mega() {  # <pass> <heads> <threads> <iname> <input> [--minus vocals.wav]
  local p=$1 h=$2 th=$3 iname=$4 in=$5; shift 5
  run "$p" "$th" "$iname" "$in" "$PY" -I "$R/research/round5/r5_mega.py" "$MSST" $M/mega53.ckpt $M/mega53.yaml "$in" "$W/m.npy" \
    --stems "$h" --device cpu --threads "$th" "$@"
}
for tag in "$@"; do case $tag in
  rust)
    for th in 8 1; do for x in "ex30:$EX" "full:$FULL"; do
      rust hyb6 ensemble $th "${x%%:*}" "${x#*:}"; rust chain chain $th "${x%%:*}" "${x#*:}"; rust kim melroformer $th "${x%%:*}" "${x#*:}"
    done; done;;
  mega)   # 5 heads on the mix (mega_mix), 4 heads on Kim's instrumental (kim_mega), 1 head (guitar) for the per-head increment
    rm -rf "$W/o"; lockf -k $LOCK "$BIN" separate -n melroformer --float32 --clip-mode none -q -o "$W/kim" "$EX"
    for th in 8; do
      mega mega5 vocal,drums,bass,guitar,piano $th ex30 "$EX"
      mega mega4_inst drums,bass,guitar,piano $th ex30 "$EX" --minus "$W/kim/melroformer/ex30/vocals.wav"
      mega mega1 guitar $th ex30 "$EX"
      mega mega5 vocal,drums,bass,guitar,piano $th full "$FULL"
    done;;
  mega1)
    mega mega5 vocal,drums,bass,guitar,piano 1 ex30 "$EX"
    mega mega1 guitar 1 ex30 "$EX";;
  mega1full)
    mega mega5 vocal,drums,bass,guitar,piano 1 full "$FULL";;
esac; done
