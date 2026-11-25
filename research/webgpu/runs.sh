#!/bin/bash
# Every export and browser run behind research/webgpu/webgpu.md, in order. Scratch: $W (models, results, venv, node_modules).
#   setup:  uv venv --python 3.11.17 $W/venv && VIRTUAL_ENV=$W/venv uv pip install -r <the list in webgpu.md>
#           (cd $W && npm install onnxruntime-web@1.31.1 puppeteer-core)
#           node research/webgpu/serve.mjs $W 8791 &        (127.0.0.1 only)
#   run:    lockf -k /private/tmp/claude-503/heavy.lock research/webgpu/runs.sh export|bench|song
set -e
W=${W:-/Users/divyansh/work/mac-testing/webgpu}; H=$(cd "$(dirname "$0")" && pwd)
PY=$W/venv/bin/python; E="$PY -I $H/export_onnx.py"
MSST=/Users/divyansh/work/mac-testing/Music-Source-Separation-Training; MD=/Users/divyansh/htdemucs-rs/research/round4/data/models
KIM=/Users/divyansh/htdemucs-rs/research/round1/data/ext/kim; KCK=/Users/divyansh/htdemucs-rs/research/round1/data/models/MelBandRoformer.ckpt
WAV="/Users/divyansh/datasets/musdb18hq/test/Al James - Schoolboy Facination/mixture.wav"
case $1 in
export)
  $E mega $MSST $MD/mega53.ckpt $MD/mega53.yaml "$WAV" $W/models --budget-mb 1024 --name mega4_20s_b1024
  $E mega $MSST $MD/mega53.ckpt $MD/mega53.yaml "$WAV" $W/models --budget-mb 256 --name mega4_20s_b256 --no-ort
  $E mega $MSST $MD/mega53.ckpt $MD/mega53.yaml "$WAV" $W/models --budget-mb 256 --chunk-s 10 --name mega4_10s_b256 --no-ort
  $E kim $KIM $KCK - "$WAV" $W/models --budget-mb 1024 --name kim_8s_b1024
  $E kim $KIM $KCK - "$WAV" $W/models --budget-mb 256 --name kim_8s_b256 --no-ort
  for n in mega4_20s_b1024 mega4_20s_b256 mega4_10s_b256 kim_8s_b1024 kim_8s_b256; do
    $PY -I $H/dump_audio.py $W/models $n
    $PY -I $H/convert_f16.py $W/models $n fp16 fp16n w16 q8   # (as run: fp16n / q8 for the b256 exports)
  done ;;
bench)
  B=$H/bench_all.sh
  $B 'model=mega4_20s_b1024&ep=webgpu&runs=3&dsp=1&save=1&tag=mega20_fp32_native' 'model=mega4_20s_b1024&ep=webgpu&runs=3&bundle=jsep&tag=mega20_fp32_jsep'
  $B 'model=mega4_20s_b1024_fp16&ep=webgpu&runs=3&dsp=1&save=1&tag=mega20_fp16' 'model=mega4_20s_b1024_q8&ep=webgpu&runs=2&dsp=1&save=1&tag=mega20_q8' \
     'model=kim_8s_b1024&ep=webgpu&runs=3&dsp=1&save=1&tag=kim8_fp32' 'model=kim_8s_b1024_fp16&ep=webgpu&runs=3&dsp=1&save=1&tag=kim8_fp16' \
     'model=kim_8s_b1024_q8&ep=webgpu&runs=2&dsp=1&save=1&tag=kim8_q8'
  $B 'model=mega4_20s_b256&ep=webgpu&runs=3&dsp=1&save=1&tag=mega20_b256_fp32' 'model=mega4_10s_b256&ep=webgpu&runs=3&dsp=1&save=1&tag=mega10_b256_fp32' \
     'model=kim_8s_b256&ep=webgpu&runs=3&dsp=1&tag=kim8_b256_fp32'
  $B 'model=kim_8s_b1024_fp16n&ep=webgpu&runs=3&dsp=1&save=1&tag=kim8_fp16n' 'model=mega4_20s_b1024_fp16n&ep=webgpu&runs=3&dsp=1&save=1&tag=mega20_fp16n' \
     'model=mega4_20s_b1024_fp16n&ep=webgpu&runs=2&dsp=1&save=1&f32acc=1&tag=mega20_fp16n_acc' \
     'model=mega4_10s_b256&ep=webgpu&runs=2&cache=disabled&tag=mega10_b256_nocache' 'model=mega4_10s_b256&ep=webgpu&runs=2&cache=lazyRelease&tag=mega10_b256_lazy' \
     'model=kim_8s_b256&ep=webgpu&runs=2&cache=lazyRelease&tag=kim8_b256_lazy' \
     'model=mega4_10s_b256_fp16n&ep=webgpu&runs=3&dsp=1&save=1&tag=mega10_b256_fp16n' 'model=kim_8s_b256_fp16n&ep=webgpu&runs=3&dsp=1&save=1&tag=kim8_b256_fp16n' \
     'model=mega4_10s_b256_fp16n&ep=webgpu&runs=2&cache=lazyRelease&tag=mega10_b256_fp16n_lazy' 'model=kim_8s_b256_fp16n&ep=webgpu&runs=2&cache=lazyRelease&tag=kim8_b256_fp16n_lazy'
  $B 'model=kim_8s_b256&ep=wasm&threads=8&runs=1&tag=kim8_wasm8' 'model=mega4_10s_b256&ep=wasm&threads=8&runs=1&tag=mega10_wasm8' ;;
song)   # whole song; $W/models/song.wav -> a local MUSDB mixture (Cristina Vane - So Easy)
  for t in 'kim=kim_8s_b256_fp16n&mega=mega4_10s_b256_fp16n&cache=lazyRelease&tag=song_fp16n' 'kim=kim_8s_b256&mega=mega4_10s_b256&cache=lazyRelease&tag=song_fp32'; do
    NODE_PATH=$W/node_modules node $H/run.mjs "$t" --page song --port ${PORT:-8791}
  done ;;
esac
# quality: $PY -I $H/analyze.py $W/models $W/results "<musdb song dir>" <export name> <tag> ...
#          $PY -I $H/analyze_song.py $W/results <tag> "<musdb song dir>"
