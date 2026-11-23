# htdemucs-rs

Music stem separation (HTDemucs v4) in Rust: the same model and weights as Meta's Demucs, without
Python or PyTorch. Built for Fretboard's song pipeline, where it replaces `demucs` (htdemucs and
htdemucs_6s, 2 seeded shifts).

Priorities, in order: quality (match PyTorch Demucs), speed, build size.

**Status:** `htdemucs-rs separate` separates whole songs with `htdemucs` (4 stems) and
`htdemucs_6s` (6 stems), reproducing `demucs.separate` - including its seeded random shifts - to
well above 100 dB SNR per stem against PyTorch CPU float32 (see Parity).

## Usage

```sh
cargo build --release                         # Linux / any CPU
cargo build --release --features accelerate   # macOS: Accelerate BLAS (faster)
htdemucs-rs separate -n htdemucs_6s --shifts 2 --seed 0 --flac --clip-mode clamp -o out song.m4a
# -> out/htdemucs_6s/song/{drums,bass,other,vocals,guitar,piano}.flac
```

Same options and defaults as demucs where they exist (`-n`, `--shifts` (1), `--overlap` (0.25),
`-o` (separated), `--float32`, `--flac`, `--clip-mode` rescale|clamp, output layout), plus
`--seed` (default 0; equals `random.seed(S); torch.manual_seed(S)` before demucs, as Fretboard's
`demucs_seeded.py` does), `--threads`, `--weights DIR` (or `$HTDEMUCS_WEIGHTS`), and
`--clip-mode none`. No `--mp3`: write FLAC/WAV and encode afterwards.

Input: WAV (PCM / float), FLAC, MP3, AAC in MP4/M4A, decoded in pure Rust (symphonia), converted
like demucs (to 44.1 kHz only if needed, mono duplicated, more than 2 channels: the first two).
MP4 edit lists are applied as ffmpeg does (symphonia ignores them), so an M4A decodes to the
same length and alignment as ffmpeg's. Output 16-bit samples are quantised exactly like demucs'
torchaudio/libsndfile writer (WAV: top 16 bits of round(x * 2^31); FLAC: round-half-even of
x * 32768).

Reproducing demucs' shifts exactly needs Python's own random numbers: `src/pyrandom.rs` is
CPython 3.11's MT19937 with `random.seed(int)`, `getrandbits`, `randrange` and `randint`. Each
shift draws `randint(0, 22050)`, and every segment forward in between draws `randrange(1)`
(the transformer's positional-embedding shift), which returns 0 but consumes generator output -
so the second offset depends on the first pass's segment count. With seed 0 the first offset is
always 12623.

Memory: one segment's activations at a time, plus the output buffers.

## How it is built, and why

The whole model is ten kinds of operation: (transposed) convolutions, GroupNorm / LayerNorm,
GELU, GLU, linear layers, multi-head attention, an STFT and its inverse. Nearly every FLOP is a
single-precision matrix product. So the crate is **hand-written kernels over a GEMM**:

- convolutions are im2col + GEMM (stride-1 1-D ones as one tap-major GEMM + shift-add), transposed
  convolutions GEMM + col2im, attention and linear layers plain GEMMs;
- the GEMM is the pure-Rust [`gemm`](https://crates.io/crates/gemm) crate (runtime dispatch to
  AVX2/FMA/AVX-512 on x86-64, NEON on ARM, multithreaded with rayon), or Apple's Accelerate BLAS
  (the AMX units) with `--features accelerate` on macOS;
- STFT / iSTFT with [`realfft`](https://crates.io/crates/realfft), in f64.

Why not a framework, given the priorities:

1. **Quality.** Owning every kernel means owning every rounding decision: normalisation statistics
   are two-pass f64, GELU is the exact erf form, the STFT reproduces `torch.stft`/`istft`'s padding
   and normalisation exactly. Result: every layer within 1e-5 relative of PyTorch, and outputs that
   are bit-identical whatever the thread count. A framework port adds a second set of numerics to
   debug (and burn / candle / ONNX have no complex STFT in the graph anyway).
2. **Speed.** The deployment target is a Linux x86-64 CPU container. There, candle and burn's CPU
   backends end in the same kind of GEMM (candle literally uses the `gemm` crate) plus framework
   overhead and unfused elementwise passes; tract is slower on large convolutions; ONNX Runtime
   is fast but a C++ dependency. Hand-fusing the hot elementwise chains (DConv's
   GroupNorm-GELU and GroupNorm-GLU-LayerScale-residual) is what made the biggest difference after
   the GEMM itself.
3. **Size.** No C/C++ dependencies, ~60 small crates; the release binaries are **~1.0 MB**
   (LTO, stripped). Weights are separate files.

What this gives up: a GPU backend. On the Mac runner PyTorch's MPS backend is ~7x its CPU, so a
Metal path (for the GEMMs and attention) is the obvious later addition if Mac speed matters; the
op set is small enough to add one behind the same functions.

## Layout

```
tools/convert.py     .th checkpoint -> weights/<name>.safetensors (float16, lossless) + <name>.json
tools/harness.py     PyTorch reference: one segment fed exactly as demucs.separate + apply_model,
                     dumping input, ~150 intermediate activations and the output (safetensors)
tools/dump_refs.sh   the reference segments used by the parity test (into ref/, gitignored)
tools/bench_torch.py PyTorch's segment forward time, for comparison
src/model.rs         HTDemucs forward (encoders, decoders, frequency embedding, branch I/O)
src/transformer.rs   the cross-domain transformer
src/stft.rs          HTDemucs' _spec / _ispec
src/ops.rs           kernels (GEMM wrapper, conv, conv-transpose, norms, activations, attention)
src/main.rs          the `htdemucs-rs separate` CLI
src/apply.rs         demucs' separate.py + apply_model: normalisation, shifts, segments, overlap-add
src/pyrandom.rs      CPython-compatible random (MT19937, randint / randrange)
src/audio.rs         decode (symphonia + MP4 edit lists), resample, WAV / FLAC writers
src/bin/parity.rs    per-layer comparison with the reference dumps
src/bin/compare.rs   full-song stems vs a PyTorch reference
src/bin/decodecmp.rs our decode vs ffmpeg's
tools/ref_separate.py, ref_songs.sh   full-song PyTorch references (demucs' own code path)
tools/parity_songs.sh, decode_effect.sh, docker_x86.sh   full-song parity, decoder effect, x86 check
src/bin/bench.rs     segment timing (+ HTDEMUCS_PROFILE=1 per-layer profile)
```

## Reproducing

The Python side uses Fretboard's pinned Demucs environment (demucs 4.0.1, torch 2.2.2):

```sh
PY="uv run --quiet --python 3.11.17 --with-requirements ../guitar-tabs/scripts/requirements-demucs.txt python"
$PY tools/convert.py htdemucs htdemucs_6s          # -> weights/
tools/dump_refs.sh                                  # -> ref/*.safetensors (~1.8 GB each)
cargo run --release --bin parity -- weights ref/*.safetensors
cargo run --release --features accelerate --bin bench -- weights htdemucs
```

Checkpoints (from demucs' own `remote/*.yaml` + `files.txt`): `htdemucs` = `955717e8-8726e21a.th`
(84 MB, 41.98 M parameters), `htdemucs_6s` = `5c90dfd2-34c22ccb.th` (55 MB, 27.41 M). Demucs
ships them in float16 and upcasts on load, so the converted safetensors are float16 too: 84.0 MB
and 54.9 MB.

## Parity, one segment (vs PyTorch CPU float32)

| segment | htdemucs | htdemucs_6s |
|---|---|---|
| System of a Down, Aerials: chunk 10 | 119.7 dB | 120.3 dB |
| Oasis, Wonderwall: chunk 5 | 126.8 dB | 126.2 dB |
| Three Days Grace, Never Too Late: last (padded) chunk of a pass shifted by 12345 samples | 130.6 dB | 128.0 dB |

`parity` prints every layer (~150 per model). All are above 100 dB except the first spectral
self-attention of htdemucs (84-95 dB): its softmax logits reach ~480 with row ranges ~720, so it
amplifies the 1e-6-level difference already in its input. Run alone on PyTorch's own input
(`parity` also does this for every transformer layer) it matches at 119 dB - better than
PyTorch's own float32 vs float64 difference for that layer, 116 dB - and the error does not
survive to the output.

Speed, one segment on an M1 Pro (8 threads, quiet machine): PyTorch 2.2.2 CPU 3.4 s (htdemucs) / 3.0 s
(htdemucs_6s); this crate 1.8 s / 1.7 s with the `gemm` crate, 1.3 s / 1.1 s with Accelerate.

## Parity, full songs (vs PyTorch CPU float32, demucs' own code path)

Fretboard's four test songs, decoded once by ffmpeg to float WAV (the same file feeds both),
seed 0, SNR in dB per stem over the whole song (`tools/ref_songs.sh`, `tools/parity_songs.sh`,
Accelerate build; the gemm build gives the same to within ~1 dB).

| htdemucs | shifts | drums | bass | other | vocals |
|---|---|---|---|---|---|
| Aerials | 2 | 116.1 | 122.0 | 125.4 | 129.4 |
| Aerials | 0 | 113.2 | 118.7 | 122.6 | 126.9 |
| The Man Who Sold the World | 2 | 119.3 | 122.1 | 125.5 | 128.0 |
| The Man Who Sold the World | 0 | 116.7 | 119.4 | 122.7 | 125.0 |
| Never Too Late | 2 | 118.7 | 125.1 | 124.2 | 129.0 |
| Never Too Late | 0 | 115.7 | 122.3 | 121.3 | 126.3 |
| Iss Tarah | 2 | 118.4 | 124.6 | 123.8 | 128.0 |
| Iss Tarah | 0 | 116.0 | 121.4 | 120.4 | 125.5 |

| htdemucs_6s | shifts | drums | bass | other | vocals | guitar | piano |
|---|---|---|---|---|---|---|---|
| Aerials | 2 | 124.0 | 120.9 | 111.8 | 128.8 | 124.4 | 101.9 |
| Aerials | 0 | 121.3 | 117.9 | 113.8 | 126.3 | 122.2 | 94.3 |
| The Man Who Sold the World | 2 | 120.9 | 120.6 | 109.2 | 126.5 | 123.4 | 94.7 |
| The Man Who Sold the World | 0 | 119.0 | 117.8 | 112.0 | 124.7 | 120.8 | 95.0 |
| Never Too Late | 2 | 125.7 | 124.8 | 108.3 | 128.4 | 124.9 | 100.0 |
| Never Too Late | 0 | 122.9 | 122.0 | 107.4 | 125.9 | 122.2 | 103.8 |
| Iss Tarah | 2 | 122.7 | 125.4 | 121.0 | 126.5 | 120.4 | 118.3 |
| Iss Tarah | 0 | 119.6 | 122.0 | 117.9 | 124.0 | 116.4 | 116.7 |

The absolute error is the same everywhere, -134 to -160 dBFS RMS (max abs error 0.4-10 x 1e-6);
SNR is relative to each stem's own level, so the lower figures are near-silent stems: the 6-stem
"piano" on the three rock songs sits at -49 to -67 dBFS RMS (e.g. The Man Who Sold the World,
shifts 2: piano at -67 dBFS, 94.7 dB SNR, error -162 dBFS, max abs 5.8e-7), and "other" at
-34 to -42 dBFS there.

Decoder: symphonia (+ edit lists) vs ffmpeg on the four m4a files: same length, aligned, 127-134
dB SNR. Stems from the m4a decoded by us vs the PyTorch reference from ffmpeg's decode
(htdemucs, shifts 2): 115.3-127.9 dB, within ~1-3 dB of the same-input figures - the decoder
choice does not matter at this level.

## Speed and memory, full song (indicative)

Aerials (243.6 s), htdemucs, shifts 2, M1 Pro, whole process (`tools/speed.sh`). The machine was
shared with other heavy jobs; the 1-minute load average (8 cores) is given for each run, and
CPU time is the less load-sensitive number. A clean benchmark is still to do.

| | threads | wall | CPU time | peak RSS | load |
|---|---|---|---|---|---|
| Rust, Accelerate | 8 | 168 s | 347 s | 1.5 GB | 12-14 |
| Rust, Accelerate | 4 | 196 s | 314 s | 1.4 GB | 12-17 |
| Rust, gemm crate | 8 | 252 s | 742 s | 1.2 GB | 12-15 |
| Rust, gemm crate | 4 | 285 s | 639 s | 1.7 GB | 10-12 |
| PyTorch 2.2.2 CPU | 8 | 400 s | 1402 s | 2.2 GB | 15-17 |
| PyTorch 2.2.2 CPU | 4 | 350 s | 1077 s | 1.8 GB | 10-11 |

Release binary (`htdemucs-rs`, LTO, stripped, with decoders, resampler and FLAC encoder): 2.2 MB
on macOS. Weights: 84 MB / 55 MB.

## Demucs behaviour worth knowing

- `separate.py` standardises the whole track first: `ref = wav.mean(0)`, `wav = (wav -
  ref.mean()) / ref.std()` (unbiased std), and undoes it on the output.
- The model then standardises again inside: the CaC spectrogram by its own mean / std (unbiased,
  `+1e-5` on the std) and the waveform branch likewise; outputs are de-standardised with them.
- Segments are always the training length (343980 samples = 39/5 s). `TensorChunk.padded`
  centres a short chunk and fills the padding from real neighbouring audio where it exists.
- The transformer's GroupNorm "norm_out" normalises over the whole token sequence and all
  channels, not per token.
- The frequency branch's DConv runs on `(B*F, C, T)`: its GroupNorms normalise each frequency
  row separately.

## Credits

HTDemucs is by Simon Rouard, Francisco Massa and Alexandre Défossez (Meta AI),
[facebookresearch/demucs](https://github.com/facebookresearch/demucs), MIT licence; this is an
independent reimplementation of its inference code from the published source, and loads its
pretrained weights. The weights are not in this repository; their terms are separate from the
code's (see demucs issue #327).
