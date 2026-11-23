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

`-n ensemble` runs htdemucs_6s and htdemucs on one decode and averages them (vocals, drums,
bass: the mean; 4-stem other: the mean of htdemucs' other and 6s' guitar + piano + other; the
6-stem set keeps 6s' guitar and piano with other = that mean minus them, so both views agree and
the stems sum as closely to the mix as each model does - `src/ensemble.rs`). It writes
`DIR/htdemucs/<track>/` (4 stems, clip mode `--clip-mode-4`, default `--clip-mode`) and
`DIR/htdemucs_6s/<track>/` (6 stems); each model gets its own seeded generator, so its part is
exactly what its own run would give. Research round 2 scored it at shifts 0 above htdemucs at
shifts 2 on every MUSDB stem at about half the CPU of running both models at shifts 2.

Input: WAV (PCM / float), FLAC, MP3, AAC-LC in MP4/M4A, decoded in pure Rust (symphonia), converted
like demucs (to 44.1 kHz only if needed, mono duplicated, more than 2 channels: the first two).
MP4 edit lists are applied as ffmpeg does (symphonia ignores them), so an M4A decodes to the
same length and alignment as ffmpeg's. Output 16-bit samples are quantised exactly like demucs'
torchaudio/libsndfile writer (WAV: top 16 bits of round(x * 2^31); FLAC: round-half-even of
x * 32768). **HE-AAC / HE-AACv2** (SBR / PS) is refused with an error: symphonia decodes only
the AAC core at half the rate, which would give wrong, late stems; decode such files with ffmpeg
first (Fretboard always hands over ffmpeg's float WAV). Silent (or constant) input gives silent
stems and a warning (demucs fails there: it divides by a zero standard deviation). Several
tracks in one call continue one random sequence, as demucs.separate does.

Reproducing demucs' shifts exactly needs Python's own random numbers: `src/pyrandom.rs` is
CPython 3.11's MT19937 with `random.seed(int)`, `getrandbits`, `randrange` and `randint`. Each
shift draws `randint(0, 22050)`, and every segment forward in between draws `randrange(1)`
(the transformer's positional-embedding shift), which returns 0 but consumes generator output -
so the second offset depends on the first pass's segment count. With seed 0 the first offset is
always 12623.

Memory: one segment's activations at a time, plus the output buffer (sources x 2 x length) and
the decoded input. Each shift pass is overlap-added into the output as it goes (a one-segment
window), and the normalised / padded mix is read on demand, never copied - in the same order of
operations as before, so the output is bit-identical. See *Speed and memory* for peak RSS.

**Determinism.** Output is bit-identical whatever the thread count (every parallel reduction is
summed in a fixed order, and the GEMMs never split their inner dimension), and run to run. It is
**not** bit-identical across CPU types: x86-64 picks AVX2+FMA or AVX-512 kernels at run time
(and macOS can use Accelerate), whose roundings differ - e.g. the same commit's synthetic-clip
parity figure differs by about 2 dB between an AVX2 (Zen 3) and an AVX-512 (Zen 5) runner. All
of them stay 110+ dB from PyTorch; but stems made on different hosts differ in their last bits.

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
3. **Size.** No C/C++ dependencies, ~60 small crates; the `htdemucs-rs` release binary is
   **2.3 MB** on macOS and **4.2 MB** on Linux x86-64 (LTO, stripped; with the audio decoders,
   resampler and FLAC encoder). Weights are separate files (84 MB / 55 MB).

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
seed 0, SNR in dB per stem over the whole song. Measured with the **default build** (the `gemm`
crate: the Linux code path; here its NEON kernels) at `f23ab23`, 4 threads
(`tools/parity_table.sh`); the earlier Accelerate-build table agreed to within ~4 dB everywhere.

| htdemucs | shifts | drums | bass | other | vocals |
|---|---|---|---|---|---|
| Aerials | 2 | 116.0 | 120.5 | 124.8 | 128.6 |
| Aerials | 0 | 113.1 | 118.1 | 122.2 | 126.2 |
| The Man Who Sold the World | 2 | 119.1 | 121.5 | 125.0 | 128.0 |
| The Man Who Sold the World | 0 | 116.3 | 118.7 | 122.3 | 125.1 |
| Never Too Late | 2 | 118.6 | 124.9 | 123.9 | 128.4 |
| Never Too Late | 0 | 115.6 | 122.1 | 121.0 | 125.6 |
| Iss Tarah | 2 | 118.4 | 124.0 | 123.4 | 127.7 |
| Iss Tarah | 0 | 116.2 | 122.0 | 120.3 | 124.8 |

| htdemucs_6s | shifts | drums | bass | other | vocals | guitar | piano |
|---|---|---|---|---|---|---|---|
| Aerials | 2 | 123.4 | 120.0 | 113.1 | 128.0 | 124.2 | 100.2 |
| Aerials | 0 | 120.6 | 117.1 | 111.2 | 125.2 | 121.5 | 93.8 |
| The Man Who Sold the World | 2 | 119.4 | 119.4 | 104.8 | 126.5 | 122.4 | 93.3 |
| The Man Who Sold the World | 0 | 118.2 | 117.0 | 106.7 | 123.9 | 120.1 | 95.3 |
| Never Too Late | 2 | 125.0 | 124.3 | 108.8 | 127.9 | 124.4 | 102.5 |
| Never Too Late | 0 | 122.3 | 121.6 | 107.9 | 125.3 | 121.7 | 100.0 |
| Iss Tarah | 2 | 121.8 | 124.5 | 119.8 | 126.0 | 117.6 | 118.5 |
| Iss Tarah | 0 | 118.8 | 121.4 | 118.5 | 124.3 | 116.5 | 116.4 |

The absolute error is the same everywhere, -134 to -160 dBFS RMS (max abs error 0.4-12 x 1e-6);
SNR is relative to each stem's own level, so the lower figures are near-silent stems: the 6-stem
"piano" on the three rock songs sits at -49 to -67 dBFS RMS (e.g. The Man Who Sold the World,
shifts 2: piano at -67 dBFS, 93.3 dB SNR, error -160 dBFS), and "other" at
-34 to -42 dBFS there.

Decoder: symphonia (+ edit lists) vs ffmpeg on the four m4a files: same length, aligned, 127-134
dB SNR. Stems from the m4a decoded by us vs the PyTorch reference from ffmpeg's decode
(htdemucs, shifts 2): 115.3-127.9 dB, within ~1-3 dB of the same-input figures - the decoder
choice does not matter at this level.

## Speed and memory

**x86-64** (GitHub Actions runners, `.github/workflows/bench.yml`; 2 vCPUs = one core with SMT,
so this is single-core cost; multi-core x86 scaling is not measured yet). CPU seconds per second
of audio for one segment forward, 1 thread, at `dff89f6`:

| CPU | htdemucs-rs | PyTorch 2.2.2 | |
|---|---|---|---|
| AMD EPYC 7763 (Zen 3, AVX2) | 0.72-0.74 | 0.76 | **parity** (within run-to-run noise) |
| AMD EPYC 9V45 (Zen 5, AVX-512) | 0.33 | 0.51 | **about 36% less CPU** |

A whole song costs about 2.7x that per second of music (overlap 0.25: 7.8 s segments every
5.85 s; 2 shifts). No Intel or Zen 4 PyTorch baseline yet.

**Apple M1 Pro**, Aerials (243.6 s), htdemucs, shifts 2, whole process (`tools/speed.sh`). The machine was
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

Release binary (`htdemucs-rs`, LTO, stripped, with decoders, resampler and FLAC encoder): 2.3 MB
on macOS, 4.2 MB on Linux x86-64. Weights: 84 MB / 55 MB.

Peak memory, M1 Pro, 4 threads, Accelerate build (`peak memory footprint` from `/usr/bin/time -l`:
what the process holds; macOS's "maximum resident set size" is lower when the system compresses
memory under load, so it is not used here), before and after streaming the overlap-add
(`4072d30`), on Aerials and on a 22-minute track (Aerials looped; Fretboard's length limit; 1
shift - a second pass adds time, not memory):

| | Aerials, htdemucs, 2 shifts | Aerials, htdemucs_6s, 2 shifts | 22 min, htdemucs | 22 min, htdemucs_6s |
|---|---|---|---|---|
| before | 1.31 GB | 1.40 GB | 4.67 GB | 6.41 GB |
| after | 0.93 GB | 0.91 GB | 2.53 GB | 3.55 GB |

What remains is mostly the output itself (sources x 2 x length float32: 2.8 GB for 6 stems at
22 min), the decoded input (0.47 GB) and the weights.

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
