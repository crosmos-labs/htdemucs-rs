# htdemucs-rs

Music stem separation (HTDemucs v4) in Rust: the same model and weights as Meta's Demucs, without
Python or PyTorch. Built for Fretboard's song pipeline, where it replaces `demucs` (htdemucs and
htdemucs_6s, 2 seeded shifts).

Priorities, in order: quality (match PyTorch Demucs), speed, build size.

**Status:** `htdemucs-rs separate` separates whole songs with `htdemucs` (4 stems) and
`htdemucs_6s` (6 stems), reproducing `demucs.separate` - including its seeded random shifts - to
well above 100 dB SNR per stem against PyTorch CPU float32 (see Parity). `-n melroformer` runs
Kim's Mel-Band RoFormer vocal model (vocals + instrumental; +3.7 dB vocal SDR over htdemucs on
MUSDB18-HQ), matching its PyTorch inference to 101-135 dB per stem on six full tracks with 37-60%
less CPU than PyTorch on x86 (see *Mel-Band RoFormer*). `-n chain` runs round 3's best recipe -
Kim's vocals, then the htdemucs + htdemucs_6s ensemble on Kim's instrumental - matching the
PyTorch pipeline to 108-132 dB per stem (see *Chain*). `-n megaguitar` runs the guitar head of
MVSEP's Mega 53 BS-RoFormer (the best guitar stem measured: +2.6 dB over htdemucs_6s on 26/26
Cambridge-MT songs), matching PyTorch to 109.7-119.4 dB (guitar) on a chunk and full tracks with 9-65% less CPU
than PyTorch on x86 (on Apple silicon only with `--features accelerate`; see *Mega 53*).
**`-n kimmega` is the recommended recipe** (research round 5): Kim's vocals, then Mega 53's
drums, bass, guitar and keys heads on Kim's instrumental from one backbone pass, other = the rest -
vocals +2.2, bass +0.9, guitar +2.65, keys +4.5 dB over the Demucs ensemble on 26 Cambridge-MT
songs, drums and other level, all weights MIT; 108-131 dB per stem against the PyTorch recipe on a
full song (see *Kim + Mega 53*). It costs about 2.5x the CPU of `-n chain`.

## Usage

```sh
cargo build --release                         # Linux / any CPU
cargo build --release --features accelerate   # macOS: Accelerate BLAS (faster)
htdemucs-rs separate -n kimmega --flac --clip-mode clamp -o out song.m4a   # recommended
# -> out/kimmega/song/{vocals,drums,bass,guitar,piano,other}.flac
htdemucs-rs separate -n htdemucs_6s --shifts 2 --seed 0 --flac --clip-mode clamp -o out song.m4a
# -> out/htdemucs_6s/song/{drums,bass,other,vocals,guitar,piano}.flac
```

Recipes (`-n`), best quality first (research round 5, 26 Cambridge-MT songs; CPU on x86 at one
thread, full song):

| `-n` | stems | models (weights licence) | quality | CPU-s per audio-s |
|---|---|---|---|---|
| **`kimmega`** | vocals, drums, bass, guitar, piano, other | Kim Mel-RoFormer + Mega 53, 4 heads (MIT) | best on vocals, bass, guitar, keys; level on drums, other | 10.6 |
| `chain` | 4-stem + 6-stem views | Kim (MIT) + htdemucs + htdemucs_6s ("for scientific purposes only", round 1) | vocals as kimmega; guitar -2.2, keys -4.5, bass -0.8 dB | 4.2 |
| `ensemble` | 4-stem + 6-stem views | htdemucs + htdemucs_6s (scientific purposes only) | vocals -2.2 dB vs kimmega | 0.8 |
| `htdemucs` / `htdemucs_6s` | 4 / 6 | one Demucs model | below the ensemble (round 2) | ~0.4 per shift (half the ensemble) |
| `melroformer`, `megaguitar`, `mega` | vocals + rest; guitar + rest; 4 heads + rest | the parts | | |

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

Memory: one segment's activations at a time and the decoded input; nothing else grows with the
track. The normalised / padded mix is read on demand, never copied. The shift passes run side by
side, each overlap-adding into its own one-segment window (`apply::Streamer`; their offsets are
drawn up front, moving the generator exactly as one pass after another would). A position goes
to the writers once every pass (and, for the ensemble, both models) has finalised it, summed in
pass order. The writers (`audio::StemWriter`) encode WAV / FLAC as samples arrive; FLAC frame by
frame, its STREAMINFO rewritten at the end. `--clip-mode rescale` needs each stem's peak, so
those stems go to a float temp file next to the output (4 bytes a sample) and are divided and
encoded at the end. The same operations in the same order as before, so the output is
bit-identical. See *Speed and memory* for peak RSS.

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
src/roformer.rs      Mel-Band RoFormer: config, STFT / iSTFT, forward, upstream demix_track
tools/melroformer.py its converter, PyTorch references (chunk dump, full song, float64), timing
tools/mr_parity_songs.sh   its full-song parity + CPU time vs PyTorch
src/bin/mr_parity.rs, mr_bench.rs   its chunk parity (+ each transformer alone) and chunk timing
src/chain.rs         -n chain: Kim vocals, then the ensemble on the instrumental (streamed)
tools/chain_ref.py   its PyTorch reference (round 3's k+chain_avg on CPU float32)
tools/chain_box.sh, chain_musdb.sh, measure.sh   its parity / uSDR / memory / CPU runs on Linux
.github/workflows/bench-roformer.yml   its x86 parity / speed runs (settings: .github/bench/mr_run.env)
tools/bsroformer.py  Mega 53 BS-RoFormer (-n megaguitar, -n mega): converter (backbone + one head or
                     a list of heads), PyTorch references (chunk dump, full song as r4_mega.py /
                     r5_mega.py, --minus / --recipe for kimmega, float64), timing
src/kimmega.rs       -n kimmega: Kim vocals, then Mega's 4 heads on the instrumental, other = rest
.github/workflows/bench-kimmega.yml   its x86 parity / CPU runs (settings: .github/bench/km_run.env)
.github/workflows/bench-megaguitar.yml   its x86 parity / speed runs (settings: .github/bench/mg_run.env)
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

What remained was mostly the output itself (sources x 2 x length float32: 2.8 GB for 6 stems at
22 min), the decoded input (0.47 GB) and the weights.

Streamed output (`a7c5ba8`): peak RSS on Linux (i5-12450H, gemm build, 8 threads, getrusage of
the child = VmHWM), the previous build (`ff85811`) against `66c76a4`, FLAC, the ensemble with
Fretboard's options (`--clip-mode clamp --clip-mode-4 rescale --ensemble-own-6s`, 16 files), the
single models at the default `rescale`. Every stem is bit-identical (cmp: 78 files, 0 differ):

| | Aerials (4:04) shifts 0 | Aerials shifts 2 | 22 min (`long22`) shifts 0 |
|---|---|---|---|
| htdemucs | 1281 -> **843** MiB | 1301 -> **886** MiB | 4036 -> **1213** MiB |
| htdemucs_6s | 1470 -> **913** MiB | 1491 -> **965** MiB | 4849 -> **1315** MiB |
| ensemble | 1830 -> **1173** MiB | 1994 -> **1271** MiB | 7203 -> **1561** MiB |

What still grows with length is the decoded input (0.35 MB per second: the 22-minute runs are
about 0.37 GB above Aerials). The rescale pass costs 7-10 s at 22 minutes (1-4% of the run) and
4 bytes x 2 channels per sample of temporary disk per rescaled stem (466 MB a stem at 22 min).
`66c76a4` also has the Mel-RoFormer merge's tensor-by-tensor weight loader.

## Mel-Band RoFormer (`-n melroformer`)

Kim Jensen's Mel-Band RoFormer vocal model, the best vocal separator in the research round
(MUSDB18-HQ, 12 test tracks, uSDR: vocals 13.73 dB vs 10.04 for htdemucs at 2 shifts;
accompaniment 18.76 vs 15.08 - `research/round1/round1.md`, round 2).

```sh
PY="uv run --quiet --python 3.11.17 --with-requirements research/round1/kim_env.txt python"
$PY tools/melroformer.py convert <Mel-Band-Roformer-Vocal-Model repo @25f44ff> MelBandRoformer.ckpt
#   (float16; add --float32 for the parity checks' melroformer-f32)
htdemucs-rs separate -n melroformer --flac --clip-mode clamp -o out song.m4a
# -> out/melroformer/song/{vocals,instrumental}.flac
```

**What it computes** (upstream `inference.py` / `utils.demix_track`, CPU, float32): chunks of
352800 samples (8 s) every 176400 (`num_overlap` 2; `--num-overlap N` as upstream), the track
reflect-padded by 4 s at both ends, linear fades of 0.8 s at each chunk's ends (not at the track's
ends), outputs overlap-added and divided by the summed windows. Per chunk: STFT (2048 / hop 441,
centred) -> 60 overlapping mel bands of bins, each RMSNorm + Linear to 384 -> 6 x (time
transformer over each band's 801 frames, frequency transformer over each frame's 60 bands;
RoPE, gated attention, 8 x 64 heads, GELU MLP) -> per-band tanh MLP + GLU -> complex mask,
averaged where bands overlap -> iSTFT. `instrumental` = mix - vocals, as upstream writes it.
There are no shifts or seeds (`--shifts` / `--overlap` are refused). Output formats and clip
modes are the Demucs path's (`--float32`, `--flac`, `--clip-mode`).

**How.** `src/roformer.rs` reuses the crate's GEMM (gemm crate, runtime AVX2 / AVX-512 dispatch)
and its elementwise style. Each transformer runs as tasks of a fixed shape - one band (801 tokens)
for the time transformer, 16 frames (960 tokens) for the frequency transformer, one band for the
mask MLPs - with every GEMM, RMSNorm, rotary, attention tile (256 query rows) and GELU of the task
done on one thread with its intermediates in cache; the frequency tasks gather their tokens and
scatter them back, so the activations stay in one (band, frame, dim) buffer. Results therefore do
not depend on the thread count (checked by `tests/roformer.rs` and `mr_bench` at 1/2/4 threads).
The track streams as in the Demucs path: the padded mix is read on demand, and the overlap-add
lives in a one-chunk window finalised as the chunks advance (`tests`: bit-identical to a literal
whole-buffer `demix_track`).

**Weights.** The checkpoint (huggingface.co/KimberleyJSN/melbandroformer, `MelBandRoformer.ckpt`,
913 MB, sha256 `87201f4d…7559e`; MIT since 2026-04-22, see `research/round1/round1.md` for the
licence history and the open training-data question) is float32 and **not** float16-exact (none
of its 684 tensors; 228.2 M parameters, 88% of them in the 60 mask MLPs). Float16 weights cost
parity - Aerials chunk 10 drops from 128 dB to 72 dB against PyTorch float32 - but not separation
quality: within 0.0004 dB uSDR on every stem of four MUSDB tracks (see *Chain*, float16). So
`convert` writes `weights/melroformer.safetensors` in **float16 (457 MB)**; `convert --float32`
writes `melroformer-f32.safetensors` (913 MB), which the parity checks below load
(`MELROFORMER_WEIGHTS=melroformer-f32`). Weights are widened to float32 on load either way.

### Parity (vs PyTorch 2.2.2 CPU float32, upstream code)

One chunk (`mr_parity`, Aerials chunk 10): output 128.1 dB; every transformer run alone on
PyTorch's own input 126-134 dB; the band split is the lowest intermediate (96.8 dB, see below).

Full songs (`tools/mr_parity_songs.sh`, the same float WAV into both; SNR per stem over the whole
song, then the reference stem's level and the error's level):

| track | vocals | instrumental | levels (vocals / instrumental), error |
|---|---|---|---|
| System of a Down, Aerials (243.6 s) | 126.7 dB | 130.6 dB | -19 / -15 dBFS; error -145 dBFS, max 2.6e-6 |
| Nirvana, The Man Who Sold the World (217.9 s) | 117.5 dB | 125.5 dB | -22 / -14 dBFS; error -140 dBFS, max 6.2e-6 |
| Three Days Grace, Never Too Late (209.3 s) | 128.3 dB | 132.5 dB | -16 / -12 dBFS; error -145 dBFS, max 8.9e-6 |
| Chaar Diwaari, Iss Tarah (323.5 s) | 100.9 dB | 103.6 dB | -14 / -11 dBFS; error -115 dBFS, max 1.9e-4 |
| MUSDB18-HQ test: Signe Jakobsen, What Have You Done To Me (177.4 s) | 131.5 dB | 133.9 dB | -23 / -20 dBFS; error -154 dBFS, max 3.4e-7 |
| MUSDB18-HQ test: Angels In Amplifiers, I'm Alright (179.6 s) | 131.4 dB | 134.5 dB | -23 / -20 dBFS; error -155 dBFS, max 3.4e-7 |

Measured at `0227a00` (the tuned defaults since change only task shapes; chunk parity unchanged at 128.1 dB). The absolute error is -140 to -155 dBFS on five tracks, so the SNR follows each stem's own level; Iss Tarah is the outlier at -115 dBFS (see below).

**Where float32 cannot agree.** Each band's features are L2-normalised (RMSNorm) before the band
projection. A band holding nothing but FFT rounding noise (a synthetic mix with no energy above
some frequency; digital near-silence) is normalised up to unit size, and the noise's direction
is arbitrary - there PyTorch float32 is itself far from PyTorch float64: on the bare CI synthetic
clip, band split 11.1 dB and output 22.8 dB (torch f32 vs f64), against 10.8 / 30.0 dB for this
crate vs f64. Real recordings have a noise floor in every band and agree to 96+ dB at the band
split; the CI clip gets a -60 dBFS noise floor, which brings it to 101.7 dB (band split) and
95.6 dB (chunk output, vocals at -100 dBFS), full path vocals 105.1 dB / instrumental 175.3 dB
(error -220 / -191 dBFS).

The same mechanism, milder, is Iss Tarah's lower figure: its error is -139 dBFS in the median
second but -95 dBFS around 3:56 (chunks 59-60). There (`mr_parity` on chunk 59) the STFT agrees
to 136.5 dB and the band split drops to 75.9 dB (max abs error 0.016 on values up to 7.8) - a
low-energy band's rounding noise scaled up by its normalisation - then propagates (69-81 dB
through the transformers, 109.6 dB at the chunk output), while every transformer run alone on
PyTorch's input stays at 126-134 dB. Nothing in the network itself diverges; the band split's
input is where float32 rounding, ours or PyTorch's, decides.

### Speed and memory

CPU seconds per second of audio **of one chunk** forward, 1 thread (`mr_bench`, `melroformer.py
bench`); a whole track costs about twice that per second of music (every sample is in two
chunks) plus the 4 s of padding at each end.

| CPU | htdemucs-rs | PyTorch 2.2.2 | |
|---|---|---|---|
| AMD EPYC 7763 (Zen 3, AVX2), GitHub runner, `44419c0` | 4.90 | 7.76 | **37% less CPU** |
| AMD EPYC 9V45 (Zen 5, AVX-512), GitHub runner, `f90a933` (first version) | 2.44 | 6.06 | 60% less |
| Intel Xeon Platinum 8573C (Emerald Rapids, AVX-512), GitHub runner, `194a6f8` | 2.80 | 5.30 | **47% less CPU** |
| Apple M1 Pro, gemm crate (NEON) build, loaded machine | 5.9-6.1 | 3.2-3.4 | **1.8x more CPU**: PyTorch's GEMMs use Accelerate (AMX) |

Whole songs on the M1 Pro (4 threads, machine shared with other jobs at load 15-20, so CPU time
is the comparable number): Aerials (243.6 s) 3046-3130 CPU-s here vs 1836 for PyTorch (12.5-12.8 vs 7.5 CPU-s per second of music); over the six parity tracks 12.9-16.3 vs 7.0-7.5 (load varied run to run) - the
`gemm`-crate NEON build reaches ~70 GFLOP/s per core where PyTorch's Accelerate GEMMs use the AMX
units. On x86 the GEMMs run at 90-95% of peak (Zen 3 ~93 GFLOP/s, Zen 5 / Xeon 6975P ~210-230
GFLOP/s per core); they are 80-85% of the time, attention softmax ~6%, GELU / tanh ~5%,
RMSNorm 2%. Two threads on a runner's two vCPUs (one core, SMT) give ~1.06x the throughput (Zen 3), as for PyTorch (1.11x).

Optimisations kept (each measured on the runners): fixed-shape single-threaded GEMM tasks; QKV
and gate projections in one GEMM; exp's integer part read from the float bits (a saturating
`as i32` left every exp in the crate scalar on x86-64 - this also speeds up the Demucs path,
bit-identically); softmax as plain max / exp / sum loops with the exp input clamped at -40 (no
subnormal lanes) and normalisation applied to the 64-wide output instead of the 801-wide weights;
AVX-512 versions of the elementwise kernels; tile sizes from A/B runs. Tried and dropped: K^T
copied per head for the QK^T GEMM (faster GEMM, paid back in copies), 64-row tiles, 8-frame
frequency tasks (1-3% slower).

Memory: weights 0.91 GB (float32) + the decoded input (2 x 4 bytes x samples) + one chunk's
activations (~0.15 GB plus ~10 MB per thread) + a one-chunk overlap-add window; both stems are
written as they are finalised (`audio::StemWriter`, as the Demucs path), so nothing else grows
with the track. The safetensors loader now reads tensor by tensor (it used to hold the whole file
and the tensors at once, ~1.8 GB at load). Aerials, M1 Pro, 4 threads: **1.29 GB peak memory
footprint** (1.44 GB max RSS); before streaming the loader and the output, 1.83 GB. PyTorch:
2.4 GB on Aerials (4 threads); on the 12 s CI clip (Linux) 2.35 GB vs 1.09 GB here.

Binary: 2.5 MB (macOS) / 4.4 MB (Linux x86-64) with both models and the AVX-512 kernels.

## Kim + Mega 53 (`-n kimmega`): the recommended recipe

```sh
PY="uv run --quiet --python 3.11.17 --with-requirements research/round4/r4_msst_env.txt python"
$PY tools/bsroformer.py convert mvsep_mega_model_bs_roformer_53_stems_v1.ckpt mvsep_mega_model_bs_roformer_53_stems.yaml --name mega
# -> weights/mega.safetensors (152 MB, float16: the backbone + the drums, bass, guitar and keys heads)
#    + weights/mega.json; Kim's weights/melroformer.* as in *Mel-Band RoFormer*
htdemucs-rs separate -n kimmega --flac --clip-mode clamp -o out song.m4a      # alias: -n kim_mega_keys
# -> out/kimmega/song/{vocals,drums,bass,guitar,piano,other}.flac
```

Research round 5's pick, `kim_mega_keys` ([research/round5/round5.md](research/round5/round5.md)),
in one command, every model MIT-licensed:

1. **vocals**: Kim's Mel-Band RoFormer (`-n melroformer`, upstream `demix_track`);
2. **drums, bass, guitar, piano**: MVSEP's Mega 53 BS-RoFormer on Kim's instrumental (mix - vocals,
   float32), its backbone run **once** per chunk and four mask heads off it: `drums`, `bass`,
   `guitar` and `keys` (written as `piano`). `keys`, not `piano`: on the 10 round-5 songs with
   keyboards it scores 5.24 dB against 3.33 for the `piano` head (both on the instrumental), because it also catches organ and
   e-piano (Impossible Colours 0.4 -> 8.4 dB), and is within 0.1 dB of `piano` on piano-only songs
   but one (Al James 2.78 vs 3.47);
3. **other** = instrumental - (drums + bass + guitar + keys), summed in float64. Mega has no
   "other" head, so the six stems sum to the mix exactly up to float32 rounding (`-v` prints the
   residual; about -150 dB).

Against the other recipes, on round 5's 26 Cambridge-MT remixes (uSDR, Δ vs `hyb6` = `-n ensemble`
at shifts 0 with its 6-stem view, 95 % CI, items better):

| stem | `-n ensemble` (hyb6) | `-n chain` | **`-n kimmega`** | Δ vs hyb6 | Δ vs chain |
|---|---|---|---|---|---|
| vocals (26) | 9.91 | 12.13 | **12.13** (Kim) | **+2.22** [+1.65, +2.80] 24/26 | same model |
| drums (26) | 9.11 | 9.48 | 9.46 | +0.35 [-0.29, +0.87] 20/26 | -0.02 |
| bass (26) | 8.82 | 8.96 | **9.74** | **+0.92** [+0.45, +1.42] 20/26 | **+0.78** |
| guitar (26) | 4.27 | 4.76 | **6.92** | **+2.65** [+2.04, +3.29] 26/26 | **+2.16** |
| piano vs true keys (10) | 0.73 | 0.77 | **5.24** | **+4.51** [+3.38, +5.71] 10/10 | **+4.47** |
| other (9) | 1.33 | 1.62 | 1.38 | +0.05 [-1.45, +1.40] 7/9 | -0.23 |

Drums and other are ties; see round 5 for the caveats (Mega's `drums` head misses brushes /
percussion on two songs; other is a residual).

Options: `--mega-chunk-size N` / `--mega-num-overlap N` set Mega's chunking (default 882000
samples = 20 s, overlap 2: the reference settings), `--chunk-size` / `--num-overlap` Kim's (8 s,
2), as for `-n melroformer`. No `--shifts` / `--overlap`. Output formats and clip modes as
elsewhere.

**Several heads, one backbone.** `src/roformer.rs` runs a model with N mask heads (`num_stems` N
in its JSON, heads listed in `instruments`): STFT, band split, the 12 transformer layers and the
final norm once per chunk; then per head its 62 band MLPs, the mask, the masked STFT and the
iSTFT. Each head's output is exactly what a one-head model of that head computes (`tests/roformer.rs`:
`mega`'s guitar stem is bit-identical to `-n megaguitar`'s). `separate_stream_n` overlap-adds the N
stems with the same operations in the same order as one stem. `-n mega` runs the four heads
alone on its input (+ `rest` = input - the four), the building block `kimmega` runs on Kim's
instrumental. `tools/bsroformer.py convert --stems a,b,...` exports any list of heads;
`megaguitar` (`convert` with no `--stems`) is unchanged, byte for byte.

**Memory**, as the chain: the instrumental is the one track-long buffer besides the decoded mix
(freed once Kim is done); Kim's weights (0.91 GB as float32) are freed before Mega's (0.30 GB);
all six stems are written as they are finalised.

### Parity (vs PyTorch CPU float32: tools/bsroformer.py, the inference of research/round5/r5_mega.py)

**One chunk, the four heads** (`mr_parity`, chunk 1 of the Al James 30 s excerpt's instrumental -
mix minus this crate's Kim vocals - as `bsroformer.py segment --name mega --minus` feeds it; M1 Pro,
NEON build):

| head | output vs PyTorch float32 | vs PyTorch float64 | PyTorch float32 vs float64 | mask |
|---|---|---|---|---|
| drums | 120.3 dB | 122.2 | 120.4 | 111.5 |
| bass | 119.4 | 120.9 | 120.8 | 111.6 |
| guitar | 110.2 | 111.8 | 110.7 | 107.4 |
| keys (piano) | 111.6 | 112.7 | 112.8 | 107.4 |

The shared part: STFT 136.5 dB, final norm 102.6 dB; each transformer run alone on PyTorch's input
to it 125-126 dB. This crate is as close to a float64 run as PyTorch float32 itself is.

**Full path** (`htdemucs-rs separate -n kimmega` vs `bsroformer.py separate --name mega --minus
<vocals> --recipe`; SNR per stem over the whole file; M1 Pro, NEON build, float32 Kim weights):

| input | reference | vocals | drums | bass | guitar | piano | other |
|---|---|---|---|---|---|---|---|
| Al James excerpt (30 s) | PyTorch Mega on this crate's Kim instrumental | identical | 122.3 | 123.4 | 113.9 | 115.2 | 111.4 |
| | PyTorch end to end (PyTorch Kim, then PyTorch Mega) | 129.2 | 110.8 | 119.5 | 105.7 | 108.2 | 101.6 |
| MUSDB18-HQ Angels In Amplifiers, I'm Alright (179.6 s) | PyTorch Mega on this crate's Kim instrumental | identical | 118.2 | 119.1 | 119.0 | 118.0 | 109.1 |
| | PyTorch end to end | 131.4 | 118.0 | 118.6 | 118.2 | 116.2 | 108.6 |
| synthetic CI clip + -60 dBFS noise (30 s), Blacksmith EPYC (AVX-512) | PyTorch Mega on this crate's Kim instrumental | identical | 117.9 | 109.3 | 98.3 | 100.3 | 108.2 |
| | PyTorch end to end | 97.4 | 116.9 | 108.4 | 98.1 | 99.6 | 107.6 |
| the same clip, M1 Pro | PyTorch Mega on this crate's Kim instrumental | identical | 113.4 | 110.8 | 99.0 | 101.3 | 108.4 |

Errors are -143 to -152 dBFS on the music. "End to end" also carries Kim's float32 differences
(vocals 129-131 dB) into Mega's input, which costs a few dB on the excerpt. The synthetic clip has
almost no guitar (-80 dBFS: the error is -178 dBFS) and near-empty bands, and Kim finds no vocals in
it (-114 dBFS), so its lower figures are the clip's, not the CPU's: the M1 Pro gives the same ones
(one chunk there: guitar 97.3, keys 101.1 dB; on the EPYC 95.7 and 98.8).

**Same computation as the parts.** On the 30 s excerpt (M1 Pro): kimmega's vocals are bit-identical
to `-n melroformer`'s; `-n mega`'s guitar is bit-identical to `-n megaguitar`'s (one chunk:
`tests/roformer.rs`); all six stems are bit-identical at 1, 4 and 8 threads, and with float32 Mega
weights (`MEGA_WEIGHTS=mega-f32`) instead of float16. `-n megaguitar` and `-n melroformer` give
bit-identical stems to `main` before this change (1afee2c).

### CPU and memory

**Four heads cost what one does.** One 20 s chunk forward (`mr_bench`), `mega` (4 heads) against
`megaguitar` (1 head), M1 Pro:

| build | threads | 1 head | 4 heads | |
|---|---|---|---|---|
| `--features accelerate` | 1 | 46.2 CPU-s | 46.7 | +1.2 % |
| same | 8 | 83.2 CPU-s, 12.2 s wall | 82.0, 12.6 s wall | |
| default (NEON) | 1 | 157.6 CPU-s | 153.7 | within noise |
| same | 8 | 211.6 CPU-s, 29.5 s wall | 210.3, 29.3 s wall | |

The profile puts the three extra heads' mask MLPs at 0.36 s of 46.7 (Accelerate) and 1.6 s of 154
(NEON) per chunk, under 1 %; their iSTFTs are a few tenths of a percent more. The backbone - 12
time + frequency transformer pairs over 62 x 1723 tokens - is everything else.

**Whole tracks**, CPU seconds per second of audio (user + sys, decode and writing included) and
peak RSS, `--flac --clip-mode clamp`, float16 Kim weights. The 30 s excerpt pays for Kim's 4 s and
Mega's 10 s of padding at each end, so its per-second figures run high; the full song is the rate to
plan with.

x86-64, Blacksmith runners (AMD EPYC, AVX-512; synthetic clips, `bench-kimmega.yml`; 8 threads on 8
vCPU = 4 cores with SMT, 1 thread on 2 vCPU):

| recipe | 30 s, 1 thread | 30 s, 8 threads (wall) | 180 s, 1 thread | 180 s, 8 threads (wall) | peak RSS |
|---|---|---|---|---|---|
| **`-n kimmega`** | **14.9** | 28.2 (115 s) | **10.6** | 14.6 (350 s) | 1.08-1.30 GB |
| `-n chain` | 6.7 | 12.6 (53 s) | 4.2 | 7.7 (194 s) | 1.09-1.29 GB |
| `-n ensemble` (shifts 0) | 0.86 | 2.2 (11 s) | 0.78 | 1.7 (50 s) | 0.92-1.12 GB |

At one thread on the full clip, Kim took 762 s of kimmega's 1905 (4.2 CPU-s per audio-s, the
chain's Kim) and Mega's four heads 1142 s (6.3 per audio-s; one head's chunk rate on this CPU, 3.02 per audio-s of chunk, makes about 6 per second of a song):
**kimmega costs 2.5x the chain** on this CPU. 8 threads on 4 SMT cores cost 1.4-2.2x the CPU of 1
thread for 2.8-5.4x less wall time (kimmega gains most: 5.4x).

Apple M1 Pro (8 cores: 6 performance + 2 efficiency; shared with another job, run one at a time;
MUSDB18-HQ Angels In Amplifiers 179.6 s and the Al James 30 s excerpt; `VECLIB_MAXIMUM_THREADS` =
threads). Peak memory footprint (what macOS charges the process) / max RSS:

| build | recipe | 30 s, 1 thread | 30 s, 8 threads (wall) | 179.6 s, 1 thread (wall) | 179.6 s, 8 threads (wall) | footprint / max RSS |
|---|---|---|---|---|---|---|
| `--features accelerate` | **`-n kimmega`** | **11.0** | 20.4 (86 s) | **7.6** (1370 s) | 14.2 (358 s) | 1.03-1.11 / 1.19-2.17 GB |
| | `-n chain` | 3.8 | 7.6 (37 s) | 3.0 (542 s) | 5.9 (179 s) | 1.03-1.15 / 1.35-2.11 GB |
| | `-n ensemble` | 0.68 | 1.2 (9 s) | 0.61 (109 s) | 1.05 (48 s) | 0.58-0.70 / 1.36-1.93 GB |
| default (NEON) | **`-n kimmega`** | 39.0 | 51.5 (224 s) | 27.4 (4936 s) | 36.5 (914 s) | 1.04-1.16 / 1.20-1.91 GB |
| | `-n chain` | 15.4 | 19.4 (90 s) | 12.1 (2179 s) | 15.8 (424 s) | 1.03-1.19 / 1.34-1.88 GB |
| | `-n ensemble` | 1.85 | 2.5 (15 s) | 1.60 (288 s) | 2.1 (76 s) | 0.61-0.69 / 1.33-1.65 GB |

On the Mac, kimmega costs 2.3-2.5x the chain, as on x86; build with `--features accelerate` (3.6x
less CPU than the default build for kimmega: Accelerate's AMX GEMMs). For comparison, PyTorch
(2.2.2 Kim, 2.6.0 Mega, CPU) took 2081 + 4173 = 6254 CPU-s for the same recipe on Angels at 8
threads, the NEON build 6435 and the Accelerate build 2556. Kim finishes at 32 % (Accelerate) to
38 % (NEON) of a full-song run; the rest is Mega. Memory: the footprint is 1.0-1.2 GB at 30 s and at 3 min (not
measured on longer tracks; what grows with length is the decoded mix + the instrumental, 2 x 2 x 4
bytes a sample, 0.17 GB at 4 min). Kim's weights widened to float32 (0.91 GB) are its largest
part, and are freed before Mega's (0.30 GB) are loaded.

### Cheaper Mega settings

`--mega-chunk-size` / `--mega-num-overlap` trade Mega's quality for CPU exactly as `-n megaguitar`'s
`--chunk-size` / `--num-overlap` do (see *Mega 53 ... Cheaper settings*: measured there on the guitar
head): 10 s chunks (`--mega-chunk-size 441000`) cost -0.03 dB guitar for 32 % less of Mega's CPU,
10 s chunks at overlap 1 (`--mega-num-overlap 1`) -0.12 dB for 69 % less. Kim's half of the recipe
is unchanged by them. The defaults are the reference settings (20 s, overlap 2), the ones round 5
scored. Measured on the whole recipe (Angels, M1 Pro Accelerate build, 8 threads; "SNR vs default"
is how far each stem moves from the default settings' stem, not its quality):

| Mega setting | CPU-s per audio-s | wall | SNR vs default: drums / bass / guitar / piano / other |
|---|---|---|---|
| default: 20 s, overlap 2 | 14.2 | 358 s | - |
| `--mega-chunk-size 441000` | 11.1 (-22 %) | 285 s | 17.8 / 22.4 / 22.3 / 22.8 / 10.4 dB |
| `--mega-chunk-size 441000 --mega-num-overlap 1` | 7.9 (-45 %) | 200 s | 16.7 / 20.6 / 20.8 / 21.1 / 9.2 dB |

(other moves most because it is quiet and carries every head's change.) The heads' quality cost
on the other stems than guitar is not measured.

## Chain (`-n chain`): Kim vocals, then the ensemble on the instrumental

```sh
htdemucs-rs separate -n chain --flac --clip-mode clamp -o out song.m4a
# -> out/htdemucs/song/{drums,bass,other,vocals}.flac           (4-stem view, --clip-mode-4)
#    out/htdemucs_6s/song/{drums,bass,other,vocals,guitar,piano}.flac
```

Round 3's best recipe, `k+chain_avg` (research/round3/round3.md, section 2): Kim's Mel-Band RoFormer
vocals; htdemucs and htdemucs_6s at shifts 0 on Kim's instrumental (mix - vocals); drums, bass and
other averaged across the two models as `-n ensemble` averages them (`ensemble::combine_with`),
guitar and piano from 6s. Against today's mixer split (htdemucs_6s, 2 shifts) on 12 MUSDB18-HQ
tracks: vocals +3.44, drums +0.65, bass +0.17, other +1.24 dB uSDR. Same folders and stems as
`-n ensemble`, vocals being Kim's in both; `--clip-mode-4` and `--ensemble-4` work as there; no
`--shifts` / `--overlap` (Demucs runs at shifts 0, overlap 0.25; Kim at `--num-overlap`, 2).

**Sum to the mix.** Not forced, as in round 3: the chain's stems sum to the mix within -24 to
-33 dB on 11 of round 3's 12 MUSDB tracks (residual energy re the mix; mean -26.4 dB), but only
-8.3 dB on Little Chicago's Finest, whose bass also loses 3.9 dB against 6s at shifts 2. Folding
the residual into "other" was neutral on typical tracks; the outlier is not understood. `-v` prints it per track (Signe Jakobsen -28.6 dB,
Angels In Amplifiers -24.2 dB - the PyTorch pipeline's -28.57 / -24.25).

**Memory.** Demucs' whole-track normalisation needs the whole instrumental before its first
segment, so one track-long buffer is unavoidable besides the decode: the instrumental
(2 x 4 bytes x samples). Everything else streams - Kim's vocals go to both views' writers as Kim
finalises them, the decoded mix is freed once the instrumental is complete, Kim's 0.91 GB of
weights are freed before the Demucs models (0.28 GB) are loaded, and Demucs runs through
`apply::Streamer` window by window. Peak RSS (VmHWM) on the Linux box, 8 threads:

| track | length | peak RSS | of which track-long (mix + instrumental, float32) |
|---|---|---|---|
| Aerials | 4.1 min | **1.34 GB** | 0.17 GB |
| Aerials looped (the Demucs path's 22-min test) | 22 min | **2.08 GB** | 0.93 GB |

The difference is exactly those two buffers growing. During Kim both exist, because Kim reads the
mix (reflect-padded at both ends) while the instrumental is being written; overwriting the mix in
place would save 0.47 GB at 22 min but needs care with that reflection, and is not done.

### Parity (vs the PyTorch pipeline round 3 scored, CPU float32: tools/chain_ref.py)

Two MUSDB18-HQ test tracks on the Linux box (i5-12450H), the same WAV into both; SNR per stem
over the whole track, and the error's level:

| track | view | vocals | drums | bass | other | guitar | piano |
|---|---|---|---|---|---|---|---|
| Signe Jakobsen, What Have You Done To Me | 4 | 131.8 | 119.5 | 123.5 | 123.4 | | |
| | 6 | 131.8 | 119.5 | 123.5 | 108.7 | 120.5 | 118.7 |
| Angels In Amplifiers, I'm Alright | 4 | 132.2 | 121.5 | 126.5 | 123.3 | | |
| | 6 | 132.2 | 121.5 | 126.5 | 108.3 | 119.4 | 114.5 |

The error is -144 to -157 dBFS on every stem; the 6-stem "other" scores lower only because it is
quiet (-38 / -39 dBFS: guitar and piano are taken out of it). On the CI runner's synthetic clip
(Xeon 6975P, -60 dBFS noise floor) every stem is 91.9-122.8 dB, the error at most -141 dBFS.

And the scores themselves: with float32 Kim weights, the chain's uSDR (r2_score.py) on four MUSDB
tracks equals round 3's PyTorch `k+chain_avg` scores - vocals to all four decimals, the other
stems within 0.002 dB (round 3 ran its Demucs passes on MPS).

### Float16 Kim weights

The checkpoint is float32, and none of its tensors is exactly float16, so float16 weights change
the model's output (Kim's chunk parity against PyTorch float32: 128 dB -> 72 dB). What that costs
in separation quality, the chain with float32 and with float16 Kim weights on four MUSDB18-HQ
tracks, scored with research/round2/r2_score.py (uSDR, float32 / float16):

| track | vocals | drums | bass | other | accompaniment |
|---|---|---|---|---|---|
| Signe Jakobsen | 14.2991 / 14.2991 | 6.7664 / 6.7663 | 3.2100 / 3.2100 | 6.6414 / 6.6413 | 16.7997 / 16.7997 |
| Angels In Amplifiers | 13.8809 / 13.8812 | 8.0086 / 8.0087 | 10.7113 / 10.7112 | 9.1856 / 9.1856 | 16.9100 / 16.9103 |
| Enda Reilly | 18.5387 / 18.5390 | 13.6319 / 13.6319 | 11.4105 / 11.4106 | 12.8993 / 12.8993 | 22.6026 / 22.6028 |
| Al James | 10.9387 / 10.9390 | 7.6269 / 7.6269 | 11.0154 / 11.0155 | 5.7101 / 5.7104 | 12.2219 / 12.2223 |

At most 0.0004 dB on any stem, in either direction - far under the 0.05 dB bar. So the converter
now writes **float16 Kim weights by default** (`weights/melroformer.safetensors`, 457 MB instead
of 913); `convert --float32` writes `melroformer-f32.safetensors`, which the parity checks against
PyTorch float32 load (`MELROFORMER_WEIGHTS=melroformer-f32`). The weights are widened to float32
on load, so this halves the download and the file, not the memory.

### CPU

CPU seconds per second of audio for the whole chain (Kim + both Demucs passes, decode and writing
included), Rust against tools/chain_ref.py on PyTorch 2.2.2 CPU, same machine, same input:

| machine | threads | input | Rust CPU-s / audio-s | PyTorch CPU-s / audio-s | Rust wall vs PyTorch wall |
|---|---|---|---|---|---|
| Linux box, i5-12450H (4 P + 4 E cores, 12 threads, AVX2) | 8 | Aerials, 2 min from 1:00 | **16.0** | 30.0 | 267 s vs 492 s |
| same | 4 | same | **9.8** | 13.7 | 308 s vs 421 s |
| same | 8 | 4 MUSDB tracks, 3-3.3 min | 15.1-15.9 | 35-53 (2 tracks, box shared) | 394-429 s per track |
| GitHub runner, Xeon 6975P-C (AVX-512) | 2 (1 core) | 30 s synthetic clip | **14.5** | 20.0 | 223 s vs 311 s |

(The 1-thread runs were cut short when the box went offline; at one thread the Kim part alone
measured 4.90 CPU-s per audio-s of chunk against PyTorch's 7.76 on a Zen 3 runner - see
*Mel-Band RoFormer*.)
Every figure uses 27-47% less CPU than PyTorch. On the box, 8 threads buy Rust 13% less wall time than 4 for
63% more CPU (the four extra threads are hyperthreads and E-cores); PyTorch gets slower (492 s vs
421 s). On a CPU-billed container, threads = physical cores is the efficient setting. The 30 s clip carries proportionally more of Kim's 4 s padding at each end, so its
per-second figures run high; a 4-minute track costs about 2 x Kim's chunk rate plus Demucs' two
passes (round 3's PyTorch estimate was 8.97 CPU-s per audio-s with Kim at one thread).

The Mac (M1 Pro) is not measured here: the machine is shared and the chain is now run only on the
Linux box. For Kim alone the Mac gemm build was 1.8x PyTorch's CPU (PyTorch uses AMX there; see
*Mel-Band RoFormer*).

## Mega 53 BS-RoFormer guitar (`-n megaguitar`)

MVSEP's 53-stem BS-RoFormer ("Mega 53"), its `guitar` mask head only: the best guitar stem
measured so far (research round 4: +2.64 dB guitar uSDR over htdemucs_6s on 26/26 Cambridge-MT
songs).

```sh
PY="uv run --quiet --python 3.11 --with-requirements research/round4/r4_msst_env.txt python"
$PY tools/bsroformer.py convert mvsep_mega_model_bs_roformer_53_stems_v1.ckpt mvsep_mega_model_bs_roformer_53_stems.yaml
# -> weights/megaguitar.safetensors (77 MB, float16) + weights/megaguitar.json
htdemucs-rs separate -n megaguitar --flac --clip-mode clamp -o out song.m4a
# -> out/megaguitar/song/{guitar,no_guitar}.flac      (no_guitar = mix - guitar)
```

**What it computes** (MSST `BSRoformer` @1523206, CPU, float32, with the chunking of
`research/round4/r4_mega.py`, i.e. MSST's generic `demix`): chunks of 882000 samples (20 s)
every 441000 (`--num-overlap 2`), the track reflect-padded by 10 s at both ends, linear fades of
2 s, outputs overlap-added and divided by the summed windows. Per chunk: STFT (2048 / hop 512,
centred; 1723 frames) -> 62 fixed, non-overlapping bands of bins (24 bands of 2 bins, 12 of 4,
8 of 12, 8 of 24, 8 of 48, then 128 and 129), each RMSNorm + Linear to 256 -> 12 x (time transformer over each band's 1723
frames, frequency transformer over each frame's 62 bands; RoPE, gated attention, 8 x 64 heads,
GELU MLP; no per-transformer output norm) -> one final RMSNorm -> per band Linear(256, 512) ->
tanh -> Linear -> GLU -> complex mask -> DC bin zeroed (`zero_dc`) -> iSTFT to 882000 samples.

It is the Mel-RoFormer code path (`src/roformer.rs`) with the differences read from the config
(`weights/megaguitar.json`): bands, `transformer_norm_output: false` + `final_norm`, a 2-layer
mask MLP 4 x narrower than Kim's, `zero_dc`, and MSST's two chunk-edge rules (a short last chunk is
reflect-padded when longer than C/2 rather than C/2 + 1; the end fade is dropped for every chunk
with i + step >= total, the first included). Thread determinism, streaming and the kernels are
shared with `-n melroformer`.

**Weights.** The checkpoint (`mvsep_mega_model_bs_roformer_53_stems_v1.ckpt` from [MSST release
v1.0.21](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21), 1.37 GB,
sha256 `c6282089…3519f`; MIT per its trainer in MSST #245) is float16: 682 M parameters, of which
the 53 mask heads are 656 M. `convert` keeps the 26.3 M backbone and the guitar head (12.4 M) as
stored, float16 (77 MB; `--stem` picks another head). The rotary frequencies are float16-rounded in
the checkpoint and PyTorch uses them so, so they are kept as stored. Because PyTorch itself
widens the float16 weights to float32, float32 weights (`convert --float32`, 155 MB,
`MEGAGUITAR_WEIGHTS=megaguitar-f32`) hold exactly the same values and give bit-identical output
(measured: the 30 s excerpt below, both stems, M1 Pro): there is nothing to gain from them.

### Parity (vs PyTorch 2.6.0 CPU float32, MSST code, r4_mega.py's chunking)

One chunk (`mr_parity`; chunk 1 of a 30 s excerpt of MUSDB18-HQ test "Al James - Schoolboy
Facination", 60-90 s, i.e. its first 20 s of music): output **109.7 dB**; band split 105.6 dB, the
transformers 102.6-113.3 dB along the stack; every transformer run alone on PyTorch's own input
122.5-131.5 dB. Against a float64 PyTorch run of the same chunk this crate is 112.5 dB and PyTorch
float32 itself 111.2 dB: the remaining difference is float32 rounding on both sides.

Full path (`htdemucs-rs separate` vs `tools/bsroformer.py separate`, the same float WAV into both;
SNR over the whole file; reference level and error level):

| input | guitar | no_guitar | levels (guitar / no_guitar), error |
|---|---|---|---|
| Al James excerpt (30 s, 5 chunks), Linux box (i5-12450H, AVX2) | 113.8 dB | 128.3 dB | -35 / -21 dBFS; error -149 dBFS, max 4.6e-7 |
| the same, Apple M1 Pro (NEON) | 112.7 dB | 127.2 dB | -35 / -21 dBFS; error -148 dBFS, max 6.5e-7 |
| MUSDB18-HQ test: Angels In Amplifiers, I'm Alright (179.6 s, 20 chunks), M1 Pro | 119.4 dB | 124.8 dB | -25 / -20 dBFS; error -144 dBFS, max 1.9e-6 |
| synthetic CI clip + -60 dBFS noise (30 s, Blacksmith EPYC, AVX-512) | 96.8 dB | 155.0 dB | -80 / -16 dBFS; error -177 dBFS (the clip has almost no "guitar") |

The chunk parity is the same 109.7 dB on the M1 Pro (NEON kernels) as on the Linux box (AVX2).

Against round 4's numbers: run on the four Cambridge-MT remixes of round 4's `cmt8` set that were
re-scored here, this port's guitar gives the same guitar uSDR as round 4's PyTorch (CUDA, fp32)
`mega_mix` to the third decimal (Angels In Amplifiers 8.029 / 8.029 dB, Juliet's Rescue 7.893 /
7.893, Forkupines 7.576 / 7.576, Al James 4.339 / 4.339).

**Determinism.** Bit-identical output (full path, both stems, the 30 s excerpt) at 1, 4 and 8
threads on the Linux box and at 1, 4 and 6 threads on the M1 Pro; one chunk bit-identical at 1, 4
and 8 threads on the box and on Blacksmith's AVX-512 EPYC (`MR_CHECK_THREADS`).

### Speed

CPU seconds per second of audio **of one 20 s chunk** forward (`mr_bench`, `bsroformer.py
bench`); a whole track costs about twice that per second of music (every sample is in two
chunks), plus the 10 s of padding at each end.

| machine | threads | htdemucs-rs | PyTorch 2.6.0 | |
|---|---|---|---|---|
| AMD EPYC (Blacksmith 8 vCPU, AVX-512) | 1 | 3.02 (60.3 CPU-s / chunk) | 8.59 (171.8) | **65% less CPU** |
| same | 8 | 3.93 (10.3 s wall / chunk) | 17.4 (47.3 s wall) | 4.6x faster wall |
| Intel i5-12450H (the Linux box, AVX2) | 1 | 5.20 (103.9 CPU-s / chunk) | 5.72 (114.3, round 4's `r4_cputime.sh`) | 9% less CPU |
| same | 8 | 9.46 (24.6 s wall / chunk) | 13.9 (278 CPU-s / chunk in a 30 s run) | 32% less CPU |
| Apple M1 Pro, default (`gemm` crate, NEON) build | 1 | 7.45 (149.0 CPU-s / chunk) | **4.64 (92.8)** | **PyTorch is faster: 1.6x less CPU** (Accelerate's AMX GEMMs) |
| same | 4 | 8.21 (41.9 s wall / chunk) | 6.99 (34.1 s wall) | PyTorch faster |
| Apple M1 Pro, `--features accelerate` build | 1 | **2.36 (47.2 CPU-s / chunk)** | 4.64 (92.8) | 49% less CPU |
| same | 4 | 3.50 (16.7 s wall / chunk) | 6.99 (34.1 s wall) | 2.0x faster wall |

The Accelerate build is bit-identical across 1 and 4 threads, but its parity against PyTorch was
not measured (its GEMM roundings differ from the `gemm` crate's; the parity figures above are the
default build's). Profile (1 thread, any machine): the time transformer is 71-75% of the forward,
attention alone (QK^T, softmax, AV over 1723 frames) 45-52%; the GEMMs run at 190-230 GFLOP/s per
thread on the EPYC, 113-128 on the box, 80-92 on the M1 Pro (NEON).

Whole tracks (`htdemucs-rs separate`, CPU-s per second of music; wall time):

| machine, threads | Al James 30 s excerpt | MUSDB Angels In Amplifiers 179.6 s | Aerials 243.6 s | 22 min (long22, 1320 s) |
|---|---|---|---|---|
| Linux box, 8 | 118 s wall (PyTorch 192 s, 46.3 CPU-s/s) | - | 21.4; 678 s wall | 19.4; 3338 s (\*) |
| M1 Pro (NEON), 6 | 28.7 (PyTorch 28.1) | 20.1; 629 s wall (PyTorch 19.2; 588 s) | 20.3; 861 s | 19.2; 4399 s |
| Blacksmith EPYC, 8 (240 s synthetic) | | | 6.5; 207 s | |

(\*) the CLI's own (monotonic) clock; the wrapper's wall clock said 3958 s, the gap matching a
pause of the machine (CPU time 25638 s). Short excerpts cost more per second than long tracks
because of the 10 s of padding at each end.

### Memory

Peak RSS, `--flac --clip-mode clamp`: weights (77 MB float16, widened to 155 MB float32 at load)
+ the decoded input (2 x 4 bytes x samples: 86 MB at 4 min, 466 MB at 22 min) + one chunk's
activations (~0.1 GB of tokens plus per-thread scratch) + the one-chunk overlap-add window; both
stems are written as they are finalised.

| | Aerials (4 min) | long22 (22 min) | 30 s / 3 min |
|---|---|---|---|
| Linux box, 8 threads (VmHWM) | **672 MB** | **1055 MB** | 557 MB (30 s); PyTorch 3.37 GB |
| M1 Pro, 6 threads (peak memory footprint / max RSS) | 530 MB / 1.22 GB | 954 MB / 1.81 GB | Angels: 520 MB / 1.19 GB; PyTorch 3.56 GB max RSS |
| Blacksmith EPYC, 8 threads (240 s synthetic) | 645 MB | | |

### Cheaper settings (chunk size, overlap): quality cost

MUSDB has no guitar stem, so these are scored on round 4's Cambridge-MT remixes, whose guitar is
known exactly: 4 of the 8 `cmt8` songs (Angels In Amplifiers, Juliet's Rescue, Forkupines, Al
James; 90 s each), on the mix, Linux box, 8 threads. Guitar uSDR as round 4 defines it; "SNR vs
full" is each variant's guitar against the full-settings output (how different, not how good).

| setting | CPU vs default | guitar uSDR (mean of 4) | Δ vs default, items worse | guitar removed uSDR | SNR vs full |
|---|---|---|---|---|---|
| default: 20 s chunks, `--num-overlap 2` | 1.00 (2094 CPU-s per song) | **6.959** | - | 13.900 | - |
| `--num-overlap 1` | 0.46 | 6.871 | -0.088, 4/4 | 13.812 | 17.6-24.0 dB |
| `--chunk-size 441000` (10 s, the training length) | 0.68 | 6.931 | -0.028, 4/4 (Juliet's Rescue -0.001) | 13.872 | 17.1-22.8 dB |
| 10 s chunks, `--num-overlap 1` | 0.31 | 6.845 | -0.115, 4/4 | 13.785 | 16.2-21.4 dB |

On four MUSDB18-HQ test songs (Angels full, 60 s excerpts of three more; M1 Pro) the variants are
14-23 dB from the full-settings output, the same range. The cost is small next to Mega's +2.6 dB
over htdemucs_6s: 10 s chunks keep all but 0.03 dB for 32% less CPU, 10 s chunks at overlap 1 all
but 0.12 dB for 69% less. With `--num-overlap 1` the chunks abut and each chunk's first and last
sample get zero weight (MSST's linear fades start and end at 0), so the two samples at every chunk
boundary (every 20 s, or 10 s) come out 0; inaudible, but it is there. The defaults stay MSST's.

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

Mega 53 BS-RoFormer: the BS-RoFormer architecture is by Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong
and Yun-Ning Hung (ByteDance, arXiv:2309.02612), the implementation by Phil Wang
(lucidrains/BS-RoFormer, MIT) as adapted in ZFTurbo's Music-Source-Separation-Training (MIT), the
53-stem model trained by ZFTurbo / MVSEP (weights MIT per MSST #245; not in this repository).

Mel-Band RoFormer: the architecture is by Ju-Chiang Wang, Wei-Tsung Lu and Minz Won (ByteDance,
arXiv:2310.01809), the PyTorch implementation by Phil Wang (lucidrains/BS-RoFormer, MIT), the
trained vocal model and its inference code by Kimberley Jensen
([KimberleyJensen/Mel-Band-Roformer-Vocal-Model](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model);
weights at huggingface.co/KimberleyJSN/melbandroformer, MIT since 2026-04-22, not in this
repository). `src/roformer.rs` is an independent reimplementation of that inference path.


HTDemucs is by Simon Rouard, Francisco Massa and Alexandre Défossez (Meta AI),
[facebookresearch/demucs](https://github.com/facebookresearch/demucs), MIT licence; this is an
independent reimplementation of its inference code from the published source, and loads its
pretrained weights. The weights are not in this repository; their terms are separate from the
code's (see demucs issue #327).
