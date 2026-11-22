# htdemucs-rs

Music stem separation (HTDemucs v4) in Rust: the same model and weights as Meta's Demucs, without
Python or PyTorch. Built for Fretboard's song pipeline, where it replaces `demucs` (htdemucs and
htdemucs_6s, 2 seeded shifts).

Priorities, in order: quality (match PyTorch Demucs), speed, build size.

**Status:** the HTDemucs forward pass for one segment (7.8 s) works for both `htdemucs` (4 stems)
and `htdemucs_6s` (6 stems) and matches PyTorch CPU float32 at 120-131 dB SNR, layer by layer.
Full-song separation (segmentation with overlap-add, shifts, normalisation, audio I/O) is next.

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
src/bin/parity.rs    per-layer comparison with the reference dumps
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

## Parity (one segment, vs PyTorch CPU float32)

| segment | htdemucs | htdemucs_6s |
|---|---|---|
| System of a Down, Aerials: chunk 10 | 119.7 dB | 120.3 dB |
| Oasis, Wonderwall: chunk 5 | 126.8 dB | 126.2 dB |
| Three Days Grace, Never Too Late: last (padded) chunk of a pass shifted by 12345 samples | 130.6 dB | 128.0 dB |

Every intermediate layer is above 95 dB (most 105-135 dB); `parity` prints the full table.

Speed, one segment on an M1 Pro (8 threads): PyTorch 2.2.2 CPU 3.4 s (htdemucs) / 3.0 s
(htdemucs_6s); this crate 1.8 s / 1.7 s with the `gemm` crate, 1.3 s / 1.1 s with Accelerate.

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
