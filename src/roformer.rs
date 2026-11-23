//! Kim's Mel-Band RoFormer vocal model (github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model,
//! lucidrains' `MelBandRoformer`), eval mode, for one chunk; `separate` is upstream's
//! `demix_track` (chunks, linear fades, overlap-add). Weights and config: tools/melroformer.py.
//!
//! Forward, for a stereo chunk of T samples (T = 352800 -> 801 STFT frames):
//! 1. STFT per channel (n_fft 2048, hop 441, periodic Hann, centred with reflect padding, not
//!    normalised): (frames, 1025 bins, 2 channels, re/im).
//! 2. Band split: 60 mel bands, each a contiguous range of bins (neighbouring bands overlap); a
//!    band's features at one frame are its bins x channels x re/im, RMSNorm'd and projected
//!    (Linear) to 384. Tokens: (band, frame).
//! 3. 6 x (time transformer over each band's 801 frames, then frequency transformer over each
//!    frame's 60 bands). Each transformer: one pre-norm block - RMSNorm, attention with rotary
//!    position embeddings (8 heads x 64) and a per-head sigmoid gate, residual; RMSNorm, GELU
//!    feed-forward (x4), residual - then a final RMSNorm.
//! 4. Mask estimator per band: Linear -> tanh -> Linear -> tanh -> Linear -> GLU, giving a
//!    complex mask per (bin, channel) of the band; bins in two bands average their two masks.
//! 5. Masked STFT -> iSTFT -> vocals.
//!
//! RMSNorm is `F.normalize(x) * sqrt(dim) * gamma` (L2-normalise, then scale): computed as
//! x / max(||x||, 1e-12) with the norm accumulated in f64, then the two multiplies in float32.
//!
//! Work is split into tasks with a fixed shape (one band for the time transformer and the mask
//! estimator, a fixed group of frames for the frequency transformer), each running its GEMMs
//! single-threaded: the result is bit-identical whatever the thread count.

use std::path::Path;

use rayon::prelude::*;
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;
use serde::Deserialize;

use crate::ops::{add_assign, exp_g, gelu1_g, sgemm_par, sigmoid_g, simd_fn, MulAdd, SendPtr};
use crate::tensor::Tensor;
use crate::weights::SafeTensors;
use crate::Tracer;

/// Op-class profiler for this model (off unless `prof::enable(true)`): thread time per (part,
/// op class), summed over every call on every thread, plus GEMM flops. Scopes are leaf-level.
pub mod prof {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
    use std::time::Instant;

    pub const CLASSES: [&str; 14] = [
        "STFT / iSTFT",
        "RMSNorm",
        "projection GEMMs (qkv+gates, out)",
        "rotary",
        "attn QK^T GEMM",
        "attn softmax",
        "attn AV GEMM",
        "gates",
        "feed-forward GEMMs",
        "GELU / tanh / GLU (+bias)",
        "residual adds, copies",
        "band-split GEMM",
        "mask-estimator GEMMs",
        "mask average + apply",
    ];
    pub const STFT: usize = 0;
    pub const NORM: usize = 1;
    pub const PROJ: usize = 2;
    pub const ROPE: usize = 3;
    pub const QK: usize = 4;
    pub const SOFTMAX: usize = 5;
    pub const AV: usize = 6;
    pub const GATES: usize = 7;
    pub const FF: usize = 8;
    pub const ACT: usize = 9;
    pub const COPY: usize = 10;
    pub const SPLIT: usize = 11;
    pub const MASKMLP: usize = 12;
    pub const APPLY: usize = 13;
    pub const PARTS: [&str; 4] = ["in/out", "time transformer", "freq transformer", "mask estimator"];
    const N: usize = CLASSES.len() * PARTS.len();
    static ON: AtomicBool = AtomicBool::new(false);
    static PART: AtomicUsize = AtomicUsize::new(0);
    static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    static FLOPS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];

    pub fn enable(on: bool) {
        ON.store(on, Relaxed);
    }
    /// Which part subsequent scopes are charged to (set from the calling thread between phases).
    pub fn part(p: usize) {
        PART.store(p, Relaxed);
    }
    pub struct Scope(usize, Option<Instant>);
    #[inline]
    pub fn scope(class: usize) -> Scope {
        Scope(class, ON.load(Relaxed).then(Instant::now))
    }
    /// A scope that also counts an (m x k) * (k x n) GEMM's flops.
    #[inline]
    pub fn gemm(class: usize, m: usize, n: usize, k: usize) -> Scope {
        if ON.load(Relaxed) {
            FLOPS[PART.load(Relaxed) * CLASSES.len() + class].fetch_add((2 * m * n * k) as u64, Relaxed);
        }
        scope(class)
    }
    impl Drop for Scope {
        fn drop(&mut self) {
            if let Some(t) = self.1 {
                NS[PART.load(Relaxed) * CLASSES.len() + self.0].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            }
        }
    }
    /// (thread seconds, GEMM flops) per [part][class] since the last call; resets.
    pub fn take() -> Vec<Vec<(f64, f64)>> {
        (0..PARTS.len())
            .map(|p| (0..CLASSES.len()).map(|c| (NS[p * CLASSES.len() + c].swap(0, Relaxed) as f64 * 1e-9, FLOPS[p * CLASSES.len() + c].swap(0, Relaxed) as f64)).collect())
            .collect()
    }
}

/// tools/melroformer.py's JSON: the upstream constructor kwargs plus inference settings and bands.
#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub name: String,
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub num_bands: usize,
    pub stereo: bool,
    pub num_stems: usize,
    pub time_transformer_depth: usize,
    pub freq_transformer_depth: usize,
    pub dim_freqs_in: usize,
    pub sample_rate: usize,
    pub stft_n_fft: usize,
    pub stft_hop_length: usize,
    pub stft_win_length: usize,
    pub stft_normalized: bool,
    pub mask_estimator_depth: usize,
    pub mask_estimator_hidden: usize,
    pub ff_mult: usize,
    pub chunk_size: usize,
    pub num_overlap: usize,
    pub target_instrument: String,
    /// Bins of each band, ascending.
    pub bands: Vec<Vec<usize>>,
    pub num_bands_per_freq: Vec<usize>,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let cfg: Config = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        cfg.check()?;
        Ok(cfg)
    }

    /// The configuration of Kim's checkpoint; refuse others rather than compute something else.
    fn check(&self) -> Result<(), String> {
        let mut bad = vec![];
        let mut need = |ok: bool, what: &str| {
            if !ok {
                bad.push(what.to_string())
            }
        };
        need(self.stereo && self.num_stems == 1, "stereo, one stem");
        need(self.time_transformer_depth == 1 && self.freq_transformer_depth == 1, "transformer depth 1");
        need(self.mask_estimator_depth == 2, "mask estimator depth 2");
        need(self.stft_win_length == self.stft_n_fft && !self.stft_normalized, "win = n_fft, not normalised");
        need(self.dim_freqs_in == self.stft_n_fft / 2 + 1, "dim_freqs_in = n_fft / 2 + 1");
        need(self.bands.len() == self.num_bands, "num_bands");
        need(self.heads * self.dim_head > 0 && self.dim_head % 2 == 0, "heads, even dim_head");
        need(self.num_overlap >= 1 && self.chunk_size >= self.num_overlap && self.chunk_size % self.stft_hop_length == 0, "chunking");
        // Every band a non-empty contiguous range (so a band's features are one slice of a frame).
        need(self.bands.iter().all(|b| !b.is_empty() && b.windows(2).all(|w| w[1] == w[0] + 1)), "contiguous bands");
        let mut count = vec![0usize; self.dim_freqs_in];
        for b in &self.bands {
            for &f in b {
                if f < count.len() {
                    count[f] += 1;
                }
            }
        }
        need(count == self.num_bands_per_freq && count.iter().all(|&c| c > 0), "bands cover every bin");
        if bad.is_empty() { Ok(()) } else { Err(format!("unsupported Mel-RoFormer config: {}", bad.join(", "))) }
    }
}

/// The STFT and iSTFT of the model (torch.stft / torch.istft, centred, reflect padding, periodic
/// Hann, onesided, not normalised), FFTs in f64. Spectra are (frames, bins, 2 channels, re/im).
pub struct Spec {
    n_fft: usize,
    hop: usize,
    window: Vec<f64>,
}

impl Spec {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        // torch.hann_window(n_fft) (periodic), float32.
        let window = (0..n_fft).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos()) as f32 as f64).collect();
        Spec { n_fft, hop, window }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    pub fn frames(&self, len: usize) -> usize {
        len / self.hop + 1
    }

    /// x: 2 channels x len. Returns (frames, bins, 2, 2).
    pub fn stft(&self, x: &[f32], len: usize) -> Vec<f32> {
        let _sc = prof::scope(prof::STFT);
        let (nfft, hop, bins) = (self.n_fft, self.hop, self.bins());
        let half = nfft / 2;
        assert!(len > half, "chunk shorter than n_fft / 2");
        let frames = self.frames(len);
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(nfft);
        let mut out = vec![0.0f32; frames * bins * 4];
        out.par_chunks_mut(bins * 4).enumerate().for_each_init(
            || (fft.make_input_vec(), fft.make_output_vec(), fft.make_scratch_vec()),
            |(inp, outp, scratch), (t, row)| {
                for s in 0..2 {
                    let ch = &x[s * len..(s + 1) * len];
                    for (i, v) in inp.iter_mut().enumerate() {
                        // Centred: frame t covers padded samples [t*hop, t*hop + nfft), i.e.
                        // signal positions t*hop - nfft/2 + i, reflected at both ends.
                        let mut j = (t * hop + i) as isize - half as isize;
                        if j < 0 {
                            j = -j;
                        }
                        if j >= len as isize {
                            j = 2 * (len as isize - 1) - j;
                        }
                        *v = ch[j as usize] as f64 * self.window[i];
                    }
                    fft.process_with_scratch(inp, outp, scratch).unwrap();
                    for (f, c) in outp.iter().enumerate() {
                        row[f * 4 + s * 2] = c.re as f32;
                        row[f * 4 + s * 2 + 1] = c.im as f32;
                    }
                }
            },
        );
        out
    }

    /// y: (frames, bins, 2, 2). Returns 2 channels x (frames - 1) * hop samples (torch.istft with
    /// center=True and no length: the overlap-added frames / the summed squared window, minus
    /// n_fft/2 samples at each end).
    pub fn istft(&self, y: &[f32], frames: usize) -> Vec<f32> {
        let _sc = prof::scope(prof::STFT);
        let (nfft, hop, bins) = (self.n_fft, self.hop, self.bins());
        let len = (frames - 1) * hop;
        let ifft = RealFftPlanner::<f64>::new().plan_fft_inverse(nfft);
        let scale = 1.0 / nfft as f64;
        // Windowed inverse FFT of every frame, both channels: (frames, 2, nfft).
        let mut blocks = vec![0f64; frames * 2 * nfft];
        blocks.par_chunks_mut(2 * nfft).enumerate().for_each_init(
            || (ifft.make_input_vec(), ifft.make_output_vec(), ifft.make_scratch_vec()),
            |(inp, outp, scratch), (t, blk)| {
                let row = &y[t * bins * 4..(t + 1) * bins * 4];
                for s in 0..2 {
                    for (f, c) in inp.iter_mut().enumerate() {
                        *c = Complex::new(row[f * 4 + s * 2] as f64, row[f * 4 + s * 2 + 1] as f64);
                    }
                    // c2r ignores the imaginary parts of the DC and Nyquist bins.
                    inp[0].im = 0.0;
                    inp[bins - 1].im = 0.0;
                    ifft.process_with_scratch(inp, outp, scratch).unwrap();
                    for ((o, v), w) in blk[s * nfft..(s + 1) * nfft].iter_mut().zip(outp.iter()).zip(&self.window) {
                        *o = v * scale * w;
                    }
                }
            },
        );
        let mut out = vec![0.0f32; 2 * len];
        out.par_chunks_mut(4096).enumerate().for_each(|(ci, chunk)| {
            for (k, o) in chunk.iter_mut().enumerate() {
                let i = ci * 4096 + k;
                let (s, i) = (i / len, i % len);
                let n = i + nfft / 2; // position in the overlap-added signal
                let f_lo = if n + 1 > nfft { (n + 1 - nfft).div_ceil(hop) } else { 0 };
                let f_hi = (n / hop).min(frames - 1);
                let (mut acc, mut env) = (0f64, 0f64);
                for t in f_lo..=f_hi {
                    let j = n - t * hop;
                    acc += blocks[(t * 2 + s) * nfft + j];
                    env += self.window[j] * self.window[j];
                }
                *o = (acc / env) as f32;
            }
        });
        out
    }
}

simd_fn! {
/// RMSNorm of one row (F.normalize(x) * scale * gamma) into `o`.
pub fn rms_norm_row(x: &[f32], o: &mut [f32], scale: f32, gamma: &[f32]) {
    let mut acc = [0f64; 8];
    let chunks = x.chunks_exact(8);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..8 {
            acc[i] += c[i] as f64 * c[i] as f64;
        }
    }
    let ss = acc.iter().sum::<f64>() + rest.iter().map(|&v| v as f64 * v as f64).sum::<f64>();
    let n = (ss.sqrt() as f32).max(1e-12);
    for i in 0..x.len() {
        o[i] = (x[i] / n) * scale * gamma[i];
    }
}
}

/// tanh in f32 (Cephes tanhf: odd polynomial below 0.625, 1 - 2 / (e^2|x| + 1) above), branch-free.
#[inline(always)]
pub fn tanh_g<F: MulAdd>(x: f32) -> f32 {
    let a = x.abs();
    let z = x * x;
    let mut p = F::fma(-5.704_988_7e-3, z, 2.063_908_9e-2);
    p = F::fma(p, z, -5.373_971_6e-2);
    p = F::fma(p, z, 1.333_144_2e-1);
    p = F::fma(p, z, -3.333_328_2e-1);
    let small = F::fma(p * z, x, x);
    let e = exp_g::<F>(2.0 * a);
    let big = (1.0 - 2.0 / (e + 1.0)).copysign(x);
    if a < 0.625 { small } else { big }
}

simd_fn! {
/// v = tanh(v + b), one row.
fn bias_tanh_row(v: &mut [f32], b: &[f32]) {
    for i in 0..v.len() {
        v[i] = tanh_g::<F>(v[i] + b[i]);
    }
}
}

simd_fn! {
/// v = GELU(v + b), one row.
fn bias_gelu_row(v: &mut [f32], b: &[f32]) {
    for i in 0..v.len() {
        v[i] = gelu1_g::<F>(v[i] + b[i]);
    }
}
}

simd_fn! {
/// x += o + b, one row.
fn residual_bias_row(x: &mut [f32], o: &[f32], b: &[f32]) {
    for i in 0..x.len() {
        x[i] += o[i] + b[i];
    }
}
}

simd_fn! {
/// GLU of (a + ba, g + bg): o = (a + ba) * sigmoid(g + bg).
fn glu_bias_row(o: &mut [f32], a: &[f32], g: &[f32], ba: &[f32], bg: &[f32]) {
    for i in 0..o.len() {
        o[i] = (a[i] + ba[i]) * sigmoid_g::<F>(g[i] + bg[i]);
    }
}
}

simd_fn! {
/// exp(scale * row - max) in place (weights below e^-40 set to 0, as ops::softmax_row: they
/// cannot change a float32 sum and would only cost subnormal arithmetic later), returning
/// 1 / their sum: the caller scales the attention output row instead of the weights (one pass
/// fewer; PyTorch's CPU flash attention also normalises at the end).
fn softmax_unnorm(row: &mut [f32], scale: f32) -> f32 {
    // Three simple passes (max, exp, sum): plain loops that the loop vectoriser handles fully;
    // fusing exp and the f64 sum into one 8-lane loop left the exp scalar on x86-64.
    let mut m8 = [f32::NEG_INFINITY; 8];
    let chunks = row.chunks_exact(8);
    for &s in chunks.remainder() {
        m8[0] = m8[0].max(s);
    }
    for c in chunks {
        for i in 0..8 {
            m8[i] = m8[i].max(c[i]);
        }
    }
    let mx = m8.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) * scale;
    for v in row.iter_mut() {
        let x = *v * scale - mx;
        // exp of the clamped value: every lane computes exp, and far below -40 it would give
        // subnormals (microcode assists); those lanes are 0 anyway.
        *v = if x < -40.0 { 0.0 } else { exp_g::<F>(x.max(-40.0)) };
    }
    (1.0 / crate::ops::sum64(row)) as f32
}
}

simd_fn! {
/// The exp pass of `softmax_unnorm` alone (micro-benchmarks).
fn exp_pass_row(row: &mut [f32], scale: f32, mx: f32) {
    for v in row.iter_mut() {
        let x = *v * scale - mx;
        *v = if x < -40.0 { 0.0 } else { exp_g::<F>(x.max(-40.0)) };
    }
}
}

simd_fn! {
/// Attention output of one head scaled by sigmoid(gate + bias), one row.
fn gate_row(o: &mut [f32], g: f32) {
    let s = sigmoid_g::<F>(g);
    for v in o.iter_mut() {
        *v *= s;
    }
}
}

/// Rotary embedding tables (rotary_embedding_torch 0.3.5, `freqs_for='lang'`, interleaved pairs):
/// angle(p, i) = float32(p) * freqs[i] in float32; cos / sin of it rounded to float32.
struct Rope {
    len: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    fn new(len: usize, freqs: &[f32]) -> Self {
        let h = freqs.len();
        let mut cos = vec![0.0f32; len * h];
        let mut sin = vec![0.0f32; len * h];
        for p in 0..len {
            for (i, &f) in freqs.iter().enumerate() {
                let a = (p as f32 * f) as f64;
                cos[p * h + i] = a.cos() as f32;
                sin[p * h + i] = a.sin() as f32;
            }
        }
        Rope { len, cos, sin }
    }

    /// Rotates the `heads` (head, dim_head) vectors of row `x` at sequence position `p`:
    /// out = t * cos + rotate_half(t) * sin, rotate_half pairing (2i, 2i+1) -> (-x[2i+1], x[2i]).
    #[inline]
    fn apply(&self, x: &mut [f32], p: usize, dim_head: usize) {
        let h = dim_head / 2;
        rope_row(x, &self.cos[p * h..(p + 1) * h], &self.sin[p * h..(p + 1) * h]);
    }
}

simd_fn! {
/// One row's heads rotated (see `Rope::apply`): pairs (a, b) -> (a c - b s, b c + a s), each
/// product rounded, as torch's t * cos + rotate_half(t) * sin.
fn rope_row(x: &mut [f32], c: &[f32], s: &[f32]) {
    let h = c.len();
    for head in x.chunks_exact_mut(2 * h) {
        for ((pr, &ci), &si) in head.chunks_exact_mut(2).zip(c).zip(s) {
            let (a, b) = (pr[0], pr[1]);
            pr[0] = a * ci + -b * si;
            pr[1] = b * ci + a * si;
        }
    }
}
}

/// One `Transformer` of depth 1: Attention (RMSNorm, qkv, rotary, attention, gates, to_out),
/// FeedForward (RMSNorm, Linear, GELU, Linear), each residual, then the output RMSNorm.
struct Block {
    attn_norm: Vec<f32>,
    /// to_qkv (3 * inner rows: q, k, v, each (head, dim_head)) then to_gates (heads rows).
    w_qkvg: Vec<f32>,
    b_gates: Vec<f32>,
    w_out: Vec<f32>,
    ff_norm: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    out_norm: Vec<f32>,
    freqs: Vec<f32>,
}

/// Per-task buffers.
#[derive(Default)]
struct Scratch {
    h: Vec<f32>,
    big: Vec<f32>,
    ao: Vec<f32>,
    o: Vec<f32>,
    scores: Vec<f32>,
    x: Vec<f32>,
    inv: Vec<f32>,
}

fn grow(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

/// A tuning knob read once from the environment (benchmarks), else `default`. Shapes only: the
/// result never depends on the thread count.
fn knob(name: &str, cell: &'static std::sync::OnceLock<usize>, default: usize) -> usize {
    *cell.get_or_init(|| std::env::var(name).ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(default))
}

/// Query rows per attention tile (the score tile is rows x seq_len); MR_ATTN_ROWS overrides.
fn attn_rows() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    knob("MR_ATTN_ROWS", &C, 128)
}

impl Block {
    fn load(st: &mut SafeTensors, p: &str, heads: usize) -> Self {
        let mut t = |n: &str| st.take(&format!("{p}.{n}")).data;
        let mut w_qkvg = t("layers.0.0.to_qkv.weight");
        w_qkvg.extend_from_slice(&t("layers.0.0.to_gates.weight"));
        let b = Block {
            attn_norm: t("layers.0.0.norm.gamma"),
            w_qkvg,
            b_gates: t("layers.0.0.to_gates.bias"),
            w_out: t("layers.0.0.to_out.0.weight"),
            ff_norm: t("layers.0.1.net.0.gamma"),
            w1: t("layers.0.1.net.1.weight"),
            b1: t("layers.0.1.net.1.bias"),
            w2: t("layers.0.1.net.4.weight"),
            b2: t("layers.0.1.net.4.bias"),
            out_norm: t("norm.gamma"),
            freqs: t("layers.0.0.rotary_embed.freqs"),
        };
        assert_eq!(b.b_gates.len(), heads);
        b
    }

    /// x: `nseq` sequences of `len` tokens, (nseq * len, dim) row-major, transformed in place.
    fn forward(&self, x: &mut [f32], len: usize, rope: &Rope, heads: usize, dim_head: usize, sc: &mut Scratch) {
        let dim = self.attn_norm.len();
        let n = x.len() / dim;
        let inner = heads * dim_head;
        let wq = 3 * inner + heads; // qkv + gates columns
        let hidden = self.b1.len();
        let scale = (dim as f64).sqrt() as f32;
        debug_assert_eq!(rope.len, len);
        // Attention.
        let h = grow(&mut sc.h, n * dim);
        rms_norm_rows(x, h, dim, scale, &self.attn_norm);
        let big = grow(&mut sc.big, n * wq.max(hidden));
        {
            let _p = prof::gemm(prof::PROJ, n, wq, dim);
            sgemm_par(n, wq, dim, h, dim, 1, &self.w_qkvg, 1, dim, big, wq, false, false);
        }
        {
            let _p = prof::scope(prof::ROPE);
            for (r, row) in big.chunks_exact_mut(wq).take(n).enumerate() {
                let p = r % len;
                rope.apply(&mut row[..inner], p, dim_head);
                rope.apply(&mut row[inner..2 * inner], p, dim_head);
            }
        }
        let ao = grow(&mut sc.ao, n * inner);
        let rows = attn_rows();
        let tile = grow(&mut sc.scores, rows.min(len) * len);
        let sm = 1.0 / (dim_head as f32).sqrt();
        for s0 in (0..n).step_by(len) {
            for hd in 0..heads {
                let q = &big[s0 * wq + hd * dim_head..];
                let k = &big[s0 * wq + inner + hd * dim_head..];
                let v = &big[s0 * wq + 2 * inner + hd * dim_head..];
                for r0 in (0..len).step_by(rows) {
                    let m = rows.min(len - r0);
                    let t = &mut tile[..m * len];
                    {
                        let _p = prof::gemm(prof::QK, m, len, dim_head);
                        sgemm_par(m, len, dim_head, &q[r0 * wq..], wq, 1, k, 1, wq, t, len, false, false);
                    }
                    let inv = &mut sc.inv;
                    {
                        let _p = prof::scope(prof::SOFTMAX);
                        inv.clear();
                        inv.extend(t.chunks_exact_mut(len).map(|row| softmax_unnorm(row, sm)));
                    }
                    let o = &mut ao[(s0 + r0) * inner + hd * dim_head..];
                    {
                        let _p = prof::gemm(prof::AV, m, dim_head, len);
                        sgemm_par(m, dim_head, len, t, len, 1, v, wq, 1, o, inner, false, false);
                    }
                    let _p = prof::scope(prof::SOFTMAX);
                    for (r, &s) in inv.iter().enumerate() {
                        o[r * inner..r * inner + dim_head].iter_mut().for_each(|x| *x *= s);
                    }
                }
            }
        }
        {
            let _p = prof::scope(prof::GATES);
            for (r, row) in ao.chunks_exact_mut(inner).enumerate() {
                let g = &big[r * wq + 3 * inner..r * wq + 3 * inner + heads];
                for (hd, o) in row.chunks_exact_mut(dim_head).enumerate() {
                    gate_row(o, g[hd] + self.b_gates[hd]);
                }
            }
        }
        let o = grow(&mut sc.o, n * dim);
        {
            let _p = prof::gemm(prof::PROJ, n, dim, inner);
            sgemm_par(n, dim, inner, ao, inner, 1, &self.w_out, 1, inner, o, dim, false, false);
        }
        {
            let _p = prof::scope(prof::COPY);
            add_assign(x, o);
        }
        // Feed-forward.
        rms_norm_rows(x, h, dim, scale, &self.ff_norm);
        let f = &mut big[..n * hidden];
        {
            let _p = prof::gemm(prof::FF, n, hidden, dim);
            sgemm_par(n, hidden, dim, h, dim, 1, &self.w1, 1, dim, f, hidden, false, false);
        }
        {
            let _p = prof::scope(prof::ACT);
            f.chunks_exact_mut(hidden).for_each(|r| bias_gelu_row(r, &self.b1));
        }
        {
            let _p = prof::gemm(prof::FF, n, dim, hidden);
            sgemm_par(n, dim, hidden, f, hidden, 1, &self.w2, 1, hidden, o, dim, false, false);
        }
        {
            let _p = prof::scope(prof::COPY);
            for (xr, or) in x.chunks_exact_mut(dim).zip(o.chunks_exact(dim)) {
                residual_bias_row(xr, or, &self.b2);
            }
        }
        // Output norm.
        rms_norm_rows(x, h, dim, scale, &self.out_norm);
        let _p = prof::scope(prof::COPY);
        x.copy_from_slice(&h[..n * dim]);
    }
}

/// RMSNorm of every `dim`-row of x into o.
fn rms_norm_rows(x: &[f32], o: &mut [f32], dim: usize, scale: f32, gamma: &[f32]) {
    let _p = prof::scope(prof::NORM);
    for (xr, hr) in x.chunks_exact(dim).zip(o.chunks_exact_mut(dim)) {
        rms_norm_row(xr, hr, scale, gamma);
    }
}

/// Band split for one band: RMSNorm(dim_in) + Linear(dim_in, dim).
struct BandIn {
    f0: usize,
    nf: usize,
    gamma: Vec<f32>,
    w: Vec<f32>,
    b: Vec<f32>,
}

/// Mask estimator MLP for one band: dim -> hidden -> hidden -> 2 * dim_in, GLU.
struct BandOut {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: Vec<f32>,
}

pub struct MelRoformer {
    pub cfg: Config,
    spec: Spec,
    band_in: Vec<BandIn>,
    layers: Vec<(Block, Block)>,
    band_out: Vec<BandOut>,
}

/// Frames per frequency-transformer task (60 bands each): fixed, so results do not depend on
/// the thread count; MR_FREQ_GROUP overrides.
fn freq_group() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    knob("MR_FREQ_GROUP", &C, 8)
}

impl MelRoformer {
    /// Loads `<dir>/melroformer.json` and `<dir>/<name>.safetensors` (float32 or float16;
    /// `$MELROFORMER_WEIGHTS` overrides the name, e.g. melroformer-f16 for measurements).
    pub fn load(dir: impl AsRef<Path>, name: &str) -> Result<Self, String> {
        let dir = dir.as_ref();
        let name = std::env::var("MELROFORMER_WEIGHTS").unwrap_or_else(|_| name.to_string());
        let cfg = Config::load(dir.join("melroformer.json"))?;
        let mut st = SafeTensors::load(dir.join(format!("{name}.safetensors")))?;
        let m = Self::from_weights(cfg, &mut st)?;
        if !st.tensors.is_empty() {
            let mut left: Vec<_> = st.tensors.keys().cloned().collect();
            left.sort();
            return Err(format!("unused weights: {left:?}"));
        }
        Ok(m)
    }

    fn from_weights(cfg: Config, st: &mut SafeTensors) -> Result<Self, String> {
        let dim = cfg.dim;
        let mut need = |name: String, shape: &[usize]| -> Result<Vec<f32>, String> {
            let t = st.tensors.remove(&name).ok_or_else(|| format!("missing weight {name}"))?;
            if t.shape != shape {
                return Err(format!("{name}: shape {:?}, expected {shape:?}", t.shape));
            }
            Ok(t.data)
        };
        let mut band_in = vec![];
        let mut band_out = vec![];
        let hid = cfg.mask_estimator_hidden;
        for (b, bins) in cfg.bands.iter().enumerate() {
            let din = 4 * bins.len();
            let p = format!("band_split.to_features.{b}");
            band_in.push(BandIn {
                f0: bins[0],
                nf: bins.len(),
                gamma: need(format!("{p}.0.gamma"), &[din])?,
                w: need(format!("{p}.1.weight"), &[dim, din])?,
                b: need(format!("{p}.1.bias"), &[dim])?,
            });
            let p = format!("mask_estimators.0.to_freqs.{b}.0");
            band_out.push(BandOut {
                w1: need(format!("{p}.0.weight"), &[hid, dim])?,
                b1: need(format!("{p}.0.bias"), &[hid])?,
                w2: need(format!("{p}.2.weight"), &[hid, hid])?,
                b2: need(format!("{p}.2.bias"), &[hid])?,
                w3: need(format!("{p}.4.weight"), &[2 * din, hid])?,
                b3: need(format!("{p}.4.bias"), &[2 * din])?,
            });
        }
        let layers = (0..cfg.depth)
            .map(|i| (Block::load(st, &format!("layers.{i}.0"), cfg.heads), Block::load(st, &format!("layers.{i}.1"), cfg.heads)))
            .collect::<Vec<_>>();
        let inner = cfg.heads * cfg.dim_head;
        for (t, f) in &layers {
            for b in [t, f] {
                if b.w_qkvg.len() != (3 * inner + cfg.heads) * dim || b.w_out.len() != dim * inner || b.w1.len() != cfg.ff_mult * dim * dim || b.freqs.len() != cfg.dim_head / 2 {
                    return Err("transformer weight shapes".into());
                }
            }
        }
        let spec = Spec::new(cfg.stft_n_fft, cfg.stft_hop_length);
        Ok(MelRoformer { cfg, spec, band_in, layers, band_out })
    }

    /// Layer i's time transformer on x (band, frame, dim): one task per band.
    fn time_layer(&self, i: usize, x: &mut [f32], frames: usize, rope: &Rope) {
        let (dim, heads, dh) = (self.cfg.dim, self.cfg.heads, self.cfg.dim_head);
        prof::part(1);
        x.par_chunks_mut(frames * dim).for_each_init(Scratch::default, |sc, xb| self.layers[i].0.forward(xb, frames, rope, heads, dh, sc));
    }

    /// Layer i's frequency transformer on x (band, frame, dim): tasks of `freq_group()` frames, each
    /// gathering its (frame, band) tokens, transforming them and scattering them back.
    fn freq_layer(&self, i: usize, x: &mut [f32], frames: usize, rope: &Rope) {
        let (dim, heads, dh, nb) = (self.cfg.dim, self.cfg.heads, self.cfg.dim_head, self.cfg.num_bands);
        assert_eq!(x.len(), nb * frames * dim);
        prof::part(2);
        let fb = &self.layers[i].1;
        let xp = SendPtr(x.as_mut_ptr());
        let fg = freq_group();
        (0..frames.div_ceil(fg)).into_par_iter().for_each_init(Scratch::default, |sc, g| {
            let t0 = g * fg;
            let gt = fg.min(frames - t0);
            let mut loc = std::mem::take(&mut sc.x);
            let l = grow(&mut loc, gt * nb * dim);
            // SAFETY: task g reads and writes only tokens (b, t) with t in [t0, t0 + gt).
            let row = |b: usize, t: usize| unsafe { std::slice::from_raw_parts_mut(xp.get().add((b * frames + t) * dim), dim) };
            {
                let _p = prof::scope(prof::COPY);
                for t in 0..gt {
                    for b in 0..nb {
                        l[(t * nb + b) * dim..(t * nb + b + 1) * dim].copy_from_slice(row(b, t0 + t));
                    }
                }
            }
            fb.forward(l, nb, rope, heads, dh, sc);
            let _p = prof::scope(prof::COPY);
            for t in 0..gt {
                for b in 0..nb {
                    row(b, t0 + t).copy_from_slice(&l[(t * nb + b) * dim..(t * nb + b + 1) * dim]);
                }
            }
            sc.x = loc;
        });
    }

    /// Runs transformer `i` (time or frequency) alone on x in PyTorch's layout for its input -
    /// (band, frame, dim) for time, (frame, band, dim) for frequency - returning the same layout
    /// (the parity check's isolated per-layer comparison).
    pub fn transformer(&self, i: usize, freq: bool, x: &Tensor) -> Tensor {
        let (nb, dim) = (self.cfg.num_bands, self.cfg.dim);
        if freq {
            let frames = x.shape[0];
            let mut b = frame_to_band(&x.data, nb, frames, dim);
            self.freq_layer(i, &mut b, frames, &Rope::new(nb, &self.layers[i].1.freqs));
            Tensor::new(&x.shape, band_to_frame(&b, nb, frames, dim))
        } else {
            let frames = x.shape[1];
            let mut b = x.data.clone();
            self.time_layer(i, &mut b, frames, &Rope::new(frames, &self.layers[i].0.freqs));
            Tensor::new(&x.shape, b)
        }
    }

    pub fn sources(&self) -> [&'static str; 2] {
        ["vocals", "instrumental"]
    }

    /// One chunk: `mix` is 2 channels x len samples (len a multiple of the hop); returns the
    /// vocals, 2 x len.
    pub fn forward(&self, mix: &[f32], tr: &mut Tracer) -> Vec<f32> {
        let cfg = &self.cfg;
        let len = mix.len() / 2;
        let (dim, nb) = (cfg.dim, cfg.num_bands);
        let bins = self.spec.bins();
        let frames = self.spec.frames(len);
        let spec = self.spec.stft(mix, len);
        if tr.on() {
            // torch layout (channel, bin, frame, re/im)
            let mut t = vec![0.0f32; spec.len()];
            for fr in 0..frames {
                for f in 0..bins {
                    for s in 0..2 {
                        for c in 0..2 {
                            t[((s * bins + f) * frames + fr) * 2 + c] = spec[((fr * bins + f) * 2 + s) * 2 + c];
                        }
                    }
                }
            }
            tr.emit("stft", &Tensor::new(&[2, bins, frames, 2], t));
        }

        // Band split -> x: (band, frame, dim).
        let mut x = vec![0.0f32; nb * frames * dim];
        prof::part(0);
        x.par_chunks_mut(frames * dim).zip(&self.band_in).for_each_init(Vec::new, |buf, (xb, bi)| {
            let din = 4 * bi.nf;
            let inp = grow(buf, frames * din);
            {
                let _p = prof::scope(prof::NORM);
                for (t, row) in inp.chunks_exact_mut(din).enumerate() {
                    let src = &spec[(t * bins + bi.f0) * 4..(t * bins + bi.f0) * 4 + din];
                    rms_norm_row(src, row, (din as f64).sqrt() as f32, &bi.gamma);
                }
            }
            let _p = prof::gemm(prof::SPLIT, frames, dim, din);
            sgemm_par(frames, dim, din, inp, din, 1, &bi.w, 1, din, xb, dim, false, false);
            for r in xb.chunks_exact_mut(dim) {
                add_assign(r, &bi.b);
            }
        });
        if tr.on() {
            tr.emit("band_split", &Tensor::new(&[frames, nb, dim], band_to_frame(&x, nb, frames, dim)));
        }

        let rope_t = Rope::new(frames, &self.layers[0].0.freqs);
        let rope_f = Rope::new(nb, &self.layers[0].1.freqs);
        for i in 0..self.layers.len() {
            self.time_layer(i, &mut x, frames, &rope_t);
            if tr.on() {
                tr.emit(&format!("layers.{i}.time"), &Tensor::new(&[nb, frames, dim], x.clone()));
            }
            self.freq_layer(i, &mut x, frames, &rope_f);
            if tr.on() {
                tr.emit(&format!("layers.{i}.freq"), &Tensor::new(&[frames, nb, dim], band_to_frame(&x, nb, frames, dim)));
            }
        }

        // Mask estimators: per band (frames, dim_in) complex masks, (bin, channel, re/im) per frame.
        prof::part(3);
        let masks: Vec<Vec<f32>> = {
            x.par_chunks(frames * dim)
                .zip(self.band_out.par_iter().zip(&self.band_in))
                .map_init(
                    || (Vec::new(), Vec::new()),
                    |(h1, h2), (xb, (bo, bi))| {
                        let hid = bo.b1.len();
                        let din = 4 * bi.nf;
                        let h1 = grow(h1, frames * hid.max(2 * din));
                        let h2 = grow(h2, frames * hid);
                        {
                            let _p = prof::gemm(prof::MASKMLP, frames, hid, dim);
                            sgemm_par(frames, hid, dim, xb, dim, 1, &bo.w1, 1, dim, h1, hid, false, false);
                        }
                        {
                            let _p = prof::scope(prof::ACT);
                            h1[..frames * hid].chunks_exact_mut(hid).for_each(|r| bias_tanh_row(r, &bo.b1));
                        }
                        {
                            let _p = prof::gemm(prof::MASKMLP, frames, hid, hid);
                            sgemm_par(frames, hid, hid, h1, hid, 1, &bo.w2, 1, hid, h2, hid, false, false);
                        }
                        {
                            let _p = prof::scope(prof::ACT);
                            h2.chunks_exact_mut(hid).for_each(|r| bias_tanh_row(r, &bo.b2));
                        }
                        let o = &mut h1[..frames * 2 * din];
                        {
                            let _p = prof::gemm(prof::MASKMLP, frames, 2 * din, hid);
                            sgemm_par(frames, 2 * din, hid, h2, hid, 1, &bo.w3, 1, hid, o, 2 * din, false, false);
                        }
                        let _p = prof::scope(prof::ACT);
                        let mut m = vec![0.0f32; frames * din];
                        for (mr, or) in m.chunks_exact_mut(din).zip(o.chunks_exact(2 * din)) {
                            glu_bias_row(mr, &or[..din], &or[din..], &bo.b3[..din], &bo.b3[din..]);
                        }
                        m
                    },
                )
                .collect()
        };
        drop(x);
        if tr.on() {
            let tot: usize = masks.iter().map(|m| m.len() / frames).sum();
            let mut t = Vec::with_capacity(frames * tot);
            for fr in 0..frames {
                for m in &masks {
                    let din = m.len() / frames;
                    t.extend_from_slice(&m[fr * din..(fr + 1) * din]);
                }
            }
            tr.emit("mask", &Tensor::new(&[frames, tot], t));
        }

        // Average the masks of bins in several bands, apply (complex multiply), iSTFT.
        prof::part(0);
        let mut y = spec;
        {
            let nbpf = &cfg.num_bands_per_freq;
            y.par_chunks_mut(bins * 4).enumerate().for_each(|(t, row)| {
                let _p = prof::scope(prof::APPLY);
                let mut acc = vec![0.0f32; bins * 4];
                for (m, bi) in masks.iter().zip(&self.band_in) {
                    let din = 4 * bi.nf;
                    for (a, v) in acc[bi.f0 * 4..bi.f0 * 4 + din].iter_mut().zip(&m[t * din..(t + 1) * din]) {
                        *a += v;
                    }
                }
                for f in 0..bins {
                    let d = nbpf[f] as f32;
                    for s in 0..2 {
                        let i = f * 4 + s * 2;
                        let (mr, mi) = (acc[i] / d, acc[i + 1] / d);
                        let (xr, xi) = (row[i], row[i + 1]);
                        row[i] = xr * mr - xi * mi;
                        row[i + 1] = xr * mi + xi * mr;
                    }
                }
            });
        }
        let out = self.spec.istft(&y, frames);
        tr.emit("output", &Tensor::new(&[2, len], out.clone()));
        out
    }
}

/// Upstream `demix_track`'s chunking: chunk length C, step C / num_overlap, fades of C / 10 at
/// both ends of each chunk's window (none at the track's two ends), outputs overlap-added and
/// divided by the summed windows; the track is reflect-padded by C - step at both ends when it
/// is longer than twice that.
pub struct Chunking {
    pub chunk: usize,
    pub step: usize,
    pub fade: usize,
    pub border: usize,
}

impl Chunking {
    pub fn new(cfg: &Config, num_overlap: usize) -> Result<Self, String> {
        let c = cfg.chunk_size;
        if num_overlap == 0 || num_overlap > c {
            return Err(format!("num_overlap must be in 1..={c}"));
        }
        let step = c / num_overlap;
        Ok(Chunking { chunk: c, step, fade: c / 10, border: c - step })
    }

    fn padded(&self, len: usize) -> bool {
        self.border > 0 && len > 2 * self.border
    }

    /// Length of the (padded) track the chunks walk over.
    pub fn total(&self, len: usize) -> usize {
        if self.padded(len) { len + 2 * self.border } else { len }
    }

    /// Number of model forwards for a track of `len` samples.
    pub fn chunks(&self, len: usize) -> usize {
        self.total(len).div_ceil(self.step)
    }

    /// get_windowing_array: ones with linspace fades (torch.linspace's float32 arithmetic).
    fn window(&self) -> Vec<f32> {
        let (c, fade) = (self.chunk, self.fade);
        let lin = |start: f32, end: f32| -> Vec<f32> {
            if fade == 1 {
                return vec![start];
            }
            let step = (end - start) / (fade - 1) as f32;
            let half = fade / 2;
            (0..fade).map(|i| if i < half { start + step * i as f32 } else { end - step * (fade - i - 1) as f32 }).collect()
        };
        let mut w = vec![1.0f32; c];
        for (v, f) in w[c - fade..].iter_mut().zip(lin(1.0, 0.0)) {
            *v *= f;
        }
        for (v, f) in w[..fade].iter_mut().zip(lin(0.0, 1.0)) {
            *v *= f;
        }
        w
    }
}

/// Separates a track (2 channels x len, 44.1 kHz) as upstream's demix_track; returns the
/// vocals, 2 x len. `progress` is called after every chunk.
///
/// Streams: the padded track is read on demand (never copied), and each chunk's weighted output
/// is added into a one-chunk window; positions no later chunk reaches are divided by their summed
/// weights and written to the output as the walk goes (same operations, same order as upstream's
/// whole-track `result` / `counter` buffers).
pub fn separate(model: &MelRoformer, wav: &[f32], ch: &Chunking, progress: &mut dyn FnMut()) -> Vec<f32> {
    separate_with(wav, ch, &|part: &[f32]| model.forward(part, &mut Tracer::none()), progress)
}

/// `separate` with any chunk model (2 x chunk samples in, 2 x chunk out).
pub fn separate_with(wav: &[f32], ch: &Chunking, model: &dyn Fn(&[f32]) -> Vec<f32>, progress: &mut dyn FnMut()) -> Vec<f32> {
    let len = wav.len() / 2;
    let c = ch.chunk;
    let total = ch.total(len);
    let border = if ch.padded(len) { ch.border } else { 0 };
    // Sample p of the (padded) track, channel s: reflect padding without edge repetition.
    let at = |s: usize, p: usize| -> f32 {
        let mut j = p as isize - border as isize;
        if j < 0 {
            j = -j;
        }
        if j >= len as isize {
            j = 2 * (len as isize - 1) - j;
        }
        wav[s * len + j as usize]
    };
    let base = ch.window();
    let mut out = vec![0.0f32; 2 * len];
    // Window [wbase, wbase + c) of upstream's result (2 rows) and counter.
    let mut res = vec![0.0f32; 2 * c];
    let mut cnt = vec![0.0f32; c];
    let mut wbase = 0usize;
    let mut flush = |res: &mut Vec<f32>, cnt: &mut Vec<f32>, wbase: &mut usize, upto: usize| {
        let w = upto - *wbase;
        if w == 0 {
            return;
        }
        assert!(w <= c);
        for s in 0..2 {
            for k in 0..w {
                let p = *wbase + k;
                if p >= border && p < border + len {
                    let v = res[s * c + k] / cnt[k];
                    out[s * len + p - border] = if v.is_nan() { 0.0 } else { v.clamp(f32::MIN, f32::MAX) };
                }
            }
            res.copy_within(s * c + w..(s + 1) * c, s * c);
            res[(s + 1) * c - w..(s + 1) * c].fill(0.0);
        }
        cnt.copy_within(w.., 0);
        cnt[c - w..].fill(0.0);
        *wbase = upto;
    };
    let mut part = vec![0.0f32; 2 * c];
    let mut i = 0;
    while i < total {
        flush(&mut res, &mut cnt, &mut wbase, i);
        let length = c.min(total - i);
        part.par_chunks_mut(c).enumerate().for_each(|(s, row)| {
            for (q, v) in row.iter_mut().enumerate() {
                *v = if q < length {
                    at(s, i + q)
                } else if length > c / 2 + 1 {
                    at(s, i + 2 * (length - 1) - q) // reflect-pad the short last chunk
                } else {
                    0.0
                };
            }
        });
        let y = model(&part);
        progress();
        let mut w = base.clone();
        if i == 0 {
            w[..ch.fade].fill(1.0);
        } else if i + c >= total {
            w[c - ch.fade..].fill(1.0);
        }
        let rel = i - wbase;
        for s in 0..2 {
            for q in 0..length {
                res[s * c + rel + q] += y[s * c + q] * w[q];
            }
        }
        for q in 0..length {
            cnt[rel + q] += w[q];
        }
        i += ch.step;
    }
    flush(&mut res, &mut cnt, &mut wbase, total);
    out
}

/// (frame, band, dim) -> (band, frame, dim).
fn frame_to_band(x: &[f32], nb: usize, frames: usize, dim: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; x.len()];
    for b in 0..nb {
        for t in 0..frames {
            o[(b * frames + t) * dim..(b * frames + t + 1) * dim].copy_from_slice(&x[(t * nb + b) * dim..(t * nb + b + 1) * dim]);
        }
    }
    o
}

/// (band, frame, dim) -> (frame, band, dim).
fn band_to_frame(x: &[f32], nb: usize, frames: usize, dim: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; x.len()];
    for b in 0..nb {
        for t in 0..frames {
            o[(t * nb + b) * dim..(t * nb + b + 1) * dim].copy_from_slice(&x[(b * frames + t) * dim..(b * frames + t + 1) * dim]);
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Fused, Unfused};

    fn tanh_worst<F: MulAdd>() -> f64 {
        let mut worst = 0f64;
        let mut x = -20.0f32;
        while x < 20.0 {
            let t = (x as f64).tanh();
            worst = worst.max((tanh_g::<F>(x) as f64 - t).abs() / t.abs().max(1e-30));
            x += 0.000_731;
        }
        worst
    }

    #[test]
    fn tanh_accuracy() {
        for (name, w) in [("unfused", tanh_worst::<Unfused>()), ("fused", tanh_worst::<Fused>())] {
            assert!(w < 4e-7, "tanh ({name}) rel err {w}");
        }
        assert_eq!(tanh_g::<Unfused>(0.0), 0.0);
        assert_eq!(tanh_g::<Unfused>(100.0), 1.0);
        assert_eq!(tanh_g::<Unfused>(-100.0), -1.0);
    }

    /// Upstream demix_track with whole-track buffers, literally (padded copy, result, counter).
    fn demix_reference(wav: &[f32], ch: &Chunking, model: &dyn Fn(&[f32]) -> Vec<f32>) -> Vec<f32> {
        let len = wav.len() / 2;
        let c = ch.chunk;
        let pad = ch.padded(len);
        let b = if pad { ch.border } else { 0 };
        let total = len + 2 * b;
        let refl = |j: isize, n: usize| -> usize {
            let j = if j < 0 { -j } else { j };
            (if j >= n as isize { 2 * (n as isize - 1) - j } else { j }) as usize
        };
        let mix: Vec<f32> = (0..2).flat_map(|s| (0..total).map(move |p| (s, p))).map(|(s, p)| wav[s * len + refl(p as isize - b as isize, len)]).collect();
        let w0 = ch.window();
        let mut result = vec![0.0f32; 2 * total];
        let mut counter = vec![0.0f32; total];
        let mut i = 0;
        while i < total {
            let length = c.min(total - i);
            let mut part = vec![0.0f32; 2 * c];
            for s in 0..2 {
                for q in 0..c {
                    part[s * c + q] = if q < length {
                        mix[s * total + i + q]
                    } else if length > c / 2 + 1 {
                        mix[s * total + i + refl(q as isize, length)]
                    } else {
                        0.0
                    };
                }
            }
            let y = model(&part);
            let mut w = w0.clone();
            if i == 0 {
                w[..ch.fade].fill(1.0);
            } else if i + c >= total {
                w[c - ch.fade..].fill(1.0);
            }
            for s in 0..2 {
                for q in 0..length {
                    result[s * total + i + q] += y[s * c + q] * w[q];
                }
            }
            for q in 0..length {
                counter[i + q] += w[q];
            }
            i += ch.step;
        }
        (0..2).flat_map(|s| (b..b + len).map(move |p| (s, p))).map(|(s, p)| {
            let v = result[s * total + p] / counter[p];
            if v.is_nan() { 0.0 } else { v }
        }).collect()
    }

    #[test]
    fn streamed_demix_matches_whole_track() {
        // A toy chunk "model" that depends on the position in the chunk, so misplaced chunks or
        // windows show; small chunks so every branch (padding, short last chunk reflect / zero
        // padded, one-chunk tracks) is reached.
        let model = |x: &[f32]| -> Vec<f32> { x.iter().enumerate().map(|(i, v)| v * 0.5 + (i % 7) as f32 * 1e-3).collect() };
        for (chunk, n) in [(100usize, 2usize), (100, 4), (100, 1), (120, 3)] {
            let step = chunk / n;
            let ch = Chunking { chunk, step, fade: chunk / 10, border: chunk - step };
            for len in [1usize, 2, 30, 52, 99, 100, 101, 150, 199, 200, 201, 260, 333, 1000, 1037] {
                let wav: Vec<f32> = (0..2 * len).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
                let mut calls = 0;
                let a = separate_with(&wav, &ch, &model, &mut || calls += 1);
                let b = demix_reference(&wav, &ch, &model);
                assert_eq!(calls, ch.chunks(len), "chunk {chunk} n {n} len {len}");
                assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "chunk {chunk} n {n} len {len}");
            }
        }
    }

    #[test]
    fn fade_window() {
        let ch = Chunking { chunk: 352800, step: 176400, fade: 35280, border: 176400 };
        let w = ch.window();
        assert_eq!((w[0], w[35279], w[35280], w[352800 - 35281], w[352799]), (0.0, 1.0, 1.0, 1.0, 0.0));
        assert!((w[1] - 1.0 / 35279.0).abs() < 1e-9);
        assert_eq!(ch.chunks(10 * 44100), 5); // padded to 441000 + 2 * 176400 samples
        assert_eq!(ch.chunks(8 * 44100), 2); // not padded (not longer than 2 * border)
    }

    #[test]
    fn stft_round_trip() {
        // iSTFT(STFT(x)) = x (Hann at hop 441 < n_fft / 4: the envelope never vanishes).
        let s = Spec::new(2048, 441);
        let len = 441 * 40;
        let x: Vec<f32> = (0..2 * len).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * 0.5).collect();
        let y = s.istft(&s.stft(&x, len), s.frames(len));
        let err = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1e-6, "round trip error {err}");
    }
}

/// Kernel entry points for micro-benchmarks (examples/).
#[doc(hidden)]
pub mod micro {
    pub fn softmax(rows: &mut [f32], n: usize) {
        rows.chunks_exact_mut(n).for_each(|r| {
            super::softmax_unnorm(r, 0.125);
        });
    }
    pub fn exp_pass(rows: &mut [f32], n: usize) {
        rows.chunks_exact_mut(n).for_each(|r| super::exp_pass_row(r, 0.125, 3.0));
    }
    pub fn sum_pass(rows: &[f32], n: usize) -> f64 {
        rows.chunks_exact(n).map(crate::ops::sum64).sum()
    }
    pub fn bias_gelu(x: &mut [f32], b: &[f32]) {
        x.chunks_exact_mut(b.len()).for_each(|r| super::bias_gelu_row(r, b));
    }
    pub fn bias_tanh(x: &mut [f32], b: &[f32]) {
        x.chunks_exact_mut(b.len()).for_each(|r| super::bias_tanh_row(r, b));
    }
    pub fn rms(x: &[f32], o: &mut [f32], g: &[f32]) {
        for (a, b) in x.chunks_exact(g.len()).zip(o.chunks_exact_mut(g.len())) {
            super::rms_norm_row(a, b, 19.6, g);
        }
    }
}
