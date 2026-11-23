//! The numeric kernels HTDemucs needs. Every FLOP-heavy op is a single-precision GEMM (convolutions
//! via im2col / col2im, linear layers, attention); the rest are cheap elementwise / reduction passes.
//!
//! Precision notes (we aim to match PyTorch CPU float32 to ~1e-6 relative per layer):
//! * normalisation statistics (GroupNorm, LayerNorm, the model's input standardisation) are
//!   accumulated in f64 with a two-pass mean/variance, so they are the correctly rounded values
//!   PyTorch's float32 reductions approximate;
//! * GELU is the exact erf form (PyTorch's default `approximate='none'`).

use rayon::prelude::*;

use crate::tensor::Tensor;

/// Op-class profiler (off unless `prof::enable(true)`): wall time per (branch, op class), summed
/// over calls. Scopes are placed at leaf level (never nested), on the calling thread.
pub mod prof {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
    use std::time::Instant;

    pub const CLASSES: [&str; 15] = [
        "conv gemm",
        "conv im2col / tap shift-add",
        "conv bias fill",
        "conv_tr gemm",
        "conv_tr col2im",
        "linear gemm",
        "attn QK^T gemm",
        "attn softmax",
        "attn AV gemm",
        "groupnorm (+gelu / glu)",
        "layernorm",
        "gelu / glu",
        "stft",
        "istft",
        "other (layout, copies, adds, standardise)",
    ];
    pub const CONV: usize = 0;
    pub const IM2COL: usize = 1;
    pub const BIAS: usize = 2;
    pub const CONVTR: usize = 3;
    pub const COL2IM: usize = 4;
    pub const LINEAR: usize = 5;
    pub const QK: usize = 6;
    pub const SOFTMAX: usize = 7;
    pub const AV: usize = 8;
    pub const GROUPNORM: usize = 9;
    pub const LAYERNORM: usize = 10;
    pub const ACT: usize = 11;
    pub const STFT: usize = 12;
    pub const ISTFT: usize = 13;
    pub const COPY: usize = 14;
    pub const BRANCHES: [&str; 3] = ["freq", "time", "transformer"];
    const N: usize = CLASSES.len() * BRANCHES.len();

    static ON: AtomicBool = AtomicBool::new(false);
    static BRANCH: AtomicUsize = AtomicUsize::new(0);
    static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    static FLOPS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    static CLASS: AtomicUsize = AtomicUsize::new(0);

    pub fn enable(on: bool) {
        ON.store(on, Relaxed);
    }

    /// Which branch subsequent scopes are charged to (0 freq, 1 time, 2 transformer).
    pub fn branch(b: usize) {
        BRANCH.store(b, Relaxed);
    }

    pub struct Scope(usize, Option<Instant>);

    #[inline]
    pub fn scope(class: usize) -> Scope {
        let on = ON.load(Relaxed);
        if on {
            CLASS.store(class, Relaxed);
        }
        Scope(class, on.then(Instant::now))
    }

    /// Counts a GEMM's floating-point operations against the current scope's class.
    #[inline]
    pub fn gemm_flops(m: usize, n: usize, k: usize) {
        if ON.load(Relaxed) {
            let i = BRANCH.load(Relaxed) * CLASSES.len() + CLASS.load(Relaxed);
            FLOPS[i].fetch_add((2 * m * n * k) as u64, Relaxed);
        }
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            if let Some(t) = self.1 {
                let i = BRANCH.load(Relaxed) * CLASSES.len() + self.0;
                NS[i].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            }
        }
    }

    /// (seconds, GEMM flops) per [branch][class] since the last call; resets.
    pub fn take() -> Vec<Vec<(f64, f64)>> {
        (0..BRANCHES.len())
            .map(|b| {
                (0..CLASSES.len())
                    .map(|c| {
                        let i = b * CLASSES.len() + c;
                        (NS[i].swap(0, Relaxed) as f64 * 1e-9, FLOPS[i].swap(0, Relaxed) as f64)
                    })
                    .collect()
            })
            .collect()
    }
}

/// True if the CPU has AVX2 and FMA (x86-64): elementwise kernels then run in versions compiled
/// for them. (Without target-cpu flags, x86-64 code is SSE2-only; the GEMMs dispatch on their own.)
/// Rust never contracts a * b + c into an FMA, so both versions compute the same bits.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn avx2_fma() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    static S: AtomicU8 = AtomicU8::new(0);
    match S.load(Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var_os("HTDEMUCS_NO_AVX2").is_none()
                && std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma");
            S.store(if on { 1 } else { 2 }, Relaxed);
            on
        }
    }
}

/// Defines a function whose body is compiled twice on x86-64 - for the baseline and for
/// AVX2 + FMA - and picks one at run time (see `avx2_fma`). Elsewhere, just the function.
macro_rules! simd_fn {
    ($(#[$m:meta])* $vis:vis fn $name:ident($($a:ident : $t:ty),* $(,)?) $(-> $r:ty)? $body:block) => {
        $(#[$m])*
        #[allow(clippy::too_many_arguments)]
        $vis fn $name($($a: $t),*) $(-> $r)? {
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            fn imp($($a: $t),*) $(-> $r)? $body
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn v3($($a: $t),*) $(-> $r)? {
                    imp($($a),*)
                }
                if crate::ops::avx2_fma() {
                    return unsafe { v3($($a),*) };
                }
            }
            imp($($a),*)
        }
    };
}
#[allow(unused_imports)]
pub(crate) use simd_fn;

/// C (row-major m x n) = A (m x k) * B (k x n) [+ C if `accumulate`], where A and B are given as
/// strided views: element (i, j) of A is a[i * a_rs + j * a_cs].
#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    a_rs: usize,
    a_cs: usize,
    b: &[f32],
    b_rs: usize,
    b_cs: usize,
    c: &mut [f32],
    c_rs: usize,
    accumulate: bool,
) {
    sgemm_par(m, n, k, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, accumulate, true)
}

/// `sgemm`, single-threaded unless `parallel` (for callers that parallelise over many GEMMs).
#[allow(clippy::too_many_arguments)]
pub fn sgemm_par(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    a_rs: usize,
    a_cs: usize,
    b: &[f32],
    b_rs: usize,
    b_cs: usize,
    c: &mut [f32],
    c_rs: usize,
    accumulate: bool,
    parallel: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(k == 0 || (m - 1) * a_rs + (k - 1) * a_cs < a.len());
    assert!(k == 0 || (k - 1) * b_rs + (n - 1) * b_cs < b.len());
    assert!((m - 1) * c_rs + n <= c.len());
    prof::gemm_flops(m, n, k);
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        let _ = parallel; // Accelerate manages its own threads
        accelerate::sgemm(m, n, k, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, accumulate);
    }
    // gemm computes dst = alpha * dst + beta * lhs * rhs. It never splits the k dimension across
    // threads, so results do not depend on the thread count.
    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            c_rs as isize,
            accumulate,
            a.as_ptr(),
            a_cs as isize,
            a_rs as isize,
            b.as_ptr(),
            b_cs as isize,
            b_rs as isize,
            1.0f32,
            1.0f32,
            false,
            false,
            false,
            if parallel { gemm::Parallelism::Rayon(rayon::current_num_threads()) } else { gemm::Parallelism::None },
        );
    }
}

/// Apple's Accelerate BLAS (runs on the AMX units of Apple silicon), behind the `accelerate`
/// feature on macOS.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
mod accelerate {
    #[allow(non_camel_case_types)]
    type c_int = i32;
    const ROW_MAJOR: c_int = 101;
    const NO_TRANS: c_int = 111;
    const TRANS: c_int = 112;
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: c_int, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: f32,
            a: *const f32, lda: c_int, b: *const f32, ldb: c_int, beta: f32, c: *mut f32, ldc: c_int,
        );
    }

    /// A strided view with one unit stride as a BLAS (transpose flag, leading dimension).
    fn layout(rs: usize, cs: usize, cols: usize) -> (c_int, c_int) {
        if cs == 1 {
            (NO_TRANS, rs.max(cols).max(1) as c_int)
        } else {
            assert_eq!(rs, 1, "BLAS needs a unit stride");
            (TRANS, cs as c_int)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
        m: usize, n: usize, k: usize, a: &[f32], a_rs: usize, a_cs: usize, b: &[f32], b_rs: usize,
        b_cs: usize, c: &mut [f32], c_rs: usize, accumulate: bool,
    ) {
        let (ta, lda) = layout(a_rs, a_cs, k);
        let (tb, ldb) = layout(b_rs, b_cs, n);
        unsafe {
            cblas_sgemm(
                ROW_MAJOR, ta, tb, m as c_int, n as c_int, k as c_int, 1.0, a.as_ptr(), lda,
                b.as_ptr(), ldb, if accumulate { 1.0 } else { 0.0 }, c.as_mut_ptr(), c_rs as c_int,
            );
        }
    }
}

/// A convolution over (C, H, W) inputs: PyTorch Conv2d, or Conv1d viewed as H = 1.
/// Weight layout (Cout, Cin, KH, KW) row-major, i.e. a (Cout x Cin*KH*KW) matrix.
pub struct Conv {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    pub k: (usize, usize),
    pub stride: (usize, usize),
    pub pad: (usize, usize),
    pub dil: (usize, usize),
    /// For stride-1 convolutions along W only (kh = 1, kw > 1): the weights regrouped tap-major,
    /// (KW * Cout x Cin), so one GEMM computes every tap and the taps are then shift-added -
    /// instead of materialising a KW-times larger im2col buffer.
    w_taps: Option<Vec<f32>>,
}

/// Upper bound on the im2col buffer (floats); larger convolutions are done in column blocks.
const COL_BLOCK: usize = 1 << 24;

impl Conv {
    #[allow(clippy::too_many_arguments)]
    pub fn new(w: Vec<f32>, b: Vec<f32>, cin: usize, cout: usize, k: (usize, usize), stride: (usize, usize), pad: (usize, usize), dil: (usize, usize)) -> Self {
        assert_eq!(w.len(), cout * cin * k.0 * k.1);
        let w_taps = (k.0 == 1 && k.1 > 1 && stride == (1, 1) && pad.0 == 0).then(|| {
            let kw = k.1;
            let mut t = vec![0.0f32; w.len()];
            for kj in 0..kw {
                for co in 0..cout {
                    for ci in 0..cin {
                        t[(kj * cout + co) * cin + ci] = w[(co * cin + ci) * kw + kj];
                    }
                }
            }
            t
        });
        Conv { w, b, cin, cout, k, stride, pad, dil, w_taps }
    }

    pub fn out_size(&self, h: usize, w: usize) -> (usize, usize) {
        let ho = (h + 2 * self.pad.0 - self.dil.0 * (self.k.0 - 1) - 1) / self.stride.0 + 1;
        let wo = (w + 2 * self.pad.1 - self.dil.1 * (self.k.1 - 1) - 1) / self.stride.1 + 1;
        (ho, wo)
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (cin, h, w) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(cin, self.cin, "conv input channels");
        let (ho, wo) = self.out_size(h, w);
        let n = ho * wo;
        let kk = cin * self.k.0 * self.k.1;
        let sc = prof::scope(prof::BIAS);
        let mut out = vec![0.0f32; self.cout * n];
        // Bias first, then accumulate the GEMM into it.
        out.par_chunks_mut(n).zip(self.b.par_iter()).for_each(|(row, &b)| row.fill(b));
        drop(sc);
        let pointwise = self.k == (1, 1) && self.stride == (1, 1) && self.pad == (0, 0);
        if let Some(wt) = &self.w_taps {
            // y[kj] = W_kj x for all taps at once, then out[co, h, o] += y[kj, co, h, o + kj*d - p].
            let kw = self.k.1;
            let hw = h * w;
            let sc = prof::scope(prof::CONV);
            let mut y = vec![0.0f32; kw * self.cout * hw];
            sgemm(kw * self.cout, hw, cin, wt, cin, 1, &x.data, hw, 1, &mut y, hw, false);
            drop(sc);
            let _sc = prof::scope(prof::IM2COL);
            let (d, p) = (self.dil.1 as isize, self.pad.1 as isize);
            let cout = self.cout;
            out.par_chunks_mut(n).enumerate().for_each(|(co, orow)| {
                for kj in 0..kw {
                    let yp = &y[(kj * cout + co) * hw..(kj * cout + co + 1) * hw];
                    let shift = kj as isize * d - p;
                    for hi in 0..h {
                        let src = &yp[hi * w..(hi + 1) * w];
                        let dst = &mut orow[hi * wo..(hi + 1) * wo];
                        // dst[o] += src[o + shift] where 0 <= o + shift < w
                        let lo = (-shift).max(0) as usize;
                        let hi_ = ((w as isize - shift).min(wo as isize)).max(0) as usize;
                        if lo < hi_ {
                            let s0 = (lo as isize + shift) as usize;
                            for (o, v) in dst[lo..hi_].iter_mut().zip(&src[s0..s0 + (hi_ - lo)]) {
                                *o += v;
                            }
                        }
                    }
                }
            });
        } else if pointwise {
            let _sc = prof::scope(prof::CONV);
            sgemm(self.cout, n, kk, &self.w, kk, 1, &x.data, n, 1, &mut out, n, true);
        } else {
            let block = (COL_BLOCK / kk).max(wo).min(n);
            let mut col = vec![0.0f32; kk * block];
            let mut j0 = 0;
            while j0 < n {
                let nb = block.min(n - j0);
                let sc = prof::scope(prof::IM2COL);
                self.im2col(x, ho, wo, j0, nb, &mut col[..kk * nb]);
                drop(sc);
                let _sc = prof::scope(prof::CONV);
                sgemm(self.cout, nb, kk, &self.w, kk, 1, &col, nb, 1, &mut out[j0..], n, true);
                j0 += nb;
            }
        }
        Tensor::new(&[self.cout, ho, wo], out)
    }

    /// col[(ci, ki, kj), j - j0] = x[ci, ho*sh + ki*dh - ph, wo*sw + kj*dw - pw] for output
    /// positions j = ho * Wo + wo in [j0, j0 + nb); zero outside the input.
    fn im2col(&self, x: &Tensor, _ho: usize, wo_n: usize, j0: usize, nb: usize, col: &mut [f32]) {
        let (h, w) = (x.shape[1] as isize, x.shape[2] as isize);
        let (kh, kw) = self.k;
        let (sh, sw) = (self.stride.0 as isize, self.stride.1 as isize);
        let (ph, pw) = (self.pad.0 as isize, self.pad.1 as isize);
        let (dh, dw) = (self.dil.0 as isize, self.dil.1 as isize);
        col.par_chunks_mut(nb).enumerate().for_each(|(r, row)| {
            let ci = r / (kh * kw);
            let ki = ((r / kw) % kh) as isize;
            let kj = (r % kw) as isize;
            let plane = &x.data[ci * (h * w) as usize..(ci + 1) * (h * w) as usize];
            let mut j = j0;
            let end = j0 + nb;
            while j < end {
                let oh = (j / wo_n) as isize;
                let ow0 = j % wo_n;
                let run = (wo_n - ow0).min(end - j); // positions left in this output row
                let dst = &mut row[j - j0..j - j0 + run];
                let ih = oh * sh + ki * dh - ph;
                if ih < 0 || ih >= h {
                    dst.fill(0.0);
                } else {
                    let src = &plane[(ih * w) as usize..((ih + 1) * w) as usize];
                    let base = kj * dw - pw;
                    if sw == 1 {
                        // Contiguous: copy the in-range span, zero the rest.
                        for (t, d) in dst.iter_mut().enumerate() {
                            let iw = (ow0 + t) as isize + base;
                            *d = if iw >= 0 && iw < w { src[iw as usize] } else { 0.0 };
                        }
                    } else {
                        for (t, d) in dst.iter_mut().enumerate() {
                            let iw = (ow0 + t) as isize * sw + base;
                            *d = if iw >= 0 && iw < w { src[iw as usize] } else { 0.0 };
                        }
                    }
                }
                j += run;
            }
        });
    }
}

/// Transposed convolution over (C, H, W) (ConvTranspose2d, or ConvTranspose1d with H = 1), no
/// padding (HTDemucs crops afterwards). Weight layout (Cin, Cout, KH, KW).
pub struct ConvTr {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    pub k: (usize, usize),
    pub stride: (usize, usize),
}

impl ConvTr {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (cin, h, w) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(cin, self.cin, "conv_tr input channels");
        let (kh, kw) = self.k;
        let (sh, sw) = self.stride;
        let (ho, wo) = ((h - 1) * sh + kh, (w - 1) * sw + kw);
        let rows = self.cout * kh * kw;
        let n = h * w;
        // cols = W^T (rows x Cin) * x (Cin x n).
        let sc = prof::scope(prof::CONVTR);
        let mut cols = vec![0.0f32; rows * n];
        sgemm(rows, n, cin, &self.w, 1, rows, &x.data, n, 1, &mut cols, n, false);
        drop(sc);
        let _sc = prof::scope(prof::COL2IM);
        // col2im: scatter-add each kernel tap, in tap order.
        let mut out = vec![0.0f32; self.cout * ho * wo];
        out.par_chunks_mut(ho * wo).enumerate().for_each(|(co, plane)| {
            plane.fill(self.b[co]);
            for ki in 0..kh {
                for kj in 0..kw {
                    let src = &cols[((co * kh + ki) * kw + kj) * n..((co * kh + ki) * kw + kj + 1) * n];
                    for ih in 0..h {
                        let orow = &mut plane[(ih * sh + ki) * wo..(ih * sh + ki + 1) * wo];
                        let srow = &src[ih * w..(ih + 1) * w];
                        if sw == 1 {
                            for (o, s) in orow[kj..kj + w].iter_mut().zip(srow) {
                                *o += s;
                            }
                        } else {
                            for (iw, s) in srow.iter().enumerate() {
                                orow[iw * sw + kj] += s;
                            }
                        }
                    }
                }
            }
        });
        Tensor::new(&[self.cout, ho, wo], out)
    }
}

simd_fn! {
/// Sum in f64 with 8 independent accumulators (vectorises).
fn sum64(x: &[f32]) -> f64 {
    let mut acc = [0f64; 8];
    let chunks = x.chunks_exact(8);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..8 {
            acc[i] += c[i] as f64;
        }
    }
    acc.iter().sum::<f64>() + rest.iter().map(|&v| v as f64).sum::<f64>()
}
}

simd_fn! {
/// Sum of squared deviations from `m`, in f64 (vectorises).
fn sqdev64(x: &[f32], m: f64) -> f64 {
    let mut acc = [0f64; 8];
    let chunks = x.chunks_exact(8);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..8 {
            let d = c[i] as f64 - m;
            acc[i] += d * d;
        }
    }
    acc.iter().sum::<f64>() + rest.iter().map(|&v| (v as f64 - m) * (v as f64 - m)).sum::<f64>()
}
}

/// Mean and biased variance of a large slice, f64 two-pass, in parallel. Partial sums are
/// collected and added in a fixed order, so the result does not depend on the thread count.
fn big_stats(x: &[f32]) -> (f64, f64) {
    const B: usize = 1 << 16;
    let n = x.len() as f64;
    let parts: Vec<f64> = x.par_chunks(B).map(sum64).collect();
    let mean = parts.iter().sum::<f64>() / n;
    let parts: Vec<f64> = x.par_chunks(B).map(|c| sqdev64(c, mean)).collect();
    (mean, parts.iter().sum::<f64>() / n)
}

/// Mean and unbiased (N-1) standard deviation, as torch.mean / torch.std.
pub fn mean_std(x: &[f32]) -> (f32, f32) {
    let (m, v) = big_stats(x);
    let n = x.len() as f64;
    (m as f32, (v * n / (n - 1.0)).sqrt() as f32)
}

/// Per-sample (mean, 1/sqrt(var + 1e-5)) for GroupNorm with ONE group over a (C, S, T) tensor
/// where each index s of the middle axis is a separate sample (PyTorch's batch): statistics over
/// (C, T) for each s. (S = 1 for an ordinary (C, T) input.)
pub fn row_stats(x: &Tensor) -> Vec<(f32, f32)> {
    let (c, s, t) = (x.shape[0], x.shape[1], x.shape[2]);
    let n = (c * t) as f64;
    let sums: Vec<Vec<f64>> = x.data.par_chunks(s * t).map(|p| p.chunks(t).map(sum64).collect()).collect();
    let means: Vec<f64> = (0..s).map(|si| sums.iter().map(|v| v[si]).sum::<f64>() / n).collect();
    let sq: Vec<Vec<f64>> = x
        .data
        .par_chunks(s * t)
        .map(|p| p.chunks(t).zip(&means).map(|(r, &m)| sqdev64(r, m)).collect())
        .collect();
    (0..s)
        .map(|si| {
            let v = sq.iter().map(|q| q[si]).sum::<f64>() / n;
            (means[si] as f32, (1.0 / (v + 1e-5).sqrt()) as f32)
        })
        .collect()
}

/// GroupNorm(1) per sample of a (C, S, T) tensor (see `row_stats`), then `f`, in place.
pub fn group_norm1_then(x: &mut Tensor, gamma: &[f32], beta: &[f32], f: impl Fn(f32) -> f32 + Sync) {
    let _sc = prof::scope(prof::GROUPNORM);
    let st = row_stats(x);
    let (s, t) = (x.shape[1], x.shape[2]);
    x.data.par_chunks_mut(s * t).enumerate().for_each(|(ci, plane)| gn_plane(plane, t, &st, gamma[ci], beta[ci], &f));
}

simd_fn! {
fn gn_plane(plane: &mut [f32], t: usize, st: &[(f32, f32)], g: f32, b: f32, f: &(impl Fn(f32) -> f32 + Sync)) {
    for (row, &(m, r)) in plane.chunks_mut(t).zip(st) {
        let scale = r * g;
        let shift = b - m * scale;
        for v in row {
            *v = f(*v * scale + shift);
        }
    }
}
}

/// GroupNorm(1) per sample of z (2C, S, T), GLU over channels, LayerScale, residual add:
/// x += ls * (a * sigmoid(b)) with (a, b) the normalised halves of z - the tail of a DConv branch.
/// If `delta` is given, the added values are also written there.
pub fn group_norm1_glu_add(z: &Tensor, gamma: &[f32], beta: &[f32], ls: &[f32], x: &mut Tensor, mut delta: Option<&mut Tensor>) {
    let _sc = prof::scope(prof::GROUPNORM);
    let st = row_stats(z);
    let (c, s, t) = (x.shape[0], x.shape[1], x.shape[2]);
    assert_eq!(z.shape, [2 * c, s, t]);
    let plane = s * t;
    match delta.as_deref_mut() {
        None => x.data.par_chunks_mut(plane).enumerate().for_each(|(ci, xp)| glu_add_plane(ci, c, t, xp, None, &z.data, &st, gamma, beta, ls)),
        Some(d) => x
            .data
            .par_chunks_mut(plane)
            .zip(d.data.par_chunks_mut(plane))
            .enumerate()
            .for_each(|(ci, (xp, dp))| glu_add_plane(ci, c, t, xp, Some(dp), &z.data, &st, gamma, beta, ls)),
    }
}

simd_fn! {
fn glu_add_plane(ci: usize, c: usize, t: usize, xp: &mut [f32], dp: Option<&mut [f32]>, z: &[f32], st: &[(f32, f32)], gamma: &[f32], beta: &[f32], ls: &[f32]) {
    let mut dp = dp;
    let plane = xp.len();
    let za = &z[ci * plane..(ci + 1) * plane];
    let zb = &z[(c + ci) * plane..(c + ci + 1) * plane];
    for (si, &(m, r)) in st.iter().enumerate() {
        let (sa, sb) = (r * gamma[ci], r * gamma[c + ci]);
        let (ha, hb) = (beta[ci] - m * sa, beta[c + ci] - m * sb);
        let rg = si * t..(si + 1) * t;
        let (xa, ya, yb) = (&mut xp[rg.clone()], &za[rg.clone()], &zb[rg.clone()]);
        match dp.as_deref_mut() {
            None => {
                for i in 0..t {
                    let d = ls[ci] * ((ya[i] * sa + ha) * sigmoid(yb[i] * sb + hb));
                    xa[i] += d;
                }
            }
            Some(dp) => {
                for i in 0..t {
                    let d = ls[ci] * ((ya[i] * sa + ha) * sigmoid(yb[i] * sb + hb));
                    dp[si * t + i] = d;
                    xa[i] += d;
                }
            }
        }
    }
}
}

/// LayerNorm over the last axis of (N, C). eps = 1e-5.
pub fn layer_norm(x: &Tensor, gamma: &[f32], beta: &[f32]) -> Tensor {
    let _sc = prof::scope(prof::LAYERNORM);
    let c = *x.shape.last().unwrap();
    let mut out = vec![0.0f32; x.numel()];
    out.par_chunks_mut(c).zip(x.data.par_chunks(c)).for_each(|(o, row)| layer_norm_row(row, o, gamma, beta));
    Tensor::new(&x.shape, out)
}

simd_fn! {
/// LayerNorm of one row into `o`.
pub fn layer_norm_row(row: &[f32], o: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = row.len() as f64;
    let m = sum64(row) / n;
    let v = sqdev64(row, m) / n;
    let r = 1.0 / (v + 1e-5).sqrt();
    for i in 0..row.len() {
        o[i] = (((row[i] as f64 - m) * r) as f32) * gamma[i] + beta[i];
    }
}
}

/// GroupNorm with one group over a whole (N, C) token sequence (Demucs' MyGroupNorm on (B, T, C)):
/// one mean / variance for everything, then per-channel affine.
pub fn group_norm_tokens(x: &mut Tensor, gamma: &[f32], beta: &[f32]) {
    let _sc = prof::scope(prof::GROUPNORM);
    let c = *x.shape.last().unwrap();
    let (m, v) = big_stats(&x.data);
    let r = (1.0 / (v + 1e-5).sqrt()) as f32;
    let m = m as f32;
    x.data.par_chunks_mut(c).for_each(|row| affine_row(row, r, m, gamma, beta));
}

simd_fn! {
fn affine_row(row: &mut [f32], r: f32, m: f32, gamma: &[f32], beta: &[f32]) {
    for (i, val) in row.iter_mut().enumerate() {
        let scale = r * gamma[i];
        *val = *val * scale + (beta[i] - m * scale);
    }
}
}

/// exp(x) in f32, ~1 ulp (Cephes expf: Cody-Waite reduction + degree-6 polynomial), written
/// branch-free so loops over it vectorise. Inputs are clamped to the finite normal range.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const C1: f32 = 0.693_359_4;
    const C2: f32 = -2.121_944_4e-4;
    const ROUND: f32 = 12_582_912.0; // 1.5 * 2^23: (v + ROUND) - ROUND rounds to nearest
    let x = x.clamp(-87.336_54, 88.722_83);
    let n = (x * LOG2E + ROUND) - ROUND;
    let r = x - n * C1 - n * C2;
    let z = r * r;
    let p = ((((1.987_569_1e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r
        + 1.666_666_5e-1)
        * r
        + 5.000_000_1e-1;
    let y = p * z + r + 1.0;
    // Scale by 2^n; n + 127 stays in [1, 254] for the clamped range, except n = 128 at the top,
    // which we split in two.
    let ni = n as i32;
    let h = ni >> 1;
    y * f32::from_bits(((h + 127) << 23) as u32) * f32::from_bits(((ni - h + 127) << 23) as u32)
}

/// GELU, exact (erf) form: 0.5 x erfc(-x / sqrt 2). erfc from Numerical Recipes' Chebyshev fit,
/// relative error < 1.2e-7 everywhere (so no cancellation for negative x, unlike 1 + erf).
#[inline(always)]
pub fn gelu1(x: f32) -> f32 {
    let u = -x * std::f32::consts::FRAC_1_SQRT_2;
    let z = u.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -1.265_512_2
        + t * (1.000_023_7
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let ans = t * exp(-z * z + poly);
    let erfc = if u >= 0.0 { ans } else { 2.0 - ans };
    0.5 * x * erfc
}

pub fn gelu(x: &mut [f32]) {
    let _sc = prof::scope(prof::ACT);
    x.par_chunks_mut(8192).for_each(gelu_slice);
}

simd_fn! {
pub fn gelu_slice(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu1(*v);
    }
}
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// GLU over the first axis: (2C, ...) -> (C, ...), a * sigmoid(b).
pub fn glu(x: &Tensor) -> Tensor {
    let _sc = prof::scope(prof::ACT);
    let c2 = x.shape[0];
    let half = x.numel() / 2;
    let mut shape = x.shape.clone();
    shape[0] = c2 / 2;
    let (a, b) = x.data.split_at(half);
    let mut out = vec![0.0f32; half];
    out.par_chunks_mut(8192)
        .zip(a.par_chunks(8192).zip(b.par_chunks(8192)))
        .for_each(|(o, (a, b))| glu_slice(o, a, b));
    Tensor::new(&shape, out)
}

simd_fn! {
fn glu_slice(o: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..o.len() {
        o[i] = a[i] * sigmoid(b[i]);
    }
}
}

/// y = x W^T + b for x (N, Cin), W (Cout, Cin) (nn.Linear).
pub fn linear(x: &Tensor, w: &[f32], b: &[f32]) -> Tensor {
    let _sc = prof::scope(prof::LINEAR);
    let (n, cin) = (x.shape[0], x.shape[1]);
    let cout = b.len();
    let mut out = vec![0.0f32; n * cout];
    out.par_chunks_mut(cout).for_each(|row| row.copy_from_slice(b));
    sgemm(n, cout, cin, &x.data, cin, 1, w, 1, cin, &mut out, cout, true);
    Tensor::new(&[n, cout], out)
}

/// Multi-head scaled dot-product attention. q (Nq, C), k and v (Nk, C), heads split C.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> Tensor {
    let (nq, c) = (q.shape[0], q.shape[1]);
    let nk = k.shape[0];
    let mut out = vec![0.0f32; nq * c];
    attention_into(&q.data, c, &k.data, c, &v.data, c, nq, nk, c, heads, &mut out);
    Tensor::new(&[nq, c], out)
}

simd_fn! {
/// softmax(scale * row), in place. (The max is over eight lanes: max is exact, so any order gives
/// the same result.)
fn softmax_row(row: &mut [f32], scale: f32) {
    let mut m8 = [f32::NEG_INFINITY; 8];
    let chunks = row.chunks_exact(8);
    for &s in chunks.remainder() {
        m8[0] = m8[0].max(s * scale);
    }
    for c in chunks {
        for i in 0..8 {
            m8[i] = m8[i].max(c[i] * scale);
        }
    }
    let mx = m8.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    for s in row.iter_mut() {
        *s = exp(*s * scale - mx);
    }
    let inv = (1.0 / sum64(row)) as f32;
    for s in row.iter_mut() {
        *s *= inv;
    }
}
}

/// Query rows per attention task (see `attention_into`); HTDEMUCS_ATTN_BLOCK overrides (0: the
/// old head-by-head full-width path).
fn attn_block() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("HTDEMUCS_ATTN_BLOCK").ok().and_then(|v| v.parse().ok()).unwrap_or(128))
}

/// A raw pointer that may cross threads: tasks write disjoint (row block, head) tiles of one
/// output matrix, which cannot be expressed as disjoint slices.
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Attention on strided views: row i of q / k / v starts at q[i * q_rs] (etc.), each row holds C =
/// heads * d values; writes out (Nq, C) with row stride C.
///
/// Blocked: one task per (query block, head), each a single-threaded QK^T GEMM, softmax and AV
/// GEMM on a (block x Nk) score tile that stays in cache - instead of materialising every head's
/// full (Nq x Nk) scores. Every value is computed the same way whatever the thread count.
#[allow(clippy::too_many_arguments)]
pub fn attention_into(q: &[f32], q_rs: usize, k: &[f32], k_rs: usize, v: &[f32], v_rs: usize, nq: usize, nk: usize, c: usize, heads: usize, out: &mut [f32]) {
    let d = c / heads;
    let scale = 1.0 / (d as f32).sqrt();
    assert!(out.len() >= nq * c);
    let softmax = |row: &mut [f32]| softmax_row(row, scale);
    let bq = attn_block();
    if bq == 0 {
        let mut scores = vec![0.0f32; nq * nk];
        for h in 0..heads {
            let sc = prof::scope(prof::QK);
            sgemm(nq, nk, d, &q[h * d..], q_rs, 1, &k[h * d..], 1, k_rs, &mut scores, nk, false);
            drop(sc);
            let sc = prof::scope(prof::SOFTMAX);
            scores.par_chunks_mut(nk).for_each(softmax);
            drop(sc);
            let _sc = prof::scope(prof::AV);
            sgemm(nq, d, nk, &scores, nk, 1, &v[h * d..], v_rs, 1, &mut out[h * d..], c, false);
        }
        return;
    }
    let _sc = prof::scope(prof::QK);
    let nb = nq.div_ceil(bq);
    let optr = SendPtr(out.as_mut_ptr());
    let olen = out.len();
    (0..nb * heads).into_par_iter().for_each_init(
        || vec![0.0f32; bq * nk],
        |scores, task| {
            let (b, h) = (task / heads, task % heads);
            let r0 = b * bq;
            let m = bq.min(nq - r0);
            let s = &mut scores[..m * nk];
            sgemm_par(m, nk, d, &q[r0 * q_rs + h * d..], q_rs, 1, &k[h * d..], 1, k_rs, s, nk, false, false);
            s.chunks_mut(nk).for_each(softmax);
            // SAFETY: tile (rows r0..r0+m, cols h*d..h*d+d) of out is written by this task only.
            let o = unsafe { std::slice::from_raw_parts_mut(optr.get().add(r0 * c + h * d), olen - r0 * c - h * d) };
            sgemm_par(m, d, nk, s, nk, 1, &v[h * d..], v_rs, 1, o, c, false, false);
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_accuracy() {
        let mut worst = 0f64;
        let mut x = -87.0f32;
        while x < 88.0 {
            let e = (x as f64).exp();
            let rel = ((exp(x) as f64 - e) / e).abs();
            worst = worst.max(rel);
            x += 0.001_37;
        }
        assert!(worst < 2.5e-7, "exp rel err {worst}");
    }

    #[test]
    fn gelu_accuracy() {
        // Against an f64 reference (libm erfc). Relative error < 2e-6 wherever |gelu| > 1e-4 (in
        // the far negative tail the f32 rounding of -x^2/2 limits relative accuracy, but the
        // values themselves are < 1e-4 there).
        let mut worst = 0f64;
        let mut x = -12.0f32;
        while x < 12.0 {
            let xd = x as f64;
            let g = 0.5 * xd * libm::erfc(-xd / std::f64::consts::SQRT_2);
            worst = worst.max((gelu1(x) as f64 - g).abs() / g.abs().max(1e-4));
            x += 0.000_97;
        }
        assert!(worst < 2e-6, "gelu rel err {worst}");
    }
}
