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

    pub const CLASSES: [&str; 17] = [
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
        "DConv, frequency rows fused",
        "attention, blocked (QK^T, softmax, AV)",
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
    pub const DCONV: usize = 15;
    pub const ATTN: usize = 16;
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
/// (Rust never contracts a * b + c into an FMA by itself; kernels opt in through `MulAdd`.)
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

/// a * b + c: fused (one rounding) in code compiled for FMA hardware (`Fused`: the AVX2+FMA
/// versions on x86-64, always on aarch64), else a multiply and an add (`Unfused`; without the
/// target feature, mul_add would be a slow library call).
pub trait MulAdd {
    fn fma(a: f32, b: f32, c: f32) -> f32;
}
pub struct Fused;
impl MulAdd for Fused {
    #[inline(always)]
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
}
pub struct Unfused;
impl MulAdd for Unfused {
    #[inline(always)]
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        a * b + c
    }
}
/// The baseline build's choice.
#[cfg(target_arch = "aarch64")]
pub type Native = Fused;
#[cfg(not(target_arch = "aarch64"))]
pub type Native = Unfused;

/// Defines a function whose body is compiled twice on x86-64 - for the baseline and for
/// AVX2 + FMA - and picks one at run time (see `avx2_fma`). Elsewhere, just the function. The
/// body sees a type `F: MulAdd` (Fused in the AVX2+FMA version and on aarch64).
macro_rules! simd_fn {
    ($(#[$m:meta])* $vis:vis fn $name:ident($($a:ident : $t:ty),* $(,)?) $(-> $r:ty)? $body:block) => {
        $(#[$m])*
        #[allow(clippy::too_many_arguments)]
        $vis fn $name($($a: $t),*) $(-> $r)? {
            // F: how a * b + c is evaluated (see MulAdd); bodies may use F::fma, exp_g::<F>, ...
            #[inline(always)]
            #[allow(clippy::too_many_arguments, unused)]
            fn imp<F: crate::ops::MulAdd>($($a: $t),*) $(-> $r)? $body
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn v3($($a: $t),*) $(-> $r)? {
                    imp::<crate::ops::Fused>($($a),*)
                }
                if crate::ops::avx2_fma() {
                    return unsafe { v3($($a),*) };
                }
            }
            imp::<crate::ops::Native>($($a),*)
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
    assert!((m - 1) * c_rs + n <= c.len());
    // SAFETY: c covers the output, as asserted.
    unsafe { sgemm_raw(m, n, k, a, a_rs, a_cs, b, b_rs, b_cs, c.as_mut_ptr(), c_rs, accumulate, parallel) }
}

/// `sgemm_par` writing through a raw pointer, for tasks that write disjoint tiles of one matrix.
///
/// # Safety
/// c[i * c_rs + j] for i < m, j < n must be valid for writes (and reads if `accumulate`), and not
/// accessed by anyone else during the call.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_raw(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    a_rs: usize,
    a_cs: usize,
    b: &[f32],
    b_rs: usize,
    b_cs: usize,
    c: *mut f32,
    c_rs: usize,
    accumulate: bool,
    parallel: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(k == 0 || (m - 1) * a_rs + (k - 1) * a_cs < a.len());
    assert!(k == 0 || (k - 1) * b_rs + (n - 1) * b_cs < b.len());
    prof::gemm_flops(m, n, k);
    #[cfg(any(all(feature = "accelerate", target_os = "macos"), feature = "openblas"))]
    {
        let _ = parallel; // the BLAS manages its own threads
        unsafe { cblas::sgemm(m, n, k, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, accumulate) };
    }
    // gemm computes dst = alpha * dst + beta * lhs * rhs. It never splits the k dimension across
    // threads, so results do not depend on the thread count.
    #[cfg(not(any(all(feature = "accelerate", target_os = "macos"), feature = "openblas")))]
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c,
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

/// A CBLAS sgemm: Apple's Accelerate (runs on the AMX units of Apple silicon) with the
/// `accelerate` feature on macOS, or OpenBLAS (`openblas`, for speed comparisons on x86).
#[cfg(any(all(feature = "accelerate", target_os = "macos"), feature = "openblas"))]
mod cblas {
    #[allow(non_camel_case_types)]
    type c_int = i32;
    const ROW_MAJOR: c_int = 101;
    const NO_TRANS: c_int = 111;
    const TRANS: c_int = 112;
    #[cfg_attr(all(feature = "accelerate", target_os = "macos"), link(name = "Accelerate", kind = "framework"))]
    #[cfg_attr(feature = "openblas", link(name = "openblas"))]
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
    pub unsafe fn sgemm(
        m: usize, n: usize, k: usize, a: &[f32], a_rs: usize, a_cs: usize, b: &[f32], b_rs: usize,
        b_cs: usize, c: *mut f32, c_rs: usize, accumulate: bool,
    ) {
        let (ta, lda) = layout(a_rs, a_cs, k);
        let (tb, ldb) = layout(b_rs, b_cs, n);
        unsafe {
            cblas_sgemm(
                ROW_MAJOR, ta, tb, m as c_int, n as c_int, k as c_int, 1.0, a.as_ptr(), lda,
                b.as_ptr(), ldb, if accumulate { 1.0 } else { 0.0 }, c, c_rs as c_int,
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

/// One DConv residual branch on a (C, F, T) input whose GroupNorms are per frequency row f (the
/// frequency branch, where PyTorch runs DConv on (B*F, C, T)): dilated conv (k = 3) -> GroupNorm
/// -> GELU -> 1x1 conv -> GroupNorm -> GLU -> LayerScale, added to x. Row f's result depends on
/// row f only, so each row is one task with every intermediate in cache: no (2C, F, T) buffers.
/// Same operations in the same order as the layer-by-layer path. `delta` (tracing) receives the
/// added values.
#[allow(clippy::too_many_arguments)]
pub fn dconv_rows(
    x: &mut Tensor,
    conv1: &Conv,
    gn1: (&[f32], &[f32]),
    conv2: &Conv,
    gn2: (&[f32], &[f32]),
    ls: &[f32],
    delta: Option<&mut Tensor>,
) {
    let (c, fr, t) = (x.shape[0], x.shape[1], x.shape[2]);
    let cm = conv1.cout;
    assert_eq!((conv1.cin, conv2.cin, conv2.cout), (c, cm, 2 * c));
    assert_eq!((conv1.k, conv1.stride, conv1.pad.0, conv1.pad.1), ((1, 3), (1, 1), 0, conv1.dil.1));
    assert!(conv2.k == (1, 1) && conv2.stride == (1, 1) && conv2.pad == (0, 0));
    let _sc = prof::scope(prof::DCONV);
    let xp = SendPtr(x.data.as_mut_ptr());
    let dp = delta.map(|d| {
        assert_eq!(d.shape, x.shape);
        SendPtr(d.data.as_mut_ptr())
    });
    (0..fr).into_par_iter().for_each_init(
        || (vec![0.0f32; c * t], vec![0.0f32; 3 * cm * t], vec![0.0f32; cm * t], vec![0.0f32; 2 * c * t]),
        |(xr, yt, y1, z), f| {
            // SAFETY: task f reads and writes only rows (ci, f) of x and delta.
            let row = |p: SendPtr, ci: usize| unsafe { std::slice::from_raw_parts_mut(p.get().add((ci * fr + f) * t), t) };
            for ci in 0..c {
                xr[ci * t..(ci + 1) * t].copy_from_slice(row(xp, ci));
            }
            dconv_row(xr, yt, y1, z, t, conv1, gn1, conv2, gn2, ls);
            // x += ls * GLU(GroupNorm(z + b2)), as `glu_add_plane`.
            for ci in 0..c {
                let d = dp.map(|p| row(p, ci));
                glu_add_row(row(xp, ci), d, &z[ci * t..(ci + 1) * t], &z[(c + ci) * t..(c + ci + 1) * t], ci, c, &conv2.b, gn2, ls, y1[0], y1[1]);
            }
        },
    );
}

simd_fn! {
/// `dconv_rows` for one row up to z = W2 y (2C, T) without its bias; leaves GroupNorm-2's
/// (mean, 1/std) of z + b2 in y1[0], y1[1].
fn dconv_row(xr: &[f32], yt: &mut [f32], y1: &mut [f32], z: &mut [f32], t: usize, conv1: &Conv, gn1: (&[f32], &[f32]), conv2: &Conv, gn2: (&[f32], &[f32]), ls: &[f32]) {
    let _ = (gn2, ls);
    let c = conv1.cin;
    let cm = conv1.cout;
    let wt = conv1.w_taps.as_ref().unwrap();
    sgemm_par(3 * cm, t, c, wt, c, 1, xr, t, 1, yt, t, false, false);
    // y1[co] = b1 + sum_kj taps shifted, as Conv's tap path.
    let (d, p) = (conv1.dil.1 as isize, conv1.pad.1 as isize);
    for co in 0..cm {
        let dst = &mut y1[co * t..(co + 1) * t];
        dst.fill(conv1.b[co]);
        for kj in 0..3 {
            let src = &yt[(kj * cm + co) * t..(kj * cm + co + 1) * t];
            let shift = kj as isize * d - p;
            let lo = (-shift).max(0) as usize;
            let hi = ((t as isize - shift).min(t as isize)).max(0) as usize;
            if lo < hi {
                let s0 = (lo as isize + shift) as usize;
                for (o, v) in dst[lo..hi].iter_mut().zip(&src[s0..s0 + (hi - lo)]) {
                    *o += v;
                }
            }
        }
    }
    // GroupNorm-1 over (Cm, T) + GELU, as row_stats + gn_plane.
    let n = (cm * t) as f64;
    let mut s = 0f64;
    for co in 0..cm {
        s += sum64(&y1[co * t..(co + 1) * t]);
    }
    let mean = s / n;
    let mut q = 0f64;
    for co in 0..cm {
        q += sqdev64(&y1[co * t..(co + 1) * t], mean);
    }
    let (m, r) = (mean as f32, (1.0 / (q / n + 1e-5).sqrt()) as f32);
    for co in 0..cm {
        let scale = r * gn1.0[co];
        let shift = gn1.1[co] - m * scale;
        for v in &mut y1[co * t..(co + 1) * t] {
            *v = gelu1_g::<F>(*v * scale + shift);
        }
    }
    sgemm_par(2 * c, t, cm, &conv2.w, cm, 1, y1, t, 1, z, t, false, false);
    // GroupNorm-2 statistics of z + b2 over (2C, T).
    let n = (2 * c * t) as f64;
    let mut s = 0f64;
    for ci in 0..2 * c {
        s += sum64b(&z[ci * t..(ci + 1) * t], conv2.b[ci]);
    }
    let mean = s / n;
    let mut q = 0f64;
    for ci in 0..2 * c {
        q += sqdev64b(&z[ci * t..(ci + 1) * t], conv2.b[ci], mean);
    }
    y1[0] = mean as f32;
    y1[1] = (1.0 / (q / n + 1e-5).sqrt()) as f32;
}
}

simd_fn! {
/// x += ls[ci] * ((za + ba) * sa + ha) * sigmoid((zb + bb) * sb + hb), one row (see `glu_add_plane`).
fn glu_add_row(x: &mut [f32], d: Option<&mut [f32]>, za: &[f32], zb: &[f32], ci: usize, c: usize, b2: &[f32], gn2: (&[f32], &[f32]), ls: &[f32], m: f32, r: f32) {
    let (gamma, beta) = gn2;
    let (sa, sb) = (r * gamma[ci], r * gamma[c + ci]);
    let (ha, hb) = (beta[ci] - m * sa, beta[c + ci] - m * sb);
    let (ba, bb) = (b2[ci], b2[c + ci]);
    match d {
        None => {
            for i in 0..x.len() {
                let v = ls[ci] * (((za[i] + ba) * sa + ha) * sigmoid_g::<F>((zb[i] + bb) * sb + hb));
                x[i] += v;
            }
        }
        Some(dd) => {
            for i in 0..x.len() {
                let v = ls[ci] * (((za[i] + ba) * sa + ha) * sigmoid_g::<F>((zb[i] + bb) * sb + hb));
                dd[i] = v;
                x[i] += v;
            }
        }
    }
}
}

/// Upper bound on the im2col buffer (floats); larger convolutions are done in column blocks.
const COL_BLOCK: usize = 1 << 24;

/// Output columns per task for the blocked im2col convolution (0: one multithreaded GEMM per
/// 2^24-float im2col block instead). Depends on the shape only - never on the thread count.
/// HTDEMUCS_CONV_BLOCK = target im2col floats per task (0 disables).
fn conv_block_cols(kk: usize, n: usize) -> usize {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let target = *T.get_or_init(|| std::env::var("HTDEMUCS_CONV_BLOCK").ok().and_then(|v| v.parse().ok()).unwrap_or(1 << 20));
    if target == 0 {
        return 0;
    }
    (target / kk).max(512).min(n)
}

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
        self.run(x, true)
    }

    /// The convolution without its bias: callers add `self.b` in their next elementwise pass
    /// (`bias_gelu`, `glu_bias`, `group_norm1_glu_add_b`).
    pub fn forward_raw(&self, x: &Tensor) -> Tensor {
        self.run(x, false)
    }

    fn run(&self, x: &Tensor, bias: bool) -> Tensor {
        let (cin, h, w) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(cin, self.cin, "conv input channels");
        let (ho, wo) = self.out_size(h, w);
        let n = ho * wo;
        let kk = cin * self.k.0 * self.k.1;
        let mut out = vec![0.0f32; self.cout * n];
        let pointwise = self.k == (1, 1) && self.stride == (1, 1) && self.pad == (0, 0);
        if let Some(wt) = &self.w_taps {
            // y[kj] = W_kj x for all taps at once, then out[co, h, o] = b + sum_kj y[kj, co, h, o + kj*d - p].
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
                orow.fill(if bias { self.b[co] } else { 0.0 });
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
                            add_assign(&mut dst[lo..hi_], &src[s0..s0 + (hi_ - lo)]);
                        }
                    }
                }
            });
        } else {
            if bias {
                // Bias first, then accumulate the GEMM into it.
                let _sc = prof::scope(prof::BIAS);
                out.par_chunks_mut(n).zip(self.b.par_iter()).for_each(|(row, &b)| row.fill(b));
            }
            let nbc = conv_block_cols(kk, n);
            if pointwise {
                let _sc = prof::scope(prof::CONV);
                sgemm(self.cout, n, kk, &self.w, kk, 1, &x.data, n, 1, &mut out, n, bias);
            } else if nbc > 0 {
                // Column blocks as tasks: each im2cols its block into a buffer that stays in
                // cache and multiplies it single-threaded. (k is never split, so per-element
                // sums are the same as one big GEMM's.)
                let _sc = prof::scope(prof::CONV);
                let op = SendPtr(out.as_mut_ptr());
                let cout = self.cout;
                (0..n.div_ceil(nbc)).into_par_iter().for_each_init(
                    || vec![0.0f32; kk * nbc],
                    |col, blk| {
                        let j0 = blk * nbc;
                        let nb = nbc.min(n - j0);
                        let col = &mut col[..kk * nb];
                        for (r, row) in col.chunks_mut(nb).enumerate() {
                            self.im2col_row(x, wo, j0, nb, r, row);
                        }
                        // SAFETY: columns [j0, j0 + nb) of every output row belong to this task.
                        unsafe { sgemm_raw(cout, nb, kk, &self.w, kk, 1, col, nb, 1, op.get().add(j0), n, bias, false) };
                    },
                );
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
                    sgemm(self.cout, nb, kk, &self.w, kk, 1, &col, nb, 1, &mut out[j0..], n, bias);
                    j0 += nb;
                }
            }
        }
        Tensor::new(&[self.cout, ho, wo], out)
    }

    /// col[(ci, ki, kj), j - j0] = x[ci, ho*sh + ki*dh - ph, wo*sw + kj*dw - pw] for output
    /// positions j = ho * Wo + wo in [j0, j0 + nb); zero outside the input.
    fn im2col(&self, x: &Tensor, _ho: usize, wo_n: usize, j0: usize, nb: usize, col: &mut [f32]) {
        col.par_chunks_mut(nb).enumerate().for_each(|(r, row)| self.im2col_row(x, wo_n, j0, nb, r, row));
    }

    /// Row r = (ci, ki, kj) of the im2col matrix, output positions [j0, j0 + nb).
    fn im2col_row(&self, x: &Tensor, wo_n: usize, j0: usize, nb: usize, r: usize, row: &mut [f32]) {
        let (h, w) = (x.shape[1] as isize, x.shape[2] as isize);
        let (kh, kw) = self.k;
        let (sh, sw) = (self.stride.0 as isize, self.stride.1 as isize);
        let (ph, pw) = (self.pad.0 as isize, self.pad.1 as isize);
        let (dh, dw) = (self.dil.0 as isize, self.dil.1 as isize);
        {
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
                        let first = ow0 as isize + base; // input index of dst[0]
                        let lo = (-first).clamp(0, run as isize) as usize;
                        let hi = (w - first).clamp(lo as isize, run as isize) as usize;
                        dst[..lo].fill(0.0);
                        if lo < hi {
                            let s0 = (first + lo as isize) as usize;
                            dst[lo..hi].copy_from_slice(&src[s0..s0 + (hi - lo)]);
                        }
                        dst[hi..].fill(0.0);
                    } else {
                        for (t, d) in dst.iter_mut().enumerate() {
                            let iw = (ow0 + t) as isize * sw + base;
                            *d = if iw >= 0 && iw < w { src[iw as usize] } else { 0.0 };
                        }
                    }
                }
                j += run;
            }
        }
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
                            add_assign(&mut orow[kj..kj + w], srow);
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
/// Sum of (x + b) in f64 (x + b rounded to f32 first, as if stored).
fn sum64b(x: &[f32], b: f32) -> f64 {
    let mut acc = [0f64; 8];
    let chunks = x.chunks_exact(8);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..8 {
            acc[i] += (c[i] + b) as f64;
        }
    }
    acc.iter().sum::<f64>() + rest.iter().map(|&v| (v + b) as f64).sum::<f64>()
}
}

simd_fn! {
/// Sum of squared deviations of (x + b) from `m`, in f64.
fn sqdev64b(x: &[f32], b: f32, m: f64) -> f64 {
    let mut acc = [0f64; 8];
    let chunks = x.chunks_exact(8);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..8 {
            let d = (c[i] + b) as f64 - m;
            acc[i] += d * d;
        }
    }
    acc.iter().sum::<f64>() + rest.iter().map(|&v| ((v + b) as f64 - m) * ((v + b) as f64 - m)).sum::<f64>()
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

/// `mean_std` of a sequence that is never materialised: `fill(start, buf)` writes elements
/// [start, start + buf.len()). Chunked exactly like `big_stats`, so the result is bit-identical to
/// `mean_std` of the same values in a Vec.
pub fn mean_std_by(len: usize, fill: impl Fn(usize, &mut [f32]) + Sync) -> (f32, f32) {
    const B: usize = 1 << 16;
    let n = len as f64;
    let chunk = |k: usize, buf: &mut Vec<f32>| {
        let a = k * B;
        buf.resize((len - a).min(B), 0.0);
        fill(a, buf);
    };
    let nb = len.div_ceil(B);
    let parts: Vec<f64> = (0..nb).into_par_iter().map_init(Vec::new, |buf, k| {
        chunk(k, buf);
        sum64(buf)
    }).collect();
    let mean = parts.iter().sum::<f64>() / n;
    let parts: Vec<f64> = (0..nb).into_par_iter().map_init(Vec::new, |buf, k| {
        chunk(k, buf);
        sqdev64(buf, mean)
    }).collect();
    let var = parts.iter().sum::<f64>() / n;
    (mean as f32, (var * n / (n - 1.0)).sqrt() as f32)
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
    row_stats_b(x, None)
}

/// `row_stats` of x + bias[c] (a convolution output given without its bias).
pub fn row_stats_b(x: &Tensor, bias: Option<&[f32]>) -> Vec<(f32, f32)> {
    let (c, s, t) = (x.shape[0], x.shape[1], x.shape[2]);
    let n = (c * t) as f64;
    let bc = |ci: usize| bias.map_or(0.0, |b| b[ci]);
    let sums: Vec<Vec<f64>> =
        x.data.par_chunks(s * t).enumerate().map(|(ci, p)| p.chunks(t).map(|r| sum64b(r, bc(ci))).collect()).collect();
    let means: Vec<f64> = (0..s).map(|si| sums.iter().map(|v| v[si]).sum::<f64>() / n).collect();
    let sq: Vec<Vec<f64>> = x
        .data
        .par_chunks(s * t)
        .enumerate()
        .map(|(ci, p)| p.chunks(t).zip(&means).map(|(r, &m)| sqdev64b(r, bc(ci), m)).collect())
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

/// GroupNorm(1) per sample of a (C, S, T) tensor, then GELU, in place.
pub fn group_norm1_gelu(x: &mut Tensor, gamma: &[f32], beta: &[f32]) {
    let _sc = prof::scope(prof::GROUPNORM);
    let st = row_stats(x);
    let (s, t) = (x.shape[1], x.shape[2]);
    x.data.par_chunks_mut(s * t).enumerate().for_each(|(ci, plane)| gn_gelu_plane(plane, t, &st, gamma[ci], beta[ci]));
}

simd_fn! {
fn gn_gelu_plane(plane: &mut [f32], t: usize, st: &[(f32, f32)], g: f32, b: f32) {
    for (row, &(m, r)) in plane.chunks_mut(t).zip(st) {
        let scale = r * g;
        let shift = b - m * scale;
        for v in row {
            *v = gelu1_g::<F>(*v * scale + shift);
        }
    }
}
}

/// GroupNorm(1) per sample of z (2C, S, T), GLU over channels, LayerScale, residual add:
/// x += ls * (a * sigmoid(b)) with (a, b) the normalised halves of z - the tail of a DConv branch.
/// If `delta` is given, the added values are also written there.
pub fn group_norm1_glu_add(z: &Tensor, gamma: &[f32], beta: &[f32], ls: &[f32], x: &mut Tensor, delta: Option<&mut Tensor>) {
    group_norm1_glu_add_b(z, None, gamma, beta, ls, x, delta)
}

/// `group_norm1_glu_add` on z + zbias[c] (a convolution output given without its bias).
pub fn group_norm1_glu_add_b(
    z: &Tensor,
    zbias: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    ls: &[f32],
    x: &mut Tensor,
    mut delta: Option<&mut Tensor>,
) {
    let _sc = prof::scope(prof::GROUPNORM);
    let st = row_stats_b(z, zbias);
    let zero = vec![0.0f32; z.shape[0]];
    let zb = zbias.unwrap_or(&zero);
    let (c, s, t) = (x.shape[0], x.shape[1], x.shape[2]);
    assert_eq!(z.shape, [2 * c, s, t]);
    let plane = s * t;
    match delta.as_deref_mut() {
        None => x.data.par_chunks_mut(plane).enumerate().for_each(|(ci, xp)| glu_add_plane(ci, c, t, xp, None, &z.data, zb, &st, gamma, beta, ls)),
        Some(d) => x
            .data
            .par_chunks_mut(plane)
            .zip(d.data.par_chunks_mut(plane))
            .enumerate()
            .for_each(|(ci, (xp, dp))| glu_add_plane(ci, c, t, xp, Some(dp), &z.data, zb, &st, gamma, beta, ls)),
    }
}

simd_fn! {
fn glu_add_plane(ci: usize, c: usize, t: usize, xp: &mut [f32], dp: Option<&mut [f32]>, z: &[f32], zbias: &[f32], st: &[(f32, f32)], gamma: &[f32], beta: &[f32], ls: &[f32]) {
    let mut dp = dp;
    let plane = xp.len();
    let za = &z[ci * plane..(ci + 1) * plane];
    let zb = &z[(c + ci) * plane..(c + ci + 1) * plane];
    for (si, &(m, r)) in st.iter().enumerate() {
        let (sa, sb) = (r * gamma[ci], r * gamma[c + ci]);
        let (ha, hb) = (beta[ci] - m * sa, beta[c + ci] - m * sb);
        let (ba, bb) = (zbias[ci], zbias[c + ci]);
        let rg = si * t..(si + 1) * t;
        let (xa, ya, yb) = (&mut xp[rg.clone()], &za[rg.clone()], &zb[rg.clone()]);
        match dp.as_deref_mut() {
            None => {
                for i in 0..t {
                    let d = ls[ci] * (((ya[i] + ba) * sa + ha) * sigmoid_g::<F>((yb[i] + bb) * sb + hb));
                    xa[i] += d;
                }
            }
            Some(dp) => {
                for i in 0..t {
                    let d = ls[ci] * (((ya[i] + ba) * sa + ha) * sigmoid_g::<F>((yb[i] + bb) * sb + hb));
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
    exp_g::<Native>(x)
}

/// `exp` with the multiply-adds evaluated by F.
#[inline(always)]
pub fn exp_g<F: MulAdd>(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const C1: f32 = 0.693_359_4;
    const C2: f32 = -2.121_944_4e-4;
    const ROUND: f32 = 12_582_912.0; // 1.5 * 2^23: (v + ROUND) - ROUND rounds to nearest
    let x = x.clamp(-87.336_54, 88.722_83);
    let t = F::fma(x, LOG2E, ROUND);
    let n = t - ROUND;
    let r = F::fma(-n, C2, F::fma(-n, C1, x));
    let z = r * r;
    let mut p = F::fma(1.987_569_1e-4, r, 1.398_199_9e-3);
    p = F::fma(p, r, 8.333_452e-3);
    p = F::fma(p, r, 4.166_579_6e-2);
    p = F::fma(p, r, 1.666_666_5e-1);
    p = F::fma(p, r, 5.000_000_1e-1);
    let y = F::fma(p, z, r) + 1.0;
    // Scale by 2^n; n + 127 stays in [1, 254] for the clamped range, except n = 128 at the top,
    // which we split in two.
    // n as an integer, read from t's bits (t = 1.5 * 2^23 + n, so its low mantissa bits are n)
    // rather than `n as i32`: Rust's saturating float-to-int cast does not vectorise on x86-64,
    // which left every loop calling exp scalar there. Same value, so bit-identical results.
    let ni = (t.to_bits() as i32).wrapping_sub(0x4B40_0000);
    let h = ni >> 1;
    y * f32::from_bits(((h + 127) << 23) as u32) * f32::from_bits(((ni - h + 127) << 23) as u32)
}

/// GELU, exact (erf) form: 0.5 x erfc(-x / sqrt 2). erfc from Numerical Recipes' Chebyshev fit,
/// relative error < 1.2e-7 everywhere (so no cancellation for negative x, unlike 1 + erf).
#[inline(always)]
pub fn gelu1(x: f32) -> f32 {
    gelu1_g::<Native>(x)
}

/// `gelu1` with the multiply-adds evaluated by F.
#[inline(always)]
pub fn gelu1_g<F: MulAdd>(x: f32) -> f32 {
    let u = -x * std::f32::consts::FRAC_1_SQRT_2;
    let z = u.abs();
    let t = 1.0 / F::fma(0.5, z, 1.0);
    let mut p = F::fma(t, 0.170_872_77, -0.822_152_23);
    p = F::fma(t, p, 1.488_515_87);
    p = F::fma(t, p, -1.135_203_98);
    p = F::fma(t, p, 0.278_868_07);
    p = F::fma(t, p, -0.186_288_06);
    p = F::fma(t, p, 0.096_784_18);
    p = F::fma(t, p, 0.374_091_96);
    p = F::fma(t, p, 1.000_023_7);
    let poly = F::fma(t, p, -1.265_512_2);
    let ans = t * exp_g::<F>(F::fma(-z, z, poly));
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
        *v = gelu1_g::<F>(*v);
    }
}
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    sigmoid_g::<Native>(x)
}

#[inline(always)]
pub fn sigmoid_g<F: MulAdd>(x: f32) -> f32 {
    1.0 / (1.0 + exp_g::<F>(-x))
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
        o[i] = a[i] * sigmoid_g::<F>(b[i]);
    }
}
}

/// y = x W^T for x (N, Cin), W (Cout, Cin): nn.Linear without its bias (callers fuse the bias into
/// the next elementwise pass; adding it afterwards gives the same bits as starting from it when
/// the GEMM sums k in one block).
pub fn matmul_t(x: &Tensor, w: &[f32], cout: usize) -> Tensor {
    let _sc = prof::scope(prof::LINEAR);
    let (n, cin) = (x.shape[0], x.shape[1]);
    assert_eq!(w.len(), cout * cin);
    let mut out = vec![0.0f32; n * cout];
    sgemm(n, cout, cin, &x.data, cin, 1, w, 1, cin, &mut out, cout, false);
    Tensor::new(&[n, cout], out)
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

simd_fn! {
/// dst += src.
pub fn add_assign(dst: &mut [f32], src: &[f32]) {
    for (o, v) in dst.iter_mut().zip(src) {
        *o += v;
    }
}
}

simd_fn! {
/// v = GELU(v + b) over a slice.
fn bias_gelu_slice(x: &mut [f32], b: f32) {
    for v in x.iter_mut() {
        *v = gelu1_g::<F>(*v + b);
    }
}
}

/// x[c, ...] = GELU(x[c, ...] + b[c]) - a convolution's bias and the GELU after it, one pass.
pub fn bias_gelu(x: &mut Tensor, b: &[f32]) {
    let _sc = prof::scope(prof::ACT);
    let plane = x.numel() / x.shape[0];
    assert_eq!(b.len(), x.shape[0]);
    x.data.par_chunks_mut(plane).zip(b.par_iter()).for_each(|(p, &bc)| p.par_chunks_mut(8192).for_each(|c| bias_gelu_slice(c, bc)));
}

/// GLU over the first axis of a convolution output given without its bias:
/// (2C, ...) -> (C, ...), (a + ba) * sigmoid(b + bb).
pub fn glu_bias(x: &Tensor, bias: &[f32]) -> Tensor {
    let _sc = prof::scope(prof::ACT);
    let c = x.shape[0] / 2;
    assert_eq!(bias.len(), 2 * c);
    let plane = x.numel() / x.shape[0];
    let mut shape = x.shape.clone();
    shape[0] = c;
    let (a, b) = x.data.split_at(c * plane);
    let mut out = vec![0.0f32; c * plane];
    out.par_chunks_mut(plane).zip(a.par_chunks(plane).zip(b.par_chunks(plane))).enumerate().for_each(|(ci, (o, (a, b)))| {
        let (ba, bb) = (bias[ci], bias[c + ci]);
        o.par_chunks_mut(8192)
            .zip(a.par_chunks(8192).zip(b.par_chunks(8192)))
            .for_each(|(o, (a, b))| glu_bias_slice(o, a, b, ba, bb));
    });
    Tensor::new(&shape, out)
}

simd_fn! {
fn glu_bias_slice(o: &mut [f32], a: &[f32], b: &[f32], ba: f32, bb: f32) {
    for i in 0..o.len() {
        o[i] = (a[i] + ba) * sigmoid_g::<F>(b[i] + bb);
    }
}
}

/// A copy of x with b[c] added to every value of channel c (for tracing un-fused values).
pub fn plus_channel_bias(x: &Tensor, b: &[f32]) -> Tensor {
    let mut o = x.clone();
    let plane = x.numel() / x.shape[0];
    o.data.par_chunks_mut(plane).zip(b.par_iter()).for_each(|(p, &bc)| p.iter_mut().for_each(|v| *v += bc));
    o
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
pub fn softmax_row(row: &mut [f32], scale: f32) {
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
    // Weights below e^-40 (4e-18 of the largest) are set to 0: they cannot change a float32 sum,
    // and left in they become subnormal after normalisation and in the AV products, which costs
    // Intel CPUs a microcode assist per operation (attention ran ~6x slower on a Xeon).
    for s in row.iter_mut() {
        let x = *s * scale - mx;
        *s = if x < -40.0 { 0.0 } else { exp_g::<F>(x) };
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
    *B.get_or_init(|| std::env::var("HTDEMUCS_ATTN_BLOCK").ok().and_then(|v| v.parse().ok()).unwrap_or(512))
}

/// K transposed per head before the blocked QK^T GEMMs (default; HTDEMUCS_ATTN_KT=0 disables).
fn attn_kt() -> bool {
    static B: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("HTDEMUCS_ATTN_KT").as_deref() != Ok("0"))
}

/// A raw pointer that may cross threads: tasks write disjoint (row block, head) tiles of one
/// output matrix, which cannot be expressed as disjoint slices.
#[derive(Clone, Copy)]
pub struct SendPtr(pub *mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    pub fn get(self) -> *mut f32 {
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
    let _sc = prof::scope(prof::ATTN);
    let nb = nq.div_ceil(bq);
    let optr = SendPtr(out.as_mut_ptr());
    let olen = out.len();
    // K transposed per head, (heads, d, Nk), so the QK^T GEMMs read it in place instead of
    // packing a strided K for every task.
    let kt: Option<Vec<f32>> = attn_kt().then(|| {
        let mut kt = vec![0.0f32; heads * d * nk];
        kt.par_chunks_mut(nk).enumerate().for_each(|(hd, row)| {
            for (j, v) in row.iter_mut().enumerate() {
                *v = k[j * k_rs + hd];
            }
        });
        kt
    });
    (0..nb * heads).into_par_iter().for_each_init(
        || vec![0.0f32; bq * nk],
        |scores, task| {
            let (b, h) = (task / heads, task % heads);
            let r0 = b * bq;
            let m = bq.min(nq - r0);
            let s = &mut scores[..m * nk];
            match &kt {
                Some(kt) => sgemm_par(m, nk, d, &q[r0 * q_rs + h * d..], q_rs, 1, &kt[h * d * nk..], nk, 1, s, nk, false, false),
                None => sgemm_par(m, nk, d, &q[r0 * q_rs + h * d..], q_rs, 1, &k[h * d..], 1, k_rs, s, nk, false, false),
            }
            s.chunks_mut(nk).for_each(softmax);
            // SAFETY: tile (rows r0..r0+m, cols h*d..h*d+d) of out is in bounds and written by
            // this task only.
            assert!((r0 + m - 1) * c + (h + 1) * d <= olen);
            unsafe { sgemm_raw(m, d, nk, s, nk, 1, &v[h * d..], v_rs, 1, optr.get().add(r0 * c + h * d), c, false, false) };
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp_worst<F: MulAdd>() -> f64 {
        let mut worst = 0f64;
        let mut x = -87.0f32;
        while x < 88.0 {
            let e = (x as f64).exp();
            let rel = ((exp_g::<F>(x) as f64 - e) / e).abs();
            worst = worst.max(rel);
            x += 0.001_37;
        }
        worst
    }

    #[test]
    fn exp_integer_part_from_bits() {
        // The exponent read from the rounding constant's bits equals the saturating cast.
        let mut x = -100.0f32;
        while x < 100.0 {
            let c = x.clamp(-87.336_54, 88.722_83);
            let t = Fused::fma(c, std::f32::consts::LOG2_E, 12_582_912.0);
            assert_eq!((t.to_bits() as i32).wrapping_sub(0x4B40_0000), (t - 12_582_912.0) as i32, "{x}");
            x += 0.003_1;
        }
    }

    #[test]
    fn exp_accuracy() {
        for (name, worst) in [("unfused", exp_worst::<Unfused>()), ("fused", exp_worst::<Fused>())] {
            assert!(worst < 2.5e-7, "exp ({name}) rel err {worst}");
        }
    }

    fn gelu_worst<F: MulAdd>() -> f64 {
        let mut worst = 0f64;
        let mut x = -12.0f32;
        while x < 12.0 {
            let xd = x as f64;
            let g = 0.5 * xd * libm::erfc(-xd / std::f64::consts::SQRT_2);
            worst = worst.max((gelu1_g::<F>(x) as f64 - g).abs() / g.abs().max(1e-4));
            x += 0.000_97;
        }
        worst
    }

    #[test]
    fn gelu_fused_accuracy() {
        for (name, worst) in [("unfused", gelu_worst::<Unfused>()), ("fused", gelu_worst::<Fused>())] {
            assert!(worst < 2e-6, "gelu ({name}) rel err {worst}");
        }
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
