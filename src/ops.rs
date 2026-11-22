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
    if m == 0 || n == 0 {
        return;
    }
    assert!(k == 0 || (m - 1) * a_rs + (k - 1) * a_cs < a.len());
    assert!(k == 0 || (k - 1) * b_rs + (n - 1) * b_cs < b.len());
    assert!((m - 1) * c_rs + n <= c.len());
    // gemm computes dst = alpha * dst + beta * lhs * rhs. It never splits the k dimension across
    // threads, so results do not depend on the thread count.
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
            gemm::Parallelism::Rayon(rayon::current_num_threads()),
        );
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
}

/// Upper bound on the im2col buffer (floats); larger convolutions are done in column blocks.
const COL_BLOCK: usize = 1 << 24;

impl Conv {
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
        let mut out = vec![0.0f32; self.cout * n];
        // Bias first, then accumulate the GEMM into it.
        out.par_chunks_mut(n).zip(self.b.par_iter()).for_each(|(row, &b)| row.fill(b));
        let pointwise = self.k == (1, 1) && self.stride == (1, 1) && self.pad == (0, 0);
        if pointwise {
            sgemm(self.cout, n, kk, &self.w, kk, 1, &x.data, n, 1, &mut out, n, true);
        } else {
            let block = (COL_BLOCK / kk).max(wo).min(n);
            let mut col = vec![0.0f32; kk * block];
            let mut j0 = 0;
            while j0 < n {
                let nb = block.min(n - j0);
                self.im2col(x, ho, wo, j0, nb, &mut col[..kk * nb]);
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
        let mut cols = vec![0.0f32; rows * n];
        sgemm(rows, n, cin, &self.w, 1, rows, &x.data, n, 1, &mut cols, n, false);
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

/// Mean and (biased) variance of a slice, f64 two-pass.
fn stats(xs: impl Iterator<Item = f32> + Clone) -> (f64, f64) {
    let mut n = 0usize;
    let mut s = 0f64;
    for x in xs.clone() {
        s += x as f64;
        n += 1;
    }
    let mean = s / n as f64;
    let mut v = 0f64;
    for x in xs {
        let d = x as f64 - mean;
        v += d * d;
    }
    (mean, v / n as f64)
}

/// Mean and unbiased (N-1) standard deviation, as torch.mean / torch.std.
pub fn mean_std(x: &[f32]) -> (f32, f32) {
    let (m, v) = stats(x.iter().copied());
    let n = x.len() as f64;
    (m as f32, (v * n / (n - 1.0)).sqrt() as f32)
}

/// GroupNorm with ONE group over a (C, S, T) tensor where each index s of the middle axis is a
/// separate sample (PyTorch's batch): statistics over (C, T) per s, then per-channel affine.
/// (S = 1 for an ordinary (C, T) input.) eps = 1e-5.
pub fn group_norm1(x: &mut Tensor, gamma: &[f32], beta: &[f32]) {
    let (c, s, t) = (x.shape[0], x.shape[1], x.shape[2]);
    let st: Vec<(f32, f32)> = (0..s)
        .into_par_iter()
        .map(|si| {
            let it = (0..c).flat_map(|ci| x.data[(ci * s + si) * t..(ci * s + si + 1) * t].iter().copied());
            let (m, v) = stats(it);
            (m as f32, (1.0 / (v + 1e-5).sqrt()) as f32)
        })
        .collect();
    x.data.par_chunks_mut(s * t).enumerate().for_each(|(ci, plane)| {
        for (si, row) in plane.chunks_mut(t).enumerate() {
            let (m, r) = st[si];
            let scale = r * gamma[ci];
            let shift = beta[ci] - m * scale;
            for v in row {
                *v = *v * scale + shift;
            }
        }
    });
}

/// LayerNorm over the last axis of (N, C). eps = 1e-5.
pub fn layer_norm(x: &Tensor, gamma: &[f32], beta: &[f32]) -> Tensor {
    let c = *x.shape.last().unwrap();
    let mut out = x.clone();
    out.data.par_chunks_mut(c).for_each(|row| {
        let (m, v) = stats(row.iter().copied());
        let r = 1.0 / (v + 1e-5).sqrt();
        for (i, val) in row.iter_mut().enumerate() {
            *val = (((*val as f64 - m) * r) as f32) * gamma[i] + beta[i];
        }
    });
    out
}

/// GroupNorm with one group over a whole (N, C) token sequence (Demucs' MyGroupNorm on (B, T, C)):
/// one mean / variance for everything, then per-channel affine.
pub fn group_norm_tokens(x: &mut Tensor, gamma: &[f32], beta: &[f32]) {
    let c = *x.shape.last().unwrap();
    let (m, v) = stats(x.data.iter().copied());
    let r = (1.0 / (v + 1e-5).sqrt()) as f32;
    let m = m as f32;
    x.data.par_chunks_mut(c).for_each(|row| {
        for (i, val) in row.iter_mut().enumerate() {
            let scale = r * gamma[i];
            *val = *val * scale + (beta[i] - m * scale);
        }
    });
}

#[inline]
pub fn gelu1(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub fn gelu(x: &mut [f32]) {
    x.par_chunks_mut(4096).for_each(|c| c.iter_mut().for_each(|v| *v = gelu1(*v)));
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// GLU over the first axis: (2C, ...) -> (C, ...), a * sigmoid(b).
pub fn glu(x: &Tensor) -> Tensor {
    let c2 = x.shape[0];
    let half = x.numel() / 2;
    let mut shape = x.shape.clone();
    shape[0] = c2 / 2;
    let (a, b) = x.data.split_at(half);
    let mut out = vec![0.0f32; half];
    out.par_chunks_mut(4096)
        .zip(a.par_chunks(4096).zip(b.par_chunks(4096)))
        .for_each(|(o, (a, b))| {
            for i in 0..o.len() {
                o[i] = a[i] * sigmoid(b[i]);
            }
        });
    Tensor::new(&shape, out)
}

/// y = x W^T + b for x (N, Cin), W (Cout, Cin) (nn.Linear).
pub fn linear(x: &Tensor, w: &[f32], b: &[f32]) -> Tensor {
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
    let d = c / heads;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; nq * c];
    let mut scores = vec![0.0f32; nq * nk];
    for h in 0..heads {
        // scores = q_h k_h^T
        sgemm(nq, nk, d, &q.data[h * d..], c, 1, &k.data[h * d..], 1, c, &mut scores, nk, false);
        scores.par_chunks_mut(nk).for_each(|row| {
            let mut mx = f32::NEG_INFINITY;
            for &s in row.iter() {
                mx = mx.max(s * scale);
            }
            let mut sum = 0f64;
            for s in row.iter_mut() {
                *s = (*s * scale - mx).exp();
                sum += *s as f64;
            }
            let inv = (1.0 / sum) as f32;
            for s in row.iter_mut() {
                *s *= inv;
            }
        });
        sgemm(nq, d, nk, &scores, nk, 1, &v.data[h * d..], c, 1, &mut out[h * d..], c, false);
    }
    Tensor::new(&[nq, c], out)
}
