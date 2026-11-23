//! Demucs' CrossTransformerEncoder (demucs/transformer.py) for the pretrained configuration:
//! layers alternate self-attention (even indices) and cross-attention (odd), each branch (spectral
//! tokens `x`, waveform tokens `xt`) with its own weights; norm_first, LayerScale, GELU
//! feed-forward, and a GroupNorm(1) "norm_out" over the WHOLE sequence after every layer.

use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::ops::{attention_into, group_norm_tokens, layer_norm, layer_norm_row, linear, matmul_t, prof, simd_fn};
use crate::tensor::Tensor;
use crate::weights::SafeTensors;
use crate::Tracer;

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn load(st: &mut SafeTensors, p: &str) -> Self {
        Norm { w: st.take(&format!("{p}.weight")).data, b: st.take(&format!("{p}.bias")).data }
    }
}

/// nn.MultiheadAttention (packed in_proj; q from one sequence, k = v from another).
struct Mha {
    /// in_proj: (3C, C) rows q, k, v - so self-attention is one GEMM, and k, v one for cross.
    w_in: Vec<f32>,
    b_in: Vec<f32>,
    w_out: Vec<f32>,
    b_out: Vec<f32>,
    heads: usize,
    dim: usize,
}

impl Mha {
    fn load(st: &mut SafeTensors, p: &str, dim: usize, heads: usize) -> Self {
        Mha {
            w_in: st.take(&format!("{p}.in_proj_weight")).data,
            b_in: st.take(&format!("{p}.in_proj_bias")).data,
            w_out: st.take(&format!("{p}.out_proj.weight")).data,
            b_out: st.take(&format!("{p}.out_proj.bias")).data,
            heads,
            dim,
        }
    }

    /// The attention heads' output, before out_proj: (Nq, C). `kv` None: self-attention.
    fn attend(&self, q: &Tensor, kv: Option<&Tensor>) -> Tensor {
        let c = self.dim;
        let nq = q.shape[0];
        let mut out = vec![0.0f32; nq * c];
        match kv {
            None => {
                let qkv = linear(q, &self.w_in, &self.b_in);
                let d = &qkv.data;
                attention_into(d, 3 * c, &d[c..], 3 * c, &d[2 * c..], 3 * c, nq, nq, c, self.heads, &mut out);
            }
            Some(kv) => {
                let nk = kv.shape[0];
                let qp = linear(q, &self.w_in[..c * c], &self.b_in[..c]);
                let kvp = linear(kv, &self.w_in[c * c..], &self.b_in[c..]);
                let d = &kvp.data;
                attention_into(&qp.data, c, d, 2 * c, &d[c..], 2 * c, nq, nk, c, self.heads, &mut out);
            }
        }
        Tensor::new(&[nq, c], out)
    }
}

struct Layer {
    cross: bool,
    attn: Mha,
    linear1: (Vec<f32>, Vec<f32>),
    linear2: (Vec<f32>, Vec<f32>),
    norm1: Norm,
    norm2: Norm,
    /// Cross layers only: norm2 normalises the keys, norm3 the feed-forward input.
    norm3: Option<Norm>,
    norm_out: Norm,
    gamma1: Vec<f32>,
    gamma2: Vec<f32>,
}

simd_fn! {
/// x1 = x + g * (o + b), then h = LayerNorm(x1) - the attention residual and the feed-forward
/// input, one row.
fn residual_ln_row(x: &[f32], o: &[f32], b: &[f32], g: &[f32], x1: &mut [f32], h: &mut [f32], nw: &[f32], nb: &[f32]) {
    for i in 0..x.len() {
        x1[i] = x[i] + g[i] * (o[i] + b[i]);
    }
    layer_norm_row(x1, h, nw, nb);
}
}

simd_fn! {
/// x += g * (o + b), one row.
fn residual_row(x: &mut [f32], o: &[f32], b: &[f32], g: &[f32]) {
    for i in 0..x.len() {
        x[i] += g[i] * (o[i] + b[i]);
    }
}
}

simd_fn! {
/// f = GELU(f + b), one row.
fn bias_gelu_row(f: &mut [f32], b: &[f32]) {
    for i in 0..f.len() {
        f[i] = crate::ops::gelu1(f[i] + b[i]);
    }
}
}

/// t + bias per row (for tracing the un-fused values).
fn plus_bias(t: &Tensor, b: &[f32]) -> Tensor {
    let mut o = t.clone();
    o.data.par_chunks_mut(b.len()).for_each(|r| r.iter_mut().zip(b).for_each(|(v, b)| *v += b));
    o
}

impl Layer {
    fn load(st: &mut SafeTensors, p: &str, dim: usize, heads: usize, cross: bool) -> Self {
        let attn_name = if cross { "cross_attn" } else { "self_attn" };
        Layer {
            cross,
            attn: Mha::load(st, &format!("{p}.{attn_name}"), dim, heads),
            linear1: (st.take(&format!("{p}.linear1.weight")).data, st.take(&format!("{p}.linear1.bias")).data),
            linear2: (st.take(&format!("{p}.linear2.weight")).data, st.take(&format!("{p}.linear2.bias")).data),
            norm1: Norm::load(st, &format!("{p}.norm1")),
            norm2: Norm::load(st, &format!("{p}.norm2")),
            norm3: cross.then(|| Norm::load(st, &format!("{p}.norm3"))),
            norm_out: Norm::load(st, &format!("{p}.norm_out")),
            gamma1: st.take(&format!("{p}.gamma_1.scale")).data,
            gamma2: st.take(&format!("{p}.gamma_2.scale")).data,
        }
    }

    /// x: (N, C) tokens; kv: the other branch's tokens for a cross layer.
    /// Biases are added in the elementwise passes after each GEMM (residual + LayerScale +
    /// LayerNorm in one pass, bias + GELU in one pass).
    fn forward(&self, x: &Tensor, kv: Option<&Tensor>, tr: &mut Tracer, p: &str) -> Tensor {
        let (n, c) = (x.shape[0], x.shape[1]);
        let h = layer_norm(x, &self.norm1.w, &self.norm1.b);
        let a = if self.cross {
            let k = layer_norm(kv.unwrap(), &self.norm2.w, &self.norm2.b);
            self.attn.attend(&h, Some(&k))
        } else {
            self.attn.attend(&h, None)
        };
        drop(h);
        let o = matmul_t(&a, &self.attn.w_out, c);
        drop(a);
        if tr.on() {
            tr.emit(&format!("{p}.{}", if self.cross { "cross_attn" } else { "self_attn" }), &plus_bias(&o, &self.attn.b_out));
        }
        let nrm = if self.cross { self.norm3.as_ref().unwrap() } else { &self.norm2 };
        let mut x1 = vec![0.0f32; n * c];
        let mut h2 = vec![0.0f32; n * c];
        {
            let _sc = prof::scope(prof::LAYERNORM);
            x1.par_chunks_mut(c)
                .zip(h2.par_chunks_mut(c))
                .zip(x.data.par_chunks(c).zip(o.data.par_chunks(c)))
                .for_each(|((x1r, hr), (xr, or))| residual_ln_row(xr, or, &self.attn.b_out, &self.gamma1, x1r, hr, &nrm.w, &nrm.b));
        }
        drop(o);
        let h2 = Tensor::new(&[n, c], h2);
        let hidden = self.linear1.1.len();
        let mut f = matmul_t(&h2, &self.linear1.0, hidden);
        drop(h2);
        {
            let _sc = prof::scope(prof::ACT);
            f.data.par_chunks_mut(hidden).for_each(|r| bias_gelu_row(r, &self.linear1.1));
        }
        let o2 = matmul_t(&f, &self.linear2.0, c);
        drop(f);
        if tr.on() {
            tr.emit(&format!("{p}.linear2"), &plus_bias(&o2, &self.linear2.1));
        }
        {
            let _sc = prof::scope(prof::COPY);
            x1.par_chunks_mut(c).zip(o2.data.par_chunks(c)).for_each(|(xr, or)| residual_row(xr, or, &self.linear2.1, &self.gamma2));
        }
        let mut x = Tensor::new(&[n, c], x1);
        group_norm_tokens(&mut x, &self.norm_out.w, &self.norm_out.b);
        tr.emit(&format!("{p}.norm_out"), &x);
        x
    }
}

pub struct CrossTransformer {
    dim: usize,
    max_period: f64,
    weight_pos_embed: f32,
    norm_in: Norm,
    norm_in_t: Norm,
    layers: Vec<Layer>,
    layers_t: Vec<Layer>,
    /// weight_pos_embed * the positional embeddings in token layout, for the last shapes seen:
    /// ((C, Fr, T1), 2-D embedding (T1 * Fr, C)) and ((C, T2), 1-D embedding (T2, C)).
    pe_cache: Mutex<PeCache>,
}

type PeCache = (Option<((usize, usize, usize), Arc<Vec<f32>>)>, Option<((usize, usize), Arc<Vec<f32>>)>);

/// (A, B, C) -> (C, B, A) permutation for the token layouts (A = channels): out[(c * B + b) * A + a]
/// = x[(a * B + b) * C + c], in parallel over blocks of the last axis.
fn tokens_from_channels(x: &[f32], a: usize, b: usize, c: usize) -> Vec<f32> {
    const BT: usize = 16;
    let mut out = vec![0.0f32; a * b * c];
    out.par_chunks_mut(BT * b * a).enumerate().for_each(|(blk, o)| {
        let c0 = blk * BT;
        let nc = (c - c0).min(BT);
        for ai in 0..a {
            for bi in 0..b {
                let src = &x[(ai * b + bi) * c + c0..(ai * b + bi) * c + c0 + nc];
                for (k, &v) in src.iter().enumerate() {
                    o[(k * b + bi) * a + ai] = v;
                }
            }
        }
    });
    out
}

/// The inverse of `tokens_from_channels`: tokens (C, B, A) -> (A, B, C).
fn channels_from_tokens(t: &[f32], a: usize, b: usize, c: usize) -> Vec<f32> {
    const BA: usize = 16;
    let mut out = vec![0.0f32; a * b * c];
    out.par_chunks_mut(BA * b * c).enumerate().for_each(|(blk, o)| {
        let a0 = blk * BA;
        let na = (a - a0).min(BA);
        for ci in 0..c {
            for bi in 0..b {
                let row = &t[(ci * b + bi) * a + a0..(ci * b + bi) * a + a0 + na];
                for (k, &v) in row.iter().enumerate() {
                    o[(k * b + bi) * c + ci] = v;
                }
            }
        }
    });
    out
}

impl CrossTransformer {
    pub fn load(st: &mut SafeTensors, dim: usize, heads: usize, n_layers: usize, max_period: f64, weight_pos_embed: f64) -> Self {
        let p = "crosstransformer";
        let mut layers = vec![];
        let mut layers_t = vec![];
        for i in 0..n_layers {
            let cross = i % 2 == 1; // classic_parity = 0 (t_cross_first = False)
            layers.push(Layer::load(st, &format!("{p}.layers.{i}"), dim, heads, cross));
            layers_t.push(Layer::load(st, &format!("{p}.layers_t.{i}"), dim, heads, cross));
        }
        CrossTransformer {
            dim,
            max_period,
            weight_pos_embed: weight_pos_embed as f32,
            norm_in: Norm::load(st, &format!("{p}.norm_in")),
            norm_in_t: Norm::load(st, &format!("{p}.norm_in_t")),
            layers,
            layers_t,
            pe_cache: Mutex::new((None, None)),
        }
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn is_cross(&self, i: usize) -> bool {
        self.layers[i].cross
    }

    /// Runs layer `i` of one branch alone on (N, C) tokens (for the parity test's isolated
    /// per-layer checks); `kv` is the other branch's tokens for a cross layer.
    pub fn layer(&self, time_branch: bool, i: usize, x: &Tensor, kv: Option<&Tensor>, tr: &mut Tracer) -> Tensor {
        let (l, p) = if time_branch { (&self.layers_t[i], "layers_t") } else { (&self.layers[i], "layers") };
        l.forward(x, kv, tr, &format!("crosstransformer.{p}.{i}"))
    }

    /// weight_pos_embed * pos_emb_2d in token layout (T1 * Fr, C), cached.
    fn pe2(&self, c: usize, fr: usize, t1: usize) -> Arc<Vec<f32>> {
        let mut g = self.pe_cache.lock().unwrap();
        if let Some((k, v)) = &g.0 {
            if *k == (c, fr, t1) {
                return v.clone();
            }
        }
        let pe = pos_emb_2d(c, fr, t1, self.max_period);
        let w = self.weight_pos_embed;
        let mut v = tokens_from_channels(&pe.data, c, fr, t1);
        v.iter_mut().for_each(|x| *x *= w);
        let v = Arc::new(v);
        g.0 = Some(((c, fr, t1), v.clone()));
        v
    }

    /// weight_pos_embed * pos_emb_1d, (T2, C), cached.
    fn pe1(&self, c: usize, t2: usize) -> Arc<Vec<f32>> {
        let mut g = self.pe_cache.lock().unwrap();
        if let Some((k, v)) = &g.1 {
            if *k == (c, t2) {
                return v.clone();
            }
        }
        let w = self.weight_pos_embed;
        let v = Arc::new(pos_emb_1d(t2, c, self.max_period).data.iter().map(|x| w * x).collect::<Vec<f32>>());
        g.1 = Some(((c, t2), v.clone()));
        v
    }

    /// x: (C, Fr, T1) spectral features; xt: (C, T2) waveform features. Returns both, same shapes.
    pub fn forward(&self, x: &Tensor, xt: &Tensor, tr: &mut Tracer) -> (Tensor, Tensor) {
        let (c, fr, t1) = (x.shape[0], x.shape[1], x.shape[2]);
        let t2 = xt.shape[1];
        assert_eq!(c, self.dim);
        // "b c fr t1 -> b (t1 fr) c": token index t * Fr + f; then norm_in and the embedding.
        let sc = prof::scope(prof::LAYERNORM);
        let tok = tokens_from_channels(&x.data, c, fr, t1);
        let mut xs = layer_norm(&Tensor::new(&[t1 * fr, c], tok), &self.norm_in.w, &self.norm_in.b);
        drop(sc);
        tr.emit("crosstransformer.norm_in", &xs);
        if tr.on() {
            tr.emit("pos_emb_2d", &pos_emb_2d(c, fr, t1, self.max_period));
        }
        let pe2 = self.pe2(c, fr, t1);
        xs.data.par_chunks_mut(c).zip(pe2.par_chunks(c)).for_each(|(r, p)| r.iter_mut().zip(p).for_each(|(v, p)| *v += p));

        let sc = prof::scope(prof::LAYERNORM);
        let tok = tokens_from_channels(&xt.data, c, 1, t2);
        let mut xts = layer_norm(&Tensor::new(&[t2, c], tok), &self.norm_in_t.w, &self.norm_in_t.b);
        drop(sc);
        tr.emit("crosstransformer.norm_in_t", &xts);
        if tr.on() {
            tr.emit("pos_emb_1d", &pos_emb_1d(t2, c, self.max_period));
        }
        let pe1 = self.pe1(c, t2);
        xts.data.par_chunks_mut(c).zip(pe1.par_chunks(c)).for_each(|(r, p)| r.iter_mut().zip(p).for_each(|(v, p)| *v += p));

        for (i, (l, lt)) in self.layers.iter().zip(&self.layers_t).enumerate() {
            let p = format!("crosstransformer.layers.{i}");
            let pt = format!("crosstransformer.layers_t.{i}");
            if l.cross {
                let old_x = xs;
                xs = l.forward(&old_x, Some(&xts), tr, &p);
                xts = lt.forward(&xts, Some(&old_x), tr, &pt);
            } else {
                xs = l.forward(&xs, None, tr, &p);
                xts = lt.forward(&xts, None, tr, &pt);
            }
            tr.emit(&p, &xs);
            tr.emit(&pt, &xts);
        }

        // "b (t1 fr) c -> b c fr t1"
        let out = Tensor::new(&[c, fr, t1], channels_from_tokens(&xs.data, c, fr, t1));
        let outt = Tensor::new(&[c, t2], channels_from_tokens(&xts.data, c, 1, t2));
        (out, outt)
    }
}

/// create_2d_sin_embedding(d_model, height, width): (C, Fr, T1). Channels [0, C/2) encode the
/// time position (even: sin, odd: cos), [C/2, C) the frequency position. Mirrors torch's float32
/// arithmetic: div_term = exp(arange(0, C/2, 2) * -(ln(max_period) / (C/2))), sin(pos * div_term).
pub fn pos_emb_2d(c: usize, fr: usize, t1: usize, max_period: f64) -> Tensor {
    let half = c / 2;
    let coef = -(max_period.ln() / half as f64) as f32;
    let div: Vec<f32> = (0..half / 2).map(|i| (((2 * i) as f32 * coef) as f64).exp() as f32).collect();
    let mut pe = Tensor::zeros(&[c, fr, t1]);
    for (i, &d) in div.iter().enumerate() {
        for f in 0..fr {
            let pf = f as f32 * d;
            for t in 0..t1 {
                let pt = t as f32 * d;
                pe.data[((2 * i) * fr + f) * t1 + t] = (pt as f64).sin() as f32;
                pe.data[((2 * i + 1) * fr + f) * t1 + t] = (pt as f64).cos() as f32;
                pe.data[((half + 2 * i) * fr + f) * t1 + t] = (pf as f64).sin() as f32;
                pe.data[((half + 2 * i + 1) * fr + f) * t1 + t] = (pf as f64).cos() as f32;
            }
        }
    }
    pe
}

/// create_sin_embedding(length, dim): (T, C) = [cos(phase), sin(phase)] with
/// phase = pos / max_period ** (i / (C/2 - 1)), each step rounded to float32 like torch.
pub fn pos_emb_1d(t: usize, c: usize, max_period: f64) -> Tensor {
    let half = c / 2;
    let den: Vec<f32> = (0..half)
        .map(|i| {
            let e = i as f32 / (half - 1) as f32;
            (max_period.powf(e as f64)) as f32
        })
        .collect();
    let mut pe = Tensor::zeros(&[t, c]);
    for p in 0..t {
        for i in 0..half {
            let ph = p as f32 / den[i];
            pe.data[p * c + i] = (ph as f64).cos() as f32;
            pe.data[p * c + half + i] = (ph as f64).sin() as f32;
        }
    }
    pe
}
