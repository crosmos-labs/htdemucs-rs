//! Demucs' CrossTransformerEncoder (demucs/transformer.py) for the pretrained configuration:
//! layers alternate self-attention (even indices) and cross-attention (odd), each branch (spectral
//! tokens `x`, waveform tokens `xt`) with its own weights; norm_first, LayerScale, GELU
//! feed-forward, and a GroupNorm(1) "norm_out" over the WHOLE sequence after every layer.

use crate::ops::{attention, gelu, group_norm_tokens, layer_norm, linear};
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
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    bq: Vec<f32>,
    bk: Vec<f32>,
    bv: Vec<f32>,
    w_out: Vec<f32>,
    b_out: Vec<f32>,
    heads: usize,
}

impl Mha {
    fn load(st: &mut SafeTensors, p: &str, dim: usize, heads: usize) -> Self {
        let w = st.take(&format!("{p}.in_proj_weight")).data;
        let b = st.take(&format!("{p}.in_proj_bias")).data;
        let n = dim * dim;
        Mha {
            wq: w[..n].to_vec(),
            wk: w[n..2 * n].to_vec(),
            wv: w[2 * n..].to_vec(),
            bq: b[..dim].to_vec(),
            bk: b[dim..2 * dim].to_vec(),
            bv: b[2 * dim..].to_vec(),
            w_out: st.take(&format!("{p}.out_proj.weight")).data,
            b_out: st.take(&format!("{p}.out_proj.bias")).data,
            heads,
        }
    }

    fn forward(&self, q: &Tensor, kv: &Tensor) -> Tensor {
        let qp = linear(q, &self.wq, &self.bq);
        let kp = linear(kv, &self.wk, &self.bk);
        let vp = linear(kv, &self.wv, &self.bv);
        let a = attention(&qp, &kp, &vp, self.heads);
        linear(&a, &self.w_out, &self.b_out)
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
    fn forward(&self, x: &Tensor, kv: Option<&Tensor>, tr: &mut Tracer, p: &str) -> Tensor {
        let h = layer_norm(x, &self.norm1.w, &self.norm1.b);
        let a = if self.cross {
            let k = layer_norm(kv.unwrap(), &self.norm2.w, &self.norm2.b);
            self.attn.forward(&h, &k)
        } else {
            self.attn.forward(&h, &h)
        };
        tr.emit(&format!("{p}.{}", if self.cross { "cross_attn" } else { "self_attn" }), &a);
        let mut x = x.clone();
        add_scaled(&mut x, &a, &self.gamma1);
        let n = if self.cross { self.norm3.as_ref().unwrap() } else { &self.norm2 };
        let h = layer_norm(&x, &n.w, &n.b);
        let mut f = linear(&h, &self.linear1.0, &self.linear1.1);
        gelu(&mut f.data);
        let f = linear(&f, &self.linear2.0, &self.linear2.1);
        tr.emit(&format!("{p}.linear2"), &f);
        add_scaled(&mut x, &f, &self.gamma2);
        group_norm_tokens(&mut x, &self.norm_out.w, &self.norm_out.b);
        tr.emit(&format!("{p}.norm_out"), &x);
        x
    }
}

/// x += gamma (per channel) * y, for (N, C) tokens.
fn add_scaled(x: &mut Tensor, y: &Tensor, gamma: &[f32]) {
    let c = gamma.len();
    for (xr, yr) in x.data.chunks_mut(c).zip(y.data.chunks(c)) {
        for i in 0..c {
            xr[i] += gamma[i] * yr[i];
        }
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

    /// x: (C, Fr, T1) spectral features; xt: (C, T2) waveform features. Returns both, same shapes.
    pub fn forward(&self, x: &Tensor, xt: &Tensor, tr: &mut Tracer) -> (Tensor, Tensor) {
        let (c, fr, t1) = (x.shape[0], x.shape[1], x.shape[2]);
        let t2 = xt.shape[1];
        assert_eq!(c, self.dim);
        // "b c fr t1 -> b (t1 fr) c": token index t * Fr + f.
        let mut tok = Tensor::zeros(&[t1 * fr, c]);
        for ci in 0..c {
            for f in 0..fr {
                for t in 0..t1 {
                    tok.data[(t * fr + f) * c + ci] = x.data[(ci * fr + f) * t1 + t];
                }
            }
        }
        let mut xs = layer_norm(&tok, &self.norm_in.w, &self.norm_in.b);
        tr.emit("crosstransformer.norm_in", &xs);
        let pe2 = pos_emb_2d(c, fr, t1, self.max_period);
        if tr.on() {
            tr.emit("pos_emb_2d", &pe2);
        }
        // pe2 is (C, Fr, T1); add it in token order.
        for ci in 0..c {
            for f in 0..fr {
                for t in 0..t1 {
                    xs.data[(t * fr + f) * c + ci] += self.weight_pos_embed * pe2.data[(ci * fr + f) * t1 + t];
                }
            }
        }

        let mut xts = layer_norm(&xt.t(), &self.norm_in_t.w, &self.norm_in_t.b);
        tr.emit("crosstransformer.norm_in_t", &xts);
        let pe1 = pos_emb_1d(t2, c, self.max_period);
        if tr.on() {
            tr.emit("pos_emb_1d", &pe1);
        }
        for (v, p) in xts.data.iter_mut().zip(&pe1.data) {
            *v += self.weight_pos_embed * p;
        }

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
        let mut out = Tensor::zeros(&[c, fr, t1]);
        for ci in 0..c {
            for f in 0..fr {
                for t in 0..t1 {
                    out.data[(ci * fr + f) * t1 + t] = xs.data[(t * fr + f) * c + ci];
                }
            }
        }
        (out, xts.t())
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
