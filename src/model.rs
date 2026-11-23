//! HTDemucs forward pass for one segment (demucs/htdemucs.py `HTDemucs.forward`, eval mode).
//!
//! Layouts: the frequency branch is (C, F, T) - channels, frequency bins, STFT frames; the time
//! branch is (C, T), handled as (C, 1, T) so both use the same convolution code.

use std::path::Path;

use rayon::prelude::*;

use crate::config::Config;
use crate::ops::{bias_gelu, dconv_rows, gelu, gelu1, glu_bias, group_norm1_glu_add_b, group_norm1_then, mean_std, plus_channel_bias, prof, Conv, ConvTr};
use crate::stft::Stft;
use crate::tensor::Tensor;
use crate::transformer::CrossTransformer;
use crate::weights::SafeTensors;
use crate::Tracer;

fn conv(st: &mut SafeTensors, p: &str, k: (usize, usize), stride: (usize, usize), pad: (usize, usize), dil: (usize, usize)) -> Conv {
    let w = st.take(&format!("{p}.weight"));
    let b = st.take(&format!("{p}.bias")).data;
    let (cout, cin) = (w.shape[0], w.shape[1]);
    assert_eq!(w.numel(), cout * cin * k.0 * k.1, "{p}: kernel shape {:?}", w.shape);
    Conv::new(w.data, b, cin, cout, k, stride, pad, dil)
}

fn conv_tr(st: &mut SafeTensors, p: &str, k: (usize, usize), stride: (usize, usize)) -> ConvTr {
    let w = st.take(&format!("{p}.weight"));
    let b = st.take(&format!("{p}.bias")).data;
    let (cin, cout) = (w.shape[0], w.shape[1]);
    assert_eq!(w.numel(), cout * cin * k.0 * k.1, "{p}: kernel shape {:?}", w.shape);
    ConvTr { w: w.data, b, cin, cout, k, stride }
}

/// One residual branch of DConv: dilated conv (C -> C/comp, k=3) -> GroupNorm(1) -> GELU ->
/// 1x1 conv (-> 2C) -> GroupNorm(1) -> GLU -> LayerScale.
struct DConvLayer {
    conv1: Conv,
    gn1: (Vec<f32>, Vec<f32>),
    conv2: Conv,
    gn2: (Vec<f32>, Vec<f32>),
    scale: Vec<f32>,
}

/// demucs.demucs.DConv. In the frequency branch PyTorch runs it on (B*F, C, T) - every frequency
/// row is a separate sample, so its GroupNorms normalise per row. We keep (C, F, T) and do the
/// same: convolutions only along T, statistics per (row f).
struct DConv {
    layers: Vec<DConvLayer>,
}

impl DConv {
    fn load(st: &mut SafeTensors, p: &str, depth: usize) -> Self {
        let layers = (0..depth)
            .map(|d| {
                let q = format!("{p}.layers.{d}");
                let dil = 1 << d;
                DConvLayer {
                    conv1: conv(st, &format!("{q}.0"), (1, 3), (1, 1), (0, dil), (1, dil)),
                    gn1: (st.take(&format!("{q}.1.weight")).data, st.take(&format!("{q}.1.bias")).data),
                    conv2: conv(st, &format!("{q}.3"), (1, 1), (1, 1), (0, 0), (1, 1)),
                    gn2: (st.take(&format!("{q}.4.weight")).data, st.take(&format!("{q}.4.bias")).data),
                    scale: st.take(&format!("{q}.6.scale")).data,
                }
            })
            .collect();
        DConv { layers }
    }

    fn forward(&self, mut x: Tensor, tr: &mut Tracer, p: &str) -> Tensor {
        // Frequency branch (F > 1): each frequency row is its own GroupNorm sample, so the whole
        // branch runs row by row (ops::dconv_rows). The choice depends only on the shape.
        if x.shape[1] > 1 {
            for (d, l) in self.layers.iter().enumerate() {
                let g1 = (&l.gn1.0[..], &l.gn1.1[..]);
                let g2 = (&l.gn2.0[..], &l.gn2.1[..]);
                if tr.on() {
                    let mut delta = Tensor::zeros(&x.shape);
                    dconv_rows(&mut x, &l.conv1, g1, &l.conv2, g2, &l.scale, Some(&mut delta));
                    tr.emit(&format!("{p}.layers.{d}"), &torch_dconv_layout(&delta));
                } else {
                    dconv_rows(&mut x, &l.conv1, g1, &l.conv2, g2, &l.scale, None);
                }
            }
            if tr.on() {
                tr.emit(p, &torch_dconv_layout(&x));
            }
            return x;
        }
        for (d, l) in self.layers.iter().enumerate() {
            let mut y = l.conv1.forward(&x);
            group_norm1_then(&mut y, &l.gn1.0, &l.gn1.1, gelu1);
            let y = l.conv2.forward_raw(&y);
            // x = x + LayerScale(GLU(GroupNorm(y + bias))), fused into one pass.
            let b = Some(&l.conv2.b[..]);
            if tr.on() {
                let mut delta = Tensor::zeros(&x.shape);
                group_norm1_glu_add_b(&y, b, &l.gn2.0, &l.gn2.1, &l.scale, &mut x, Some(&mut delta));
                tr.emit(&format!("{p}.layers.{d}"), &torch_dconv_layout(&delta));
            } else {
                group_norm1_glu_add_b(&y, b, &l.gn2.0, &l.gn2.1, &l.scale, &mut x, None);
            }
        }
        if tr.on() {
            tr.emit(p, &torch_dconv_layout(&x));
        }
        x
    }
}

/// PyTorch runs DConv on (B*F, C, T); for comparisons, (C, F, T) -> (F, C, T).
fn torch_dconv_layout(x: &Tensor) -> Tensor {
    if x.shape[1] == 1 { x.clone().reshape(&[x.shape[0], x.shape[2]]) } else { x.swap01() }
}

/// HEncLayer: conv (stride 4, kernel 8, pad 2; along F in the frequency branch, T in the time
/// branch) -> GELU -> DConv -> 1x1 conv -> GLU. (No GroupNorm: norm_starts >= depth.)
struct EncLayer {
    freq: bool,
    stride: usize,
    conv: Conv,
    dconv: DConv,
    rewrite: Conv,
}

impl EncLayer {
    fn forward(&self, x: &Tensor, tr: &mut Tracer, p: &str) -> Tensor {
        let mut xin;
        let x = if !self.freq && x.shape[2] % self.stride != 0 {
            // Time branch: right-pad with zeros to a multiple of the stride.
            let (c, t) = (x.shape[0], x.shape[2]);
            let tp = t + self.stride - t % self.stride;
            xin = Tensor::zeros(&[c, 1, tp]);
            xin.data.par_chunks_mut(tp).zip(x.data.par_chunks(t)).for_each(|(o, i)| o[..t].copy_from_slice(i));
            &xin
        } else {
            x
        };
        // Convolution biases are added in the following elementwise pass (GELU, GLU).
        let mut y = self.conv.forward_raw(x);
        if tr.on() {
            tr.emit(&format!("{p}.conv"), &plus_channel_bias(&y, &self.conv.b));
        }
        bias_gelu(&mut y, &self.conv.b);
        let y = self.dconv.forward(y, tr, &format!("{p}.dconv"));
        let z = self.rewrite.forward_raw(&y);
        drop(y);
        if tr.on() {
            tr.emit(&format!("{p}.rewrite"), &plus_channel_bias(&z, &self.rewrite.b));
        }
        let z = glu_bias(&z, &self.rewrite.b);
        tr.emit(p, &z);
        z
    }
}

/// HDecLayer: (x + skip) -> context conv (3x3 freq / 3 time) -> GLU -> DConv -> transposed conv
/// (stride 4, kernel 8) -> crop `pad` = 2 -> GELU unless last.
struct DecLayer {
    freq: bool,
    last: bool,
    pad: usize,
    rewrite: Conv,
    dconv: DConv,
    conv_tr: ConvTr,
}

impl DecLayer {
    fn forward(&self, x: &Tensor, skip: Tensor, length: usize, tr: &mut Tracer, p: &str) -> Tensor {
        // x + skip, into the skip's buffer.
        let mut x_in = skip;
        x_in.data.par_chunks_mut(8192).zip(x.data.par_chunks(8192)).for_each(|(a, b)| crate::ops::add_assign(a, b));
        let y = self.rewrite.forward_raw(&x_in);
        drop(x_in);
        if tr.on() {
            tr.emit(&format!("{p}.rewrite"), &plus_channel_bias(&y, &self.rewrite.b));
        }
        let y = glu_bias(&y, &self.rewrite.b);
        let y = self.dconv.forward(y, tr, &format!("{p}.dconv"));
        let z = self.conv_tr.forward(&y);
        tr.emit(&format!("{p}.conv_tr"), &z);
        let (c, h, w) = (z.shape[0], z.shape[1], z.shape[2]);
        let mut z = if self.freq {
            // z[..., pad:-pad, :] along F
            let ho = h - 2 * self.pad;
            let mut out = vec![0.0f32; c * ho * w];
            let pad = self.pad;
            out.par_chunks_mut(ho * w)
                .zip(z.data.par_chunks(h * w))
                .for_each(|(o, i)| o.copy_from_slice(&i[pad * w..(pad + ho) * w]));
            Tensor::new(&[c, ho, w], out)
        } else {
            // z[..., pad:pad + length] along T
            let mut out = vec![0.0f32; c * length];
            let pad = self.pad;
            out.par_chunks_mut(length).zip(z.data.par_chunks(w)).for_each(|(o, i)| o.copy_from_slice(&i[pad..pad + length]));
            Tensor::new(&[c, 1, length], out)
        };
        if !self.last {
            gelu(&mut z.data);
        }
        tr.emit(p, &z);
        z
    }
}

/// glibc serves every allocation above 32 MB (and, by its adaptive threshold, many smaller ones)
/// with a fresh mmap, so each forward's multi-GB of short-lived activations came back as new
/// zero-filled pages: ~200k page faults and ~6% of the CPU time per segment on x86 Linux. Keeping
/// freed memory in the heap instead (no mmap, no trimming) makes the buffers reusable. Process-wide;
/// HTDEMUCS_MALLOC_TUNE=0 leaves glibc's defaults. (No effect on other platforms / C libraries.)
fn tune_malloc() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            if std::env::var("HTDEMUCS_MALLOC_TUNE").as_deref() != Ok("0") {
                unsafe {
                    libc::mallopt(libc::M_MMAP_MAX, 0);
                    libc::mallopt(libc::M_TRIM_THRESHOLD, i32::MAX);
                }
            }
        });
    }
}

pub struct HTDemucs {
    pub cfg: Config,
    stft: Stft,
    encoder: Vec<EncLayer>,
    tencoder: Vec<EncLayer>,
    decoder: Vec<DecLayer>,
    tdecoder: Vec<DecLayer>,
    /// freq_emb_scale * (embedding weight * emb_scale), (Fr0, C0), precomputed in torch's order.
    freq_emb: Option<Tensor>,
    /// channel_upsampler, channel_downsampler, and their _t twins (bottom_channels > 0).
    bottom: Option<[Conv; 4]>,
    pub transformer: CrossTransformer,
}

impl HTDemucs {
    /// Loads `<dir>/<name>.json` and `<dir>/<name>.safetensors` (from tools/convert.py).
    pub fn load(dir: impl AsRef<Path>, name: &str) -> Result<Self, String> {
        tune_malloc();
        let dir = dir.as_ref();
        let cfg = Config::load(dir.join(format!("{name}.json")))?;
        let mut st = SafeTensors::load(dir.join(format!("{name}.safetensors")))?;
        let m = Self::from_weights(cfg, &mut st);
        if !st.tensors.is_empty() {
            let mut left: Vec<_> = st.tensors.keys().cloned().collect();
            left.sort();
            return Err(format!("unused weights: {left:?}"));
        }
        Ok(m)
    }

    fn from_weights(cfg: Config, st: &mut SafeTensors) -> Self {
        let k = cfg.kernel_size;
        let s = cfg.stride;
        let pad = k / 4;
        let mut encoder = vec![];
        let mut tencoder = vec![];
        let mut decoder = vec![];
        let mut tdecoder = vec![];
        for i in 0..cfg.depth {
            let last = i == 0;
            let p = format!("encoder.{i}");
            encoder.push(EncLayer {
                freq: true,
                stride: s,
                conv: conv(st, &format!("{p}.conv"), (k, 1), (s, 1), (pad, 0), (1, 1)),
                dconv: DConv::load(st, &format!("{p}.dconv"), cfg.dconv_depth),
                rewrite: conv(st, &format!("{p}.rewrite"), (1, 1), (1, 1), (0, 0), (1, 1)),
            });
            let p = format!("tencoder.{i}");
            tencoder.push(EncLayer {
                freq: false,
                stride: s,
                conv: conv(st, &format!("{p}.conv"), (1, k), (1, s), (0, pad), (1, 1)),
                dconv: DConv::load(st, &format!("{p}.dconv"), cfg.dconv_depth),
                rewrite: conv(st, &format!("{p}.rewrite"), (1, 1), (1, 1), (0, 0), (1, 1)),
            });
            // Decoders are stored outermost-last: decoder.{depth-1-i} mirrors encoder.i.
            let j = cfg.depth - 1 - i;
            let p = format!("decoder.{j}");
            decoder.push(DecLayer {
                freq: true,
                last,
                pad,
                rewrite: conv(st, &format!("{p}.rewrite"), (3, 3), (1, 1), (1, 1), (1, 1)),
                dconv: DConv::load(st, &format!("{p}.dconv"), cfg.dconv_depth),
                conv_tr: conv_tr(st, &format!("{p}.conv_tr"), (k, 1), (s, 1)),
            });
            let p = format!("tdecoder.{j}");
            tdecoder.push(DecLayer {
                freq: false,
                last,
                pad,
                rewrite: conv(st, &format!("{p}.rewrite"), (1, 3), (1, 1), (0, 1), (1, 1)),
                dconv: DConv::load(st, &format!("{p}.dconv"), cfg.dconv_depth),
                conv_tr: conv_tr(st, &format!("{p}.conv_tr"), (1, k), (1, s)),
            });
        }
        decoder.reverse();
        tdecoder.reverse();

        let freq_emb = (cfg.freq_emb > 0.0).then(|| {
            // ScaledEmbedding.forward = embedding(x) * scale; HTDemucs adds freq_emb_scale * that.
            let mut w = st.take("freq_emb.embedding.weight");
            let (scale, fscale) = (cfg.emb_scale as f32, cfg.freq_emb as f32);
            w.data.iter_mut().for_each(|v| *v = fscale * (*v * scale));
            w
        });

        let bottom = (cfg.bottom_channels > 0).then(|| {
            let c = |st: &mut SafeTensors, n: &str| conv(st, n, (1, 1), (1, 1), (0, 0), (1, 1));
            [
                c(st, "channel_upsampler"),
                c(st, "channel_downsampler"),
                c(st, "channel_upsampler_t"),
                c(st, "channel_downsampler_t"),
            ]
        });
        let tdim = if cfg.bottom_channels > 0 { cfg.bottom_channels } else { cfg.channels * cfg.growth.pow(cfg.depth as u32 - 1) };
        let transformer = CrossTransformer::load(st, tdim, cfg.t_heads, cfg.t_layers, cfg.t_max_period, cfg.t_weight_pos_embed);
        HTDemucs { stft: Stft::new(cfg.nfft), cfg, encoder, tencoder, decoder, tdecoder, freq_emb, bottom, transformer }
    }

    pub fn sources(&self) -> &[String] {
        &self.cfg.sources
    }

    /// Separate one segment. `mix` is (audio_channels, training_length) row-major, already
    /// normalised the way demucs/separate.py does for the whole track. Returns
    /// (sources, audio_channels, training_length).
    pub fn forward(&self, mix: &[f32], tr: &mut Tracer) -> Tensor {
        let cfg = &self.cfg;
        let ch = cfg.audio_channels;
        let len = cfg.training_length;
        assert_eq!(mix.len(), ch * len, "forward takes exactly one training-length segment");
        let nsrc = cfg.sources.len();

        // --- Spectrogram (complex as channels): (2*ch, F, T), channel order [c0 re, c0 im, c1 re, ...]
        prof::branch(0);
        let sc = prof::scope(prof::STFT);
        let specs: Vec<(Vec<f32>, usize)> = (0..ch).into_par_iter().map(|c| self.stft.spec(&mix[c * len..(c + 1) * len])).collect();
        drop(sc);
        let bins = cfg.nfft / 2;
        let frames = specs[0].1;
        if tr.on() {
            let mut z = Vec::with_capacity(ch * bins * frames * 2);
            for (s, _) in &specs {
                z.extend_from_slice(s);
            }
            tr.emit("spec", &Tensor::new(&[ch, bins, frames, 2], z));
        }
        let mut mag = Tensor::zeros(&[2 * ch, bins, frames]);
        let plane = bins * frames;
        mag.data.par_chunks_mut(frames).enumerate().for_each(|(r, row)| {
            // row r = (channel c, re/im part, bin b)
            let (c, part, b) = (r / (2 * bins), (r / bins) % 2, r % bins);
            let s = &specs[c].0[b * frames * 2..(b + 1) * frames * 2];
            for (f, v) in row.iter_mut().enumerate() {
                *v = s[2 * f + part];
            }
        });
        let _ = plane;
        tr.emit("mag", &mag);

        // --- Standardise both branch inputs (torch .std is unbiased; eps added to std).
        let (mean, std) = mean_std(&mag.data);
        let mut x = mag;
        x.data.par_iter_mut().for_each(|v| *v = (*v - mean) / (1e-5 + std));
        tr.emit("x_norm", &x);
        let (meant, stdt) = mean_std(mix);
        let mut xt = Tensor::new(&[ch, 1, len], mix.par_iter().map(|v| (v - meant) / (1e-5 + stdt)).collect());
        tr.emit("xt_norm", &xt);

        // --- Encoders.
        let mut saved = vec![];
        let mut saved_t = vec![];
        let mut lengths_t = vec![];
        for (i, (enc, tenc)) in self.encoder.iter().zip(&self.tencoder).enumerate() {
            // Each layer's output is kept as the decoder's skip and read from there.
            let xt_in: &Tensor = if i == 0 { &xt } else { &saved_t[i - 1] };
            lengths_t.push(xt_in.shape[2]);
            prof::branch(1);
            let yt = tenc.forward(xt_in, tr, &format!("tencoder.{i}"));
            saved_t.push(yt);
            prof::branch(0);
            let x_in: &Tensor = if i == 0 { &x } else { &saved[i - 1] };
            let mut x = enc.forward(x_in, tr, &format!("encoder.{i}"));
            if i == 0 {
                if let Some(emb) = &self.freq_emb {
                    // x[c, f, t] += emb[f, c]
                    let (c, f, t) = (x.shape[0], x.shape[1], x.shape[2]);
                    assert_eq!(emb.shape, [f, c]);
                    x.data.par_chunks_mut(f * t).enumerate().for_each(|(ci, plane)| {
                        for fi in 0..f {
                            let e = emb.data[fi * c + ci];
                            plane[fi * t..(fi + 1) * t].iter_mut().for_each(|v| *v += e);
                        }
                    });
                    tr.emit("encoder.0+emb", &x);
                }
            }
            saved.push(x);
        }
        x = saved.last().unwrap().clone();
        xt = saved_t.last().unwrap().clone();

        // --- Cross-transformer (with 1x1 channel up/down-sampling around it if bottom_channels).
        let (fr, t1) = (x.shape[1], x.shape[2]);
        prof::branch(2);
        if let Some([up, _, up_t, _]) = &self.bottom {
            // rearrange "b c f t -> b c (f t)" then a 1x1 conv: layout-agnostic for us.
            x = up.forward(&x);
            tr.emit("channel_upsampler", &x);
            xt = up_t.forward(&xt);
            tr.emit("channel_upsampler_t", &xt);
        }
        let (xo, xto) = self.transformer.forward(&x, &xt.clone().reshape(&[xt.shape[0], xt.shape[2]]), tr);
        tr.emit("crosstransformer", &xo);
        x = xo;
        xt = xto.reshape(&[xt.shape[0], 1, xt.shape[2]]);
        if let Some([_, down, _, down_t]) = &self.bottom {
            x = down.forward(&x);
            tr.emit("channel_downsampler", &x);
            xt = down_t.forward(&xt);
            tr.emit("channel_downsampler_t", &xt);
        }
        assert_eq!((x.shape[1], x.shape[2]), (fr, t1));

        // --- Decoders.
        for (i, (dec, tdec)) in self.decoder.iter().zip(&self.tdecoder).enumerate() {
            let skip = saved.pop().unwrap();
            prof::branch(0);
            x = dec.forward(&x, skip, 0, tr, &format!("decoder.{i}"));
            let skip_t = saved_t.pop().unwrap();
            let lt = lengths_t.pop().unwrap();
            prof::branch(1);
            xt = tdec.forward(&xt, skip_t, lt, tr, &format!("tdecoder.{i}"));
        }

        // --- Frequency branch output: un-standardise, complex-as-channels -> iSTFT.
        // x is (S * 2*ch, F, T) with channel order (source, audio channel, re/im).
        prof::branch(0);
        x.data.par_iter_mut().for_each(|v| *v = *v * std + mean);
        tr.emit("mask_in", &x);
        let plane = bins * frames;
        if tr.on() {
            let mut z = vec![0.0f32; nsrc * ch * plane * 2];
            for sc in 0..nsrc * ch {
                for i in 0..plane {
                    z[(sc * plane + i) * 2] = x.data[(2 * sc) * plane + i];
                    z[(sc * plane + i) * 2 + 1] = x.data[(2 * sc + 1) * plane + i];
                }
            }
            tr.emit("zout", &Tensor::new(&[nsrc, ch, bins, frames, 2], z));
        }
        let sc = prof::scope(prof::ISTFT);
        let wav: Vec<Vec<f32>> = (0..nsrc * ch)
            .into_par_iter()
            .map(|sc| {
                let re = &x.data[(2 * sc) * plane..(2 * sc + 1) * plane];
                let im = &x.data[(2 * sc + 1) * plane..(2 * sc + 2) * plane];
                self.stft.ispec(re, im, frames, len)
            })
            .collect();
        drop(sc);
        let mut out = Tensor::zeros(&[nsrc, ch, len]);
        out.data.par_chunks_mut(len).zip(wav.par_iter()).for_each(|(o, w)| o.copy_from_slice(w));
        tr.emit("ispec", &out);

        // --- Time branch output, un-standardised, plus the spectral estimate: x = xt + x.
        assert_eq!(xt.numel(), nsrc * ch * len);
        out.data.par_iter_mut().zip(&xt.data).for_each(|(o, t)| *o = (t * stdt + meant) + *o);
        tr.emit("output", &out);
        out
    }
}
