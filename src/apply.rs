//! Full-track separation: demucs/separate.py's normalisation and demucs/apply.py's `apply_model`
//! (random shifts, segmentation with overlap-add), reproduced step for step in float32.
//!
//! * Normalise by the whole track: ref = mean over channels; wav = (wav - mean(ref)) / std(ref)
//!   (unbiased std); undone on the output.
//! * Shifts (if > 0): zero-pad by max_shift = samplerate / 2 on both sides; each pass draws
//!   offset = random.randint(0, max_shift), separates the view [offset, offset + len + max_shift -
//!   offset) of the padded track, and keeps its output from max_shift - offset on; passes are
//!   averaged.
//! * Split: chunks of segment_length = training length at offsets range(0, len, stride), stride =
//!   int((1 - overlap) * segment_length). Each chunk is centred in a training-length window that
//!   reads REAL samples of the underlying (padded) track on both sides where they exist (zeros only
//!   past its ends) - TensorChunk.padded - and the output is centre-trimmed back. Outputs are
//!   overlap-added with a triangle weight (weight[:chunk_length], 1..L/2, L-L/2..1, / max) and
//!   divided by the summed weights.
//! * Every segment forward draws random.randrange(1) (see pyrandom.rs), which moves the generator.
//!
//! Memory: one chunk's activations at a time, plus the output buffers (sources x 2 x length).

use rayon::prelude::*;

use crate::ops::mean_std;
use crate::pyrandom::PyRandom;
use crate::tensor::Tensor;
use crate::{HTDemucs, Tracer};

pub struct Options {
    pub shifts: usize,
    pub seed: i128,
    pub overlap: f32,
}

impl Default for Options {
    fn default() -> Self {
        Options { shifts: 1, seed: 0, overlap: 0.25 }
    }
}

/// Number of model forwards `separate` will run for a track of `len` samples (for progress).
pub fn forwards(model: &HTDemucs, len: usize, opts: &Options) -> usize {
    let seg = model.cfg.training_length;
    let stride = ((1.0 - opts.overlap) * seg as f32) as usize;
    if opts.shifts == 0 {
        return len.div_ceil(stride);
    }
    // Upper bound per pass (the view length depends on the random offset).
    opts.shifts * (len + model.cfg.samplerate / 2).div_ceil(stride)
}

/// Separates a stereo track `wav` (channels x len, at the model's sample rate). Returns
/// (sources, channels, len). `progress` is called after every segment forward.
pub fn separate(model: &HTDemucs, wav: &[f32], channels: usize, opts: &Options, progress: &mut dyn FnMut()) -> Tensor {
    let cfg = &model.cfg;
    assert_eq!(channels, cfg.audio_channels);
    let len = wav.len() / channels;
    // separate.py: ref = wav.mean(0); wav -= ref.mean(); wav /= ref.std()
    let refm: Vec<f32> = (0..len).map(|i| (0..channels).map(|c| wav[c * len + i]).sum::<f32>() / channels as f32).collect();
    let (mean, std) = mean_std(&refm);
    drop(refm);
    let mix: Vec<f32> = wav.par_iter().map(|&v| (v - mean) / std).collect();

    let mut rng = PyRandom::new(opts.seed);
    let nsrc = cfg.sources.len();
    let mut out = if opts.shifts > 0 {
        let ms = cfg.samplerate / 2;
        let n = len + 2 * ms;
        let mut padded = vec![0.0f32; channels * n];
        for c in 0..channels {
            padded[c * n + ms..c * n + ms + len].copy_from_slice(&mix[c * len..(c + 1) * len]);
        }
        drop(mix);
        let mut acc = vec![0.0f32; nsrc * channels * len];
        for _ in 0..opts.shifts {
            let offset = rng.randint(0, ms as u32) as usize;
            let view_len = len + ms - offset;
            let pass = split(model, &padded, channels, offset, view_len, opts, &mut rng, progress);
            // out += shifted_out[..., max_shift - offset:]
            acc.par_chunks_mut(len).zip(pass.par_chunks(view_len)).for_each(|(a, p)| {
                for (x, y) in a.iter_mut().zip(&p[ms - offset..]) {
                    *x += y;
                }
            });
        }
        let s = opts.shifts as f32;
        acc.par_iter_mut().for_each(|v| *v /= s);
        acc
    } else {
        split(model, &mix, channels, 0, len, opts, &mut rng, progress)
    };
    // sources *= ref.std(); sources += ref.mean()
    out.par_iter_mut().for_each(|v| *v = *v * std + mean);
    Tensor::new(&[nsrc, channels, len], out)
}

/// apply_model's split branch on the view [start, start + view_len) of `base` (channels x N).
/// Returns (sources x channels x view_len), flattened.
#[allow(clippy::too_many_arguments)]
fn split(
    model: &HTDemucs,
    base: &[f32],
    channels: usize,
    start: usize,
    view_len: usize,
    opts: &Options,
    rng: &mut PyRandom,
    progress: &mut dyn FnMut(),
) -> Vec<f32> {
    let seg = model.cfg.training_length;
    let n = base.len() / channels;
    let nsrc = model.cfg.sources.len();
    let stride = ((1.0 - opts.overlap) * seg as f32) as usize;
    // th.cat([arange(1, L//2 + 1), arange(L - L//2, 0, -1)]) / max, in float32.
    let half = seg / 2;
    let wmax = half.max(seg - half) as f32;
    let weight: Vec<f32> = (1..=half).chain((1..=seg - half).rev()).map(|v| v as f32 / wmax).collect();

    let mut out = vec![0.0f32; nsrc * channels * view_len];
    let mut sum_weight = vec![0.0f32; view_len];
    let mut input = vec![0.0f32; channels * seg];
    for off in (0..view_len).step_by(stride) {
        let chunk_len = seg.min(view_len - off);
        // TensorChunk.padded(seg): centre the chunk, real samples where the tensor has them.
        let delta = seg - chunk_len;
        let s0 = (start + off) as isize - (delta / 2) as isize;
        for c in 0..channels {
            let plane = &base[c * n..(c + 1) * n];
            for (i, v) in input[c * seg..(c + 1) * seg].iter_mut().enumerate() {
                let j = s0 + i as isize;
                *v = if j >= 0 && (j as usize) < n { plane[j as usize] } else { 0.0 };
            }
        }
        // The transformer's positional embedding draws random.randrange(sin_random_shift + 1).
        rng.randrange(1);
        let y = model.forward(&input, &mut Tracer::none());
        progress();
        // center_trim, then out[..., off:off + L] += weight[:chunk_len] * chunk_out
        let t0 = delta / 2;
        out.par_chunks_mut(view_len).enumerate().for_each(|(sc, o)| {
            let src = &y.data[sc * seg + t0..sc * seg + t0 + chunk_len];
            for ((o, &v), &w) in o[off..off + chunk_len].iter_mut().zip(src).zip(&weight) {
                *o += w * v;
            }
        });
        for (s, &w) in sum_weight[off..off + chunk_len].iter_mut().zip(&weight) {
            *s += w;
        }
    }
    out.par_chunks_mut(view_len).for_each(|o| {
        for (v, s) in o.iter_mut().zip(&sum_weight) {
            *v /= s;
        }
    });
    out
}
