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
//! Memory: the output (sources x channels x length) and one segment's activations. The normalised
//! and padded mix is never materialised (chunk inputs are read from the caller's samples), and
//! each pass's overlap-add lives in a window of one segment: positions no later chunk can reach
//! are finalised (divided by their summed weights) and added into the output as the pass goes.
//! Every value is computed with the same operations in the same order as the whole-buffer
//! version, so the output is bit-identical to it.

use rayon::prelude::*;

use crate::ops::mean_std_by;
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

/// The segment stride: int((1 - overlap) * segment_length), as apply_model computes it. Errors
/// when the overlap is outside [0, 1) or leaves a stride of 0.
pub fn stride(model: &HTDemucs, overlap: f32) -> Result<usize, String> {
    if !(0.0..1.0).contains(&overlap) {
        return Err(format!("overlap must be in [0, 1), not {overlap}"));
    }
    let s = ((1.0 - overlap) * model.cfg.training_length as f32) as usize;
    if s == 0 {
        return Err(format!("overlap {overlap} leaves no stride between segments"));
    }
    Ok(s)
}

/// Number of model forwards `separate` will run for a track of `len` samples, replaying the shift
/// offsets with a copy of the generator (each pass: randint, then one randrange(1) per segment).
pub fn forwards(model: &HTDemucs, len: usize, opts: &Options, rng: &PyRandom) -> usize {
    let stride = ((1.0 - opts.overlap) * model.cfg.training_length as f32) as usize;
    if opts.shifts == 0 {
        return len.div_ceil(stride);
    }
    let ms = model.cfg.samplerate / 2;
    let mut rng = rng.clone();
    let mut total = 0;
    for _ in 0..opts.shifts {
        let offset = rng.randint(0, ms as u32) as usize;
        let n = (len + ms - offset).div_ceil(stride);
        for _ in 0..n {
            rng.randrange(1);
        }
        total += n;
    }
    total
}

/// What `separate` returns.
pub struct Separated {
    /// (sources, channels, len).
    pub stems: Tensor,
    /// The input was digital silence (or constant): its standard deviation is 0, so demucs'
    /// normalisation would divide by zero (demucs itself fails here). The stems are all zero.
    pub silent: bool,
}

/// Separates a track `wav` (channels x len, at the model's sample rate). `rng` is Python's
/// `random` as demucs sees it: seeded once per run and carried on from track to track, as in
/// demucs.separate. `progress` is called after every segment forward.
pub fn separate(model: &HTDemucs, wav: &[f32], channels: usize, opts: &Options, rng: &mut PyRandom, progress: &mut dyn FnMut()) -> Separated {
    let cfg = &model.cfg;
    assert_eq!(channels, cfg.audio_channels);
    let len = wav.len() / channels;
    let nsrc = cfg.sources.len();
    // separate.py: ref = wav.mean(0); wav -= ref.mean(); wav /= ref.std()
    let (mean, std) = mean_std_by(len, |a, buf| {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (0..channels).map(|c| wav[c * len + a + i]).sum::<f32>() / channels as f32;
        }
    });
    if len < 2 || std == 0.0 {
        return Separated { stems: Tensor::zeros(&[nsrc, channels, len]), silent: true };
    }
    // The normalised mix, read on demand: (wav - mean) / std at sample j, zero outside the track.
    // With shifts the track sits max_shift samples into a zero-padded view.
    let mix = Mix { wav, len, mean, std };
    let stride = ((1.0 - opts.overlap) * cfg.training_length as f32) as usize;
    let mut out = vec![0.0f32; nsrc * channels * len];
    if opts.shifts > 0 {
        let ms = cfg.samplerate / 2;
        for _ in 0..opts.shifts {
            let offset = rng.randint(0, ms as u32) as usize;
            let view_len = len + ms - offset;
            // out += shifted_out[..., max_shift - offset:]; view position p is padded position
            // offset + p, i.e. track sample offset + p - max_shift.
            let skip = ms - offset;
            split(model, &mix, channels, offset as isize - ms as isize, view_len, stride, rng, progress, &mut |p0, w, rows| {
                out.par_chunks_mut(len).zip(rows.par_chunks(w)).for_each(|(o, r)| {
                    for (i, v) in r.iter().enumerate() {
                        let p = p0 + i;
                        if p >= skip {
                            o[p - skip] += v;
                        }
                    }
                });
            });
        }
        let s = opts.shifts as f32;
        out.par_iter_mut().for_each(|v| *v /= s);
    } else {
        split(model, &mix, channels, 0, len, stride, rng, progress, &mut |p0, w, rows| {
            out.par_chunks_mut(len).zip(rows.par_chunks(w)).for_each(|(o, r)| o[p0..p0 + r.len()].copy_from_slice(r));
        });
    }
    // sources *= ref.std(); sources += ref.mean()
    out.par_iter_mut().for_each(|v| *v = *v * std + mean);
    Separated { stems: Tensor::new(&[nsrc, channels, len], out), silent: false }
}

/// `separate` as a stream: the same output, handed out in track order as soon as it is final, so
/// nothing track-long is held besides the input. The shift passes run side by side (each with its
/// own one-segment window) rather than one after another; their offsets are drawn up front, from
/// the generator exactly as `separate` would move it (the per-segment draws never change the
/// output). Each finished position is the passes' sum in pass order, as `separate` adds them, so
/// the output is bit-identical to it.
///
///     let mut s = Streamer::new(model, wav, channels, opts, rng);
///     while s.step(progress) {}  // or interleave steps with `ready` / `take`
pub struct Streamer<'a> {
    model: &'a HTDemucs,
    mix: Mix<'a>,
    rows: usize,
    stride: usize,
    weight: Vec<f32>,
    passes: Vec<Pass>,
    /// opts.shifts (0: one unshifted pass, not averaged).
    shifts: usize,
    silent: bool,
    /// Track positions handed out so far.
    done: usize,
    input: Vec<f32>,
}

/// One shift pass: its view of the track, its overlap-add window, and the positions it has
/// finalised but nobody has taken yet.
struct Pass {
    /// View position p is track sample start + p; outputs before `skip` fall before the track.
    start: isize,
    view_len: usize,
    skip: usize,
    next: usize,
    finished: bool,
    win: Vec<f32>,
    wsum: Vec<f32>,
    wbase: usize,
    /// Finalised output for track positions [ring_base, ring_base + ring[0].len()), per row.
    ring: Vec<Vec<f32>>,
    ring_base: usize,
}

impl Pass {
    /// Track positions finalised so far (exclusive end).
    fn end(&self) -> usize {
        self.ring_base + self.ring[0].len()
    }
}

impl<'a> Streamer<'a> {
    /// Draws the passes' offsets from `rng` and moves it past every segment's draw, as `separate`
    /// does (not at all for a silent track).
    pub fn new(model: &'a HTDemucs, wav: &'a [f32], channels: usize, opts: &Options, rng: &mut PyRandom) -> Streamer<'a> {
        let cfg = &model.cfg;
        assert_eq!(channels, cfg.audio_channels);
        let len = wav.len() / channels;
        let rows = cfg.sources.len() * channels;
        let (mean, std) = mean_std_by(len, |a, buf| {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (0..channels).map(|c| wav[c * len + a + i]).sum::<f32>() / channels as f32;
            }
        });
        let silent = len < 2 || std == 0.0;
        let seg = cfg.training_length;
        let stride = ((1.0 - opts.overlap) * seg as f32) as usize;
        let half = seg / 2;
        let wmax = half.max(seg - half) as f32;
        let weight: Vec<f32> = (1..=half).chain((1..=seg - half).rev()).map(|v| v as f32 / wmax).collect();
        let pass = |start: isize, view_len: usize, skip: usize| Pass {
            start,
            view_len,
            skip,
            next: 0,
            finished: false,
            win: vec![0.0; rows * seg],
            wsum: vec![0.0; seg],
            wbase: 0,
            ring: vec![vec![]; rows],
            ring_base: 0,
        };
        let mut passes = vec![];
        if !silent {
            if opts.shifts > 0 {
                let ms = cfg.samplerate / 2;
                for _ in 0..opts.shifts {
                    let offset = rng.randint(0, ms as u32) as usize;
                    let view_len = len + ms - offset;
                    for _ in 0..view_len.div_ceil(stride) {
                        rng.randrange(1);
                    }
                    passes.push(pass(offset as isize - ms as isize, view_len, ms - offset));
                }
            } else {
                for _ in 0..len.div_ceil(stride) {
                    rng.randrange(1);
                }
                passes.push(pass(0, len, 0));
            }
        }
        Streamer { model, mix: Mix { wav, len, mean, std }, rows, stride, weight, passes, shifts: opts.shifts, silent, done: 0, input: vec![0.0; channels * seg] }
    }

    /// The input was silent (or constant): the output is all zeros (see `Separated::silent`).
    pub fn silent(&self) -> bool {
        self.silent
    }

    /// Track length in samples.
    pub fn len(&self) -> usize {
        self.mix.len
    }

    pub fn is_empty(&self) -> bool {
        self.mix.len == 0
    }

    /// Track positions final in every pass (exclusive end): `take` may go this far.
    pub fn ready(&self) -> usize {
        if self.silent {
            return self.mix.len;
        }
        self.passes.iter().map(|p| p.end()).min().unwrap_or(self.mix.len).min(self.mix.len)
    }

    /// Positions handed out by `take` so far.
    pub fn taken(&self) -> usize {
        self.done
    }

    /// Runs the next segment forward of the pass furthest behind (or finishes it). False when
    /// every pass is done.
    pub fn step(&mut self, progress: &mut dyn FnMut()) -> bool {
        let Some(k) = (0..self.passes.len()).filter(|&k| !self.passes[k].finished).min_by_key(|&k| self.passes[k].end()) else {
            return false;
        };
        let seg = self.model.cfg.training_length;
        let p = &mut self.passes[k];
        if p.next >= p.view_len {
            flush(p, seg, p.view_len);
            p.finished = true;
            return true;
        }
        let off = p.next;
        // Everything before this chunk is final.
        flush(p, seg, off);
        let chunk_len = seg.min(p.view_len - off);
        // TensorChunk.padded(seg): centre the chunk, real samples where the tensor has them.
        let delta = seg - chunk_len;
        let s0 = p.start + off as isize - (delta / 2) as isize;
        let mix = &self.mix;
        self.input.par_chunks_mut(seg).enumerate().for_each(|(c, row)| {
            for (i, v) in row.iter_mut().enumerate() {
                *v = mix.at(c, s0 + i as isize);
            }
        });
        let y = self.model.forward(&self.input, &mut Tracer::none());
        progress();
        let t0 = delta / 2;
        let rel = off - p.wbase;
        let weight = &self.weight;
        p.win.par_chunks_mut(seg).enumerate().for_each(|(sc, o)| {
            let src = &y.data[sc * seg + t0..sc * seg + t0 + chunk_len];
            for ((o, &v), &w) in o[rel..rel + chunk_len].iter_mut().zip(src).zip(weight) {
                *o += w * v;
            }
        });
        for (s, &w) in p.wsum[rel..rel + chunk_len].iter_mut().zip(weight) {
            *s += w;
        }
        p.next += self.stride;
        true
    }

    /// The final output for track positions [taken(), upto) (upto <= ready()), as rows (source,
    /// channel) of width upto - taken() back to back, into `out`.
    pub fn take(&mut self, upto: usize, out: &mut Vec<f32>) {
        assert!(upto <= self.ready() && upto >= self.done);
        let (t0, w) = (self.done, upto - self.done);
        out.clear();
        out.resize(self.rows * w, 0.0);
        if !self.silent && w > 0 {
            let (mean, std, s) = (self.mix.mean, self.mix.std, self.shifts as f32);
            let passes = &self.passes;
            let shifts = self.shifts;
            out.par_chunks_mut(w).enumerate().for_each(|(r, o)| {
                for (i, v) in o.iter_mut().enumerate() {
                    let t = t0 + i;
                    let x = if shifts > 0 {
                        // out += pass output, pass after pass; out /= shifts
                        let mut a = 0.0f32;
                        for p in passes {
                            a += p.ring[r][t - p.ring_base];
                        }
                        a / s
                    } else {
                        passes[0].ring[r][t - passes[0].ring_base]
                    };
                    // sources *= ref.std(); sources += ref.mean()
                    *v = x * std + mean;
                }
            });
            for p in &mut self.passes {
                let n = upto - p.ring_base;
                p.ring.par_iter_mut().for_each(|r| {
                    r.drain(..n);
                });
                p.ring_base = upto;
            }
        }
        self.done = upto;
    }
}

/// Finalises view positions [wbase, upto) of a pass (sum of weighted chunks / sum of weights) into
/// its ring, at track positions (dropping those before the track), and slides its window.
fn flush(p: &mut Pass, seg: usize, upto: usize) {
    let w = upto - p.wbase;
    if w == 0 {
        return;
    }
    // View positions [wbase, upto) are track positions from wbase - skip; keep those >= 0.
    let from = p.skip.saturating_sub(p.wbase).min(w);
    let (win, wsum) = (&p.win, &p.wsum);
    p.ring.par_iter_mut().zip(win.par_chunks(seg)).for_each(|(f, r)| {
        f.extend((from..w).map(|i| r[i] / wsum[i]));
    });
    let keep = seg - w.min(seg);
    p.win.par_chunks_mut(seg).for_each(|r| {
        r.copy_within(w.min(seg).., 0);
        r[keep..].fill(0.0);
    });
    p.wsum.copy_within(w.min(seg).., 0);
    p.wsum[keep..].fill(0.0);
    p.wbase = upto;
}

struct Mix<'a> {
    wav: &'a [f32],
    len: usize,
    mean: f32,
    std: f32,
}

impl Mix<'_> {
    /// The normalised track at sample j of channel c (0 outside the track).
    #[inline]
    fn at(&self, c: usize, j: isize) -> f32 {
        if j >= 0 && (j as usize) < self.len { (self.wav[c * self.len + j as usize] - self.mean) / self.std } else { 0.0 }
    }
}

/// apply_model's split branch over a view of `view_len` positions, view position p being track
/// sample `start + p`. Finalised output (sum of weighted chunks / sum of weights) is handed to
/// `emit(p0, w, rows)`: positions [p0, p0 + w) of every (source, channel) row, rows of width w
/// back to back, in increasing p0.
#[allow(clippy::too_many_arguments)]
fn split(
    model: &HTDemucs,
    mix: &Mix,
    channels: usize,
    start: isize,
    view_len: usize,
    stride: usize,
    rng: &mut PyRandom,
    progress: &mut dyn FnMut(),
    emit: &mut dyn FnMut(usize, usize, &[f32]),
) {
    let seg = model.cfg.training_length;
    let rows = model.cfg.sources.len() * channels;
    // th.cat([arange(1, L//2 + 1), arange(L - L//2, 0, -1)]) / max, in float32.
    let half = seg / 2;
    let wmax = half.max(seg - half) as f32;
    let weight: Vec<f32> = (1..=half).chain((1..=seg - half).rev()).map(|v| v as f32 / wmax).collect();

    // Window [wbase, wbase + seg) of the pass's weighted sums and summed weights.
    let mut win = vec![0.0f32; rows * seg];
    let mut wsum = vec![0.0f32; seg];
    let mut wbase = 0usize;
    let mut fin = vec![];
    // Finalise view positions [wbase, upto) and slide the window to start at `upto`.
    let mut flush = |win: &mut Vec<f32>, wsum: &mut Vec<f32>, wbase: &mut usize, upto: usize, fin: &mut Vec<f32>| {
        let w = upto - *wbase;
        if w == 0 {
            return;
        }
        fin.resize(rows * w, 0.0);
        fin.par_chunks_mut(w).zip(win.par_chunks(seg)).for_each(|(f, r)| {
            for i in 0..w {
                f[i] = r[i] / wsum[i];
            }
        });
        emit(*wbase, w, fin);
        let keep = seg - w.min(seg);
        win.par_chunks_mut(seg).for_each(|r| {
            r.copy_within(w.min(seg).., 0);
            r[keep..].fill(0.0);
        });
        wsum.copy_within(w.min(seg).., 0);
        wsum[keep..].fill(0.0);
        *wbase = upto;
    };

    let mut input = vec![0.0f32; channels * seg];
    for off in (0..view_len).step_by(stride) {
        // Everything before this chunk is final.
        flush(&mut win, &mut wsum, &mut wbase, off, &mut fin);
        let chunk_len = seg.min(view_len - off);
        // TensorChunk.padded(seg): centre the chunk, real samples where the tensor has them.
        let delta = seg - chunk_len;
        let s0 = start + off as isize - (delta / 2) as isize;
        input.par_chunks_mut(seg).enumerate().for_each(|(c, row)| {
            for (i, v) in row.iter_mut().enumerate() {
                *v = mix.at(c, s0 + i as isize);
            }
        });
        // The transformer's positional embedding draws random.randrange(sin_random_shift + 1).
        rng.randrange(1);
        let y = model.forward(&input, &mut Tracer::none());
        progress();
        // center_trim, then out[..., off:off + L] += weight[:chunk_len] * chunk_out
        let t0 = delta / 2;
        let rel = off - wbase; // 0: the window starts at this chunk
        win.par_chunks_mut(seg).enumerate().for_each(|(sc, o)| {
            let src = &y.data[sc * seg + t0..sc * seg + t0 + chunk_len];
            for ((o, &v), &w) in o[rel..rel + chunk_len].iter_mut().zip(src).zip(&weight) {
                *o += w * v;
            }
        });
        for (s, &w) in wsum[rel..rel + chunk_len].iter_mut().zip(&weight) {
            *s += w;
        }
    }
    flush(&mut win, &mut wsum, &mut wbase, view_len, &mut fin);
}
