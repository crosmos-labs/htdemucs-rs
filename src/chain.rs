//! `-n chain`: Kim's Mel-Band RoFormer vocals, then htdemucs and htdemucs_6s (shifts 0) on Kim's
//! instrumental (mix - vocals), averaged as `-n ensemble` averages them (src/ensemble.rs). Round 3
//! of the research (`k+chain_avg`, research/round3/round3.md section 2) ranked this first on 12
//! MUSDB18-HQ tracks: vocals +3.44, drums +0.65, bass +0.17, other +1.24 dB against the 6-stem
//! split at 2 shifts.
//!
//! What it computes, as round 3's `r3_sep.py` / `r3_score.py` did:
//! * vocals: Kim's (upstream demix_track, src/roformer.rs);
//! * instrumental = mix - vocals, in float32;
//! * each Demucs model on the instrumental exactly as `demucs.separate` on a track: whole-track
//!   normalisation by the instrumental's mono mean / std, shifts 0, overlap 0.25;
//! * drums, bass: the mean of the two models; 4-stem other: mean of htdemucs' other and 6s'
//!   guitar + piano + other; 6-stem set: 6s' guitar and piano, other = that 4-stem other minus them
//!   (`ensemble::combine_with`). The Demucs models' own (leftover) vocals are dropped.
//!
//! The stems are not forced to sum to the mix (round 3: the chain's residual is about -26 dB re the
//! mix, and folding it into "other" was neutral on typical tracks); `separate` returns the
//! residual's energy so callers can report it.
//!
//! Memory: Demucs' normalisation needs the whole instrumental before its first segment, so one
//! track-long buffer is unavoidable besides the decoded mix: the instrumental. The vocals go to
//! their writers as Kim finalises them (never held), the mix is dropped once the instrumental is
//! complete, and the Demucs models are loaded only after Kim's weights are freed. Peak: the mix +
//! the instrumental + Kim (0.91 GB) during the first phase; the instrumental + both Demucs models
//! (0.28 GB) + one segment window per model during the second.

use std::path::Path;

use crate::apply::{self, Options, Streamer};
use crate::pyrandom::PyRandom;
use crate::roformer::{self, Chunking, MelRoformer};
use crate::tensor::Tensor;
use crate::{HTDemucs, Tracer};

/// Model forwards a chain run of a `len`-sample track takes (Kim chunks + both Demucs passes), for
/// progress bars. Demucs at shifts 0 takes ceil(len / stride) segments.
pub fn forwards(kim_chunks: &Chunking, demucs_stride: usize, len: usize) -> usize {
    kim_chunks.chunks(len) + 2 * len.div_ceil(demucs_stride)
}

/// What the stems leave of the mix: energy of mix - (vocals + drums + bass + other4) and of the mix.
#[derive(Clone, Copy, Debug, Default)]
pub struct Residual {
    pub resid: f64,
    pub mix: f64,
}

impl Residual {
    /// 10 log10(residual / mix energy), as round 3's resid_dB.
    pub fn db(&self) -> f64 {
        10.0 * (self.resid / self.mix.max(1e-30)).max(1e-30).log10()
    }
}

/// Runs the chain on one stereo track (`mix`: 2 x len, consumed so it can be freed early).
///
/// * `vocals(left, right)` receives Kim's vocals in track order (both views' vocals stem);
/// * `window(four, six)` receives each finalised window of the Demucs stems, as
///   `ensemble::combine_with` returns them (sources in htdemucs' / htdemucs_6s' order, the vocals
///   rows being Demucs' own leftovers, which the caller ignores);
/// * `progress` is called after every model forward.
///
/// `rngs` are Python's `random` for htdemucs_6s and htdemucs, as `-n ensemble` carries them across
/// tracks (at shifts 0 they only advance; the output never depends on them).
#[allow(clippy::too_many_arguments)]
pub fn separate(
    weights: &Path,
    mix: Vec<f32>,
    kim_overlap: Option<usize>,
    rngs: &mut [PyRandom; 2],
    average_four: bool,
    progress: &mut dyn FnMut(),
    vocals: &mut dyn FnMut(&[f32], &[f32]) -> Result<(), String>,
    window: &mut dyn FnMut(&Tensor, &Tensor) -> Result<(), String>,
) -> Result<Residual, String> {
    let len = mix.len() / 2;
    let mut res = Residual::default();
    // Phase 1: Kim. Vocals out as they are finalised; instrumental = mix - vocals kept.
    let mut inst = vec![0.0f32; 2 * len];
    {
        let kim = MelRoformer::load(weights, "melroformer")?;
        let ch = Chunking::new(&kim.cfg, kim_overlap.unwrap_or(kim.cfg.num_overlap))?;
        let mut err = None;
        // HTDEMUCS_CHAIN_DRYRUN=1 (memory measurements only): Kim's "vocals" are the mix itself, so
        // the instrumental is silent and Demucs runs no segments - every track-long buffer is
        // allocated and filled as in a real run, in seconds instead of hours.
        let dry = std::env::var_os("HTDEMUCS_CHAIN_DRYRUN").is_some();
        let f = |part: &[f32]| if dry { part.to_vec() } else { kim.forward(part, &mut Tracer::none()) };
        roformer::separate_stream(&mix, &ch, &f, progress, &mut |t0, l, r| {
            for (c, v) in [l, r].into_iter().enumerate() {
                for (i, &x) in v.iter().enumerate() {
                    let m = mix[c * len + t0 + i];
                    inst[c * len + t0 + i] = m - x;
                    res.mix += m as f64 * m as f64;
                }
            }
            if err.is_none() {
                err = vocals(l, r).err();
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
    }
    drop(mix);
    // Phase 2: htdemucs_6s and htdemucs on the instrumental, shifts 0, averaged window by window.
    let six = HTDemucs::load(weights, "htdemucs_6s")?;
    let four = HTDemucs::load(weights, "htdemucs")?;
    let opts = Options { shifts: 0, seed: 0, overlap: 0.25 };
    let stride = apply::stride(&four, opts.overlap)?;
    let [r6, r4] = rngs;
    let mut streams = [Streamer::new(&six, &inst, 2, &opts, r6), Streamer::new(&four, &inst, 2, &opts, r4)];
    let (n6, n4) = (six.sources().len(), four.sources().len());
    let at = |names: &[String], s: &str| names.iter().position(|n| n == s).ok_or(format!("chain: no '{s}' stem"));
    let rows4: Vec<usize> = ["drums", "bass", "other"].iter().map(|s| at(four.sources(), s)).collect::<Result<_, _>>()?;
    let mut bufs = [vec![], vec![]];
    loop {
        let ready = streams[0].ready().min(streams[1].ready());
        let taken = streams[0].taken();
        if ready > taken && (ready - taken >= stride || ready == len) {
            let upto = ready.min(taken + stride);
            let w = upto - taken;
            for (s, b) in streams.iter_mut().zip(bufs.iter_mut()) {
                s.take(upto, b);
            }
            let t6 = Tensor::new(&[n6, 2, w], std::mem::take(&mut bufs[0]));
            let t4 = Tensor::new(&[n4, 2, w], std::mem::take(&mut bufs[1]));
            let (v4, v6) = crate::ensemble::combine_with(t4, four.sources(), t6, six.sources(), average_four)?;
            // Residual: mix - kim vocals - (drums + bass + other4) = instrumental - (d + b + o).
            for c in 0..2 {
                for i in 0..w {
                    let s: f32 = rows4.iter().map(|&r| v4.data[(r * 2 + c) * w + i]).sum();
                    let d = (inst[c * len + taken + i] - s) as f64;
                    res.resid += d * d;
                }
            }
            window(&v4, &v6)?;
            (bufs[0], bufs[1]) = (v6.data, v4.data);
            continue;
        }
        if taken == len {
            break;
        }
        let k = if streams[0].ready() <= streams[1].ready() { 0 } else { 1 };
        if !streams[k].step(progress) {
            return Err("internal: a stream stopped short of the track's end".into());
        }
    }
    Ok(res)
}

/// Same as `apply::stride` for htdemucs at overlap 0.25, without loading it (for progress counts).
pub fn demucs_stride(weights: &Path) -> Result<usize, String> {
    let cfg = crate::config::Config::load(weights.join("htdemucs.json"))?;
    Ok((0.75f32 * cfg.training_length as f32) as usize)
}
