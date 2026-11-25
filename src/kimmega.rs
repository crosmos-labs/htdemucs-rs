//! `-n kimmega` (alias `kim_mega_keys`): research round 5's recommended recipe
//! (research/round5/round5.md, `kim_mega_keys`), all MIT weights:
//!
//! * vocals: Kim's Mel-Band RoFormer (upstream demix_track, src/roformer.rs, `-n melroformer`);
//! * instrumental = mix - vocals, in float32 (as r5_mega.py's `--minus`);
//! * MVSEP's Mega 53 BS-RoFormer on the instrumental, its backbone run once per chunk and four mask
//!   heads off it: `drums`, `bass`, `guitar`, `keys` (weights/mega.json, tools/bsroformer.py convert
//!   --name mega), with MSST's generic demix (20 s chunks, overlap 2; `--mega-chunk-size`,
//!   `--mega-num-overlap`);
//! * other = instrumental - (drums + bass + guitar + keys), summed in float64: the six stems sum to
//!   the mix up to float32 rounding.
//!
//! Stems: vocals, drums, bass, guitar, piano (the `keys` head: round 5 found it catches organ and
//! e-piano the `piano` head misses, +1.7 dB on the keys group), other - the 6-stem layout.
//!
//! Memory, as the chain (src/chain.rs): the instrumental is the one track-long buffer kept besides the
//! decoded mix, which is freed once Kim is done; Kim's weights are freed before Mega's are loaded;
//! vocals go to their writer as Kim finalises them and Mega's stems as its chunks finalise them.

use std::path::Path;

use crate::Tracer;
use crate::chain::Residual;
use crate::roformer::{self, Chunking, Config, Roformer};

/// Kim's and Mega's weights names (`<name>.json` + `<name>.safetensors` in the weights directory;
/// `$MELROFORMER_WEIGHTS` / `$MEGA_WEIGHTS` override the safetensors name).
pub const KIM: &str = "melroformer";
pub const MEGA: &str = "mega";

/// Chunk settings: `None` keeps each model's default (Kim: 8 s, overlap 2; Mega: 20 s, overlap 2).
#[derive(Clone, Copy, Debug, Default)]
pub struct Settings {
    pub kim_chunk: Option<usize>,
    pub kim_overlap: Option<usize>,
    pub mega_chunk: Option<usize>,
    pub mega_overlap: Option<usize>,
}

/// What a run needs to know before loading anything: both chunkings and the output stems.
pub struct Plan {
    pub kim: Chunking,
    pub mega: Chunking,
    /// vocals, then each Mega head's stem (`keys` named `piano`), then other.
    pub sources: Vec<String>,
}

impl Plan {
    pub fn new(weights: &Path, s: &Settings) -> Result<Self, String> {
        let kc = Config::load(weights.join(format!("{KIM}.json")))?;
        let mc = Config::load(weights.join(format!("{MEGA}.json")))?;
        let kim = Chunking::with_chunk(&kc, s.kim_chunk.unwrap_or(kc.chunk_size), s.kim_overlap.unwrap_or(kc.num_overlap)).map_err(|e| format!("--chunk-size / --num-overlap (Kim): {e}"))?;
        let mega = Chunking::with_chunk(&mc, s.mega_chunk.unwrap_or(mc.chunk_size), s.mega_overlap.unwrap_or(mc.num_overlap)).map_err(|e| format!("--mega-chunk-size / --mega-num-overlap: {e}"))?;
        let mut sources = vec!["vocals".to_string()];
        sources.extend(mc.heads().iter().map(|h| if h == "keys" { "piano".to_string() } else { h.clone() }));
        sources.push("other".into());
        for (i, s) in sources.iter().enumerate() {
            if sources[..i].contains(s) {
                return Err(format!("kimmega: two stems named {s} (heads {:?} in {MEGA}.json)", mc.heads()));
            }
        }
        Ok(Plan { kim, mega, sources })
    }

    /// Model forwards for a `len`-sample track (Kim chunks, then Mega chunks), for progress bars.
    pub fn forwards(&self, len: usize) -> usize {
        self.kim.chunks(len) + self.mega.chunks(len)
    }
}

/// Runs the recipe on one stereo track (`mix`: 2 x len, consumed so it can be freed early).
///
/// * `vocals(left, right)` receives Kim's vocals in track order;
/// * `stems(rows)` receives, in track order, rows[2 * k + channel] for k over the Mega heads'
///   stems then other (`Plan::sources` after vocals), each the same span of samples;
/// * `progress` is called after every model forward.
///
/// Returns what the stems leave of the mix (an upper bound: phase 1's float32 instrumental and
/// phase 2's float32 other each round once).
pub fn separate(
    weights: &Path,
    mix: Vec<f32>,
    plan: &Plan,
    progress: &mut dyn FnMut(),
    vocals: &mut dyn FnMut(&[f32], &[f32]) -> Result<(), String>,
    stems: &mut dyn FnMut(&[&[f32]]) -> Result<(), String>,
) -> Result<Residual, String> {
    let len = mix.len() / 2;
    let (mut mix_e, mut r1, mut r2) = (0f64, 0f64, 0f64);
    // Phase 1: Kim. Vocals out as they are finalised; instrumental = mix - vocals kept.
    let mut inst = vec![0.0f32; 2 * len];
    {
        let kim = Roformer::load(weights, KIM)?;
        let mut err = None;
        let f = |part: &[f32]| kim.forward(part, &mut Tracer::none());
        roformer::separate_stream(&mix, &plan.kim, &f, progress, &mut |t0, l, r| {
            for (c, v) in [l, r].into_iter().enumerate() {
                for (i, &x) in v.iter().enumerate() {
                    let m = mix[c * len + t0 + i];
                    let d = m - x;
                    inst[c * len + t0 + i] = d;
                    mix_e += m as f64 * m as f64;
                    let e = m as f64 - x as f64 - d as f64;
                    r1 += e * e;
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
    // Phase 2: Mega's heads on the instrumental; other = instrumental - their sum.
    let mega = Roformer::load(weights, MEGA)?;
    let n = mega.num_heads();
    if n + 2 != plan.sources.len() {
        return Err(format!("{MEGA}: {n} heads, the plan has {}", plan.sources.len() - 2));
    }
    let mut err = None;
    let mut other = vec![];
    let f = |part: &[f32]| mega.forward(part, &mut Tracer::none());
    roformer::separate_stream_n(&inst, &plan.mega, n, &f, progress, &mut |t0, rows| {
        if err.is_some() {
            return;
        }
        let w = rows[0].len();
        other.clear();
        other.resize(2 * w, 0.0f32);
        for c in 0..2 {
            for i in 0..w {
                let x = inst[c * len + t0 + i] as f64;
                let s: f64 = (0..n).map(|k| rows[2 * k + c][i] as f64).sum();
                let o = (x - s) as f32;
                other[c * w + i] = o;
                let e = x - s - o as f64;
                r2 += e * e;
            }
        }
        let mut all: Vec<&[f32]> = rows.to_vec();
        all.push(&other[..w]);
        all.push(&other[w..]);
        err = stems(&all).err();
    });
    if let Some(e) = err {
        return Err(e);
    }
    Ok(Residual { resid: (r1.sqrt() + r2.sqrt()).powi(2), mix: mix_e })
}
