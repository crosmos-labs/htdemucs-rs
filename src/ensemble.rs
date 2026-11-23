//! The htdemucs + htdemucs_6s ensemble: both models on one decode, averaged.
//!
//! Round 2 of the research (MUSDB18-HQ, 12 tracks) scored the per-stem mean of the two models at
//! shifts 0 above htdemucs alone at shifts 2 on every stem, at about half the CPU of running both
//! models at shifts 2. What it averaged, and so what this reproduces:
//!
//! * vocals, drums, bass: mean of the two models' stems;
//! * the 4-stem "other": mean of htdemucs' other and htdemucs_6s' guitar + piano + other.
//!
//! The 6-stem set keeps htdemucs_6s' own guitar and piano (the mixer plays guitar on its own), and
//! its "other" is whatever makes guitar + piano + other equal that averaged 4-stem other:
//!
//!   other6 = mean(other4_htdemucs, guitar + piano + other_6s) - guitar - piano
//!          = mean(other4_htdemucs - guitar - piano, other_6s)
//!
//! i.e. htdemucs' other with 6s' guitar and piano taken out, averaged with 6s' own other. This is
//! additive, not a rescaling of guitar / piano / other: a per-sample scale factor blows up wherever
//! their sum is near zero, and would change the guitar stem. With it, the 4-stem and 6-stem views
//! agree exactly on vocals, drums and bass and on everything else in total, and the six stems sum
//! to the mean of what each model's stems sum to - as close to the mix as either model alone, so
//! Fretboard's residual-of-the-sum gate still measures how cleanly the track was separated
//! (forcing an exact sum would make that gate vacuous).

use rayon::prelude::*;

use crate::tensor::Tensor;

/// Combines one track's htdemucs (`four`, sources `four_names`) and htdemucs_6s (`six`, sources
/// `six_names`) stems, each (sources, channels, len), in place (no extra copy of either). Returns
/// (4-stem view in htdemucs' source order, 6-stem set in htdemucs_6s' source order).
pub fn combine(mut four: Tensor, four_names: &[String], mut six: Tensor, six_names: &[String]) -> Result<(Tensor, Tensor), String> {
    let plane = four.numel() / four.shape[0].max(1);
    if six.numel() / six.shape[0].max(1) != plane {
        return Err("ensemble: the two models' outputs differ in length".into());
    }
    let at = |names: &[String], s: &str| names.iter().position(|n| n == s).ok_or(format!("ensemble: no '{s}' stem"));
    let mut planes4: Vec<&mut [f32]> = four.data.chunks_mut(plane.max(1)).collect();
    let mut planes6: Vec<&mut [f32]> = six.data.chunks_mut(plane.max(1)).collect();
    if plane == 0 {
        return Ok((four, six));
    }
    for s in ["vocals", "drums", "bass"] {
        let (i4, i6) = (at(four_names, s)?, at(six_names, s)?);
        let (a, b) = (&mut *planes4[i4], &mut *planes6[i6]);
        a.par_iter_mut().zip(b.par_iter_mut()).for_each(|(x, y)| {
            let m = (*x + *y) * 0.5;
            *x = m;
            *y = m;
        });
    }
    let (o4, o6, g, p) = (at(four_names, "other")?, at(six_names, "other")?, at(six_names, "guitar")?, at(six_names, "piano")?);
    if [o6, g, p].iter().collect::<std::collections::HashSet<_>>().len() != 3 {
        return Err("ensemble: htdemucs_6s stems are not distinct".into());
    }
    // 4-stem other = mean(htdemucs other, 6s guitar + piano + other); 6-stem other = that - guitar - piano.
    let gg: Vec<f32> = planes6[g].to_vec();
    let pp: &[f32] = &planes6[p].to_vec();
    let a = &mut *planes4[o4];
    let b = &mut *planes6[o6];
    a.par_iter_mut().zip(b.par_iter_mut()).enumerate().for_each(|(k, (x, y))| {
        let other4 = (*x + (gg[k] + pp[k] + *y)) * 0.5;
        *x = other4;
        *y = other4 - gg[k] - pp[k];
    });
    drop((planes4, planes6));
    Ok((four, six))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_agree_and_sum_like_the_models() {
        let n4: Vec<String> = ["drums", "bass", "other", "vocals"].map(String::from).to_vec();
        let n6: Vec<String> = ["drums", "bass", "other", "vocals", "guitar", "piano"].map(String::from).to_vec();
        let (ch, len) = (2, 5);
        let v = |seed: f32, i: usize| ((i as f32 + 1.0) * seed).sin();
        let four = Tensor::new(&[4, ch, len], (0..4 * ch * len).map(|i| v(0.37, i)).collect());
        let six = Tensor::new(&[6, ch, len], (0..6 * ch * len).map(|i| v(0.91, i)).collect());
        let (a, b) = combine(four.clone(), &n4, six.clone(), &n6).unwrap();
        let pl = ch * len;
        let st = |t: &Tensor, i: usize| t.data[i * pl..(i + 1) * pl].to_vec();
        for (i4, i6) in [(0, 0), (1, 1), (3, 3)] {
            assert_eq!(st(&a, i4), st(&b, i6));
            for k in 0..pl {
                assert_eq!(a.data[i4 * pl + k], (four.data[i4 * pl + k] + six.data[i6 * pl + k]) * 0.5);
            }
        }
        assert_eq!(st(&b, 4), st(&six, 4));
        assert_eq!(st(&b, 5), st(&six, 5));
        for k in 0..pl {
            // guitar + piano + other6 == other4, and the six stems sum to the mean of the models' sums
            let gpo = b.data[4 * pl + k] + b.data[5 * pl + k] + b.data[2 * pl + k];
            assert!((gpo - a.data[2 * pl + k]).abs() < 1e-6);
            let sum6: f32 = (0..6).map(|s| b.data[s * pl + k]).sum();
            let models: f32 = ((0..4).map(|s| four.data[s * pl + k]).sum::<f32>() + (0..6).map(|s| six.data[s * pl + k]).sum::<f32>()) * 0.5;
            assert!((sum6 - models).abs() < 1e-5);
        }
    }
}
