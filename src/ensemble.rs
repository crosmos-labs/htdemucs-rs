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
/// `six_names`) stems, each (sources, channels, len). Returns (4-stem view in htdemucs' source
/// order, 6-stem set in htdemucs_6s' source order).
pub fn combine(four: &Tensor, four_names: &[String], six: &Tensor, six_names: &[String]) -> Result<(Tensor, Tensor), String> {
    let plane = four.numel() / four.shape[0];
    if six.numel() / six.shape[0] != plane {
        return Err("ensemble: the two models' outputs differ in length".into());
    }
    let at = |names: &[String], s: &str| names.iter().position(|n| n == s).ok_or(format!("ensemble: no '{s}' stem"));
    fn f(t: &Tensor, i: usize) -> &[f32] {
        let plane = t.numel() / t.shape[0];
        &t.data[i * plane..(i + 1) * plane]
    }
    let mut out4 = Tensor::zeros(&four.shape);
    let mut out6 = Tensor::zeros(&six.shape);
    for s in ["vocals", "drums", "bass"] {
        let (i4, i6) = (at(four_names, s)?, at(six_names, s)?);
        let (a, b) = (f(four, i4), f(six, i6));
        let mut m = vec![0.0f32; plane];
        m.par_iter_mut().enumerate().for_each(|(k, v)| *v = (a[k] + b[k]) * 0.5);
        out4.data[i4 * plane..(i4 + 1) * plane].copy_from_slice(&m);
        out6.data[i6 * plane..(i6 + 1) * plane].copy_from_slice(&m);
    }
    let (o4, o6, g, p) = (at(four_names, "other")?, at(six_names, "other")?, at(six_names, "guitar")?, at(six_names, "piano")?);
    let (a, b, gg, pp) = (f(four, o4), f(six, o6), f(six, g), f(six, p));
    // 4-stem other = mean(htdemucs other, 6s guitar + piano + other); 6-stem other = that - guitar - piano.
    let other4: Vec<f32> = (0..plane).into_par_iter().map(|k| (a[k] + (gg[k] + pp[k] + b[k])) * 0.5).collect();
    let other6: Vec<f32> = (0..plane).into_par_iter().map(|k| other4[k] - gg[k] - pp[k]).collect();
    out4.data[o4 * plane..(o4 + 1) * plane].copy_from_slice(&other4);
    out6.data[o6 * plane..(o6 + 1) * plane].copy_from_slice(&other6);
    out6.data[g * plane..(g + 1) * plane].copy_from_slice(gg);
    out6.data[p * plane..(p + 1) * plane].copy_from_slice(pp);
    Ok((out4, out6))
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
        let (a, b) = combine(&four, &n4, &six, &n6).unwrap();
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
