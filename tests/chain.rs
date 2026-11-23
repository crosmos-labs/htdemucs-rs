//! `-n chain` (src/chain.rs) with the converted weights (melroformer, htdemucs, htdemucs_6s in
//! ./weights; skipped when missing). Release builds only.

use std::path::Path;

use htdemucs::chain;
use htdemucs::pyrandom::PyRandom;
use htdemucs::roformer::{self, Chunking, MelRoformer};

fn weights() -> Option<std::path::PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
    let ok = ["melroformer", "htdemucs", "htdemucs_6s"].iter().all(|m| dir.join(format!("{m}.safetensors")).exists());
    if !ok {
        eprintln!("skipped: chain weights missing in {}", dir.display());
    }
    ok.then_some(dir)
}

fn clip(len: usize) -> Vec<f32> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    (0..2 * len)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let t = (i % len) as f32 / 44100.0;
            (2.0 * std::f32::consts::PI * 196.0 * t).sin() * 0.3 + (2.0 * std::f32::consts::PI * 55.0 * t).sin() * 0.3 + ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05
        })
        .collect()
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release only (slow)")]
fn chain_stems_are_kim_vocals_plus_the_ensemble_on_the_instrumental() {
    let Some(dir) = weights() else { return };
    let len = 3 * 44100;
    let mix = clip(len);
    let mut voc = vec![vec![]; 2];
    let (mut four, mut six) = (vec![], vec![]);
    let mut calls = 0;
    let mut rngs = [PyRandom::new(0), PyRandom::new(0)];
    let res = chain::separate(&dir, mix.clone(), None, &mut rngs, true, &mut || calls += 1, &mut |l, r| {
        voc[0].extend_from_slice(l);
        voc[1].extend_from_slice(r);
        Ok(())
    }, &mut |a, b| {
        four.push(a.clone());
        six.push(b.clone());
        Ok(())
    })
    .unwrap();
    // Vocals are exactly `-n melroformer`'s.
    let kim = MelRoformer::load(&dir, "melroformer").unwrap();
    let ch = Chunking::new(&kim.cfg, 2).unwrap();
    let v = roformer::separate(&kim, &mix, &ch, &mut || {});
    assert_eq!([voc[0].clone(), voc[1].clone()].concat(), v);
    // Windows cover the track; the views agree on drums / bass and on guitar + piano + other.
    let w: usize = four.iter().map(|t| t.shape[2]).sum();
    assert_eq!(w, len);
    let s4 = htdemucs::config::Config::load(dir.join("htdemucs.json")).unwrap().sources;
    let s6 = htdemucs::config::Config::load(dir.join("htdemucs_6s.json")).unwrap().sources;
    let at = |s: &[String], n: &str| s.iter().position(|x| x == n).unwrap();
    for (a, b) in four.iter().zip(&six) {
        let w = a.shape[2];
        let row = |t: &htdemucs::Tensor, i: usize| t.data[i * 2 * w..(i + 1) * 2 * w].to_vec();
        for n in ["drums", "bass"] {
            assert_eq!(row(a, at(&s4, n)), row(b, at(&s6, n)));
        }
        let (o4, o6, g, p) = (row(a, at(&s4, "other")), row(b, at(&s6, "other")), row(b, at(&s6, "guitar")), row(b, at(&s6, "piano")));
        for k in 0..o4.len() {
            assert!((o6[k] + g[k] + p[k] - o4[k]).abs() < 1e-5);
        }
        assert!(a.data.iter().chain(&b.data).all(|x| x.is_finite()));
    }
    assert_eq!(calls, chain::forwards(&ch, chain::demucs_stride(&dir).unwrap(), len));
    assert!(res.db() < -10.0, "stems far from summing to the mix: {:.1} dB", res.db());
}
