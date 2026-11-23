//! Mel-RoFormer behaviour that needs the converted weights (./weights/melroformer.*, made by
//! tools/melroformer.py; skipped with a note when they are missing). Release builds only: one
//! chunk is ~3e12 flops.

use std::path::Path;

use htdemucs::roformer::{self, Chunking, MelRoformer};

fn model() -> Option<MelRoformer> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
    if !dir.join("melroformer.safetensors").exists() {
        eprintln!("skipped: no melroformer weights in {}", dir.display());
        return None;
    }
    Some(MelRoformer::load(&dir, "melroformer").unwrap())
}

/// A short stereo test signal (a chord with vibrato and some noise).
fn clip(len: usize) -> Vec<f32> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = vec![0.0f32; 2 * len];
    for c in 0..2 {
        for i in 0..len {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let t = i as f32 / 44100.0;
            let v = (2.0 * std::f32::consts::PI * (220.0 * t + 0.02 * (5.0 * t).sin())).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 277.2 * t).sin() * 0.2
                + ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.02 * (1 + c) as f32;
            out[c * len + i] = v;
        }
    }
    out
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release only (slow)")]
fn bit_identical_across_thread_counts() {
    let Some(m) = model() else { return };
    let len = 3 * 44100;
    let x = clip(len);
    let ch = Chunking::new(&m.cfg, 2).unwrap();
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| roformer::separate(&m, &x, &ch, &mut || {}))
    };
    let a = run(1);
    let b = run(3);
    assert_eq!(a.len(), 2 * len);
    assert!(a.iter().all(|v| v.is_finite()));
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()), "1 vs 3 threads differ");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release only (slow)")]
fn silent_input_gives_silent_vocals() {
    let Some(m) = model() else { return };
    let ch = Chunking::new(&m.cfg, 2).unwrap();
    for len in [0usize, 1, 5000] {
        let mut calls = 0;
        let v = roformer::separate(&m, &vec![0.0f32; 2 * len], &ch, &mut || calls += 1);
        assert_eq!(v.len(), 2 * len);
        assert!(v.iter().all(|x| *x == 0.0), "len {len}");
        assert_eq!(calls, ch.chunks(len));
    }
}
