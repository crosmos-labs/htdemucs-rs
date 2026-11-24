//! RoFormer behaviour that needs the converted weights (./weights/melroformer.*, made by
//! tools/melroformer.py; ./weights/megaguitar.* and ./weights/mega.*, made by tools/bsroformer.py; each test is
//! skipped with a note when they are missing). Release builds only: one chunk is ~3e12 flops
//! (Kim) / ~1e13 (Mega).

use std::path::Path;

use htdemucs::roformer::{self, Chunking, Roformer};

fn load(name: &str) -> Option<Roformer> {
    let dir = std::env::var_os("HTDEMUCS_WEIGHTS").map(std::path::PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("weights"));
    if !dir.join(format!("{name}.safetensors")).exists() {
        eprintln!("skipped: no {name} weights in {}", dir.display());
        return None;
    }
    Some(Roformer::load(&dir, name).unwrap())
}

fn model() -> Option<Roformer> {
    load("melroformer")
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

#[test]
#[cfg_attr(debug_assertions, ignore = "release only (slow)")]
fn megaguitar_bit_identical_across_thread_counts() {
    let Some(m) = load("megaguitar") else { return };
    // 3 s: one chunk (zero-padded, as MSST pads a chunk no longer than half the chunk size).
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
    assert!(a.iter().all(|v| v.is_finite()) && a.iter().any(|v| *v != 0.0));
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()), "1 vs 3 threads differ");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release only (slow)")]
fn megaguitar_silent_input_gives_silence() {
    let Some(m) = load("megaguitar") else { return };
    let ch = Chunking::new(&m.cfg, 2).unwrap();
    for len in [0usize, 1, 5000] {
        let mut calls = 0;
        let v = roformer::separate(&m, &vec![0.0f32; 2 * len], &ch, &mut || calls += 1);
        assert_eq!(v.len(), 2 * len);
        assert!(v.iter().all(|x| *x == 0.0), "len {len}");
        assert_eq!(calls, ch.chunks(len));
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release only (slow)")]
fn mega_heads_equal_one_head_models() {
    // ./weights/mega.* (tools/bsroformer.py convert --name mega: drums, bass, guitar, keys) runs the
    // backbone once and every head off it; each head must give exactly what a one-head model of it
    // gives (megaguitar: the guitar head), at any thread count.
    let (Some(mega), Some(mg)) = (load("mega"), load("megaguitar")) else { return };
    let g = mega.cfg.heads().iter().position(|h| h == "guitar").expect("mega has a guitar head");
    let len = 3 * 44100; // one 3 s forward (the model takes any length above n_fft / 2)
    let x = clip(len);
    let tr = || htdemucs::Tracer::none();
    let one = mg.forward(&x, &mut tr());
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| mega.forward(&x, &mut tr()))
    };
    let all = run(3);
    assert_eq!(all.len(), mega.num_heads() * 2 * len);
    let guitar = &all[g * 2 * len..(g + 1) * 2 * len];
    assert!(guitar.iter().zip(&one).all(|(p, q)| p.to_bits() == q.to_bits()), "mega's guitar head differs from megaguitar");
    assert!(all.iter().all(|v| v.is_finite()));
    let all1 = run(1);
    assert!(all.iter().zip(&all1).all(|(p, q)| p.to_bits() == q.to_bits()), "1 vs 3 threads differ");
    // The heads differ from each other (each mask head really is its own).
    for k in 1..mega.num_heads() {
        assert!(all[..2 * len].iter().zip(&all[k * 2 * len..(k + 1) * 2 * len]).any(|(p, q)| p != q), "head {k} equals head 0");
    }
}
