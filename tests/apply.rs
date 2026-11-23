//! Full-track behaviour that needs the model config (and its weights in ./weights, as made by
//! tools/convert.py; skipped with a note when they are missing).

use std::path::Path;

use htdemucs::apply::{self, Options};
use htdemucs::pyrandom::PyRandom;
use htdemucs::HTDemucs;

fn model() -> Option<HTDemucs> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
    if !dir.join("htdemucs_6s.safetensors").exists() {
        eprintln!("skipped: no weights in {}", dir.display());
        return None;
    }
    Some(HTDemucs::load(&dir, "htdemucs_6s").unwrap())
}

#[test]
fn silent_input_gives_silent_stems() {
    let Some(m) = model() else { return };
    for len in [0usize, 1, 1000, 500_000] {
        let wav = vec![0.0f32; 2 * len];
        let mut rng = PyRandom::new(0);
        let mut calls = 0;
        let s = apply::separate(&m, &wav, 2, &Options { shifts: 2, ..Default::default() }, &mut rng, &mut || calls += 1);
        assert!(s.silent);
        assert_eq!(s.stems.shape, [6, 2, len]);
        assert!(s.stems.data.iter().all(|v| *v == 0.0), "len {len}: not all zero");
        assert_eq!(calls, 0);
    }
    // A constant (DC) track has std 0 too.
    let s = apply::separate(&m, &vec![0.25f32; 2 * 1000], 2, &Options::default(), &mut PyRandom::new(0), &mut || {});
    assert!(s.silent && s.stems.data.iter().all(|v| *v == 0.0));
}

#[test]
fn overlap_is_validated() {
    let Some(m) = model() else { return };
    assert_eq!(apply::stride(&m, 0.25), Ok(257_985));
    assert!(apply::stride(&m, 0.0).is_ok());
    assert!(apply::stride(&m, 1.0).is_err());
    assert!(apply::stride(&m, -0.1).is_err());
    assert!(apply::stride(&m, f32::NAN).is_err());
    assert!(apply::stride(&m, 0.999_999_9).is_err()); // rounds to a stride of 0
}

#[test]
fn segment_count_replays_the_generator_without_moving_it() {
    let Some(m) = model() else { return };
    // demucs on a 20 s clip, seed 0 (logged): offsets 12623 then 13268, 4 segments per pass.
    let rng = PyRandom::new(0);
    let opts = Options { shifts: 2, ..Default::default() };
    assert_eq!(apply::forwards(&m, 20 * 44100, &opts, &rng), 8);
    let mut a = rng.clone();
    let mut b = PyRandom::new(0);
    assert_eq!(a.randint(0, 22050), b.randint(0, 22050));
}

/// The streamed output (taken in uneven pieces, steps interleaved) is bit-identical to
/// `separate`'s, and leaves the generator where `separate` does: with 3 shifts, where the order of
/// the passes' sum matters.
#[test]
fn streamer_matches_separate() {
    let Some(m) = model() else { return };
    let len = 20 * 44100 + 123;
    let wav: Vec<f32> = (0..2 * len).map(|i| ((i % len) as f32 * 0.013 + (i / len) as f32).sin() * 0.3 + ((i * 7919) % 1000) as f32 * 1e-4).collect();
    for shifts in [0, 3] {
        let opts = Options { shifts, ..Default::default() };
        let mut ra = PyRandom::new(7);
        let whole = apply::separate(&m, &wav, 2, &opts, &mut ra, &mut || {});
        let mut rb = PyRandom::new(7);
        let mut s = apply::Streamer::new(&m, &wav, 2, &opts, &mut rb);
        let rows = 12;
        let mut got = vec![vec![]; rows];
        let mut buf = vec![];
        let mut piece = 1;
        loop {
            let ready = s.ready();
            if ready > s.taken() {
                let upto = ready.min(s.taken() + piece * 10_007);
                let w = upto - s.taken();
                s.take(upto, &mut buf);
                for (r, g) in got.iter_mut().enumerate() {
                    g.extend_from_slice(&buf[r * w..(r + 1) * w]);
                }
                piece = piece % 7 + 1;
                continue;
            }
            if s.taken() == len || !s.step(&mut || {}) {
                break;
            }
        }
        assert_eq!(s.taken(), len, "shifts {shifts}");
        for (r, g) in got.iter().enumerate() {
            let w = &whole.stems.data[r * len..(r + 1) * len];
            assert!(g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()), "shifts {shifts}, row {r} differs");
        }
        assert_eq!(ra.randint(0, 1 << 30), rb.randint(0, 1 << 30), "shifts {shifts}: generator state");
    }
}
