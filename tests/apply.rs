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
