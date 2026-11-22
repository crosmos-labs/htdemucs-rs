//! Time one segment forward: `bench <weights dir> <model> [runs]`. With HTDEMUCS_PROFILE=1, also
//! prints the time spent between consecutive trace points (coarse per-layer profile).

use std::time::Instant;

use htdemucs::{HTDemucs, Tensor, Tracer};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = HTDemucs::load(&args[0], &args[1]).unwrap();
    let runs: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
    let len = model.cfg.training_length;
    // Deterministic pseudo-noise input (standardised-ish), stereo.
    let mut s = 0x2545F4914F6CDD1Du64;
    let mix: Vec<f32> = (0..2 * len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect();
    let first = model.forward(&mix, &mut Tracer::none());
    // Results must not depend on the thread count: compare with a 3-thread run.
    let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
    let other = pool.install(|| model.forward(&mix, &mut Tracer::none()));
    assert!(first.data == other.data, "output differs between thread counts");
    let mut best = f64::MAX;
    for _ in 0..runs {
        let t = Instant::now();
        let _ = model.forward(&mix, &mut Tracer::none());
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("{}: best of {runs}: {best:.3}s per segment ({} threads)", args[1], rayon::current_num_threads());
    if std::env::var("HTDEMUCS_PROFILE").is_ok() {
        let mut last = Instant::now();
        let mut rows: Vec<(String, f64)> = vec![];
        let mut sink = |n: &str, _: &Tensor| {
            rows.push((n.to_string(), last.elapsed().as_secs_f64()));
            last = Instant::now();
        };
        let t = Instant::now();
        let _ = model.forward(&mix, &mut Tracer(Some(&mut sink)));
        let total = t.elapsed().as_secs_f64();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (n, s) in rows.iter().take(40) {
            println!("{:8.1} ms  {n}", s * 1e3);
        }
        println!("traced total {total:.3}s");
    }
}
