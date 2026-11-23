//! Time one Mel-RoFormer chunk forward: `mr_bench <weights dir> [runs] [weights name]`, on
//! `RAYON_NUM_THREADS` threads (default: all cores), on a synthetic chunk. Prints the best wall
//! time and the process CPU time (user + sys) per forward, and a `RESULT` line.
//!
//! * `MR_PROFILE=1`: thread time per op class and model part (summed over threads), GEMM GFLOP/s.
//! * `MR_CHECK_THREADS=1,2,4`: assert the output is bit-identical on each of those thread counts.

use std::time::Instant;

use htdemucs::roformer::{prof, MelRoformer};
use htdemucs::Tracer;

fn cpu() -> f64 {
    unsafe {
        let mut u: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut u);
        let t = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 * 1e-6;
        t(u.ru_utime) + t(u.ru_stime)
    }
}

/// A deterministic stereo test signal: a few decaying partials plus noise.
pub fn synth(len: usize) -> Vec<f32> {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut out = vec![0.0f32; 2 * len];
    for c in 0..2 {
        for i in 0..len {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let noise = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let t = i as f32 / 44100.0;
            let env = (-(t % 0.5) * 6.0).exp();
            let tone = (2.0 * std::f32::consts::PI * (220.0 + 110.0 * c as f32) * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 660.0 * t).sin() * 0.1 * env;
            out[c * len + i] = tone + 0.05 * noise;
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let runs: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(3);
    let name = args.get(2).map(String::as_str).unwrap_or("melroformer");
    let model = MelRoformer::load(&args[0], name).unwrap();
    let x = synth(model.cfg.chunk_size);
    let threads = rayon::current_num_threads();
    let _ = model.forward(&x, &mut Tracer::none()); // warm-up
    let profile = std::env::var_os("MR_PROFILE").is_some();
    prof::take();
    prof::enable(profile);
    let (mut best, c0) = (f64::INFINITY, cpu());
    let mut y = vec![];
    for _ in 0..runs {
        let t = Instant::now();
        y = model.forward(&x, &mut Tracer::none());
        best = best.min(t.elapsed().as_secs_f64());
    }
    let per = (cpu() - c0) / runs as f64;
    prof::enable(false);
    let audio = model.cfg.chunk_size as f64 / model.cfg.sample_rate as f64;
    println!("threads {threads}: best wall {best:.2}s, CPU {per:.2}s per forward ({:.3} CPU-s per audio-s of chunk)", per / audio);
    if profile {
        let p = prof::take();
        let total: f64 = p.iter().flatten().map(|v| v.0).sum();
        println!("\nthread time per op class (sum over threads, per forward), GEMM GFLOP/s per thread:");
        for (pi, part) in p.iter().enumerate() {
            let ps: f64 = part.iter().map(|v| v.0).sum();
            println!("  {} {:.2}s ({:.0}%)", prof::PARTS[pi], ps / runs as f64, 100.0 * ps / total);
            for (ci, &(s, fl)) in part.iter().enumerate() {
                if s > 0.0 {
                    let gf = if fl > 0.0 { format!(", {:.1} GFLOP/s", fl / s * 1e-9) } else { String::new() };
                    println!("    {:<36} {:7.3}s {:5.1}%{gf}", prof::CLASSES[ci], s / runs as f64, 100.0 * s / total);
                }
            }
        }
    }
    if let Ok(list) = std::env::var("MR_CHECK_THREADS") {
        for t in list.split(',').filter_map(|s| s.parse::<usize>().ok()) {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
            let z = pool.install(|| model.forward(&x, &mut Tracer::none()));
            assert!(z.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()), "output differs on {t} threads");
            println!("bit-identical on {t} threads");
        }
    }
    println!("RESULT model={name} threads={threads} wall_best={best:.3} cpu={per:.3} cpu_per_audio_s={:.4}", per / audio);
}
