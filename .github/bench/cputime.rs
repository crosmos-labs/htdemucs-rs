//! Segment-forward CPU time using only the crate's stable API (HTDemucs::load / forward), so the
//! same file can be dropped into any commit as src/bin/cputime.rs to time it:
//!
//!     cputime <weights dir> <model> [runs]     (threads: RAYON_NUM_THREADS)
//!
//! Prints process CPU seconds (user + sys, all threads) and wall seconds per forward, after one
//! warm-up forward. getrusage is declared by hand (no libc dependency in older commits); the
//! timeval layout below matches 64-bit Linux and macOS on little-endian machines.

use std::time::Instant;

use htdemucs::{HTDemucs, Tracer};

#[repr(C)]
#[derive(Clone, Copy)]
struct TimeVal {
    sec: i64,
    usec: i32,
    _pad: i32,
}

#[repr(C)]
struct RUsage {
    utime: TimeVal,
    stime: TimeVal,
    _rest: [i64; 14],
}

unsafe extern "C" {
    fn getrusage(who: i32, usage: *mut RUsage) -> i32;
}

fn cpu() -> f64 {
    let mut u = RUsage { utime: TimeVal { sec: 0, usec: 0, _pad: 0 }, stime: TimeVal { sec: 0, usec: 0, _pad: 0 }, _rest: [0; 14] };
    unsafe { getrusage(0, &mut u) };
    let t = |v: TimeVal| v.sec as f64 + v.usec as f64 * 1e-6;
    t(u.utime) + t(u.stime)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = HTDemucs::load(&args[0], &args[1]).unwrap();
    let runs: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
    let len = model.cfg.training_length;
    let mut s = 0x2545F4914F6CDD1Du64;
    let mix: Vec<f32> = (0..2 * len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect();
    let _ = model.forward(&mix, &mut Tracer::none());
    let (c0, w0) = (cpu(), Instant::now());
    for _ in 0..runs {
        let _ = model.forward(&mix, &mut Tracer::none());
    }
    let c = (cpu() - c0) / runs as f64;
    let w = w0.elapsed().as_secs_f64() / runs as f64;
    let seg = len as f64 / model.cfg.samplerate as f64;
    println!("CPUTIME model={} threads={} cpu={c:.4} wall={w:.4} seg_secs={seg:.4}", args[1], rayon::current_num_threads());
}
