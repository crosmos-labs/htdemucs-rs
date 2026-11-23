//! Time one segment forward: `bench <weights dir> <model> [runs]`, on `RAYON_NUM_THREADS` threads
//! (default: all cores). Prints the best wall time and the process CPU time (user + sys) per
//! forward, plus one machine-readable `RESULT` line.
//!
//! * `HTDEMUCS_PROFILE=1`: also the time per op class and branch (convs, linear layers, attention,
//!   norms, STFT, ...), memory allocated and page faults per forward, and the coarse per-layer
//!   profile between trace points.
//! * `HTDEMUCS_CHECK_THREADS=1,2,4,8`: assert the output is bit-identical on each of those thread
//!   counts (default: 3 only).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use htdemucs::ops::prof;
use htdemucs::{HTDemucs, Tensor, Tracer};

/// Counts bytes and calls of every allocation.
struct Counting;
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(l.size() as u64, Relaxed);
        ALLOC_CALLS.fetch_add(1, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(l.size() as u64, Relaxed);
        ALLOC_CALLS.fetch_add(1, Relaxed);
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOC_BYTES.fetch_add(n.saturating_sub(l.size()) as u64, Relaxed);
        ALLOC_CALLS.fetch_add(1, Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// (process CPU seconds, user + sys; minor page faults).
fn usage() -> (f64, i64) {
    unsafe {
        let mut u: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut u);
        let t = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 * 1e-6;
        (t(u.ru_utime) + t(u.ru_stime), u.ru_minflt as i64)
    }
}

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
    // Results must not depend on the thread count.
    let check = std::env::var("HTDEMUCS_CHECK_THREADS").unwrap_or_else(|_| "3".into());
    for t in check.split(',').filter(|t| !t.is_empty()) {
        let t: usize = t.parse().unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
        let other = pool.install(|| model.forward(&mix, &mut Tracer::none()));
        assert!(first.data == other.data, "output differs between thread counts ({t} threads)");
        eprintln!("identical output on {t} threads");
    }
    let threads = rayon::current_num_threads();
    let mut best = f64::MAX;
    let (c0, _) = usage();
    let w0 = Instant::now();
    for _ in 0..runs {
        let t = Instant::now();
        let _ = model.forward(&mix, &mut Tracer::none());
        best = best.min(t.elapsed().as_secs_f64());
    }
    let mean = w0.elapsed().as_secs_f64() / runs as f64;
    let cpu = (usage().0 - c0) / runs as f64;
    let seg_secs = len as f64 / model.cfg.samplerate as f64;
    println!(
        "{}: {threads} threads: wall best {best:.3}s mean {mean:.3}s, CPU {cpu:.3}s per segment ({:.3} CPU-s per audio-s of segment)",
        args[1],
        cpu / seg_secs
    );
    println!("RESULT model={} threads={threads} wall_best={best:.4} wall_mean={mean:.4} cpu={cpu:.4} seg_secs={seg_secs:.4}", args[1]);

    if std::env::var("HTDEMUCS_PROFILE").is_ok() {
        // Op classes.
        prof::take();
        prof::enable(true);
        let (b0, a0, (_, f0)) = (ALLOC_BYTES.load(Relaxed), ALLOC_CALLS.load(Relaxed), usage());
        let t = Instant::now();
        let _ = model.forward(&mix, &mut Tracer::none());
        let total = t.elapsed().as_secs_f64();
        let (b1, a1, (_, f1)) = (ALLOC_BYTES.load(Relaxed), ALLOC_CALLS.load(Relaxed), usage());
        prof::enable(false);
        let tab = prof::take();
        let tracked: f64 = tab.iter().flatten().map(|v| v.0).sum();
        let gflop: f64 = tab.iter().flatten().map(|v| v.1).sum::<f64>() * 1e-9;
        println!("\nop class profile (wall time of one forward, {threads} threads, total {total:.3}s, {gflop:.1} GFLOP in GEMMs):");
        println!("| op class | {} | all | share | GFLOP | GFLOP/s |\n|---|---|---|---|---|---|---|---|", prof::BRANCHES.join(" | "));
        let mut rows: Vec<(usize, f64, f64)> =
            (0..prof::CLASSES.len()).map(|c| (c, tab.iter().map(|b| b[c].0).sum(), tab.iter().map(|b| b[c].1).sum())).collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (c, all, fl) in rows {
            let per: Vec<String> = tab.iter().map(|b| format!("{:.0} ms", b[c].0 * 1e3)).collect();
            let rate = if fl > 0.0 { format!("{:.1} | {:.0}", fl * 1e-9, fl * 1e-9 / all) } else { " | ".into() };
            println!("| {} | {} | {:.0} ms | {:.1}% | {rate} |", prof::CLASSES[c], per.join(" | "), all * 1e3, 100.0 * all / total);
        }
        let rest = total - tracked;
        println!("| untracked (copies, layout, adds) | | | | {:.0} ms | {:.1}% | | |", rest * 1e3, 100.0 * rest / total);
        println!(
            "allocations per forward: {:.2} GB in {} calls; minor page faults {} ({:.2} GB at 4 KiB, {:.2} GB at 16 KiB)",
            (b1 - b0) as f64 / 1e9,
            a1 - a0,
            f1 - f0,
            (f1 - f0) as f64 * 4096.0 / 1e9,
            (f1 - f0) as f64 * 16384.0 / 1e9
        );

        // Coarse per-layer profile between trace points.
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
        println!("\nper-layer (time since the previous trace point, top 30):");
        for (n, s) in rows.iter().take(30) {
            println!("{:8.1} ms  {n}", s * 1e3);
        }
        println!("traced total {total:.3}s");
    }
}
