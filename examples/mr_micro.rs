//! Micro-benchmarks of the elementwise kernels (ns per element): `cargo run --release --example mr_micro`.
use std::time::Instant;

use htdemucs::ops::{gelu_slice, softmax_row};
use htdemucs::roformer::micro;

fn time(name: &str, elems: usize, reps: usize, mut f: impl FnMut()) {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    println!("{name:<28} {:7.3} ns/elem", t.elapsed().as_secs_f64() * 1e9 / (reps * elems) as f64);
}

fn main() {
    let n = 801;
    // Attention-like logits: a spread of ~100 after scaling, like the time transformer's.
    let orig: Vec<f32> = (0..128 * n).map(|i| ((i * 7919) % 1000) as f32 * 0.8 - 400.0).collect();
    let mut rows = orig.clone();
    let e = 128 * n;
    time("ops::softmax_row", e, 100, || {
        rows.copy_from_slice(&orig);
        rows.chunks_exact_mut(n).for_each(|r| softmax_row(r, 0.125));
    });
    time("softmax_unnorm", e, 100, || {
        rows.copy_from_slice(&orig);
        micro::softmax(&mut rows, n);
    });
    time("  exp pass", e, 100, || {
        rows.copy_from_slice(&orig);
        micro::exp_pass(&mut rows, n);
    });
    time("  sum pass", e, 100, || {
        std::hint::black_box(micro::sum_pass(&rows, n));
    });
    time("  copy only", e, 100, || rows.copy_from_slice(&orig));
    let mut g: Vec<f32> = (0..1 << 20).map(|i| ((i * 7919) % 1000) as f32 * 0.01 - 5.0).collect();
    let b = vec![0.1f32; 1536];
    time("ops::gelu_slice", 1 << 20, 10, || gelu_slice(&mut g));
    time("bias_gelu_row", 1 << 20, 10, || micro::bias_gelu(&mut g, &b));
    time("bias_tanh_row", 1 << 20, 10, || micro::bias_tanh(&mut g, &b));
    let mut o = vec![0f32; 1 << 20];
    let gm = vec![1.0f32; 384];
    time("rms_norm_row", 1 << 20, 10, || micro::rms(&g, &mut o, &gm));
}
