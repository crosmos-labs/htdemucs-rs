//! Parity check against the PyTorch reference dumps (tools/harness.py):
//!
//!     cargo run --release --bin parity -- weights ref/aerials-htdemucs-10.safetensors [...]
//!
//! For every named activation both sides produce: max abs error and SNR (dB) of ours vs PyTorch
//! CPU float32. Fails (exit 1) if the final output's SNR is below 80 dB.

use std::time::Instant;

use htdemucs::weights::SafeTensors;
use htdemucs::{HTDemucs, Tensor, Tracer};

fn squeeze(s: &[usize]) -> Vec<usize> {
    s.iter().copied().filter(|&d| d != 1).collect()
}

struct Row {
    name: String,
    shape: Vec<usize>,
    max_abs: f64,
    ref_max: f64,
    snr: f64,
}

fn compare(name: &str, ours: &Tensor, theirs: &Tensor) -> Row {
    // Same element order required; shapes may differ by a reshape (e.g. (C, F, T) vs (C, F*T)).
    assert_eq!(ours.numel(), theirs.numel(), "{name}: shape ours {:?} torch {:?}", ours.shape, theirs.shape);
    let (mut sig, mut err, mut max_abs, mut ref_max) = (0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in ours.data.iter().zip(&theirs.data) {
        let (a, b) = (a as f64, b as f64);
        sig += b * b;
        err += (a - b) * (a - b);
        max_abs = max_abs.max((a - b).abs());
        ref_max = ref_max.max(b.abs());
    }
    let snr = if err == 0.0 { f64::INFINITY } else { 10.0 * (sig / err).log10() };
    Row { name: name.to_string(), shape: squeeze(&theirs.shape), max_abs, ref_max, snr }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: parity <weights dir> <ref.safetensors>...");
        std::process::exit(2);
    }
    let mut failed = false;
    let mut models: Vec<(String, HTDemucs)> = vec![];
    for path in &args[1..] {
        let t0 = Instant::now();
        let r = SafeTensors::load(path).unwrap();
        let name = r.metadata["model"].clone();
        if !models.iter().any(|(n, _)| *n == name) {
            let t = Instant::now();
            let m = HTDemucs::load(&args[0], &name).unwrap();
            eprintln!("loaded {name} in {:.2}s", t.elapsed().as_secs_f64());
            models.push((name.clone(), m));
        }
        let model = &models.iter().find(|(n, _)| *n == name).unwrap().1;
        let input = r.get("input").unwrap();
        eprintln!("{path}: read in {:.1}s", t0.elapsed().as_secs_f64());
        println!("\n## {path}  (model {name}, chunk {} offset {} length {}, shift {}, torch {:.2}s/{} threads)",
            r.metadata["chunk"], r.metadata["offset"], r.metadata["chunk_length"], r.metadata["shift_offset"],
            r.metadata["torch_seconds"].parse::<f64>().unwrap(), r.metadata["torch_threads"]);

        // Untraced runs: warm-up + timed.
        let _ = model.forward(&input.data, &mut Tracer::none());
        let t = Instant::now();
        let plain = model.forward(&input.data, &mut Tracer::none());
        let secs = t.elapsed().as_secs_f64();
        println!("rust forward: {secs:.2}s on {} threads", rayon::current_num_threads());

        let mut rows: Vec<Row> = vec![];
        let mut sink = |n: &str, t: &Tensor| {
            if let Some(theirs) = r.get(n) {
                rows.push(compare(n, t, theirs));
            } else {
                eprintln!("  (no reference for {n})");
            }
        };
        let traced = model.forward(&input.data, &mut Tracer(Some(&mut sink)));
        assert_eq!(traced.data, plain.data, "tracing changed the result");
        let missing: Vec<&String> = r.order.iter().filter(|n| *n != "input" && !rows.iter().any(|row| &row.name == *n)).collect();
        if !missing.is_empty() {
            println!("torch activations without a Rust counterpart: {missing:?}");
        }
        println!("| layer | shape | max abs err | max abs ref | SNR dB |\n|---|---|---|---|---|");
        for row in &rows {
            println!("| {} | {:?} | {:.3e} | {:.3e} | {:.1} |", row.name, row.shape, row.max_abs, row.ref_max, row.snr);
        }
        let out = rows.iter().find(|r| r.name == "output").unwrap();
        println!("final output SNR {:.1} dB, max abs err {:.3e}", out.snr, out.max_abs);
        if out.snr < 80.0 {
            failed = true;
        }
    }
    if failed {
        eprintln!("FAIL: output SNR below 80 dB");
        std::process::exit(1);
    }
}
