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

/// Each transformer layer run alone on PyTorch's own input to it: separates the layer's own error
/// from error propagated from earlier layers (attention with very sharp softmaxes amplifies input
/// differences, so the propagated figure alone can look worse than the layer is).
fn isolated_transformer(model: &HTDemucs, r: &SafeTensors) {
    let get = |n: &str| r.get(n).unwrap_or_else(|| panic!("{n}"));
    let mut x = get("crosstransformer.norm_in").clone();
    let pe = get("pos_emb_2d");
    let (c, fr, t1) = (pe.shape[1], pe.shape[2], pe.shape[3]);
    let w = model.cfg.t_weight_pos_embed as f32;
    for ci in 0..c {
        for f in 0..fr {
            for t in 0..t1 {
                x.data[(t * fr + f) * c + ci] += w * pe.data[(ci * fr + f) * t1 + t];
            }
        }
    }
    let mut xt = get("crosstransformer.norm_in_t").clone();
    for (v, p) in xt.data.iter_mut().zip(&get("pos_emb_1d").data) {
        *v += w * p;
    }
    let squeeze2 = |t: &Tensor| t.clone().reshape(&[t.numel() / c, c]);
    let (mut x, mut xt) = (squeeze2(&x), squeeze2(&xt));
    println!("isolated transformer layers (input = PyTorch's input to that layer):");
    println!("| layer | attention SNR dB | output SNR dB |\n|---|---|---|");
    let tf = &model.transformer;
    for i in 0..tf.n_layers() {
        for (tb, inp, other) in [(false, &x, &xt), (true, &xt, &x)] {
            let p = format!("crosstransformer.{}.{i}", if tb { "layers_t" } else { "layers" });
            let an = format!("{p}.{}", if tf.is_cross(i) { "cross_attn" } else { "self_attn" });
            let mut att = None;
            let mut sink = |n: &str, t: &Tensor| {
                if n == an {
                    att = Some(compare(n, t, get(n)).snr);
                }
            };
            let out = tf.layer(tb, i, inp, tf.is_cross(i).then_some(other), &mut Tracer(Some(&mut sink)));
            println!("| {p} | {:.1} | {:.1} |", att.unwrap(), compare(&p, &out, get(&p)).snr);
        }
        x = squeeze2(get(&format!("crosstransformer.layers.{i}")));
        xt = squeeze2(get(&format!("crosstransformer.layers_t.{i}")));
    }
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
        isolated_transformer(model, &r);
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
