//! Mel-RoFormer parity against a PyTorch reference chunk (tools/melroformer.py segment):
//!
//!     mr_parity <weights dir> <ref.safetensors> [weights name, default melroformer]
//!
//! Prints, for every activation both sides produce, SNR (dB) and max abs error of ours vs PyTorch
//! CPU float32, then each transformer run alone on PyTorch's own input to it. Exits 1 if the
//! output's SNR is below 90 dB (several mask heads: `mask.<h>` / `output.<h>` per head, the worst
//! output counts).

use std::collections::HashMap;
use std::time::Instant;

use htdemucs::roformer::MelRoformer;
use htdemucs::weights::SafeTensors;
use htdemucs::{Tensor, Tracer};

fn snr(ours: &[f32], theirs: &[f32]) -> (f64, f64, f64) {
    assert_eq!(ours.len(), theirs.len());
    let (mut sig, mut err, mut mx, mut rmax) = (0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in ours.iter().zip(theirs) {
        let (a, b) = (a as f64, b as f64);
        sig += b * b;
        err += (a - b) * (a - b);
        mx = mx.max((a - b).abs());
        rmax = rmax.max(b.abs());
    }
    (if err == 0.0 { f64::INFINITY } else { 10.0 * (sig / err).log10() }, mx, rmax)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.get(2).map(String::as_str).unwrap_or("melroformer");
    let t0 = Instant::now();
    let model = MelRoformer::load(&args[0], name).unwrap();
    let r = SafeTensors::load(&args[1]).unwrap();
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let get = |n: &str| r.get(n).unwrap_or_else(|| panic!("reference has no {n}"));
    let input = get("input");
    let mut ours: HashMap<String, Tensor> = HashMap::new();
    let mut order = vec![];
    let t1 = Instant::now();
    let mut sink = |n: &str, t: &Tensor| {
        order.push(n.to_string());
        ours.insert(n.to_string(), t.clone());
    };
    model.forward(&input.data, &mut Tracer(Some(&mut sink)));
    eprintln!("forward (traced) in {:.1}s", t1.elapsed().as_secs_f64());
    println!("| activation | shape | SNR dB | max abs err | ref max |");
    println!("|---|---|---|---|---|");
    let mut out_snr = f64::INFINITY;
    let mut outputs = 0;
    for n in &order {
        let Some(theirs) = r.get(n) else { continue };
        let (s, mx, rm) = snr(&ours[n].data, &theirs.data);
        println!("| {n} | {:?} | {s:.1} | {mx:.2e} | {rm:.2e} |", theirs.shape);
        if n == "output" || n.starts_with("output.") {
            out_snr = f64::min(out_snr, s);
            outputs += 1;
        }
    }
    println!("\nisolated (each transformer on PyTorch's own input):\n");
    println!("| transformer | SNR dB | max abs err |");
    println!("|---|---|---|");
    for i in 0..model.cfg.depth {
        // time i: input = freq i-1 output (frame-major) or band_split; output (band, frame, dim)
        let prev = if i == 0 { get("band_split") } else { get(&format!("layers.{}.freq", i - 1)) };
        let (frames, nb, dim) = (prev.shape[0], prev.shape[1], prev.shape[2]);
        let mut bm = vec![0.0f32; prev.numel()];
        for t in 0..frames {
            for b in 0..nb {
                bm[(b * frames + t) * dim..(b * frames + t + 1) * dim].copy_from_slice(&prev.data[(t * nb + b) * dim..(t * nb + b + 1) * dim]);
            }
        }
        let y = model.transformer(i, false, &Tensor::new(&[nb, frames, dim], bm));
        let (s, mx, _) = snr(&y.data, &get(&format!("layers.{i}.time")).data);
        println!("| layers.{i}.time | {s:.1} | {mx:.2e} |");
        let tin = get(&format!("layers.{i}.time"));
        let mut fm = vec![0.0f32; tin.numel()];
        for b in 0..nb {
            for t in 0..frames {
                fm[(t * nb + b) * dim..(t * nb + b + 1) * dim].copy_from_slice(&tin.data[(b * frames + t) * dim..(b * frames + t + 1) * dim]);
            }
        }
        let y = model.transformer(i, true, &Tensor::new(&[frames, nb, dim], fm));
        let (s, mx, _) = snr(&y.data, &get(&format!("layers.{i}.freq")).data);
        println!("| layers.{i}.freq | {s:.1} | {mx:.2e} |");
    }
    println!("\nfinal output: {out_snr:.1} dB{}", if outputs > 1 { format!(" (worst of {outputs} heads)") } else { String::new() });
    if outputs == 0 || out_snr < 90.0 {
        std::process::exit(1);
    }
}
