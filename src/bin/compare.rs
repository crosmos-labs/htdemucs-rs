//! Compare separated stems with a PyTorch reference (tools/ref_separate.py):
//!
//!     compare <ref.safetensors> <stem dir>      (stems as float32 WAV, --clip-mode none)
//!
//! Prints, per stem, SNR (dB), max abs error over the full length, and the reference stem's RMS
//! level and the error's RMS level against full scale (a near-silent stem has a low SNR for the
//! same absolute error).

use std::path::Path;

use htdemucs::audio;
use htdemucs::weights::SafeTensors;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = SafeTensors::load(&args[0]).unwrap();
    let stems = r.get("stems").unwrap();
    let names: Vec<&str> = r.metadata["sources"].split(',').collect();
    let (ch, len) = (stems.shape[1], stems.shape[2]);
    let mut worst = f64::INFINITY;
    let mut row = vec![];
    for (i, n) in names.iter().enumerate() {
        let a = audio::decode(&Path::new(&args[1]).join(format!("{n}.wav"))).unwrap();
        assert_eq!((a.channels, a.frames()), (ch, len), "{n}: shape");
        let theirs = &stems.data[i * ch * len..(i + 1) * ch * len];
        let (mut sig, mut err, mut mx) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.data.iter().zip(theirs) {
            let d = x as f64 - y as f64;
            sig += (y as f64) * (y as f64);
            err += d * d;
            mx = mx.max(d.abs());
        }
        let snr = if err == 0.0 { f64::INFINITY } else { 10.0 * (sig / err).log10() };
        worst = worst.min(snr);
        let rms_db = 10.0 * (sig / theirs.len() as f64).max(1e-30).log10();
        let err_db = 10.0 * (err / theirs.len() as f64).max(1e-30).log10();
        row.push(format!("{n} {snr:.1} dB (max {mx:.1e}, level {rms_db:.0} dBFS, error {err_db:.0} dBFS)"));
    }
    println!("{}: {} | min {worst:.1} dB", args[1], row.join(", "));
}
