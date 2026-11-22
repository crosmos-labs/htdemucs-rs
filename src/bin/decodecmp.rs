//! Compare our decode of a file with a reference decode (e.g. ffmpeg's float WAV):
//!
//!     decodecmp <file> <reference.wav>
//!
//! Prints both lengths, the SNR of ours vs the reference, and the best lag in +-4096 samples (to
//! catch an unhandled encoder delay).

use std::path::Path;

use htdemucs::audio;

fn snr(a: &[f32], b: &[f32]) -> f64 {
    let (mut s, mut e) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        s += (y as f64).powi(2);
        e += (x as f64 - y as f64).powi(2);
    }
    if e == 0.0 { f64::INFINITY } else { 10.0 * (s / e).log10() }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a = audio::to_channels(audio::decode(Path::new(&args[0])).unwrap(), 2).unwrap();
    let b = audio::to_channels(audio::decode(Path::new(&args[1])).unwrap(), 2).unwrap();
    let (fa, fb) = (a.frames(), b.frames());
    let n = fa.min(fb);
    let l = |x: &audio::Audio, f: usize, c: usize, off: usize, n: usize| x.data[c * f + off..c * f + off + n].to_vec();
    let s0: Vec<f64> = (0..2).map(|c| snr(&l(&a, fa, c, 0, n), &l(&b, fb, c, 0, n))).collect();
    // Best lag on a 10 s window in the middle.
    let (w0, wl) = (n / 2, 441000.min(n / 4));
    let mut best = (0isize, f64::NEG_INFINITY);
    for lag in -4096isize..=4096 {
        let s = snr(&a.data[(w0 as isize + lag) as usize..(w0 as isize + lag) as usize + wl], &b.data[w0..w0 + wl]);
        if s > best.1 {
            best = (lag, s);
        }
    }
    println!("{}: ours {fa} frames @ {} Hz, ref {fb} @ {} Hz; SNR L {:.1} dB R {:.1} dB; best lag {} ({:.1} dB)",
        args[0], a.rate, b.rate, s0[0], s0[1], best.0, best.1);
}
