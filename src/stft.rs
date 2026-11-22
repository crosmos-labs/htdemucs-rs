//! HTDemucs' spectrogram front end and back end (`HTDemucs._spec` / `_ispec`, demucs/spec.py),
//! reproduced exactly:
//!
//! `_spec`: reflect-pad the signal by 3*hop/2 on the left and 3*hop/2 + (le*hop - len) on the
//! right (le = ceil(len / hop)), then `torch.stft(n_fft, hop, hann (periodic), normalized=True,
//! center=True, pad_mode='reflect')` - which reflect-pads n_fft/2 more on each side - then drop the
//! Nyquist bin and keep frames [2, 2 + le). `normalized=True` scales by 1/sqrt(n_fft).
//!
//! `_ispec`: append a zero Nyquist bin, add 2 zero frames on each side, `torch.istft` with the
//! same window, normalized, center=True, `length = hop*ceil(len/hop) + 2*pad`, and keep samples
//! [pad, pad + len) with pad = 3*hop/2. istft = inverse real FFT scaled by 1/sqrt(n_fft) (the
//! imaginary parts of the DC and Nyquist bins are ignored, as pocketfft's c2r does), times the
//! window, overlap-added, divided by the overlap-added squared window, minus n_fft/2 at the start.
//!
//! FFTs are computed in f64 (cheap next to the network) so our only error is the final rounding.

use rayon::prelude::*;
use realfft::RealFftPlanner;

pub struct Stft {
    pub n_fft: usize,
    pub hop: usize,
    window: Vec<f64>,
}

/// Reflect padding without edge repetition (torch 'reflect'): out[-i] = x[i].
fn reflect_pad(x: &[f32], left: usize, right: usize) -> Vec<f32> {
    let n = x.len();
    assert!(left < n && right < n, "reflect pad larger than the signal");
    let mut out = Vec::with_capacity(n + left + right);
    for i in (1..=left).rev() {
        out.push(x[i]);
    }
    out.extend_from_slice(x);
    for i in 0..right {
        out.push(x[n - 2 - i]);
    }
    out
}

impl Stft {
    pub fn new(n_fft: usize) -> Self {
        // torch.hann_window(n_fft) (periodic): 0.5 - 0.5 cos(2 pi n / N), computed by torch in
        // float32; round ours to float32 too.
        let window = (0..n_fft)
            .map(|i| {
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos();
                w as f32 as f64
            })
            .collect();
        Stft { n_fft, hop: n_fft / 4, window }
    }

    /// `_spec` for one channel: returns (n_fft/2 bins x le frames) complex, as interleaved
    /// [re, im] pairs, row-major [bin][frame][2].
    pub fn spec(&self, x: &[f32]) -> (Vec<f32>, usize) {
        let (nfft, hop) = (self.n_fft, self.hop);
        let len = x.len();
        let le = len.div_ceil(hop);
        let pad = hop / 2 * 3;
        let x1 = reflect_pad(x, pad, pad + le * hop - len);
        let y = reflect_pad(&x1, nfft / 2, nfft / 2);
        let bins = nfft / 2;
        let norm = 1.0 / (nfft as f64).sqrt();
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(nfft);
        // Frames 2 .. 2 + le of the centred STFT.
        let frames: Vec<Vec<(f64, f64)>> = (0..le)
            .into_par_iter()
            .map_init(
                || (fft.make_input_vec(), fft.make_output_vec(), fft.make_scratch_vec()),
                |(inp, outp, scratch), f| {
                    let start = (f + 2) * hop;
                    for i in 0..nfft {
                        inp[i] = y[start + i] as f64 * self.window[i];
                    }
                    fft.process_with_scratch(inp, outp, scratch).unwrap();
                    outp[..bins].iter().map(|c| (c.re * norm, c.im * norm)).collect()
                },
            )
            .collect();
        let mut out = vec![0.0f32; bins * le * 2];
        for (f, fr) in frames.iter().enumerate() {
            for (b, &(re, im)) in fr.iter().enumerate() {
                out[(b * le + f) * 2] = re as f32;
                out[(b * le + f) * 2 + 1] = im as f32;
            }
        }
        (out, le)
    }

    /// `_ispec` for one channel: `z` is (n_fft/2 bins x frames) as separate re / im planes
    /// (row-major [bin][frame]); returns `length` samples.
    pub fn ispec(&self, re: &[f32], im: &[f32], frames: usize, length: usize) -> Vec<f32> {
        let (nfft, hop) = (self.n_fft, self.hop);
        let bins = nfft / 2;
        let pad = hop / 2 * 3;
        let le = hop * length.div_ceil(hop) + 2 * pad;
        let nfr = frames + 4; // 2 zero frames each side
        let full = nfft + hop * (nfr - 1);
        let start = nfft / 2;
        assert!(start + le <= full);
        let norm = 1.0 / (nfft as f64).sqrt();
        let ifft = RealFftPlanner::<f64>::new().plan_fft_inverse(nfft);
        // Inverse FFT of each non-zero frame, windowed.
        let blocks: Vec<Vec<f64>> = (0..frames)
            .into_par_iter()
            .map_init(
                || (ifft.make_input_vec(), ifft.make_output_vec(), ifft.make_scratch_vec()),
                |(inp, outp, scratch), f| {
                    for b in 0..bins {
                        inp[b] = realfft::num_complex::Complex::new(
                            re[b * frames + f] as f64,
                            im[b * frames + f] as f64,
                        );
                    }
                    inp[0].im = 0.0; // c2r ignores the imaginary part of DC
                    inp[bins] = realfft::num_complex::Complex::new(0.0, 0.0); // padded Nyquist
                    ifft.process_with_scratch(inp, outp, scratch).unwrap();
                    // realfft's inverse is unnormalised; torch's by_root_n is 1/sqrt(n).
                    outp.iter().zip(&self.window).map(|(v, w)| v * norm * w).collect()
                },
            )
            .collect();
        // Overlap-add the signal and the squared-window envelope over all nfr frames
        // (the padding frames contribute zero signal but their window to the envelope).
        let mut out = vec![0.0f32; length];
        out.par_chunks_mut(4096).enumerate().for_each(|(ci, chunk)| {
            for (j, o) in chunk.iter_mut().enumerate() {
                let i = ci * 4096 + j; // output sample
                let n = start + pad + i; // position in the full overlap-added signal
                let mut y = 0f64;
                let mut env = 0f64;
                let f_lo = if n + 1 > nfft { (n + 1 - nfft).div_ceil(hop) } else { 0 };
                let f_hi = (n / hop).min(nfr - 1);
                for fr in f_lo..=f_hi {
                    let k = n - fr * hop;
                    let w = self.window[k];
                    env += w * w;
                    if fr >= 2 && fr < 2 + frames {
                        y += blocks[fr - 2][k];
                    }
                }
                *o = (y / env) as f32;
            }
        });
        out
    }
}
