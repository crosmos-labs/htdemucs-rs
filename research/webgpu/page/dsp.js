// STFT / complex mask / iSTFT outside the ONNX graph, matching torch.stft / torch.istft with center=True (reflect pad),
// a periodic Hann window, normalized=False, onesided. Layout of the spectrum: (T, F, channel, re/im), as the models use.
function fftPlan(n) {
  const rev = new Uint32Array(n); const bits = Math.log2(n);
  for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
  const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
  return { n, rev, cos, sin, re: new Float64Array(n), im: new Float64Array(n) };
}
function fft(p, inv) {            // in place on p.re / p.im (already bit-reversed by the caller)
  const { n, cos, sin, re, im } = p; const sg = inv ? 1 : -1;
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let i = 0; i < n; i += size) for (let j = 0, k = 0; j < half; j++, k += step) {
      const a = i + j, b = a + half, wr = cos[k], wi = sg * sin[k];
      const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
      re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
    }
  }
}
const hann = n => { const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n); return w; };

// chans: [Float32Array L, Float32Array R] of equal length; returns Float32Array (T * F * C * 2)
export function stft(chans, nfft, hop) {
  const C = chans.length, len = chans[0].length, pad = nfft / 2, T = Math.floor(len / hop) + 1, F = nfft / 2 + 1;
  const p = fftPlan(nfft), w = hann(nfft), out = new Float32Array(T * F * C * 2);
  const at = (x, i) => x[i < 0 ? -i : i >= len ? 2 * (len - 1) - i : i];      // reflect padding
  for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) {
    const s0 = t * hop - pad;
    for (let i = 0; i < nfft; i++) { const r = p.rev[i]; p.re[r] = at(chans[c], s0 + i) * w[i]; p.im[r] = 0; }
    fft(p, false);
    for (let f = 0; f < F; f++) { const o = ((t * F + f) * C + c) * 2; out[o] = p.re[f]; out[o + 1] = p.im[f]; }
  }
  return { X: out, T, F, C };
}

// X: (T, F, C, 2); M: (n, T, F, C, 2) masks; returns n x C Float32Arrays of `length` samples
export function maskIstft(X, M, n, T, F, C, nfft, hop, length, zeroDc) {
  const p = fftPlan(nfft), w = hann(nfft), pad = nfft / 2, full = (T - 1) * hop + nfft;
  const norm = new Float64Array(full); for (let t = 0; t < T; t++) for (let i = 0; i < nfft; i++) norm[t * hop + i] += w[i] * w[i];
  const outs = [];
  for (let s = 0; s < n; s++) for (let c = 0; c < C; c++) {
    const acc = new Float64Array(full);
    for (let t = 0; t < T; t++) {
      p.re.fill(0); p.im.fill(0);
      for (let f = 0; f < F; f++) {
        const xo = ((t * F + f) * C + c) * 2, mo = (((s * T + t) * F + f) * C + c) * 2;
        let yr = X[xo] * M[mo] - X[xo + 1] * M[mo + 1], yi = X[xo] * M[mo + 1] + X[xo + 1] * M[mo];
        if (zeroDc && f === 0) yr = yi = 0;
        p.re[p.rev[f]] = yr; p.im[p.rev[f]] = yi;
        if (f > 0 && f < F - 1) { p.re[p.rev[nfft - f]] = yr; p.im[p.rev[nfft - f]] = -yi; }   // Hermitian
      }
      if (true) { /* onesided: torch ignores the imaginary part of DC and Nyquist */ p.im[p.rev[0]] = 0; p.im[p.rev[F - 1]] = 0; }
      fft(p, true);
      for (let i = 0; i < nfft; i++) acc[t * hop + i] += p.re[i] / nfft * w[i];
    }
    const y = new Float32Array(length);
    for (let i = 0; i < length; i++) { const k = i + pad; y[i] = norm[k] > 1e-11 ? acc[k] / norm[k] : 0; }
    outs.push(y);
  }
  return outs;
}
