// Whole-song prototype of the round-5 recipe in the browser: decode -> Kim (vocals) -> instrumental = mix - vocals ->
// Mega 4 heads (drums, bass, guitar, keys) -> other = mix - the five. Chunking as MSST's generic demix (step = chunk / 2,
// reflect padding of chunk - step at both ends, linear fades of chunk / 10, overlap-add / summed window).
// Query: kim=<model>, mega=<model> (names in /models, metadata from the matching .json), wav=/models/song.wav,
// cache=lazyRelease, tag. POSTs <tag>.json (timings, per-chunk ms) and <tag>_stems.bin (int16, vocals drums bass, 2 ch each).
import { stft, maskIstft } from './dsp.js';
const ort = await import('/ort/ort.webgpu.min.mjs');
const q = new URLSearchParams(location.search);
const tag = q.get('tag') || 'song';
const logEl = document.getElementById('log'); const lines = [];
const log = (...a) => { lines.push(a.join(' ')); logEl.textContent = lines.join('\n'); console.log(...a); };
const now = () => performance.now();
const R = { tag, kim: q.get('kim'), mega: q.get('mega'), cache: q.get('cache') };
const base = n => n.replace(/_(fp16n?|w16|q8)$/, '');

async function session(name) {
  const meta = await (await fetch(`/models/${base(name)}.json`)).json();
  let t = now(); const buf = new Uint8Array(await (await fetch(`/models/${name}.onnx`)).arrayBuffer()); const fetch_ms = now() - t;
  const epo = { name: 'webgpu' }; if (q.get('cache')) epo.storageBufferCacheMode = q.get('cache');
  t = now(); const s = await ort.InferenceSession.create(buf, { executionProviders: [epo], graphOptimizationLevel: 'all' });
  return { s, meta, fetch_ms, session_ms: now() - t, mb: buf.byteLength / 1e6 };
}

function reflectPad(x, p) {          // torch 'reflect' padding of p samples at both ends
  const n = x.length, y = new Float32Array(n + 2 * p);
  for (let i = 0; i < n + 2 * p; i++) { let j = i - p; if (j < 0) j = -j; if (j >= n) j = 2 * (n - 1) - j; y[i] = x[j]; }
  return y;
}

async function demix(M, chans, label) {
  const { s, meta } = M, C = meta.chunk, step = C / 2, border = C - step, fade = Math.floor(C / 10), n = meta.out_shape[1];
  const n0 = chans[0].length, pad = n0 > 2 * border ? border : 0;
  const x = chans.map(c => pad ? reflectPad(c, pad) : c), N = x[0].length;
  const out = Array.from({ length: n * 2 }, () => new Float32Array(N)), cnt = new Float32Array(N);
  const win = new Float32Array(C).fill(1);
  for (let i = 0; i < fade; i++) { win[i] = i / (fade - 1); win[C - 1 - i] = i / (fade - 1); }   // torch.linspace(0, 1, fade)
  const per = []; let gpu = 0, dsp = 0;
  for (let i = 0; i < N; i += step) {
    const L = Math.min(C, N - i);
    const part = x.map(c => { const p = new Float32Array(C); p.set(c.subarray(i, i + L));
      if (L < C) { if (L > C / 2) for (let k = L; k < C; k++) p[k] = c[i + 2 * (L - 1) - k]; } return p; });   // reflect / zero tail
    let t = now(); const S = stft(part, meta.n_fft, meta.hop); dsp += now() - t;
    t = now(); const o = await s.run({ X: new ort.Tensor('float32', S.X, meta.in_shape) }); const Mk = await o.masks.getData(); const g = now() - t; gpu += g; per.push(Math.round(g));
    t = now(); const ys = maskIstft(S.X, Mk, n, meta.T, meta.F, 2, meta.n_fft, meta.hop, C, meta.zero_dc); dsp += now() - t;
    o.masks.dispose?.();
    const w = win.slice(); if (i === 0) w.fill(1, 0, fade); if (i + step >= N) w.fill(1, C - fade);
    for (let k = 0; k < n * 2; k++) { const y = ys[k], d = out[k]; for (let j = 0; j < L; j++) d[i + j] += y[j] * w[j]; }
    for (let j = 0; j < L; j++) cnt[i + j] += w[j];
    if (per.length % 10 === 1) log(`${label}: chunk ${per.length} at ${(i / 44100).toFixed(0)} s, ${g.toFixed(0)} ms`);
  }
  const res = out.map(d => { const r = new Float32Array(n0); for (let j = 0; j < n0; j++) r[j] = d[j + pad] / cnt[j + pad]; return r; });
  return { res, per, gpu_ms: gpu, dsp_ms: dsp, chunks: per.length };
}

try {
  let t0 = now();
  const ab = await (await fetch(q.get('wav') || '/models/song.wav')).arrayBuffer();
  const audio = await new OfflineAudioContext(2, 1, 44100).decodeAudioData(ab);
  const mix = [audio.getChannelData(0), audio.getChannelData(audio.numberOfChannels > 1 ? 1 : 0)];
  R.audio_s = mix[0].length / 44100; R.decode_ms = now() - t0; log(`decoded ${R.audio_s.toFixed(1)} s`);
  const tStart = now();
  let K = await session(R.kim); R.kim_load = { fetch_ms: K.fetch_ms, session_ms: K.session_ms, mb: K.mb };
  const kv = await demix(K, mix, 'kim'); await K.s.release(); K = null;
  R.kim_run = { gpu_ms: kv.gpu_ms, dsp_ms: kv.dsp_ms, chunks: kv.chunks, per_chunk_ms: kv.per };
  const voc = kv.res, inst = mix.map((c, k) => { const r = new Float32Array(c.length); for (let j = 0; j < c.length; j++) r[j] = c[j] - voc[k][j]; return r; });
  let M = await session(R.mega); R.mega_load = { fetch_ms: M.fetch_ms, session_ms: M.session_ms, mb: M.mb };
  const mv = await demix(M, inst, 'mega'); await M.s.release(); M = null;
  R.mega_run = { gpu_ms: mv.gpu_ms, dsp_ms: mv.dsp_ms, chunks: mv.chunks, per_chunk_ms: mv.per };
  R.total_ms = now() - tStart; R.s_per_audio_s = R.total_ms / 1000 / R.audio_s;
  log(`total ${(R.total_ms / 1000).toFixed(0)} s for ${R.audio_s.toFixed(0)} s of audio (${R.s_per_audio_s.toFixed(2)} s per audio-s)`);
  // vocals, drums, bass as int16 (for uSDR against the dataset's stems)
  const keep = [voc[0], voc[1], mv.res[0], mv.res[1], mv.res[2], mv.res[3]], n = keep[0].length, pcm = new Int16Array(6 * n);
  keep.forEach((c, k) => { for (let j = 0; j < n; j++) pcm[k * n + j] = Math.max(-32768, Math.min(32767, Math.round(c[j] * 32767))); });
  await fetch(`/result?name=${tag}_stems.bin`, { method: 'POST', body: pcm });
} catch (e) { R.error = String(e && e.stack || e); log('ERROR', R.error); }
await fetch(`/result?name=${tag}.json`, { method: 'POST', body: JSON.stringify(R) });
window.__result = R; log('done');
