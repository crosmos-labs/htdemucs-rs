// One-chunk benchmark of an exported separator core in onnxruntime-web. Query: model=<name in /models>, ep=webgpu|wasm,
// threads (wasm), runs, save=1 (POST the output masks too), tag. Result JSON is POSTed to /result and left in window.__result.
import * as ort from '/ort/ort.all.min.mjs';
const q = new URLSearchParams(location.search);
const model = q.get('model'), ep = q.get('ep') || 'webgpu', runs = +(q.get('runs') || 3), threads = +(q.get('threads') || 1);
const ref = q.get('ref') || model.replace(/_(fp16|w16|q8)$/, '');
const tag = q.get('tag') || `${model}_${ep}${ep === 'wasm' ? threads : ''}`;
const logEl = document.getElementById('log'); const lines = [];
const log = (...a) => { lines.push(a.join(' ')); logEl.textContent = lines.join('\n'); console.log(...a); };
const R = { tag, model, ep, threads, ua: navigator.userAgent, hw: navigator.hardwareConcurrency, deviceMemory: navigator.deviceMemory ?? null };
const now = () => performance.now();
async function bin(u) { const r = await fetch(u); if (!r.ok) throw new Error(u + ' ' + r.status); return r.arrayBuffer(); }
function snr(a, b) { let s = 0, e = 0; for (let i = 0; i < a.length; i++) { s += a[i] * a[i]; const d = a[i] - b[i]; e += d * d; } return 10 * Math.log10(s / Math.max(e, 1e-300)); }
try {
  if (navigator.gpu) {
    const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (ad) {
      const L = ad.limits;
      R.adapter = { vendor: ad.info?.vendor, architecture: ad.info?.architecture, device: ad.info?.device, description: ad.info?.description,
        isFallbackAdapter: ad.info?.isFallbackAdapter ?? null, features: [...ad.features].sort(),
        limits: { maxBufferSize: L.maxBufferSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
          maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup } };
    } else R.adapter = null;
  } else R.adapter = 'no navigator.gpu';
  log('adapter', JSON.stringify(R.adapter));
  const meta = await (await fetch(`/models/${ref}.json`)).json(); R.meta = { chunk_s: meta.chunk_s, T: meta.T, out_shape: meta.out_shape };
  let t = now(); const mbuf = await bin(`/models/${model}.onnx`); R.model_mb = mbuf.byteLength / 1e6; R.fetch_ms = now() - t;
  const X = new Float32Array(await bin(`/models/${ref}_in.bin`)); const Mref = new Float32Array(await bin(`/models/${ref}_ref.bin`));
  ort.env.wasm.wasmPaths = '/ort/'; ort.env.wasm.numThreads = ep === 'wasm' ? threads : 1;
  const opts = { executionProviders: [ep], graphOptimizationLevel: 'all' };
  if (ep === 'webgpu') ort.env.webgpu.powerPreference = 'high-performance';
  t = now(); const sess = await ort.InferenceSession.create(new Uint8Array(mbuf), opts); R.session_ms = now() - t;
  log(`session ${R.session_ms.toFixed(0)} ms, model ${R.model_mb.toFixed(0)} MB`);
  const feeds = { X: new ort.Tensor('float32', X, meta.in_shape) };
  R.run_ms = []; let out;
  for (let i = 0; i < runs + 1; i++) {
    t = now(); const o = await sess.run(feeds); out = o.masks; const d = await out.getData(); R.run_ms.push(now() - t);
    log(`run ${i}: ${R.run_ms.at(-1).toFixed(0)} ms`);
    if (i < runs) out.dispose?.();
  }
  const Y = await out.getData();
  R.mask_snr_db = snr(Mref, Y); R.nan = Y.some(Number.isNaN);
  const steady = R.run_ms.slice(1).sort((a, b) => a - b); R.median_ms = steady[Math.floor(steady.length / 2)];
  R.s_per_audio_s_chunk = R.median_ms / 1000 / meta.chunk_s;
  log(`mask SNR vs PyTorch ${R.mask_snr_db.toFixed(1)} dB, median ${R.median_ms.toFixed(0)} ms per ${meta.chunk_s} s chunk`);
  if (q.get('save') === '1') await fetch(`/result?name=${tag}_out.bin`, { method: 'POST', body: Y });
  await sess.release();
} catch (e) { R.error = String(e && e.stack || e); log('ERROR', R.error); }
await fetch(`/result?name=${tag}.json`, { method: 'POST', body: JSON.stringify(R, null, 1) });
window.__result = R; log('done');
