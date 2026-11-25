# Can visitors run stem separation in their own browser? (WebGPU feasibility)

Date: 2026-11-25. Branch `webgpu` (not merged). Prototype: `research/webgpu/` (export, page, driver, analysis). Every number labelled **measured** ran on this Mac (M1 Pro, 14-core GPU, 16 GB, shared with another agent's jobs, every run under the heavy-job lock) in Google Chrome 158 (headless, real Metal adapter) with onnxruntime-web 1.31.1. **Estimate** means arithmetic from those measurements and published peak FLOPS.

## Headline

**Yes, on a decent laptop GPU.** The whole round-5 recipe (`kim_mega_keys`) runs in the browser in fp16:
- Kim's vocals.
- Mega 53's `drums`, `bass`, `guitar` and `keys` heads on Kim's instrumental.
- other = mix − the five stems.

**Speed and resources (measured, whole song).** A 254 s song took **6.3 min on this M1 Pro, 1.49 s per audio second**.
- GPU memory: 2.5 GB.
- Download: 611 MB of weights, once.
- No thermal slowdown over the run: per-chunk times stayed flat within ±1 %.

**Quality (measured).**
- **Whole song, fp16 against the browser's fp32: uSDR within 0.004 dB** on vocals, drums and bass.
- One chunk each, fp16 against PyTorch fp32: within 0.03 dB.
- The fp32 path matches PyTorch to 59–119 dB SNR.

**What is ready and what is not.**
- The route is **ONNX Runtime Web's native WebGPU EP**, with STFT and iSTFT in JavaScript and the network exported with WebGPU-safe ops.
- WASM on the CPU is no fallback: Mega was 8× slower than WebGPU fp16, and Kim's fp32 graph didn't fit WASM's 4 GB heap.
- Qualifying devices (estimate): Apple M-series (M1 Pro measured; base M1/M2 ~1.7× slower), RTX 3060/4060-class discrete GPUs (~2–3× faster than this Mac), and recent iGPUs at roughly realtime ×2–3 or slower.
- Phones and 8 GB Intel laptops are out.
- Client-computed stems can be trusted for everyone with a **commit → random-chunk challenge → recompute** check. That costs about **1.2–2.5 vCPU-min per song (estimate), against ~82 for separating the song on the server**. Optionally add **agreement between two independent clients**, which needs almost no compute.

## What was built

1. **`export_onnx.py`: the network as ONNX, with the DSP outside.**
   - Input: X (1, T, F·2·2), the STFT in the models' own layout. Output: the complex masks (1, n, T, F·2·2).
   - The page does the STFT, the complex multiply, Mega's DC zeroing and the iSTFT in JS (`page/dsp.js`), because ORT's WebGPU EP has no STFT/DFT and no complex types.
   - Kim's overlapping mel bands are a Gather in, and the average of overlapping band masks is a Gather + weighted sum out. The upstream scatter_add is gone.
   - **PyTorch core + outside STFT vs the full PyTorch model: 107–125 dB SNR** on every stem. ORT CPU vs PyTorch: 114.6 dB on Mega's masks, 119.3 dB on Kim's.
2. **Four fixes the browser needed.** Each was found by running it.
   - **Graph size: 985 MB and 140k nodes became 305 MB and 11k nodes.** Tracing MSST's attention per slice baked a rotary cos/sin table into every slice (660 MB of constants). I re-wrote the transformer block:
     - weights pre-transposed once;
     - rotary cos/sin precomputed once per sequence length;
     - `rotate_half` as a 64×64 matmul;
     - the whole block run on batch slices whose biggest intermediate (attention matrix or FF hidden) fits a byte budget (`--budget-mb`).
   - **Too many outputs.** A 63-output `Split` (band split, mask-head unbind) failed WebGPU validation: 63 storage buffers against the default limit of 8 per stage. Now it is Slice / Gather, with concats in groups of 6.
   - **GLU.** ORT-web's WebGPU `Split` gave the GLU a zero-sized second half, so the GLU is now two Slices.
   - **fp16 overflow.** Plain fp16 broke Kim: 14 dB SNR, −1 dB uSDR. The RMSNorm's ReduceL2 (a sum of 384 squares) overflows fp16's 65504. The `fp16n` variant keeps that one op in fp32 (`convert_f16.py`).
3. **The pages.**
   - `page/bench.html` runs one real chunk and reports the session time, run times, mask and audio SNR against PyTorch, and the JS DSP time.
   - `page/song.html` runs a whole song with the recipe, using MSST's generic demix chunking (step = chunk/2, reflect border, linear fades).
   - `run.mjs` drives the installed Chrome through puppeteer-core and samples the GPU and renderer processes' physical footprint with macOS `footprint`.
   - `serve.mjs` serves on 127.0.0.1 only.
   - No audio, weights or ONNX files are in git. All audio was local MUSDB18-HQ.

## Measurements

### One chunk (measured)

Test input: Al James – Schoolboy Facination (MUSDB test), the chunk starting at 60 s. Times are the median of 2–3 runs after one warm-up. "SNR vs PyTorch" is each stem's audio against the PyTorch fp32 full model on the same chunk. "uSDR" is against MUSDB's true stem: drums and bass for Mega (on the mix here), vocals for Kim.

| model, chunk | weights (download) | precision | ms per chunk | s per audio-s of chunk | SNR vs PyTorch fp32 | uSDR (PyTorch fp32 = ref) | GPU footprint |
|---|---|---|---|---|---|---|---|
| Mega 4 heads, 20 s (budget 1024 MB), **JSEP** EP | 305 MB | fp32 | 20 660 | 1.03 | 78 dB (mask) | – | 10.1 GB |
| Mega 4 heads, 20 s (budget 1024) | 305 MB | fp32 | 14 848 | 0.742 | 75–87 dB | drums 8.640 / bass 11.822 = ref | 9.5 GB |
| Mega 4 heads, 20 s (budget 256) | 307 MB | fp32 | 14 928 | 0.746 | 109–119 dB | = ref | 4.7 GB |
| Mega 4 heads, 20 s | 154 MB | fp16 (all ops) | 9 115 | 0.456 | 32–40 dB | 8.621 / 11.852 | 5.0 GB |
| Mega 4 heads, 20 s | 154 MB | fp16n | 9 168 | 0.458 | 32–40 dB | 8.621 / 11.847 | 5.2 GB |
| Mega 4 heads, 20 s | 154 MB | fp16n + `enableMatmulFp32Accumulation` | 9 230 | 0.462 | 32–40 dB | – | 5.2 GB |
| Mega 4 heads, 20 s | 79 MB | int8 weights, fp32 compute | 14 916 | 0.746 | 30–38 dB | 8.607 / 11.830 | 9.3 GB |
| **Mega 4 heads, 10 s** (budget 256) | 305 MB | fp32 | 5 455 | 0.546 | 59–78 dB | drums 6.998 / bass 10.310 = ref | 4.6 GB (3.9 lazyRelease) |
| **Mega 4 heads, 10 s** (budget 256) | **154 MB** | **fp16n** | **3 525** | **0.352** | 33–41 dB | 6.997 / 10.309 | 2.6 GB (**2.1** lazyRelease) |
| Kim, 8 s (budget 1024) | 914 MB | fp32 | 3 557 | 0.445 | 109 dB | vocals 9.118 = ref | 8.2 GB |
| Kim, 8 s (budget 256) | 914 MB | fp32 | 3 538 | 0.442 | 99 dB | = ref | 5.3 GB (4.5 lazyRelease) |
| Kim, 8 s | 457 MB | fp16 (all ops) | 2 274 | 0.284 | **14 dB: broken** | **8.134 (−0.98)** | 4.0 GB |
| **Kim, 8 s** (budget 256) | **457 MB** | **fp16n** | **2 321** | **0.290** | 45 dB | 9.124 (+0.006) | 2.9 GB (**2.4** lazyRelease) |
| Kim, 8 s | 230 MB | int8 weights, fp32 compute | 3 564 | 0.445 | 37 dB | 9.079 (−0.04) | 7.4 GB |
| Mega 4 heads, 10 s, **WASM CPU, 8 threads** | 305 MB | fp32 | 29 120 | 2.91 | 113 dB (mask) | – | renderer 4.3 GB |
| Kim, 8 s, WASM CPU, 8 threads | 914 MB | fp32 | **fails**: `std::bad_alloc` (4 GB WASM heap) | | | | |

Also measured:
- **Session creation:** 1.2–2.6 s. It was 8 s for the 30k-node budget-256 20 s graph.
- **JS STFT** (main thread): 87–95 ms per 8 s chunk and 166–184 ms per 20 s chunk. It matches `torch.stft` to 136.7 dB.
- **JS mask + iSTFT:** 88–98 ms for Kim's 8 s chunk and 676–893 ms for Mega's 4 stems × 20 s.
- **Reference speeds on the same chunk** (this Mac, 4 threads, shared machine):
  - PyTorch CPU: Mega core 43–52 s per 20 s chunk, Kim 9–11 s.
  - ORT native CPU: Mega 41.7 s, Kim 12.1 s.
  - WebGPU fp16n is about 5× faster than both of them per chunk (Mega 20 s: 9.2 s).
- **Buffer cache.** `storageBufferCacheMode: 'lazyRelease'` cut the footprint by 15–20 % at the same speed. `'disabled'` crashed ORT ("Buffer used in submit while destroyed").
- **Precision path.** The fp16 path runs at 1.55–1.6× the fp32 speed. fp32 accumulation in fp16 MatMuls changed nothing, so Mega's 32–41 dB is fp16 rounding elsewhere (attention, softmax), and harmless by uSDR.
- **Slicing budget and speed.** Budget 1024 vs 256 made no speed difference, but doubled the memory. Smaller budgets were not tried.

**Effective throughput.**
- Mega's backbone is 5.04 TMAC (10.1 TFLOP) per 20 s chunk (round 5), so 14.85 s fp32 is **0.68 TFLOPS, ~15 % of the M1 Pro GPU's ~4.5 TFLOPS fp32 peak**.
- Kim works out the same: ~2.4 TFLOP per 8 s chunk by my count, in 3.55 s, so 0.67 TFLOPS.
- fp16n reaches ~1.05–1.1 TFLOPS.

### Whole song (measured)

Cristina Vane – So Easy (MUSDB test, 254.1 s), `song.html`:
- Kim fp16n on 8 s chunks, then Mega fp16n with 4 heads on 10 s chunks of mix − vocals.
- Overlap 2 for both, storage buffer cache `lazyRelease`.

| | fp16n (recommended) | fp32 |
|---|---|---|
| total, decode → 6 stems | **379 s = 1.49 s per audio-s** | 566 s = 2.23 s per audio-s |
| Kim: 66 chunks, GPU / JS DSP | 153 s / 10 s (2.32 s per chunk, max 2.52) | 235 s / 11 s (3.54 s per chunk; the first took 4.54) |
| Mega: 53 chunks, GPU / JS DSP | 187 s / 22 s (3.53 s per chunk, max 3.56) | 290 s / 22 s (5.46 s per chunk, max 5.74) |
| model fetch + session (local server) | 0.55 + 1.7 s (Kim), 0.19 + 2.1 s (Mega) | |
| peak footprint, GPU / renderer process | 2.5 GB / 3.2 GB | 4.5 GB / 4.2 GB |
| uSDR vs MUSDB truth: vocals / drums / bass | **15.08 / 15.06 / 9.63** | 15.08 / 15.07 / 9.63 |

- **No thermal or memory drift.** The first and last of 119 chunks took the same time (fp16n), and the fp32 run's 10 min held steady too, and the M1 Pro has fans. A fanless MacBook Air would throttle; that is untested.
- **The JS DSP is serial on the main thread:** 33 s, 9 % of the total. A worker pipelined with the GPU would hide it.
- **The renderer's 3.2 GB** is mostly whole-song Float32 buffers (6 stems × 2 ch × 254 s) plus the model bytes. Streaming the output and dropping the model bytes after session creation would shrink it.

### Full-song projection for other settings (estimate, from the chunk times above)

4-min song (240 s), MSST chunking (chunks = (L + chunk) / step):
- **fp16n as measured** (Kim 8 s, Mega 10 s, overlap 2): 62 × 2.32 + 50 × 3.53 = 320 s of GPU + ~30 s of DSP ≈ **5.8 min**.
- **fp32:** 62 × 3.54 + 50 × 5.46 = 492 s + DSP ≈ **8.7 min**.
- **Mega 20 s chunks** (the round-5 setting): 26 × 9.17 = 238 s for Mega instead of 177 s, so +1 min in fp16.
  - Per the README, 10 s chunks keep all but 0.03 dB of Mega's guitar (10 s is Mega's training length).
- **Mega at overlap 1** (README: −0.12 dB guitar) halves Mega to ~90 s, so ≈ **4.3 min** in fp16.

**Download.** Kim fp16 457 MB + Mega 4 heads fp16 154 MB = **611 MB**, once, cached in the Cache API or OPFS.
- With int8 weights it would be 230 + 79 = **309 MB**, measured at −0.04 dB uSDR (Kim) and −0.03 (Mega drums) with fp32 compute.
- int8 weights feeding fp16 compute weren't tried.
- Kim is 228 M parameters, so 914 MB fp32 and 457 MB fp16, not 228 MB.

## Which devices qualify (estimate)

**Method.** I scaled the measured 4-min fp16n time (~5.8 min on the M1 Pro at ~4.5 TFLOPS fp32 peak) by fp32 peak, assuming the same ~15–25 % efficiency.

**Caveats:**
- This is an estimate. Efficiency differs by vendor and driver, and D3D12 / Vulkan were not tested.
- The real gate must be the on-device benchmark below.
- The peaks come from the research section's aggregator figures.

**Memory needed:** ~2.5 GB of GPU or unified memory plus ~1–3 GB for the tab.

| device class | fp32 peak | 4-min song, fp16n (estimate) | memory | verdict |
|---|---|---|---|---|
| Apple M1 Pro (14c), **measured** | ~4.5 | **5.8 min** (measured 6.3 min for 254 s) | 16 GB unified | yes |
| Apple M2 Pro / M3 Pro / M4 Pro | 5.7–9 | ~3–4.5 min | 16 GB+ | yes |
| Apple M1 / M2 base (8c) | 2.6–2.8 | ~9–10 min | often 8 GB, tight | opt-in only if the benchmark ETA is acceptable; skip at 8 GB |
| Apple M3 / M4 base | ~3.5–4.3 | ~6–7.5 min | 8–16 GB | yes at 16 GB |
| NVIDIA RTX 3060 / 4060 class (desktop or laptop) | 12.7–15 | ~1.7–2.5 min | 8 GB VRAM | yes |
| AMD Radeon 780M, Intel Arc 140V (2024+ iGPUs) | ~4–4.5 | ~6–7 min | shared, 16 GB+ | yes at 16 GB+ |
| Intel Iris Xe (2020–23 laptops) | ~2 | ~13 min | shared, often 8 GB | no by default; long jobs risk Windows TDR (2 s per dispatch) |
| Phones (A17 Pro / A18 Pro, Adreno 830) | 2.2–3.6 | ~7–12 min | iOS kills WebContent at ~2 GB (forum reports); Android's 256 MiB binding limits are common | **no** |
| no WebGPU, or the WASM CPU path | – | Mega alone 29 s per 10 s chunk on 8 M1 Pro threads: **~25 min** for Mega, and Kim doesn't fit the heap | – | **no**: use the server queue |

## Design sketch for Fretboard

### 1. Gate (no download, under 1 s)

Turn the feature on only if all of these hold:
- `navigator.gpu` exists, `requestAdapter({powerPreference: 'high-performance'})` returns an adapter, and `adapter.info.isFallbackAdapter` is false.
- maxStorageBufferBindingSize and maxBufferSize are both ≥ 256 MiB. The budget-256 export's biggest tensor is ≤ 256 MB; a budget-128 export is the fallback for adapters at the spec defaults, and is untested.
- `shader-f16` is present. Without it, fp32 is 1.6× slower: offer it only if the benchmark ETA still passes.
- It is not a phone: `navigator.userAgentData?.mobile`, or iOS / Android in the UA.
- `navigator.deviceMemory` is ≥ 8 (Chromium only, bucketed; when it is missing, let the benchmark decide).
- If `navigator.getBattery` exists (Chromium only), the device is charging. When it isn't, ask the visitor first.

### 2. Quick benchmark (~1–3 s, a few MB)

- Ship a **probe graph**: Mega's band split plus one time and one frequency transformer layer on a 10 s chunk. That is about 2 M parameters, ~4 MB in fp16, and about 1/12 of a Mega chunk.
- Run it twice and take the second time t.
- **ETA ≈ t × 12 × chunks(Mega) + (Kim estimate scaled the same way) + 10 %.**
  - Calibrate the probe against the full chunk times on a few machines; for the M1 Pro, the target is 3.5 s per Mega chunk.
- Offer the split only if the ETA is ≤ ~15 min **and** the server queue's wait is longer.
- This probe is a design proposal; I did not build or validate it.

### 3. Opt-in UI

Ask in one line: "Split this song on your computer: about N min, uses your graphics card, downloads 611 MB once (309 MB with int8 weights)."
- Never start without that click.
- Show progress per chunk, with Pause and Cancel.
- Otherwise, or on decline, join the existing server queue.

### 4. Running the split without overloading the machine

- **Run everything in a dedicated Worker.** WebGPU and ORT-web work in workers.
  - Put the JS STFT / iSTFT in that worker or a second one, pipelined with the GPU. It was 9 % of the time, serial on the main thread.
- **One `session.run` per chunk**, 2.3–3.5 s on this Mac. Between chunks:
  - Yield.
  - Check the pause flag.
  - Read `PressureObserver('cpu')` (Chromium only; CPU pressure, not GPU). At *serious*, sleep as long as the last chunk took; at *critical*, stop.
  - **Detect thermal throttling from the chunk times themselves.** If a chunk takes over 1.5× the median of the first three, insert a cool-down of the same length (a 50 % duty cycle).
- **Release Kim's session before creating Mega's.** Use `storageBufferCacheMode: 'lazyRelease'`. Keep the budget-256 exports: their biggest buffer is ≤ 256 MB, and each dispatch is short enough for the Windows 2 s TDR on a fast GPU. On slow iGPUs, export a smaller budget.
- **Hold a Web Lock** while running, so Chrome's Energy Saver doesn't freeze the tab.
  - Background tabs may still be throttled. Say "keep this tab open".
  - Persist progress (finished chunks' masks, or the output so far) in OPFS so a reload resumes.
- **Cache the models** in the Cache API or OPFS, and call `navigator.storage.persist()`.
  - Safari evicts script storage after 7 days without a visit. A re-download is acceptable.

### 5. What the client uploads

- **The stems as Opus** via WebCodecs `AudioEncoder`: vocals, drums, bass, guitar, keys, at ~160–192 kbps each, about 25–30 MB for a 4-min song. Safari's WebCodecs Opus encoder is unverified.
- The runner turns the five stems into the existing `multi.v*.webm` (Opus, mapping family 255) and computes "other" itself. That costs seconds of CPU.
- **A commitment manifest:**
  - the recipe and model hashes (ONNX sha256), the chunk settings and precision;
  - the sha256 of the decoded input PCM;
  - for every chunk and stem, the sha256 of its **lossless** output, kept in OPFS until verified.

### 6. Trust: making client stems safe to show to everyone

The threat is a client uploading wrong, garbage or abusive audio (anything in place of stems), or a buggy GPU or driver producing garbage.

**Checks** (vCPU figures are estimates from round 5's x86 1-thread rates: Kim 4.9 CPU-s per audio-s of chunk; Mega 4 heads ≈ 36 CPU-s per 10 s chunk):

| check | what it catches | cost per song |
|---|---|---|
| **a. Random-chunk challenge** (recommended) | After the commitment arrives, the server picks one Kim chunk and one Mega chunk at random. The client uploads those windows **losslessly** (float or FLAC: the input PCM, its vocals, the stems; ~4–8 MB), and they must hash to the commitment. The verifier then: checks the input window against its own decode of the source (≥ 60 dB SNR after aligning by cross-correlation, which also catches decoder priming offsets); recomputes both chunks (Mega on the client's committed instrumental); compares each stem at an SNR threshold; and checks that the Opus upload decodes to within codec noise of the lossless window. | **~75 CPU-s ≈ 1.2–2.5 vCPU-min** (container vCPUs may be slower than the i5 the rates came from); **0 on the owner's Mac runner** when it is online (Rust Accelerate, seconds) |
| **b. Thresholds** | Honest fp32 vs PyTorch: 59–119 dB. Honest fp16n: 32–45 dB. Different chunk settings: 14–23 dB (README), so the recipe must match exactly. Garbage or a different model: ≲ 10 dB. **Accept at ≥ 25 dB** on every stem of both chunks. | – |
| **c. More chunks** | Partial tampering. If a fraction p of chunks is altered, k challenged chunks catch it with probability 1 − (1−p)^k. Fabricating the whole song is caught with k = 1. Raise k for new, unknown or low-reputation clients, or for songs that get played a lot. | k × the above |
| **d. Agreement between two independent clients** | A different session, IP and account computed the same song. Compare the decoded uploads per chunk (codec noise limits this to roughly 15–25 dB, so the threshold must be calibrated on honest pairs, untested). Two agreeing uploads cover every chunk, not a sample, and need only Opus decodes and a correlation. | a few CPU-s (decode + SNR) |
| **e. Visibility** | Until (a) passes, the stems are shown **only to the uploader**: their own result, playable at once from local memory. They are published to everyone after (a), or (a) + (d) for higher assurance. A failure discards the upload and rate-limits the session (Turnstile cookie / account). | – |

**Budget.**
- At ~1.2–2.5 vCPU-min per verified song, the ~375 included vCPU-min/month cover about **150–300 client-split songs per month**.
- Separating songs on the server instead (~20.5 CPU-s per audio-s × 240 s ≈ 82 vCPU-min) covers about **4–5 songs per month**.
- Container memory: the Rust kimmega pass needs ~1.3 GB RSS (README chain figure). One check of ~2–3 min at 2 GiB is ~0.1 GiB-h, so 25 GiB-h covers ~250 checks.
- Any check the owner's Mac runner does is free. The container is only the fallback when the runner is offline.
- Sending the client the input PCM as FLAC instead (~30 MB per song, served from R2) would remove decoder doubt entirely, at the price of storage and bandwidth. That is the fallback if the decode check in (a) fails often.

### 7. Volunteer chunks (later, if ever)

Splitting a song into independent chunk jobs would let weaker devices contribute one ~3 s chunk each. Several visitors could also finish one song in parallel.

Costs:
- Mega's chunks depend on Kim's vocals, so chunks come in two phases.
- Every contributor needs their own spot checks.
- Redundancy 2 (two volunteers per chunk, then (d)) doubles the client compute but needs no server compute.
- A song stalls until the stragglers finish.

**Not for v1.** A capable visitor finishes a whole song in ~6 min anyway.

## Risks and open questions

- **One machine measured.**
  - Every timing is one M1 Pro in Chrome 158.
  - Windows (D3D12), NVIDIA, AMD and Intel iGPUs, Safari 26 and Firefox are untested. Safari and Firefox run their own WebGPU implementations, and ORT-web's support for them is less exercised.
  - The device tiers are scaling estimates.
- **fp16 quality evidence is thin.**
  - It is one MUSDB song and the stems MUSDB has (vocals, drums, bass), all within 0.004 dB of fp32 over the whole song.
  - Guitar and keys (32–34 dB vs fp32) can't be scored on MUSDB. Score them on round 4/5's 26 Cambridge-MT remixes before shipping.
- **ORT-web fragility.** Three op-level bugs or limits were hit and worked around:
  - many-output Split;
  - the GLU Split with a zero-sized output;
  - `storageBufferCacheMode: 'disabled'` crashing.
  - Pin the ORT version, and keep a browser smoke test on the exported graphs.
- **Memory.** ~2.5 GB GPU plus 1–3 GB renderer is fine on 16 GB machines and marginal on 8 GB. A smaller slicing budget and streaming the outputs should cut it; untested.
- **Download.** 611 MB (fp16) or 309 MB (int8 weights) is large for a casual visitor. It only suits people who will split several songs, or who want a song the queue is slow on.
- **Mega's GPU parity at budget 1024 was 75–87 dB, against 109–119 at budget 256.** Both are far above any threshold, but the accumulation order clearly depends on the slicing. Verification must use **exactly** the client's recipe settings, and threshold well below the fp16 floor.
- **Input decode.**
  - Fretboard serves AAC (m4a). The browser's AAC decoder and its priming handling may differ from the server's ffmpeg.
  - The challenge's input-window check catches that, but a systematic offset would fail every honest client. Measure on Chrome / Safari / Firefox before relying on it, or serve FLAC.
- **Thermals and battery.** Fanless laptops will throttle over ~6 min; the duty cycle in §4 handles it, but it is untested. Running on battery drains it fast.
- **Licences.** Kim's and Mega 53's weights are MIT (round 5), so redistributing them as browser downloads is fine. Some RoFormer ONNX builds on Hugging Face carry NC licences: don't use them.

## Web research (late 2026)

Gathered 2026-11-25; every claim has its source. Where a figure comes from an aggregator or a forum, it says so.

### WebGPU availability

Sources: the gpuweb implementation-status wiki (edited 2026-10-02), https://github.com/gpuweb/gpuweb/wiki/Implementation-Status, and web.dev, https://web.dev/blog/webgpu-supported-major-browsers.

| browser | on by default | not yet |
|---|---|---|
| Chrome / Edge | Windows (D3D12), macOS, ChromeOS since 113; Android 12+ on Qualcomm / ARM / Intel GPUs since 121 (Imagination: Android 16+, Chrome 139); Linux Intel Gen12+ since 144, NVIDIA (driver ≥ 535.183.01, Wayland) since 147 | Android on Samsung Xclipse; other Linux GPUs (flag); Windows on ARM64 (flag) |
| Safari 26 | macOS, iOS, iPadOS, visionOS | – |
| Firefox | Windows since 141; macOS on Apple Silicon since 145 (macOS 26) / 147 (all versions) | Intel Macs, Linux (Nightly only); Android (Beta / Nightly behind a pref) |

**How many visitors have it:** web3dsurvey (https://web3dsurvey.com/webgpu; the collection date isn't shown) says 82.9 % of reports got a working adapter.
- By OS: ChromeOS 92.7 %, macOS 90.9 %, iOS 90.0 %, Windows 88.1 %, Android 69.8 %, Linux 18.0 %.
- By browser: Edge 93 %, Safari 84 %, Chrome 84 %, Samsung Internet 75 %, Firefox 59 %.

### Adapter limits and features

**Spec defaults.** maxBufferSize is 256 MiB and maxStorageBufferBindingSize is 128 MiB. A page gets more only by passing `requiredLimits` to `requestDevice`, and only up to `adapter.limits`. Browsers report the limits in buckets. https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits

**What real adapters report** (web3dsurvey):
- **maxStorageBufferBindingSize** (https://web3dsurvey.com/webgpu/limits/maxStorageBufferBindingSize):
  - Overall, 97 % report ≥ 256 MiB, 92 % ≥ ~819 MiB and 76 % ≥ 1.5 GiB.
  - On Android only 73 % reach 256 MiB, and 52 % reach 512 MiB.
- **maxBufferSize** ≥ 2 GiB on 76 % (https://web3dsurvey.com/webgpu/limits/maxBufferSize).
- **`shader-f16`** on 94.3 % (https://web3dsurvey.com/webgpu/features/shader-f16): iOS 100 %, macOS 99.7 %, Windows 91.3 %, Android 88.5 %, Linux 78.5 %.

**This Mac.** Headless Chrome 158 on the M1 Pro reports:
- an `apple` / `metal-3` adapter;
- maxBufferSize and maxStorageBufferBindingSize of 4 GiB;
- `shader-f16`, `subgroups`, `chromium-experimental-subgroup-matrix` and `timestamp-query`;
- `isFallbackAdapter: false`.

### Timeouts and memory caps

- **Windows TDR.** By default Windows resets the GPU when one piece of work runs longer than 2 s, and more than 5 resets in 60 s crash the system. https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys
- **Chrome's GPU watchdog.** Chrome also has a watchdog that restarts its GPU process. I found no published threshold.
- **What to do about both:** keep each dispatch short and handle `device.lost`.
- **iOS** kills the WebContent process at about 2 GB (ActiveHard 2048 MB on an iPhone 17 Pro). The WebKit GPU process has also been killed for lack of memory on iPadOS 26. Both are forum and bug reports, not documented limits:
  - https://developer.apple.com/forums/thread/823061
  - https://bugs.webkit.org/show_bug.cgi?id=303203

### Separation projects that already run in a browser

| project | model / runtime | reported speed | notes |
|---|---|---|---|
| monteslu/htdemucs-web-onnx (https://huggingface.co/monteslu/htdemucs-web-onnx) | htdemucs, ORT-web WebGPU, mixed fp16 (Conv in fp32) | ~22× realtime back to back on a Radeon RX 7600; ~8× when paced so the page stays smooth | see below |
| monteslu/htdemucs-ft-webgpu (https://huggingface.co/monteslu/htdemucs-ft-webgpu) | htdemucs_ft, 4 × 84 MB fp16 | – | the dynamo exporter gave NaNs, so it used the legacy exporter at opset 17; STFT and ScatterND in the graph break the WebGPU EP |
| bakkot/demucs-js (https://github.com/bakkot/demucs-js) | Demucs | ~3× realtime with WebGPU (hardware not given) | "much, much slower" without WebGPU |
| stemsplit demucs-onnx (https://stemsplit.github.io/demucs-onnx/browser/) | htdemucs, ORT WASM only | M4 Pro: a 7.8 s chunk in ~6 s on 1 thread, ~2.5 s on 4; iPhone 15 Pro: 5–10 s | – |
| aralroca (https://dev.to/aralroca/i-ran-a-neural-network-in-a-browser-tab-to-split-a-song-into-stems-10mk) | htdemucs, ORT WASM | a 4-min song in 3–5 min on a "modern laptop" | – |
| nikhilunni/demucs-rs (https://github.com/nikhilunni/demucs-rs) | Rust Burn + wgpu, WASM + WebGPU | "significantly slower" than native; no numbers | – |
| freemusicdemixer (https://github.com/sevagh/freemusicdemixer.com/releases) | demucs.cpp in WASM | – | ~4 GB per worker |
| audiojs/mel-roformer (https://huggingface.co/audiojs/mel-roformer) | Kim-type Mel-RoFormer, ORT-web 1.30 | M4 Max, one 8 s segment: 2.4 s WebGPU fp32, 3.0 s WebGPU fp16, 9 s CPU; its custom `roformer.js` (subgroup-matrix): 1.5–1.7 s | see below |
| Ryan5453/unblend (https://huggingface.co/Ryan5453/unblend) | BS-RoFormer and Mel-RoFormer in the browser | – | fp16 weights, fp32 compute "to avoid audible degradation"; the weights it ships are labelled CC-BY-NC-SA |

More detail on two of them:
- **monteslu/htdemucs-web-onnx:**
  - fp16 everywhere wrecked drums and bass, so Conv stays fp32.
  - The fp16 normalisation gave NaNs and was pinned to the CPU.
  - The model is split into 21 pieces of 10–30 ms each, with the tensors between them kept on the GPU, to avoid freezes of ~330 ms in the page.
- **audiojs/mel-roformer:** a Softmax over a tensor larger than 128 MiB "came out wrong one run in two". The fix was to keep every tensor under 100 MB.

**What they have in common:**
- Every project keeps STFT and iSTFT outside the ONNX graph. ORT's WebGPU EP has no STFT, DFT or complex types: https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/docs/webgpu-operators.md
- No project I found publishes a BS-RoFormer timing on WebGPU, let alone Mega 53.

### onnxruntime-web

**Keeping data on the GPU and avoiding overhead** (https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html, https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html):
- Tensors can stay on the GPU between runs: `Tensor.fromGpuBuffer` for inputs, `preferredOutputLocation: 'gpu-buffer'` for outputs.
- `enableGraphCapture` replays a recorded run, but only when every shape is static and every kernel runs on WebGPU.
- `forceCpuNodeNames` pins chosen nodes to the CPU.

**Size limits** (https://onnxruntime.ai/docs/tutorials/web/large-models.html):
- 2 GB per protobuf, so larger models need external data.
- The WASM heap tops out at 4 GB.
- Cache the model in the Cache API or OPFS.

**fp16 overflow and NaN reports:** https://github.com/microsoft/onnxruntime/issues/26732

**WebNN** is still behind a flag or in origin trial, and not on Android: https://onnxruntime.ai/docs/tutorials/web/ep-webnn.html

### Detecting capability and staying gentle

- **Fallback adapter.** `GPUAdapterInfo.isFallbackAdapter` (Chrome 136) replaces `GPUAdapter.isFallbackAdapter`, which was removed. Chrome ships no software adapter, so the value is false there. https://developer.chrome.com/blog/new-in-webgpu-136
- **`navigator.deviceMemory`** is Chromium only, needs HTTPS and is bucketed (desktop up to 32, Android capped at 8). https://developer.mozilla.org/en-US/docs/Web/API/Navigator/deviceMemory, https://groups.google.com/a/chromium.org/g/blink-dev/c/vXtAmrVZeDk
- **Compute Pressure** (`PressureObserver`):
  - It shipped in Chrome 125; Firefox and Safari don't have it.
  - It reports the CPU only, not the GPU, in four states: nominal, fair, serious and critical.
  - Only a focused page gets readings.
  - https://developer.chrome.com/docs/web-platform/compute-pressure, https://w3c.github.io/compute-pressure/
- **Battery Status:** Chromium only, and not in workers. https://caniuse.com/battery-status
- **Tab freezing.** Chrome's Energy Saver does not freeze a page that holds a Web Lock, but Android freezes background tabs after 5 min. https://developer.chrome.com/blog/freezing-on-energy-saver
- **Storage quotas.** Chrome allows up to 60 % of the disk. Firefox allows the smaller of 10 % or 10 GiB, or 50 % if storage is persisted. Safari deletes script-written storage after 7 days without a visit. https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria

### Peak GPU throughput, for scaling

Theoretical fp32 peaks, from aggregator sites that sometimes disagree:

| GPU | TFLOPS |
|---|---|
| Apple M1 (8 core) | 2.6 |
| **M1 Pro (14 core), this machine** | **4.6** |
| M2 | 2.8 |
| M2 Pro | 5.7–6.8 |
| M4 | 4.3 |
| M4 Pro | 6.4–9.2 |
| Radeon 780M | ~4.1–4.5 (single issue) |
| Intel Iris Xe 96 EU | ~2 (my arithmetic) |
| Intel Arc 140V | ~4 (my arithmetic) |
| RTX 3060 | 12.7 |
| RTX 4060 | 15.1 |
| Apple A17 Pro | 2.15 |
| A18 Pro | 3.6 |
| Adreno 830 | 3.4 |

Sources:
- https://hmc-tech.com/lists/gpu/apple/family/m-series
- https://flopper.io/compare/apple-m4-vs-apple-m4-pro
- https://www.cpu-monkey.com/en/igpu-amd_radeon_780m
- https://www.gpu-monkey.com/en/compare_gpu-nvidia_geforce_rtx_4060-vs-nvidia_geforce_rtx_3060_8gb
- https://www.cpu-monkey.com/en/igpu-apple_a17_pro_6_gpu_cores

A hand-tuned WebGPU matmul reached about 1 TFLOPS on an M2 Pro, roughly 17 % of its peak: https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel

## How to rerun

**Scratch.** Scratch lives in `/Users/divyansh/work/mac-testing/webgpu`:
- `venv` (uv, Python 3.11.17) with torch 2.6.0, numpy 1.26.4, soundfile, pyyaml, beartype 0.14.1, rotary_embedding_torch 0.3.5, einops 0.8.1, ml_collections, librosa 0.10.2.post1, safetensors, onnx 1.17.0, onnxruntime 1.20.1 and onnxconverter-common;
- `npm install onnxruntime-web@1.31.1 puppeteer-core`.

**Inputs:**
- MSST at `/Users/divyansh/work/mac-testing/Music-Source-Separation-Training`;
- Mega 53 from `research/round4/data/models`;
- Kim's upstream repo and checkpoint from `research/round1/data`.

```bash
node research/webgpu/serve.mjs /Users/divyansh/work/mac-testing/webgpu 8791 &      # 127.0.0.1 only
L="lockf -k /private/tmp/claude-503/heavy.lock"
$L research/webgpu/runs.sh export     # ONNX + reference chunk + fp16/fp16n/w16/q8 variants
$L research/webgpu/runs.sh bench      # one-chunk table (results/bench.jsonl, one JSON per run)
$L research/webgpu/runs.sh song       # whole song, fp16n and fp32
```

**Files:**
- `export_onnx.py`, `convert_f16.py`, `dump_audio.py`: export and variants.
- `page/bench.*`, `page/song.*`, `page/dsp.js`: the pages.
- `run.mjs`, `bench_all.sh`, `serve.mjs`: the driver.
- `analyze.py`, `analyze_song.py`: SNR and uSDR.
