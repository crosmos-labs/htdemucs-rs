# Can visitors run stem separation in their own browser? (WebGPU feasibility)

Date: 2026-11-25. Branch `webgpu`. Status: DRAFT, measurements pending.

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
