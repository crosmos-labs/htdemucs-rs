# Music source separation: survey for htdemucs-rs (November 2026)

Scope: what could replace or improve Fretboard's two PyTorch Demucs 4.0.1 calls (`htdemucs_6s` ×2 seeded shifts for the mixer and chord chroma; `htdemucs` ×2 seeded shifts for the tab aligner), running on a CPU-only Cloudflare Container (up to 4 vCPU / 12 GiB, CPU billed per active vCPU-second). Priorities: quality, then speed, then build size.

Unless a number is marked *measured here*, it comes from the linked source and has not been reproduced. **SDR on different test sets is not comparable**: MUSDB18-HQ test (50 songs, full length, museval median-of-medians), MVSEP Multisong (100 × 1 min, "global" SDR over the whole clip) and MVSEP Guitar / Piano (hidden stems) give different absolute numbers for the same model.

## 0. What Fretboard actually uses from the stems

Read from `/Users/divyansh/guitar-tabs` (read-only):

| Consumer | Stems it reads | Where |
|---|---|---|
| Mixer (user-facing) | 6s: vocals, drums, bass, guitar; "other" = original − those four (piano is never played on its own) | `server/audio.ts` `OTHER`, `stemsStage` in `server/pipeline/build.ts` |
| Chord alignment | `harmonic.flac` = original − vocals − drums from the 6s split; chroma CQT per beat | `ensureHarmonic` in `server/audio.ts`, `align_chords` in `scripts/sync.py` |
| Tab aligner, voice parts | 4-stem `vocals.mp3` → pYIN pitch track → `sungPitchMatch`, `onSungNote` | `pitch_track`, `align_notes` in `scripts/sync.py` |
| Tab aligner, instrument parts | 4-stem `other` (guitar parts), `bass`, `drums` → onsets, chroma, kick/snare z | `build.ts` line ~537, `thresholds.ts` `tab.approx` |
| Stems gate | sum of the 6 stems vs original (residual dB) | `thresholds.ts` `stems` |

Consequences for research:

- **Vocals** quality drives three things at once: the sung-pitch match, the chord chroma (vocal leakage into `harmonic`) and the mixer. It is the stem to improve first.
- **Drums** matter for the chord chroma (drum leakage) and the kick/snare tempo gates.
- **Guitar** matters only for the mixer today (the "no guitar" backing track). It could also feed the guitar-tab onset gate, which currently reads the 4-stem `other`.
- **Piano** quality is irrelevant today: it is folded back into `other` and `harmonic`.
- The two Demucs calls overlap heavily: both produce vocals, drums and bass. One good 6-stem pass could feed every consumer and halve the compute.

## 1. Licence finding that affects the current pipeline

**The Demucs pretrained weights are probably not MIT.** The code is MIT, but the maintainer wrote in [facebookresearch/demucs#327](https://github.com/facebookresearch/demucs/issues/327) (2022-05-23): "The model weights are not covered by the MIT license, and are provided only for scientific purposes." The statement predates v4 (htdemucs). Nothing later supersedes it, a 2024 follow-up question got no maintainer answer, and the repository was archived on 2025-01-01. A contributor in that thread suggests the restriction comes from training on MUSDB18, whose licence is non-commercial. The weights' re-upload at [adefossez/HTDemucs](https://huggingface.co/adefossez/HTDemucs) carries no licence tag. A third-party licence review (see [xeon_split_audio model-licenses.md](https://github.com/galenoferreira/xeon_split_audio/blob/main/docs/research/model-licenses.md)) reaches the same reading.

This applies to `htdemucs`, `htdemucs_6s` and `htdemucs_ft` as Fretboard uses them **today**, whether from Python or from the Rust port. It is not a reason to stop the port: the port's code is ours, and the same engine can run other weights. But shipping any Demucs checkpoint in a public service is a decision the owner should take knowingly. (This is not legal advice.)

## 2. Candidate models

Legend for weights licence: **green** = an explicit permissive licence from the rights holder; **amber** = permissive tag on a mirror or code-only licence, or terms never stated; **red** = non-commercial, "scientific purposes only", or provenance in doubt.

### 2.1 Multi-stem models (4 or 6 stems)

| Model | Stems | Reported quality | Size / compute | Weights licence | Non-Python implementations |
|---|---|---|---|---|---|
| **htdemucs** (Demucs v4) | 4 | MUSDB18-HQ test: 8.80 avg (drums 10.05, bass 9.78, other 6.42, vocals 8.93) [paper Tab. 1](https://arxiv.org/abs/2211.08553). Multisong: 9.16 avg (vocals 8.24, drums 10.88, bass 11.76, other 5.74) [ZFTurbo list](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md) | **42.0 M params, stored f16, 84 MB** (*measured here* from the safetensors header). Paper: CPU RTF 1.77 on one 2.2 GHz Xeon core (dim 512, 7.8 s segments) | **red/amber**: "scientific purposes only" (see §1) | [demucs.cpp](https://github.com/sevagh/demucs.cpp) (MIT, Eigen), [demucs.onnx](https://github.com/sevagh/demucs.onnx) (MIT, STFT outside the graph), [demucs-wgpu](https://github.com/eclipse005/demucs-wgpu) (Rust CPU + wgpu, MIT), [demucs-rs](https://github.com/nikhilunni/demucs-rs) (Rust/Burn, Apache-2.0), charon-audio and stem-splitter-core (Rust + ONNX Runtime) |
| **htdemucs_ft** | 4 (bag of 4 single-target models) | MUSDB: 9.00 avg (vocals 9.20, drums 10.08, bass 10.39, other 6.32) [paper](https://arxiv.org/abs/2211.08553) | 4 × 84 MB; 4× htdemucs compute for all stems, but **one shard alone gives one stem at htdemucs cost** (noted by demucs-wgpu) | red/amber (as above) | demucs.cpp, demucs-wgpu, demucs-rs |
| **htdemucs_6s** | 6 (+guitar, piano) | Multisong: vocals 8.05, drums 10.22, bass 11.22. **MVSEP Guitar: 5.25** (rank 45). **MVSEP Piano: 2.23** (rank 48). README: "the piano source is not working great" | **55 MB f16 file → ≈27 M params** (*measured here*: file size; config smaller than htdemucs) | red/amber | demucs.cpp, demucs-rs (not demucs-wgpu) |
| **BS-RoFormer SW** ("shared weights") | 6 | Multisong: vocals 11.30, bass 14.62, drums 14.11, other 8.72 [leaderboard](https://mvsep.com/quality_checker/multisong_leaderboard). **MVSEP Guitar 9.01, Piano 7.80**, the best public 6-stem model; scores nearly identical to *Logic Pro 11.2 Stem Splitter* (guitar 9.002, piano 7.789) | ≈0.2 B params; ONNX fp16 353 MB / fp32 701 MB ([elicwhite ONNX](https://huggingface.co/elicwhite/bs-roformer-sw-6stem-onnx)). Stock fp32 ONNX on CPU ≈ 44 s per 13.35 s chunk (host unknown) and **int8 failed the quality bar** on the 6-stem graph ([koidn ONNX](https://huggingface.co/koidn/bs-roformer-sw-onnx)) | **red**: no licence; trainer unknown; a contributor states it "is property of Apple" and comes from Logic Pro (unsourced). Open issue [MSST#248](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/248) has no maintainer reply. MIT tags on ONNX/GGUF mirrors cover their code, not the weights | ONNX (fp16/fp32), GGUF f16 ([studio409](https://huggingface.co/studio409/BSRoformer-SW-6Stem-GGUF)), Core ML, browser via [bs-roformer-web](https://github.com/elicwhite/bs-roformer-web), MLX ports on the leaderboard |
| **SubRoformer** (Nov 2026) | 4 | MUSDB18-HQ: 11.59 avg (vocals 12.69, bass 11.50, drums 12.23, other 9.94); MUSDB-only target-specific models 10.63 ([arXiv 2611.08130](https://arxiv.org/abs/2611.08130), [repo](https://github.com/ArchiMickey/SubRoformer)). **Multisong: vocals 10.96, bass 11.19, drums 11.74, other 7.22**, so its bass is *below* htdemucs there | fp16 207 MB → ≈104 M params; **48 kHz** model; evaluated with 2 s chunks at hop ratio 4 (75% overlap: heavy). GGUF q8_0 and Core ML exports exist in the [HF repo](https://huggingface.co/archimickey/SubRoformer) | **amber**: code MIT, no weights licence on HF; trained on MUSDB + 500 *internal* songs | GGUF, Core ML |
| **SCNet** (Small / Large / XL / XL-IHF) | 4 | MUSDB-only training: Small 9.03 MUSDB / 8.87 Multisong; XL-IHF 10.08 / 9.92 ([ZFTurbo list](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)). Paper: 9.0 dB MUSDB, "CPU inference time is only 48% of HT Demucs" ([arXiv 2401.13276](https://arxiv.org/abs/2401.13276)); ~7× fewer params than BS-RoFormer | Small ≈10 M params | **amber**: official repo code MIT, weights unstated ([starrytong/SCNet](https://github.com/starrytong/SCNet)); MSST releases "ambiguous, leaning MIT" | Core ML (SCNet base on HF); no Rust/C++ found |
| **BS-RoFormer (MUSDB-only, MSST)** | 4 | 9.65 MUSDB / 9.38 Multisong | – | amber (MSST release) | – |
| **MDX23C / TFC-TDF** (MSST, MUSDB-only) | 4 | 7.15 MUSDB | – | amber | ONNX (UVR) |
| **Moises-Light** (WASPAA 2025) | 4 | 9.96 MUSDB avg with ≈5 M params per stem (13× fewer than BS-RoFormer's 72 M) ([arXiv 2510.06785](https://arxiv.org/abs/2510.06785)) | very small | **not released** (paper CC BY-NC-SA) | – |
| **Spleeter** | 2/4/5 | MUSDB 5.91 avg | tiny, fast | **green** (MIT; trained on Deezer's own catalogue) | many (TF Lite, ONNX) |
| **Open-Unmix** umxhq / umxl | 4 | MUSDB vocals 6.25 / 7.21 | small | umxhq: repo MIT, trained on MUSDB18-HQ (amber); **umxl: CC BY-NC-SA 4.0 (red)** ([repo](https://github.com/sigsep/open-unmix-pytorch)) | umx.cpp |
| **Meta SAM Audio** (Dec 2025) | text-prompted, any | no MUSDB SDR; subjective scores and a reference-free "judge" ([repo](https://github.com/facebookresearch/sam-audio), [paper](https://arxiv.org/html/2512.18099v1)) | large flow-matching transformer; multi-step generation: not a CPU fit | SAM License (custom; terms not reviewed) | – |
| Commercial only: MVSep ensembles, MLSLABS, SAMI-ByteDance, Moises, LALAL.AI, Apple Logic Pro, Cubase | | top of the leaderboards | | not available | |

### 2.2 Vocals-only models (could replace the 4-stem pass)

| Model | Multisong vocals SDR | Size | Weights licence | Notes |
|---|---|---|---|---|
| BS-RoFormer 124 bands (MVSep, 2026-07) | 12.33 | – | not released (MVSep service) | current top single model |
| BS PolarFormer (MSST) | 11.00 | – | amber (MSST release; mirror [hunterhogan/bs_polarformer](https://huggingface.co/hunterhogan/bs_polarformer) has no licence) | polar-coordinate position embeddings |
| **Mel-Band RoFormer, Kim** ([HF](https://huggingface.co/KimberleyJSN/melbandroformer)) | **10.98** | `MelBandRoformer.ckpt` 913 MB fp32 (≈228 M params, *measured here*: file size) | **amber-green**: `license: mit` in HF metadata; empty model card; training data not stated | the most cleanly licensed strong vocal model found; GGUF ports via [BSRoformer.cpp](https://github.com/chenmozhijin/BSRoformer.cpp) |
| BS-RoFormer viperx ep_317 (12.97 on viperx's own set) | 10.87 | 639 MB ckpt (*measured here*: HTTP content-length) | **red/amber**: never licensed; first released on a paywalled channel | widely used via UVR |
| Community fine-tunes (unwa, Gabox, becruily, SYH...) | 10.9–11.3 | 200–900 MB | mixed: some GPL-3.0, CC BY-NC-SA, or "non-commercial per author" | unwa's "Resurrection" is the one turbo-roformer optimised |
| MDX23C vocals | 10.17 | – | amber | |
| UVR MDX-Net Voc FT / Inst HQ | lower than RoFormers (not looked up) | ~50–70 MB ONNX | **green-ish**: UVR README asks for MIT-style credit for UVR-team models | ONNX already |
| htdemucs_ft vocals shard | (MUSDB 9.20) | 84 MB | red/amber | same engine as our port |
| htdemucs (4-stem) | 8.24 | 84 MB | red/amber | today's baseline |

### 2.3 Guitar and piano

The MVSEP Guitar leaderboard (30 × 1 min, acoustic and electric) and Piano leaderboard (50 × 1 min) are the only public benchmarks found ([guitar](https://mvsep.com/quality_checker/leaderboard/guitar/), [piano](https://mvsep.com/quality_checker/leaderboard/piano/)). The best guitar scores are about 9.1 (BS-RoFormer SW derivatives, MLSLABS, Logic Pro). Dedicated open models score 5–7.4 (MVSep Guitar Mel/BS-RoFormer ensembles 7.1–7.5, Moises 6.94, becruily Mel-RoFormer guitar 5.3–5.6). **htdemucs_6s scores 5.25** on guitar and **2.23** on piano. ZFTurbo's pretrained list has no open guitar or piano model ([list](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)). The becruily guitar model is non-commercial per its author. MVSep's own guitar models are served by MVSep, not released.

**Bottom line for guitar: no strong guitar model with a clean licence exists.** The only big jump (5.25 → 9.0) is BS-RoFormer SW, and its provenance is red. Realistic routes: (a) keep htdemucs_6s guitar for the mixer; (b) ask MVSep / ZFTurbo about licensing a guitar model; (c) train or fine-tune our own guitar head (needs licensed multitrack data; see §4).

### 2.4 Restoration, ensembles, generative refinement

- **Ensembles** dominate the leaderboards (the MVSep ensemble tops Multisong), usually 0.2–0.6 dB over the best member. For us the relevant cheap ensemble is **averaging the vocals/drums/bass we already get from both Demucs passes** (see the A/B plan).
- **Apollo** (MP3/low-bitrate restoration, MSST list) and diffusion/consistency refinement on top of BS-RoFormer (+0.07 dB, [arXiv 2412.06965](https://arxiv.org/abs/2412.06965)) improve perceptual quality at large compute cost. Not worth it for alignment; maybe later for the mixer.
- Generative separators (FLOMSS flow matching [2610.17821](https://arxiv.org/abs/2610.17821), discrete-token LM separation [2604.09371](https://arxiv.org/html/2604.09371)) are competitive perceptually but not on SDR, and are multi-step. Not a CPU fit.
- Leaderboard entries show the cost of cheaper inference on BS-RoFormer SW: **overlap 4 → 2 costs ≈0.15 dB on guitar** (9.12 → 8.97); **fp16 on MLX costs ≈0.0 dB** (O2 fp16 8.9736 vs fp32 8.9740); TTA (test-time augmentation, as in Demucs shifts) adds ≈0.03 dB.

## 3. Implementations and CPU inference tricks

| Project | What it shows |
|---|---|
| [demucs.cpp](https://github.com/sevagh/demucs.cpp) (MIT, C++17 + Eigen + OpenMP) | htdemucs, 6s, ft, v3. ggml-format f16 weights (53 MB for 6s, 81 MB for 4-stem). Quality "practically identical" to PyTorch. **Speed** ([PERFORMANCE.md](https://github.com/sevagh/demucs.cpp/blob/main/.github/PERFORMANCE.md), Ryzen 5950X, 4-min song, 4-stem): single-threaded segments 10 m 23 s wall; 4 segment threads × 4 BLAS threads 4 m 9 s (19 min user CPU); "V3" 4 threads 2 m 36 s (10.5 min user CPU). Lessons: many small matmuls make GPUs and BLAS threading poor fits; **parallelism across segments beats threading inside a GEMM**; watch OpenBLAS oversubscription. |
| [demucs.onnx](https://github.com/sevagh/demucs.onnx) (MIT) | STFT/iSTFT moved outside the graph so the core exports; minimal ORT build. Claims "significantly faster" than demucs.cpp; no numbers. |
| [demucs-wgpu](https://github.com/eclipse005/demucs-wgpu) (Rust, MIT) | Hand-written CPU (rayon + gemm) and wgpu paths for htdemucs/htdemucs_ft. **CPU: 176 s track in 48.7 s (3.6× real time)**, threads and CPU not stated. Parity ≈125 dB SNR vs PyTorch. ft weights stored f16, upcast to f32. The closest prior art to our port. |
| [demucs-rs](https://github.com/nikhilunni/demucs-rs) (Rust + Burn, Apache-2.0) | All three v4 models incl. 6s, f16 safetensors, WebGPU/Metal/Vulkan. The author reports "subtle loss of signal" from small DSP deviations and built a layer-by-layer bench. Lesson: **per-layer parity tests against PyTorch are essential**. |
| [BSRoformer.cpp](https://github.com/chenmozhijin/BSRoformer.cpp) (MIT, GGML) | BS-/Mel-RoFormer inference, F16/Q8_0/Q5/Q4 GGUF, own radix-2 FFT aligned with torch.stft, golden tests. No published speed or quality-loss numbers per quant. |
| [turbo-roformer](https://github.com/LoggeL/turbo-roformer) (MIT) | BS-RoFormer (single stem, unwa "Resurrection", 204 MB) on CPU. **Stock fp32 RTF 4.88 → 0.73** on an 8-core Zen 4 (4:12 song in 3:05) via bf16 GEMMs (oneDNN, fp32 accumulate), an AVX512-BF16 attention kernel, fused QKV/RoPE/RMSNorm epilogues, malloc tuning. bf16 error ≈54 dB below signal. Thread scaling: RTF 1.35 at 4 threads, 0.75 at 8; SMT no help. **Requires AVX512-BF16** (Zen 4+, Sapphire Rapids+); otherwise ≈3.9× slower. Overlap 1.5 → RTF 0.56; overlap 3 → 1.09. |
| HTDemucs INT8 ONNX ([itamiArika](https://huggingface.co/itamiArika/htdemucs-int8-memory)) | Self-reported **−0.03 dB MUSDB avg (8.34 vs 8.37) with int8 only on the cross-transformer matmuls**; full int8 "destroys separation quality". OpenVINO cut memory to 0.98 GB vs 2.6–3.9 GB for ORT. Unverified. |
| HTDemucs ONNX f16 ([StemSplitio](https://huggingface.co/StemSplitio/htdemucs-onnx)) | f16 *storage* halves the file but gives no CPU speedup (upcast at load). |

Takeaways for the Rust port:

1. **Store f16, compute f32** is free in quality and halves the download (Demucs files already are f16). bf16 *compute* helps only on CPUs with AVX512-BF16/AMX. We don't know Cloudflare's container CPU model, so this must be probed in the container (`/proc/cpuinfo` flags).
2. **int8** is only safe on the transformer matmuls (≈−0.03 dB self-reported for HTDemucs), and it failed outright on the large 6-stem RoFormer. Treat it as a late optimisation with a quality gate.
3. **Overlap and shifts are the biggest knobs.** Cost scales as `shifts × 1/(1−overlap)`. Quality gains are small and per-stem: ≈0.15 dB for overlap 2→4 on RoFormer guitar. For Demucs, Fretboard measured +0.05 sung-pitch match from 0 → 2 shifts on one song.
4. **Parallelise across segments**, one segment per core, rather than threading inside small GEMMs (demucs.cpp, turbo-roformer thread scaling).
5. **Parity first.** Per-layer comparison against PyTorch (demucs-rs's lesson; demucs-wgpu reports ≈125 dB SNR).
6. A band-split RoFormer is architecturally **simpler** than HTDemucs (STFT → per-band linear → alternating time/frequency transformer with RoPE → per-band MLP mask → iSTFT; no waveform branch). If the port's STFT, attention and GEMM pieces are generic, a RoFormer is a modest addition, not a rewrite.

## 4. What "quality" should mean for us

### 4.1 Objective separation metrics on ground truth

- **SDR, MDX/MVSEP style** ("global" SDR = 10·log10(Σs² / Σ(s−ŝ)²) per track and stem): cheap, and the metric the MVSEP leaderboards use. Make it the primary number.
- **museval SDR (BSS Eval v4, median of 1-s frames, median over tracks)** for comparability with papers. It is slow, so run it only for final numbers.
- **SI-SDR** as a secondary check (insensitive to gain).
- Always report **per stem** and with **paired per-track differences + bootstrap CI**: averages hide that a model can win on vocals and lose on bass (SubRoformer: +2.7 dB vocals but −0.6 dB bass vs htdemucs on Multisong).
- SDR does not capture everything downstream: onset timing tracks SI-SDR (ρ = 0.62), but drum attack and dynamics distortion barely correlate with it (|ρ| ≤ 0.29), and model rankings invert ([arXiv 2609.04224](https://arxiv.org/abs/2609.04224)). That is why the downstream checks below matter.

### 4.2 Data

| Set | What | Size | Licence / access | Use |
|---|---|---|---|---|
| **MUSDB18-HQ** ([Zenodo 3338373](https://zenodo.org/records/3338373)) | 150 songs (100 train / 50 test), 4 stems, WAV | **22.7 GB zip** (no test-only download) | "Other (Non-Commercial)", educational use; some tracks CC BY-NC-SA. Direct download | 4-stem SDR. **Caveat**: Demucs, SCNet and most community models train on MUSDB18 *train*, so test is fair for them; `mdx_extra` and some community models trained on *test* too |
| MUSDB18-HQ test subset | e.g. 12 of the 50 test songs (balanced genres), full length, or 60 s excerpts of all 50 | ~2 GB of WAV after extraction | same | quick loop; the full 50 for final numbers |
| **MVSEP Multisong** | 100 × 1 min, 5 stems, stems hidden | mixtures **1.8 GB** | evaluated by uploading outputs; **results appear on a public leaderboard** | apples-to-apples with the community |
| **MVSEP Guitar** | 30 × 1 min, acoustic + electric | **0.5 GB** | upload; public leaderboard | the only guitar benchmark found |
| **MVSEP Piano** | 50 × 1 min | **0.9 GB** | upload; public leaderboard | piano (low priority) |
| **MVSEP MUSDB18 (Test) checker** | upload separations of the 50 test songs | – | public leaderboard | cross-check of local museval |
| **MoisesDB** ([repo](https://github.com/moises-ai/moises-db), [paper](https://arxiv.org/abs/2307.15913)) | 240 tracks, ≈14.4 h, 11 top-level stems incl. **guitar and piano/keys**, mostly pop/rock | official size not published (one third-party tool estimates ~149 GB; unverified) | **CC BY-NC-SA 4.0**, access by request at music.ai/research | local guitar/piano ground truth; download a subset if possible |
| Slakh2100 / GuitarSet | synthesized multitracks (Slakh, CC BY 4.0); solo guitar (GuitarSet, CC BY 4.0) | large / small | permissive | only if we ever train our own guitar model (synthetic domain gap) |

All of these are research/non-commercial data. Using them to *evaluate* internally is a different question from shipping weights trained on them, but the owner should know none of them is commercial-clean.

### 4.3 Fretboard downstream checks (no ground truth needed)

These are already computed by `scripts/sync.py` and gated in `server/pipeline/thresholds.ts`:

| Metric | Reads | Current reference values |
|---|---|---|
| `sungPitchMatch` (tab notes on the sung pitch class, pYIN on the vocals stem) | 4-stem vocals | First Love 0.683 (0 shifts) / 0.714–0.738 (1 random shift) / **0.734 (2 seeded)**; gate 0.6 |
| `onSungNote` | vocals | – |
| `onsetWithin60ms` (instrument parts) | 4-stem `other` / `bass` | Wicked Game lead 0.80, Nirvana 0.84–0.92; gate 0.7 |
| `kickSnareZ`, `kickTempo` ratio | 4-stem drums | Aerials z 15.56; gate 3 |
| chords `spotRank`, `spotBass`, `fitOverEven` | `harmonic` = mix − 6s vocals − 6s drums | spotRank Wicked Game 1.00, First Love 0.94; spotBass 0.89–0.99 |
| stems residual dB | sum of 6s stems | −20.5 to −25.7 dB; First Love −14.8 |

Caveats: there are only 11 songs in `data/songs` (two polished: Wicked Game, First Love). The numbers move by ±0.03 from shift randomness alone, and the thresholds were calibrated on these same songs. Use them as **no-regression gates and tie-breakers**, not as the main ranking signal. `npm run song -- verify <id>` rebuilds from scratch and compares within `VERIFY_TOLERANCE` (0.005 on fractions). For A/B we only need the aligner part: `scripts/sync.py` takes a JSON request on stdin with stem paths (`vocals`, `other`, `chroma`), so a harness can swap stems without touching guitar-tabs (see the A/B plan).

### 4.4 Listening

The mixer is user-facing. For the final 2–3 candidates, a blind A/B listen of the "no vocals" and "no guitar" mixes on 5–6 songs catches artifacts SDR misses (bleed, phasiness, chunk-boundary clicks).

## 5. Uncertainties and what was not checked

- CPU speed of every model on Cloudflare's actual container CPU is unknown. No source reports a like-for-like CPU RTF across HTDemucs, SCNet and RoFormers.
- The Kim Mel-RoFormer's MIT tag is metadata on an empty card. Its training data (likely including MUSDB) is unknown.
- The BS-RoFormer SW provenance claims (Apple/Logic Pro) are unsourced, but its scores match Logic Pro's to three decimals.
- MoisesDB size, and whether a subset can be downloaded, are unverified.
- SubRoformer's Multisong row is a single leaderboard entry (submitter unknown).
- "Not found" means not found in this search pass. Discord-hosted community models (the main channel for UVR/MVSEP releases) were not searched.
