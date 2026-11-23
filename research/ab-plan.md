# A/B plan: what to try, how to measure it, what to run first

Companion to [survey.md](survey.md). Status: Round 1 done, see [round1/round1.md](round1/round1.md). It found the 4-stem split 25 ms late (a pipeline bug); variant 1 (6s-only / average) is not a clean win; Kim's vocals show no downstream gain at ≈2× CPU. MUSDB18-HQ 12-track subset fetched (approved): next is SDR. Original plan below, unchanged.

Status at writing: **plan only**. Nothing here has been run yet, except the small checks the survey marks *measured here*.

## 1. Decision rules (fixed before any result)

- **Primary metric:** per-stem global SDR (MVSEP/MDX definition), paired per track against the baseline, reported as median Δ with a 95% bootstrap CI. A variant "wins a stem" when the CI excludes 0 and the median Δ ≥ +0.2 dB. It "loses" when the median Δ ≤ −0.1 dB.
- **Stem weights for Fretboard:** vocals first (aligner pitch, chord chroma, mixer), then drums (chroma leakage, kick/snare gates), then guitar (mixer only), then bass. Piano doesn't count today.
- **Downstream gate (must not regress):** on the 11 songs in `guitar-tabs/data/songs`, every verdict stays the same. `sungPitchMatch`, `onsetWithin60ms`, `spotRank` and `spotBass` may not drop by more than 0.03, which is the shift-noise spread measured on First Love. A downstream *gain* is a tie-breaker, never the main evidence (11 songs, thresholds tuned on them).
- **Cost:** vCPU-seconds per 4-minute song, measured with 4 threads, plus peak RSS (must stay well under 12 GiB).
- **Licence:** a variant with **red** weights can be measured as a reference ceiling but is never recommended for shipping. **Amber** needs a written answer from the rights holder first.
- **Determinism:** every variant must be repeatable (seeded or no shifts), as the pipeline's `verify` requires.

## 2. Cost model (estimates; the first run replaces them with measurements)

vCPU-seconds ≈ `song_seconds × passes × 1/(1 − overlap) × k`, where `k` is core-seconds per audio-second for one forward pass of one model.

- **htdemucs**: k ≈ 1.5–2.5. Sources: the paper's 1.77 on one 2.2 GHz Xeon core; demucs.cpp's 10.5 min user CPU for a 4-min song at overlap 0.25. The 6s model is smaller (≈27 M vs 42 M params): k ≈ 1.2–2.
- **Today's recipe** (htdemucs ×2 shifts + 6s ×2 shifts, overlap 0.25): 240 × 1.33 × (2×2 + 2×1.6) ≈ **2,300 vCPU-s (range 1,200–3,000)** ≈ 38 vCPU-min ≈ **$0.046 CPU** at $0.000020/vCPU-s, plus memory (12 GiB × ~10 min wall ≈ $0.018). Wall ≈ 8–12 min on 4 vCPU.
- **BS/Mel-RoFormer, stock fp32**: turbo-roformer measured RTF 4.88 on 8 Zen 4 cores at overlap 2 for a ~51 M-param single-stem BS-RoFormer, i.e. ≈39 core-s per audio-s. Its optimised bf16 path needs AVX512-BF16 and reached ≈5.8 core-s per audio-s. Kim's Mel-RoFormer is bigger (913 MB fp32 checkpoint). **Estimate 5–40 core-s per audio-s → 1,200–9,600 vCPU-s for vocals alone (≈$0.02–0.19)**. This range is too wide to plan on, so it is the first thing to measure.

## 3. Ranked shortlist

| # | Variant | Hypothesis | Est. cost / 4-min song (vCPU-s) | Download | Licence risk |
|---|---|---|---|---|---|
| **1** | **Demucs restructure**: (a) one `htdemucs_6s` pass feeds everything (aligner reads 6s vocals/drums/bass and guitar+other); (b) keep both passes but **average** the two vocals/drums/bass estimates | (a) costs ≈45% of today with ≤0.2 dB vocals loss (Multisong: 6s vocals 8.05 vs 4-stem 8.24) and no downstream regression. (b) The free two-model ensemble gains +0.2–0.4 dB on vocals/drums at no extra separation cost | (a) ≈1,000; (b) ≈2,300 (same as today) | 0 (weights already cached) | unchanged from today (Demucs = red/amber, survey §1) |
| **2** | **Knob sweep** on htdemucs and 6s: shifts {0, 1, 2, 4} × overlap {0.1, 0.25, 0.5}; weights f16-stored/f32-compute vs bf16 compute vs int8 transformer matmuls | Overlap 0.1 and 1 shift lose < 0.1 dB; f16 storage is lossless; int8 on transformer matmuls loses ≈0.03 dB (self-reported elsewhere). Sets the Rust port's defaults and tells us how much speed is free | sweep itself ≈ 12 configs × 12 songs × ~2 min ≈ 5 h of Mac CPU; result could cut production cost 30–60% | 0 | none new |
| **3** | **Mel-Band RoFormer (Kim) for vocals**, replacing the 4-stem pass. Also feeds `harmonic` = mix − RoFormer vocals − 6s drums | +2.7 dB vocals on Multisong (10.98 vs 8.24). Better sung-pitch match and cleaner chord chroma (less vocal leakage) | 1,200–9,600 (must measure); on top of variant 1a's ≈1,000 | 913 MB fp32 (≈460 MB as f16) | **amber-green**: MIT tag on HF metadata, empty card, training data unknown. The best-licensed strong vocal model found |
| **4** | **htdemucs_ft vocals shard** in place of the 4-stem pass | MUSDB vocals 9.20 vs 8.93 (+0.27 dB) at one-pass cost. Our Rust engine runs it unchanged | ≈ same as the 4-stem pass (≈1,300 with 2 shifts) | 84 MB | same as Demucs (red/amber) |
| **5** | **Efficient 4-stem alternatives**: SCNet-XL-IHF (MSST) and SubRoformer | SCNet: ≈ +0.8 dB avg over htdemucs (Multisong 9.92 vs 9.16) at ≈50% of HTDemucs CPU time (paper). SubRoformer: +2.7 dB vocals, +0.9 drums, −0.6 bass on Multisong; 48 kHz; heavy overlap at eval | SCNet ≈500–1,000; SubRoformer unknown (≈104 M params, 2 s chunks, 75% overlap → likely > RoFormer) | SCNet XL ≈ hundreds of MB (unchecked); SubRoformer 207 MB f16 | **amber** for both: ask ZFTurbo (SCNet release) and ArchiMickey (SubRoformer weights) for a written licence before investing port work |
| **6** | **BS-RoFormer SW 6-stem**, as a **quality ceiling only** | Tells us how much headroom exists: Multisong vocals 11.30, drums 14.11, bass 14.62; guitar 9.01 vs 6s 5.25 | high (≈0.2 B params; int8 failed on it) | 353 MB fp16 ONNX | **red**: no licence, provenance disputed (claimed Apple/Logic Pro). Measure, never ship |

**Guitar specifically:** no licence-clean model beats htdemucs_6s by much. The step change (5.25 → 9.0 on MVSEP Guitar) exists only in red-licensed weights. The plan keeps 6s guitar, measures the gap (variant 6), and lists two follow-ups that need the owner: ask MVSep/ZFTurbo about licensing a guitar model, or fine-tune a guitar head on licensed data (a training project, not an inference one).

**Not on the list, and why:** Spleeter / Open-Unmix (licence-cleaner but 2–3 dB worse than today); SAM Audio and generative separators (multi-step, not CPU-viable); Apollo / diffusion refinement (perceptual gains, large cost, no benefit to alignment); community vocal fine-tunes with NC/GPL licences.

## 4. How each run is measured

**Harness (to build in `research/`; never touching guitar-tabs or the port's sources):**

1. `research/eval/sdr.py`: global SDR and SI-SDR per stem and track, museval optional; paired bootstrap Δ vs baseline; writes one JSON per (variant, track).
2. `research/eval/downstream.ts` (or `.py`): for each of the 11 songs, builds the same aligner requests the pipeline sends (`chordsRequestFor` / `tabRequestFor` from `guitar-tabs/server/pipeline/build.ts`, imported read-only), swaps in the variant's stem paths (`vocals`, `other`, `chroma`), pipes them to `guitar-tabs/scripts/sync.py` in its pinned uv env, and records `sungPitchMatch`, `onSungNote`, `onsetWithin60ms`, kick/snare z, `spotRank`, `spotBass`, `fitOverEven`. `harmonic` for a variant = original − variant vocals − variant drums (same ffmpeg recipe as `ensureHarmonic`).
3. `research/eval/cost.sh`: runs a separation with 4 threads (`OMP_NUM_THREADS=4`, `torch.set_num_threads(4)`), records user+sys CPU seconds and peak RSS (`/usr/bin/time -l`). Later, the same in the Cloudflare container, plus its `/proc/cpuinfo` flags (AVX-512, AVX512-BF16, AMX decide whether bf16 is worth anything).

**Data:**

- 4-stem SDR: **MUSDB18-HQ test**, 12-song subset for the loop, all 50 for final numbers. Download 22.7 GB (zip; no test-only download), non-commercial/educational licence.
- Guitar/piano SDR: **MVSEP Guitar (0.5 GB mixtures, 30 × 1 min) and Piano (0.9 GB, 50 × 1 min)** by uploading outputs. **Results appear on a public leaderboard**, so this needs the owner's OK. Alternative: a MoisesDB subset locally (CC BY-NC-SA 4.0, access by request; full size unverified, possibly >100 GB).
- Downstream: the 11 committed songs, with recordings and both Demucs splits **already cached** in `guitar-tabs/.cache/stems` (1.8 GB). Variant 1 needs **no new separation at all**.

## 5. What to run first

**Run 1 (cheap, no downloads, ≈1–2 h of Mac time): variant 1 on downstream metrics + cost baseline.**

1. From the cached stems, build three stem sets per song: today's (4-stem vocals/other/drums + 6s harmonic), 6s-only (6s vocals/drums/bass, other = 6s guitar+other), and the two-model average for vocals/drums/bass. Run the downstream harness on all 11 songs.
2. Measure today's real CPU cost: re-split 2 songs with PyTorch Demucs at 4 threads (htdemucs and 6s, 2 seeded shifts) and record vCPU-s and peak RSS. This replaces the §2 estimate.
3. Time Kim's Mel-RoFormer on one 60 s excerpt at 4 threads (PyTorch via MSST, fp32). The checkpoint is 913 MB, under the 1 GB limit. This one number decides whether variant 3 is a $0.02 or a $0.20 per-song proposition.

**Run 2 (needs approval: 22.7 GB MUSDB18-HQ download, ~5–8 h CPU):** SDR for the baseline, variant 1a/1b, the knob sweep (variant 2) and variant 4 on the 12-song MUSDB subset; Mel-RoFormer vocals (variant 3) if Run 1's timing is tolerable.

**Run 3 (needs approval: public MVSEP submissions, or MoisesDB access):** guitar SDR for 6s vs variant 6's ceiling. This quantifies what the licence problem is costing us on guitar.

**Owner decisions needed before shipping anything** (not before measuring):

- the Demucs weights' "scientific purposes only" statement, which applies to today's pipeline as well;
- whether to write to ZFTurbo / SubRoformer's author / MVSep about weights licences.

## 6. Honest uncertainty

- Every cost number above is an estimate built from other people's hardware. Run 1 exists to replace them.
- Multisong and MUSDB gains don't transfer one-to-one to Fretboard's songs. The downstream gate guards against this, but 11 songs is a small sample.
- Model "wins" of < 0.2 dB are inside the spread between test sets, so they don't justify extra port work.
- A RoFormer only pays off if the container CPU has fast bf16 or we accept ≈5–20× today's CPU bill for vocals. Until the container's CPU flags are known, plan the Rust port around HTDemucs, with generic STFT/attention/GEMM pieces a RoFormer could reuse.
