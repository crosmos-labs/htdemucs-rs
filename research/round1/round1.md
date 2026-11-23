# Round 1: Demucs restructure, shift noise, CPU cost, Kim Mel-RoFormer, licences

Date: 2026-11-23. Machine: Apple M1 Pro (8 cores, 16 GB), **shared with the port agent's Rust/PyTorch parity runs and a VM the whole time (load average 23–33 during every timing below)**. All numbers come from Fretboard's own checks (`guitar-tabs/scripts/sync.py`, `scripts/probe.py chordscan`), run read-only on swapped stems. Full tables: [tables.md](tables.md).

## Headline

1. **Pipeline bug found: the tab aligner's 4-stem stems are 25 ms late.** Every 4-stem mp3 in `.cache/stems/htdemucs` lags the recording by exactly **1,105 samples (25.1 ms)** on all 12 songs. This is the mp3 encoder delay: Demucs' mp3 writer leaves no gapless header, so ffmpeg doesn't trim it. The 6-stem FLACs are sample-exact (lag 0). The pipeline's lag gate only checks the 6-stem split. Consequences:
   - tab note times come out **23–24 ms late** against the recording on 6 of 12 songs (the other 6 were already identical: their notes land on the beat grid, not on stem onsets);
   - First Love's `sungPitchMatch` reads **0.636 / 0.662 / 0.621** (three Demucs runs) with the delay and **0.682 / 0.714 / 0.725** without it. That is **+0.067, beyond the 0.041 run-to-run range**.

   Fix (in guitar-tabs, not done here): write the 4-stem split as FLAC/WAV like the 6-stem one, or trim 1,105 samples. A Rust port that writes WAV/FLAC avoids it by construction.
2. **Shift noise, measured** (V0 = today's recipe, 3 Demucs runs: pipeline cache + seeds 1 and 2; range = max − min per song):

   | metric | mean range | max range |
   |---|---|---|
   | chords fit − evenlySpread | 0.001 | 0.004 |
   | spotRank / spotBass | 0.007 / 0.011 | 0.043 / 0.036 |
   | tab `onsetWithin60ms` | 0.012 | 0.025 |
   | First Love `sungPitchMatch` | 0.041 | 0.041 |

   So the ±0.03 I assumed is about right for sung pitch (0.04), too wide for onsets (≤0.025) and far too wide for chord fit (≤0.004).
3. **One 6-stem pass instead of two (V1a) is not free.** Chords are unchanged by construction: they already read only the 6-stem split. But compared with the delay-fixed baseline:
   - First Love `sungPitchMatch` −0.050, larger than the noise;
   - at a fixed placement, 6-stem vocals score −0.05 on First Love and −0.02 on Nirvana (consistent across all three runs);
   - guitar-part onsets are mixed. Wonderwall +0.13, Hallelujah +0.04 and Alvvays +0.03 gain; Wicked Game −0.05 and American Wedding −0.03 lose; the mean is +0.012.

   It saves the 4-stem pass's CPU (≈930 CPU-s per 4-min song). **Not a clean win; I wouldn't adopt it on this evidence.**
4. **Averaging the two models (V1b) doesn't pay.**
   - Chords: slightly worse, within noise (fit −0.002, spotBass −0.005).
   - First Love: −0.024 vs the delay-fixed baseline.
   - Onsets: +0.002.
5. **Kim Mel-RoFormer vocals (K): no measurable downstream gain, at about 2× the CPU of a 4-stem pass.**
   - First Love, at a fixed placement and read 100 ms after the note: **0.909 vs 0.901–0.917** for 4-stem vocals.
   - Aerials: **0.449 vs 0.443–0.455**.
   - Chords on 3 songs: within noise (spotBass +0.007).
   - CPU: **7.0–7.7 CPU-seconds per audio-second** at 4 threads (fp32, overlap 2, torch 2.2.2), peak RSS 2.55 GB.

   Its claimed SDR lead (+2.7 dB on Multisong) doesn't show in what Fretboard measures today. That doesn't rule out a mixer-quality gain (unmeasured here: next round, MUSDB SDR and listening).
6. **No gate verdict changed** in any variant on any song.

## Variants

All variants are built from splits already on disk (`make_variants.py`). The chords always read `harmonic`.

| | vocals / other / bass for the tab aligner | harmonic (chords) |
|---|---|---|
| **V0** | today: 4-stem mp3s (25 ms late) | mix − 6s vocals − 6s drums |
| **V0L** | V0 with the 1,105-sample delay removed | same |
| **V1a** | 6s vocals; other = 6s guitar+piano+other; 6s bass | same |
| **V1b** | mean of 4s (delay-fixed) and 6s | mix − mean vocals − mean drums |
| **K** | Kim vocals; 4s other/bass (delay-fixed) | mix − Kim vocals − 6s drums |

Songs: the 12 with both splits cached (the 11 in `data/songs` minus Iss Tarah, which has no 4-stem split and no tab, plus Alvvays and Arctic Monkeys from `.cache/eval`; their cached recordings differ from the eval recipes', which doesn't matter for a paired A/B on the same audio). Only First Love's pinned part is a voice part. For more vocal evidence, each Songsterr transcription's first vocal part was also aligned on its own (`vocalPart`, not a pipeline gate). Most of those land near chance (0.15–0.31: the offset search places them wrongly), so only Nirvana (0.54) and Aerials (0.46) are informative. For those, and for First Love, `fixed_pitch.py` scores every variant's vocals at one fixed placement (seed-0 V0L note times), at the note time and 100 ms later. That separates stem quality from the aligner's search noise.

Harness check: V0 on the cached splits reproduces First Love's committed report exactly (chords 0.327 / 0.946 / 0.964, margin 0.169; `sungPitchMatch` 0.636). The pipeline's `harmonic.flac` and mine (numpy) agree to 73 dB SNR.

### Fixed-placement sung-pitch match (three Demucs runs; K is one run, Kim is deterministic)

| song | read at | V0 (late) | V0L | V1a (6s) | V1b (mean) | K (Kim) |
|---|---|---|---|---|---|---|
| First Love | note time | 0.49 / 0.50 / 0.48 | 0.70 / 0.69 / 0.71 | 0.65 / 0.68 / 0.66 | 0.69 / 0.68 / 0.71 | 0.70 |
| First Love | +100 ms | 0.91 / 0.91 / 0.91 | 0.92 / 0.90 / 0.91 | **0.86 / 0.87 / 0.86** | 0.89 / 0.89 / 0.90 | 0.91 |
| Nirvana (vocal part) | +100 ms | 0.62 / 0.63 / 0.62 | 0.63 / 0.62 / 0.62 | 0.61 / 0.61 / 0.61 | 0.60 / 0.61 / 0.59 | – |
| Aerials (vocal part) | note time | 0.42 / 0.42 / 0.42 | 0.48 / 0.48 / 0.48 | 0.48 / 0.49 / 0.48 | 0.47 / 0.48 / 0.48 | 0.51 |
| Aerials (vocal part) | +100 ms | 0.47 / 0.48 / 0.47 | 0.44 / 0.46 / 0.45 | 0.44 / 0.45 / 0.44 | 0.43 / 0.44 / 0.44 | 0.45 |

The "note time" reading is the pipeline's rule, and it is very sensitive to a 25 ms stem delay: V0 loses 0.2 on First Love there. The +100 ms reading is not. On that one, the 4-stem vocals (V0/V0L) are best or tied, 6-stem vocals are consistently worst, and Kim ties.

### Odd case

Hallelujah: with 6-stem stems (V1a/V1b), the aligner places the tab **66.8 s** from where V0 puts it, on all three runs. Its onset gate fails either way (0.29–0.34 vs 0.7), and its chords fail (spotRank 0.17), as in the committed report. It's a song where the aligner has no stable answer.

## Determinism note

- Two fresh seed-0 runs today are byte-identical (both models).
- The cached 6-stem split for First Love is byte-identical to today's.
- **The cached 4-stem split is not**: 30–34 dB SNR between cached and fresh, same seed and device. Something changed between the cached run (2026-11-22 17:43) and today, possibly a concurrent MPS job then. `verify` would flag this as "stems differ".

## CPU cost (PyTorch 2.2.2 CPU, 4 threads, `OMP_NUM_THREADS=4`; System of a Down – Aerials, 243.6 s)

Measured under load average 23–33 on 8 cores (the port agent and a VM were busy the whole session; the machine was never quiet). The timings are **upper bounds**: the CPU-seconds are inflated less than the wall time. Re-measure in the Cloudflare container, which is the real target anyway.

| run | wall | CPU (user+sys) | CPU-s per audio-s | peak RSS |
|---|---|---|---|---|
| htdemucs, 2 shifts (pipeline args, `-d cpu`) | 586 s | 933 s | 3.83 | 1.83 GB |
| htdemucs_6s, 2 shifts | 517 s | 850 s | 3.49 | 2.04 GB |
| **today's two splits together** | **1,103 s (18.4 min)** | **1,783 s (29.7 vCPU-min)** | 7.3 | 2.04 GB |
| Kim Mel-RoFormer, 60 s excerpt, overlap 2, fp32 | 218 s | 463 s | 7.7 | 2.55 GB |
| Kim, full songs (First Love / Wicked Game / Aerials) | 1,090 / 1,003 / 1,646 s | 1,955 / 2,067 / 1,827 s | 7.0 / 7.1 / 7.5 | – |

At Cloudflare's $0.000020 per vCPU-second, today's recipe costs ≈ **$0.036 of CPU per 4-minute song** on this evidence. Kim vocals alone would cost ≈ $0.035. Replacing the 4-stem pass with Kim roughly doubles that pass. For reference, Fretboard's MPS splits take 128–270 s for 4-stem and 45–145 s for 6-stem under the same load.

## Licence follow-up

**Demucs (adefossez/demucs fork, v4.1.0, 2026-07-11).** Weights moved to Hugging Face: `adefossez/HTDemucs`, `HTDemucs-6s`, `HTDemucs-ft`, plus 8 older models, all created 2026-07-11. Their model cards have **no `license` field**, only tags. The fork's README now says, verbatim:

> "The license of the weights is not clear: MusDB contains some data that were only usable for research purpose. I sadly cannot help granting a license, both coming from Meta custom data, or from the original MusDB data. The 6 source variant of htdemucs was trained only on Meta internal dataset, which likely means CC-NC-BY license is applicable, as it was the case for all models released by FAIR. Do not write me to ask about it, I won't be able to clarify the situation further, either for MusDB or Meta internal dataset."

The training-metadata sidecars agree:
- `htdemucs` and the `htdemucs_ft` shard read `musdbhq` + `allstems_44` + `mmi44_goodclean`;
- `htdemucs_6s` reads `musdbhq` + `mmi44_6_goodclean` (Meta internal).

**Net:** htdemucs/ft weights have no grant; **htdemucs_6s is likely CC BY-NC**, per its author. Fretboard's mixer depends on it today.

**Kim Mel-RoFormer (`KimberleyJSN/melbandroformer`).**
- **Who:** Kimberley Jensen trained it on ZFTurbo's MSST code, on one H100 for about 3 weeks (her repo, issues #1 and #3).
- **Data:** about 1,400 vocal/instrumental tracks assembled by the UVR Discord community (aufr33, Anjok, bascurtiz). Not public; it **includes MUSDB18 and MoisesDB** (issue #3, 2024-08-17).
- **Licence history:**
  - 2024-08-20: "released the model weights so people can use it for personal / research usage" (HF discussion #1);
  - 2025-06-17: GPL-3.0 added for Audacity's plugin (HF #2);
  - **2026-04-22: changed to MIT**, with "you are free to use these models and redistribute them in any way you like" (GitHub issue #18).

  The current card is only `license: mit`. Five open requests (2026-08 to 2026-11) ask whether that covers commercial use, given MUSDB/MoisesDB in training. None has an answer yet.

**Net:** Kim's weights carry an explicit MIT grant from the rights holder, which Demucs' don't. The open question is the training data, the same one every MUSDB-trained model has. Still the cleanest strong model found, but **it didn't improve anything Fretboard measures** (above).

## What I'd run next

1. **Fix the 25 ms delay in guitar-tabs** (4-stem split to FLAC/WAV), then re-verify the songs. This is a cheap, measured win, and it belongs to whoever owns guitar-tabs.
2. **MUSDB18-HQ SDR on the 12-track subset** (fetched, 2.3 GB): htdemucs vs htdemucs_6s (vocals, drums, bass, other = guitar+piano+other) vs Kim vocals, 2 shifts vs 0, overlap 0.25 vs 0.1. This answers what downstream checks can't: whether Kim's or 6s's stems are better *for the mixer*. Cost: about 12 × (190 s MPS + 280 s Kim MPS-or-CPU).
3. **Knob sweep for the port's defaults** (shifts 0/1/2, overlap 0.1/0.25) with these same downstream checks plus MUSDB SDR. The noise numbers above say which differences would be real.
4. Drop Kim from the production shortlist unless MUSDB/listening shows a mixer gain worth ≈2× CPU. Keep htdemucs 4-stem for the aligner's vocals.

## How to rerun

```bash
R=/Users/divyansh/htdemucs-rs/research/round1; GT=/Users/divyansh/guitar-tabs
# 1. the pipeline's requests (network: UG + Songsterr), into data/ (git-ignored: derived from tabs)
cd $GT && npx vite-node -c scripts/song.config.ts $R/dump_requests.ts $R/data/requests.json <ids…>
# 2. re-split with other seeds, Fretboard's exact Demucs args, under Fretboard's Demucs lock (MPS)
cd $R && ./run_splits.sh mps 1 <ids…>; ./run_splits.sh mps 2 <ids…>
# 3. Kim vocals (CPU, 4 threads; the 913 MB checkpoint from huggingface.co/KimberleyJSN/melbandroformer, sha256 87201f4d…559e,
#    code from github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model @25f44ff in data/ext/kim)
uv run --python 3.11.17 --with-requirements kim_env.txt python -I kim_sep.py data/ext/kim data/models/MelBandRoformer.ckpt $GT/.cache/audio/<id>.m4a data/kim/<id>.vocals.wav --device cpu --threads 4
# 4. variants + Fretboard's checks (seed0 = the pipeline cache), fixed-placement pitch, tables
./run_set.sh seed0 V0 V0L V1a V1b K; ./run_set.sh seed1 V0 V0L V1a V1b; ./run_set.sh seed2 V0 V0L V1a V1b
./fixed_all.sh seed0; ./fixed_all.sh seed1; ./fixed_all.sh seed2
uv run --python 3.11.17 --with-requirements kim_env.txt python -I summarize.py --md > tables.md
find data/variants -name '*.wav' -delete   # ≈43 GB of float WAVs otherwise
# 5. CPU cost (check `uptime` first)
./timing.sh run1
```
