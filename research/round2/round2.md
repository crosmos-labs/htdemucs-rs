# Round 2: MUSDB18-HQ SDR for shifts, overlap, weight precision, the two-model average and Kim vocals

Date: 2026-11-23. Machine: Apple M1 Pro (8 cores, 16 GB), shared with two other agents the whole time (load average 16–33). Full tables, per-track numbers and bootstrap intervals: [tables.md](tables.md) (`r2_summary.py`).

## Setup

- **Data:** the 12 MUSDB18-HQ test tracks listed in [../datasets.md](../datasets.md), full length (2,812 s of audio in all). Research / non-commercial licence: used only to measure.
- **Separations:** PyTorch Demucs 4.0.1, CPU, f32 compute (`r2_sep.py`), 2 threads. Shift passes reproduce `apply_model(shifts=2)` with `random.seed(0)` exactly: `_s1` is the first random shift alone, `_s2` the mean of both. For htdemucs_6s, "other" is scored as guitar + piano + other. Kim Mel-RoFormer through round 1's `kim_sep.py` (upstream `demix_track`, overlap 2), CPU, 1 thread (`r2_kim.sh`).
- **Metrics** (`r2_score.py`):
  - **uSDR** = 10·log10(Σs² / Σ(s − ŝ)²) over the whole track, both channels (the MDX / MVSEP "SDR"). This is the headline.
  - **cSDR** = museval BSSEval v4, median over 1 s windows.
  - **accompaniment** = mix − estimated vocals, scored against drums + bass + other. Its error is exactly the vocals error, so its Δ always equals the vocals Δ.
- **Statistics:** every comparison is a paired difference per track. Brackets are 95 % bootstrap intervals over tracks; "k/12" is how many tracks improved.
- **CPU:** `process_time` of the separating process (all threads), divided by audio seconds. The machine was loaded, so treat these as relative.
- The overlap 0.5 shift runs (`hdo50_s1`, `hdo50_s2`) were stopped on purpose after 2 tracks. They are in the tables but carry no weight.

## Results (mean uSDR over 12 tracks, dB)

| variant | vocals | drums | bass | other | CPU-s / audio-s |
|---|---|---|---|---|---|
| htdemucs, shifts 0 (`hd_s0`) | 10.00 | 9.98 | 7.87 | 5.66 | 1.61 |
| htdemucs, shifts 1 | 9.96 | 9.98 | 7.81 | 5.64 | 1.54 |
| **htdemucs, shifts 2 (today's 4-stem)** (`hd_s2`) | 10.04 | 10.04 | 7.95 | 5.75 | 3.07 |
| htdemucs, shifts 0, overlap 0.1 | 9.78 | 9.89 | 7.50 | 5.55 | 1.30 |
| htdemucs, shifts 2, overlap 0.1 | 10.10 | 10.00 | 8.08 | 5.75 | 2.74 |
| htdemucs, shifts 0, overlap 0.5 | 9.95 | 10.03 | 7.77 | 5.69 | 2.34 |
| htdemucs, shifts 0, bf16 weights | 9.98 | 9.98 | 7.85 | 5.65 | 1.56 |
| htdemucs, shifts 0, int8 weights | 9.89 | **8.42** | 7.86 | 5.27 | 1.67 |
| htdemucs_6s, shifts 0 | 10.25 | 9.71 | 8.07 | 5.64 | 1.65 |
| **htdemucs_6s, shifts 2 (today's mixer)** (`6s_s2`) | 10.29 | 9.79 | 8.09 | 5.70 | 3.15 |
| **mean of htdemucs + 6s, shifts 0** (`avg_s0`) | 10.41 | 10.15 | 8.26 | 6.03 | 3.26 |
| mean of htdemucs + 6s, shifts 2 (`avg_s2`) | 10.44 | 10.19 | 8.31 | 6.09 | 6.22 |
| htdemucs_ft vocals model, shifts 0 (`ftv_s0`) | 10.35 | – | – | – | 1.40 |
| **Kim Mel-RoFormer** (`kim`) | **13.73** | – | – | – | 5.71 (1 thread) |

Today's pipeline runs both `hd_s2` and `6s_s2`: **6.22 CPU-s per audio-s**.

### Paired Δ uSDR vs today's `hd_s2` (95 % CI, tracks improved)

| variant | vocals | drums | bass | other |
|---|---|---|---|---|
| `hd_s0` | −0.05 [−0.09, −0.02] 1/12 | −0.06 [−0.08, −0.04] 0/12 | −0.07 [−0.11, −0.04] 0/12 | −0.09 [−0.20, −0.03] 0/12 |
| `hdo10_s0` | −0.27 [−0.69, −0.03] 2/12 | −0.16 [−0.26, −0.07] 0/12 | −0.44 [−1.08, −0.09] 1/12 | −0.19 [−0.46, −0.04] 4/12 |
| `hdo50_s0` | −0.09 [−0.31, +0.03] 8/12 | −0.02 [−0.09, +0.04] 8/12 | −0.17 [−0.52, +0.02] 5/12 | −0.06 [−0.20, +0.02] 7/12 |
| `hdo10_s2` | +0.06 [−0.09, +0.27] 5/12 | −0.04 [−0.13, +0.03] 4/12 | +0.13 [−0.07, +0.52] 2/12 | −0.00 [−0.06, +0.06] 7/12 |
| `hdint8_s0` | −0.15 [−0.20, −0.10] 0/12 | **−1.62 [−2.01, −1.23] 0/12** | −0.08 [−0.11, −0.06] 0/12 | −0.48 [−0.60, −0.37] 0/12 |
| `6s_s2` | +0.25 [+0.04, +0.52] 6/12 | −0.26 [−0.42, −0.09] 2/12 | +0.14 [−0.75, +1.49] 4/12 | −0.04 [−0.54, +0.61] 4/12 |
| **`avg_s0`** | **+0.37 [+0.20, +0.63] 12/12** | **+0.11 [+0.04, +0.17] 10/12** | +0.31 [−0.22, +1.13] 7/12 | +0.28 [−0.05, +0.78] 9/12 |
| `avg_s2` | +0.40 [+0.21, +0.68] 12/12 | +0.14 [+0.07, +0.21] 10/12 | +0.36 [−0.16, +1.18] 7/12 | +0.34 [−0.01, +0.91] 10/12 |
| `ftv_s0` | +0.31 [+0.20, +0.45] 12/12 | – | – | – |
| **`kim`** | **+3.68 [+2.85, +4.67] 12/12** | – | – | – |

## What this says

1. **Kim Mel-RoFormer is the only big quality lever: vocals +3.7 dB on 12 of 12 tracks.** Per track the gain is +1.7 (Forkupines) to +7.3 dB (Little Chicago's Finest). It costs 5.71 CPU-s per audio-s on one thread (round 1 measured 7.0–7.7 at 4 threads under load), i.e. about 3.5 htdemucs passes. Its checkpoint is 913 MB f32, 456 MB as fp16.
2. **Averaging htdemucs and htdemucs_6s at shifts 0 beats today on every stem, at about half today's CPU.**
   - vs today's splits: vocals +0.37 (12/12), drums +0.11 (10/12), bass +0.31 (7/12), other +0.28 (9/12).
   - Its CPU is 3.26 CPU-s per audio-s against 6.22 today. Size is both models, which Fretboard already ships.
   - The bass and other intervals cross zero. One track (Little Chicago's Finest – My Own) swings bass by up to 11 dB between variants, so bass means are fragile.
3. **Random shifts are nearly worthless.**
   - Shifts 2 against 0 gain only +0.05 to +0.09 dB per stem. The gain is consistent (11–12 of 12 tracks) but costs 2× the CPU.
   - Shifts 1 is *worse* than shifts 0 on every stem: one random offset is not an average.
   - On the average, shifts 2 adds +0.03 to +0.06 dB over `avg_s0` for another 3 CPU-s per audio-s.
4. **Keep overlap 0.25.**
   - Overlap 0.1 at shifts 0 loses 0.1–0.4 dB (bass −0.37) for a 19 % CPU saving. At shifts 2 it ties `hd_s2` at 11 % less CPU, but shifts 2 isn't worth having anyway.
   - Overlap 0.5 at shifts 0 is within noise (drums +0.05, bass −0.10 vs `hd_s0`) for +45 % CPU.
5. **Weight precision.**
   - **int8** per-channel weights (no calibration) cost drums −1.6 dB and other −0.4/−0.5 dB on every track. Rejected.
   - **bf16** costs −0.01 to −0.02 dB on average and up to −0.28 on one track. It buys nothing, because the shipped weights are already exact fp16, the same 2 bytes per parameter (htdemucs 84 MB, 6s 55 MB). Rejected.
6. **htdemucs_ft's vocals model: +0.31 dB vocals (12/12) at 1.40 CPU-s.** That is a cheaper vocal gain than Kim, but `avg_s0` already gets +0.37 on vocals without another model. Kim dominates it on quality.

## Recommendation (quality, then CPU, then size)

| rank | recipe | vocals | drums | bass | other | CPU-s / audio-s | weights (fp16) |
|---|---|---|---|---|---|---|---|
| 1 | **Kim vocals + `avg_s0` for drums / bass / other / guitar / piano** | +3.68 | +0.11 | +0.31 | +0.28 | 5.71 + 3.26 = **8.97** | 456 + 84 + 55 MB |
| 2 | **`avg_s0` alone** (htdemucs + 6s, shifts 0, stems averaged) | +0.37 | +0.11 | +0.31 | +0.28 | **3.26** (−48 % vs today) | 84 + 55 MB |
| 3 | today (`hd_s2` + `6s_s2`) | 0 | 0 | 0 | 0 | 6.22 | 84 + 55 MB |

(Δ uSDR vs `hd_s2`.)

- **Do in any case:** shifts 0 (and overlap 0.25, fp16 weights).
- **Rank 2 is the safe default.** It is better on every stem than today and halves the CPU.
- **Rank 1 trades +2.75 CPU-s per audio-s (+44 % vs today, +175 % vs rank 2) for +3.3 dB on vocals.** Round 3 checks whether chaining Demucs on Kim's instrumental also improves the other stems. Kim's weights are MIT (round 1); the Demucs weights still have no clear grant, and htdemucs_6s is likely CC BY-NC.

### The caveat from round 1

None of these SDR gains moved Fretboard's own downstream checks (chord fit, tab onset sync, sung-pitch match: [../round1/round1.md](../round1/round1.md)). Kim's vocals tied the 4-stem vocals on sung pitch. The two-model average (V1b there) was within noise on chords and onsets. It also **lost one song's tab-sync verdict**, which another agent is still investigating. So the case for both recipes is mixer quality (what a listener hears when soloing or muting a stem), which uSDR measures and the downstream checks don't. Don't ship `avg_s0` to the tab aligner until that verdict is explained.

## How to rerun

```bash
cd /Users/divyansh/htdemucs-rs/research/round2
M=/Users/divyansh/datasets/musdb18hq/test
THREADS=2 ./r2_queue.sh &                                                     # Demucs variants, 2 threads
THREADS=1 ./r2_kim.sh &                                                       # Kim vocals, 1 thread
uv run --python 3.11.17 --with-requirements r2_env.txt python -I r2_loop.py $M &   # scoring + averages, 1 thread; deletes WAVs once scored
touch data/stop                                                               # lets r2_loop.py exit when the queue is empty
uv run --python 3.11.17 --with-requirements r2_env.txt python -I r2_summary.py > tables.md
```
