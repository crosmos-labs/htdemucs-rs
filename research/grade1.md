# htdemucs-rs: independent grade 1 (2026-11-23)

Scope: HEAD `dff89f6` of htdemucs-rs, Fretboard branch `htdemucs-rs` (`fc95d08`) in `/Users/divyansh/guitar-tabs-rs`.

How it was run:
- Rust was rebuilt from HEAD into a scratch target dir. This is the default build: `gemm` backend, no Accelerate, so it is the Linux code path.
- PyTorch was Fretboard's pinned environment: demucs 4.0.1, torch 2.2.2, Python 3.11.17, via `uv --offline`.
- Everything ran at 2 threads under `nice` on the shared M1 Pro (load 10–15).
- No Actions runs were triggered; the existing bench logs were downloaded and read.
- Scratch files are in `/private/tmp/claude-503/-Users-divyansh-guitar-tabs/ff8f5c1b-f8e9-48f3-a1c0-dc07bd42ab36/scratchpad/grade/`.

## Grades

| # | Claim | Grade |
|---|---|---|
| 1 | Parity ≥ 94 dB, 4 songs × 2 models × shifts 2/0; Aerials 116.0 / 120.5 / 124.8 / 128.6 | **Holds with caveats** |
| 2 | Seeded shifts reproduce CPython `random` exactly | **Holds** |
| 3 | Bit-identical across thread counts | **Holds** (only across thread counts, not across CPUs) |
| 4 | Decoding equivalent to ffmpeg; 16-bit WAV/FLAC bit-exact | **Holds for AAC-LC / mp3 / wav / flac. False for HE-AAC** |
| 5 | Speed: Zen 3 0.721 vs 0.763, Zen 5 0.325 vs 0.510 CPU-s per audio-s, 1 thread | **Numbers hold; the framing is overstated** |
| 6 | Size: 2.2 MB macOS, 4.2 MB Linux, weights 84 / 55 MB, lossless fp16 | **Holds** (README has a stale "~1.0 MB" line) |
| 7 | Fretboard: same verdicts, eval passes, 4-stem FLAC removes 25 ms delay | **Holds; the switch-off path is unchanged** |

### 1. Parity: holds with caveats

**The Aerials full song reproduces.** Aerials, htdemucs, shifts 2, seed 0, with my gemm build at 2 threads, against `ref/songs/system-of-a-down-aerials.htdemucs.s2.safetensors`, using `compare`:
- drums 116.0 dB, bass 120.5 dB, other 124.8 dB, vocals 128.6 dB;
- that is exactly the claim. Max abs error 0.8–1.4e-6, error RMS −141 to −147 dBFS.

**Edge cases the builders didn't test.** Each is my own short input, compared with `tools/ref_separate.py` (htdemucs, shifts 2, seed 0 unless noted). Minimum stem SNR / notes:

| Input | Min SNR | Notes |
|---|---|---|
| 0.3 s clip (shorter than a shift) | 107.8 dB | lowest stems at −48…−57 dBFS |
| 100 samples | 64.9 dB | ill-conditioned input (see below) |
| odd length, 100001 samples | 87.6 dB | vocals at −73 dBFS, max abs error 4e-8; the others 119–130 dB |
| exactly 1 segment, shifts 0 | 112.9 dB | |
| exactly 2 strides, shifts 0 | 113.1 dB | |
| mono WAV | 113.8 dB | |
| −80 dBFS with 1 s of digital silence | 116.5 dB | |
| hard-clipped ×8 | 93.5 dB | bass at −53 dBFS; the loud stems 123–127 dB |
| float WAV peaking at +12 dBFS (unclipped) | 116.2 dB | |
| 4-channel WAV (first two taken) | 116.2 dB | |
| htdemucs_6s, 0.3 s | 99.0 dB | |
| htdemucs_6s, odd length | 86.9 dB | only the −75…−79 dBFS stems are low |
| **48 kHz input** | **61.8 dB** | see below |
| all-zero input | — | see bug 3 |

- **48 kHz input.** Same length, lag 0, but 61.8–75 dB SNR. The resampler differs (rubato vs ffmpeg swresample): max error 1.5e-2 at sample 0 (start-up transient) and ~5e-4 in the middle. This is documented as "close, not identical". Fretboard only feeds 44.1 kHz.
- **100 samples.** This is ill-conditioning, not a bug. A float64 PyTorch run gives 59.7–70.3 dB for Rust and 61.6–76.4 dB for torch float32, so torch float32 is itself only ~62–76 dB from float64.

**Caveats:**
- **Most of the 16-row parity table was not run on the Linux code path.**
  - `ref/songs/parity-accelerate.txt` has all 16 rows. `ref/songs/parity-gemm.txt` (the Linux path) has only Aerials s2 plus one empty row.
  - The full table was measured with the macOS Accelerate build, mostly before `dd54f23`, `dff89f6` and the AVX-512 kernels.
  - On x86 at HEAD, only the 12 s synthetic clip was checked (Actions runs 45111459014 / 45111465908): 114.5–119.3 dB, vocals 88.5 dB at −72 dBFS.
  - So the gemm path at HEAD is shown on 1 real song plus a synthetic clip, not 16 runs. It is very likely fine (same kernels, same error level), but strictly the claim needs `parity_songs.sh` re-run with the default build.
- **"≥ 94 dB on every stem" is an SNR on near-silent stems.** The README explains this well: the absolute error is −134…−160 dBFS. The same file also has `ntl30 htdemucs_6s s2 other 80.7 dB`, which is outside the claimed set.

### 2. Seeded shifts: holds

- **`PyRandom` against CPython 3.11.17.** 511 seeds, including 0, negatives, 2^31−1, 2^32, 2^63, 2^64+3, 2^100+17, −2^70 and 300 random 80-bit seeds. For each, I replayed demucs' exact pattern: `randint(0, 22050)`, then one `randrange(1)` per segment, over 1–6 shifts and random lengths up to 30 min. All offsets matched.
- **End to end, separations with seed 7 and seed 2^40+1** match PyTorch at 116.1 dB and 116.2 dB. A wrong offset gives ~20 dB.
- **Nothing else draws from Python's `random` first.** Importing demucs, building `get_parser()`, and loading htdemucs and htdemucs_6s consume no draws: the first `randint` after them is still 12623.
- **CPython 3.12.15 has the same `_randbelow_with_getrandbits`** and the same sequence.
- **One nit:** with several tracks in one CLI call, Rust re-seeds for each track, while demucs continues one stream (bug 6). Fretboard passes one track per call, so it is unaffected.

### 3. Determinism: holds, only across thread counts

- Rust at 1 thread vs 2 threads gives byte-identical float32 stems:
  - clip10, htdemucs, seed 7;
  - odd.wav, htdemucs_6s.
- CI's bench checks that one segment forward is identical at 1/2/4/8 threads. But the runners have 2 vCPUs (1 core with SMT).
- The code supports the claim: every parallel float reduction is collected into a Vec in fixed order and summed sequentially, and gemm never splits k.
- **Caveat: outputs differ between CPU types.** The same commit's synthetic-clip parity is drums 114.5 dB on Zen 3 vs 116.6 dB on Zen 5, because the AVX2 and AVX-512 kernels are picked at run time.
- So Cloudflare containers landing on different hosts will give stems that are not bit-identical. Fretboard's `verify` "stems not bit-identical" path will fire. This needs to be stated next to the claim.

### 4. Decoding: holds for AAC-LC / mp3 / wav / flac, false for HE-AAC

- **All 17 cached Fretboard m4a files** (all AAC-LC 44.1 kHz) against ffmpeg's f32 decode: identical length on every file, lag 0, 116.5–134.2 dB. The builders tested 4 of these; I tested all 17.
- **LAME mp3:** same length, lag 0, 125.4 dB. Gapless trim works.
- **16-bit WAV, 24-bit FLAC and 4-channel WAV:** bit-identical to ffmpeg.
- **A corrupt AAC packet:** both decoders drop it, so the timeline is unchanged.
- **HE-AAC (made with `afconvert -d aach`): silent wrong output.** See bug 1.
- **Opus / Ogg:** refused with a clear error. That is fine, because Fretboard stores m4a.
- **Writers are bit-exact.** I ran demucs' own `save_audio` (torchaudio 2.2.2; `soundfile` is the only backend in this environment) on Rust's float stems of a +12 dBFS input. Compared with Rust's own output:
  - WAV16 rescale, FLAC16 clamp and FLAC16 rescale: 0 differing samples of 882,000 per stem;
  - this covers the `prevent_clip` maths as well as the quantisation.

### 5. Speed: numbers reproduce from the logs; the framing is overstated

From `gh run view --log`, CPU seconds for one segment forward (7.8 s of audio):

| Runner | Rust (dff89f6) | PyTorch | Claimed | Gap |
|---|---|---|---|---|
| EPYC 7763 (Zen 3), run 45111459014 | cputime 5.69 / 5.56 s → 0.721; same-run `bench` RESULT 5.73 / 5.75 s → 0.736 | 5.955 s → 0.763 | 0.721 vs 0.763 | **4–6 %** |
| EPYC 9V45 (Zen 5), run 45111465908 | 2.59 / 2.47 s → 0.325 (RESULT 0.323 / 0.332) | 3.975 s → 0.510 | 0.325 vs 0.510 | **36 % less CPU** |

- **The Zen 3 gap is within noise.** The same commit moves 2–3 % between rounds, and PyTorch was measured once per run. Call it "at parity on Zen 3", not a win.
- **The metric is per audio-second of one segment forward, not of a song.** With overlap 0.25 and shifts 2, a song costs 2 × 7.8 / 5.85 ≈ 2.7× that per second of music. Example on Zen 3: ≈ 1.9 CPU-s per song-second.
  - The full-path timing in the same logs: the 12 s synthetic clip took 59.7 CPU-s on Zen 3 and 25.5 CPU-s on Zen 5 at the default thread count.
- **Multi-core scaling is untested on x86.** Every runner had `nproc 2`, `Core(s) per socket: 1`, so "2 threads" is SMT on one core. On the M1 (README), the gemm build used 639 CPU-s at 4 threads and 742 CPU-s at 8: CPU cost grows with threads.
- **No PyTorch baseline on Intel or Zen 4.** The Xeon 6975P was 9.0 s per segment (v4) before the `dff89f6` subnormal fix and was never re-run after it. Cloudflare's host CPU is unknown.
- My own Mac number (loaded machine, not comparable): Aerials full song took 318 s wall, 457 s user + 56 s sys at 2 threads, peak RSS 1.09 GB.

### 6. Size: holds

- **macOS binary:** 2,374,400 B (2.26 MiB).
- **Linux binary (Actions log):** 4,181,200 B.
- **Weights:** 84,023,608 B and 54,883,920 B.
- **The fp16 weights are lossless.** I loaded the original `.th` checkpoints (both stored as `torch.float16`): 533 / 525 tensors, 0 mismatches against the safetensors, and no extra tensors.
- **README nit:** the "Inference approach" section still says the binaries are "~1.0 MB", which contradicts the 2.2 MB stated later.

### 7. Fretboard integration: holds

- **Same gate verdicts.** `scripts/compare_reports.ts` on `.cache/committed` vs `.cache/reports`:
  - every gate verdict is the same;
  - tab bar times and chord times are identical except one Iss Tarah chord change (1045 ms);
  - this matches the doc.
- **One move the doc doesn't mention.** It says the tab attack gates "move up". The Nirvana McNemar z went **down**, 2.03 → 1.79. It FAILs (threshold 3) either way.
- **Eval.** `.cache/eval.log` ends with "every expectation holds (17 request(s))". Every split in its cache was made by `htdemucs-rs`.
- **The 25 ms claim** is inferred, not measured directly. The control (Python Demucs on CPU, mp3) reproduces the committed numbers, and Rust matches Python Demucs to ~115 dB, so attributing the gate moves to mp3→FLAC is reasonable.
- **Unit test:** `server/separator.test.ts` passes (3 tests).
- **With the switch off, behaviour is identical to today.**
  - `STEMS` is the same path as `STEMS4`.
  - `stem4File` falls back to `.mp3`.
  - `separatorOf` treats a missing, field-less or unparsable `separated.json` as `demucs`.
  - Device strings and the nondeterminism text are unchanged.
- **What does change with the switch off** (cosmetic only):
  - two log lines;
  - `separated.json` gains `separator` and `format` fields, which then appear in reports' `stems.measurements`;
  - `stemsDevice` returns null rather than `''` for an empty device string.
- **Not covered by the branch:**
  - In rust mode there is no fallback to Python Demucs if `htdemucs-rs` fails.
  - The Cloudflare container deployment (Linux binary, weights in the image) is not part of the branch. The Mac runner still does the work.

## Bugs, by severity

1. **(Medium) HE-AAC input gives silently wrong, mis-timed stems.**
   - Symphonia decodes only the AAC core at 22,050 Hz, without SBR.
   - The edit-list trim is then computed at that rate, and rubato upsamples a band-limited signal.
   - Reproducer: `afconvert -f m4af -d aach -b 64000 clip10.wav he.m4a`, then `htdemucs-rs separate -n htdemucs --shifts 0 …`.
   - Result: 446,464 samples instead of 441,000, delayed 3262 samples (74 ms), 24.6 dB from Demucs at the best lag. Exit code 0.
   - None of the 17 cached files is HE-AAC, but nothing stops one arriving (e.g. a YouTube itag 139 fallback).
   - Fix, either:
     - refuse when the decoded rate is not the container's rate or the profile is HE; or
     - give the 4-stem split the same ffmpeg-decoded f32 WAV the 6-stem path already uses. This also removes the dependency on symphonia's AAC.
2. **(Low–medium) Multi-core CPU cost and x86 scaling are unmeasured, and outputs differ across host CPU types.** This is not a correctness bug, but it matters for the "efficiency" priority and for Fretboard's determinism story on Cloudflare.
3. **(Low) Digital-silence input exits 0 with NaN stems.**
   - `--float32` writes all-NaN files.
   - 16-bit and FLAC write zeros, because NaN casts to 0.
   - Demucs fails with an assertion here (NaN after `wav /= ref.std()`).
   - Better to fail with an error, or output zeros, consistently.
4. **(Low) Peak memory grows linearly with track length and is high at Fretboard's 22-minute limit.**
   - `apply::separate` holds the whole `acc` (S×2×len) plus a whole pass buffer (S×2×(len+0.5 s)) plus a padded copy and the input.
   - htdemucs_6s on a 22-minute track: ≈ 2.8 + 2.8 + 0.5 + 0.5 GB ≈ 6.5 GB.
   - Overlap-adding each pass straight into `acc` would remove one S×2×len buffer.
   - PyTorch is similar, so this is not a regression, but it sets the container size.
5. **(Low) Malformed-input robustness.**
   - `mp4_edit` indexes `elst[0..]`, `elst[p..p+8]` and `full_box[20..24]` without length checks, and allocates `size − hdr` from file data.
   - With `panic = "abort"`, a crafted file would abort rather than error. In my test (an inflated `elst` count), symphonia rejected the file first.
   - The weights loader also panics (`unwrap`, slicing) on a truncated weights file.
6. **(Nit) Behaviour differences from demucs.**
   - Several tracks per call re-seed the RNG per track; demucs continues one stream.
   - `--overlap` close to 1 gives `step_by(0)` and a panic; demucs gives a ValueError.
   - The progress bar covers all passes as one bar, unlike demucs.

## Before production vs nice-to-have

**Must fix before production:**
- **Bug 1:** refuse or correctly handle non-LC AAC. Simplest: decode the 4-stem input with ffmpeg, as the 6-stem path does.
- **Re-run `tools/parity_songs.sh` with the default (gemm) build at HEAD, on x86** (Actions or docker) for all 16 rows. The headline table currently comes from the Accelerate build at older commits.
- **Measure on the real target.** Use a Cloudflare container, or at least a ≥4-physical-core x86 box, at the thread count you will deploy:
  - full-song CPU-s and wall time for Rust vs PyTorch;
  - peak RSS on a long track.
- Size the container memory from that.
- **Document that stems are deterministic per CPU type, not across hosts,** and decide whether Fretboard's verify tolerances cover it.
- **Decide what rust mode does when the binary fails.** Today the split just errors, with no Demucs fallback.

**Nice to have:**
- Error on whole-track silence (bug 3).
- Bounds-checked MP4 parsing (bug 5).
- Stream the overlap-add into `acc` to cut peak memory (bug 4).
- Add a PyTorch baseline on Intel and Zen 4 runners after `dff89f6`.
- Fix the README "~1.0 MB" line.
- Mention the Nirvana McNemar decrease in the Fretboard doc.

## Untested by anyone (as far as I can find)

- Multi-core x86 scaling, and CPU cost at the thread count that will be deployed.
- Any run inside a Cloudflare container.
- The gemm path at HEAD on the full 16-row song table.
- HE-AAC and other non-LC inputs (now tested: broken). Opus/Ogg is unsupported.
- Long tracks near the 22-minute limit (memory).
- Determinism across CPU models. It is known not to hold, but this isn't written down.
- Seeds other than 0 inside Fretboard. This doesn't matter; the RNG is verified independently above.
