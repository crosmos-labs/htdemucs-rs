# Round 3: test-time augmentation, the Kim → Demucs chain, best-of stems, weighted averages, Wiener post-filter

Date: 2026-11-23. Same 12 MUSDB18-HQ test tracks and the same uSDR as round 2 ([../round2/round2.md](../round2/round2.md)). Everything is paired per track against round 2's scores. Full tables: [tables.md](tables.md) (`r3_summary.py`).

## Headline

1. **Running Demucs on Kim's instrumental (the UVR chain) is the find of the round: better drums and much better "other", at no extra CPU.**
   - Against Demucs on the full mix with the same Kim vocals (`k+chain_avg` vs `k+avg_s0`): drums **+0.29 dB (11/12)**, other **+0.91 dB (11/12)**, bass +0.00.
   - It costs the same, because the Demucs passes just get a different input.
   - The 6-stem guitar stem carries **2.1 dB less vocals/drums/bass bleed (9/9 songs with guitar)**.
2. **Best recipe: Kim vocals, then htdemucs + htdemucs_6s on the instrumental, averaged, shifts 0.**
   - vs today's mixer split (`6s_s2`): vocals **+3.44**, drums **+0.65**, bass +0.17, other **+1.24** dB.
   - vs today's 4-stem split (`hd_s2`): vocals +3.68, drums +0.39, bass +0.31, other +1.19.
   - Cost: **8.97 CPU-s per audio-s**, against 6.52 today (+38 %).
3. **Test-time augmentation (swap / polarity) is no better per CPU-second than random shifts.**
   - Polarity inversion beats shifts 2 on the 12-track mean, at the same cost. That edge comes from one unstable track (Little Chicago's Finest).
   - On the other 11 both give +0.01 to +0.06 dB, so neither is worth a second pass.
   - Swap + polarity on htdemucs_6s *lost* 10 dB of bass on that track.
4. **Rejected:**
   - **Wiener post-filter:** −0.4 to −2.7 dB.
   - **Tuned htdemucs/6s weights:** weights fitted on one half never beat 0.5 on the other.
   - **Forcing other = mix − the rest for Demucs-only stems:** median −0.01, mean −0.63.

## Setup

- **Reproducibility.** Every round 2 variant was re-separated, because round 2 deleted its WAVs. Reproduction on the same 60 track × stem pairs: largest |Δ uSDR| is 0.006 dB (`hd_s0`) and 0.000 dB (`kim`).
- **Where it ran.** Demucs passes ran on MPS (`r3_sep.py`, `r3_queue.sh`). Kim ran on MPS for the transformer, with the STFT, complex mask averaging and iSTFT on the CPU (`kim_sep_mps.py`), because MPS has no complex `scatter_add`. Two tracks ran Kim on the CPU at 2 threads (`r3_kim.sh`). This kept my CPU use within 4 threads on a machine at load 20–33 with 16–18 GB of swap in use.
- **CPU-s per audio-s** for every variant is the sum of its passes' rates, as measured on the CPU in round 2:

  | pass | CPU-s per audio-s |
  |---|---|
  | htdemucs | 1.61 |
  | htdemucs_6s | 1.65 |
  | Kim (1 thread) | 5.71 |

  A swapped, inverted or instrumental pass is the same network call on an input of the same length. The MPS runs' host CPU per pass agrees: 0.10–0.13 for every pass type.

  Kim on the CPU at 2 threads under this memory pressure measured **9.0 and 10.6** CPU-s per audio-s. That is a reminder to re-time everything in the target container.
- **Scoring.** `r3_score.py` builds 97 variants per track from 10 Demucs passes and Kim's vocals, accumulating sums of squares over 7 s chunks. Only uSDR was computed: museval cSDR took more than 30 min per track on this machine. Round 2 has cSDR for the base variants, and it ranks them the same way.
- **The outlier.** Little Chicago's Finest – My Own is the one unstable track: bass between 3.8 and 15.3 dB depending on chunking. Where a mean hinges on it, I give the median or the mean without it ("w/o LCF").

## 1. Test-time augmentation vs random shifts (same CPU: one extra pass)

Δ uSDR against the same model at shifts 0, mean over 12 tracks. Brackets are the median, then the mean without LCF.

| model | extra passes | vocals | drums | bass | other |
|---|---|---|---|---|---|
| htdemucs | shift ×2 (`s2`, today) | +0.04 (+0.02 / +0.03) | +0.06 (+0.06 / +0.06) | +0.07 (+0.06 / +0.06) | +0.09 (+0.04 / +0.04) |
| htdemucs | L/R swap | +0.08 (+0.01 / +0.01) | +0.02 (+0.02 / +0.01) | −0.02 (+0.02 / +0.01) | +0.04 (+0.01 / +0.01) |
| htdemucs | polarity | +0.12 (+0.01 / +0.01) | +0.11 (+0.06 / +0.06) | +0.31 (+0.06 / +0.05) | +0.11 (+0.03 / +0.02) |
| htdemucs | swap + polarity | +0.14 (+0.01 / +0.02) | +0.09 (+0.05 / +0.05) | +0.30 (+0.05 / +0.04) | +0.09 (+0.02 / +0.02) |
| htdemucs | all four (3 extra) | +0.19 (+0.02 / +0.02) | +0.12 (+0.06 / +0.06) | +0.37 (+0.06 / +0.05) | +0.13 (+0.03 / +0.03) |
| htdemucs_6s | shift ×2 (today) | +0.04 | +0.07 | +0.02 | +0.07 |
| htdemucs_6s | polarity | +0.03 | +0.07 | +0.11 | +0.03 |
| htdemucs_6s | swap + polarity | +0.02 | +0.09 | **−0.77** (LCF −10.1) | −0.30 |
| average of both | shift ×2 | +0.03 | +0.04 | +0.05 | +0.06 |
| average of both | polarity | +0.05 | +0.06 | +0.21 | +0.06 |
| average of both | all four | +0.07 | +0.07 | +0.10 | +0.04 |

**Verdict.**
- Polarity inversion is the best augmentation and matches or beats random shifts on 11–12 of 12 tracks. That agrees with the MSST paper's +0.00 to +0.06 dB for RoFormers.
- On typical tracks it is worth **+0.01 to +0.06 dB per doubled CPU**, the same as shifts.
- The mean advantage is LCF, where averaging over polarity stabilises bass (+3 dB on htdemucs).
- For Fretboard: **shifts 0, no TTA.** If a future model is chunk-unstable like LCF, polarity TTA is a better second pass than a random shift. Don't use swap+polarity on 6s.

## 2. The Kim → Demucs chain (UVR-style)

Kim vocals in every row. Drums, bass and other come from Demucs either on the full mix (`k+<m>_s0`) or on Kim's instrumental, mix − Kim vocals (`k+chain_<m>`). Mean uSDR over 12 tracks:

| Demucs | input | drums | bass | other | CPU-s / audio-s |
|---|---|---|---|---|---|
| htdemucs | mix | 9.98 | 7.87 | 5.66 | 7.32 |
| htdemucs | **Kim instrumental** | **10.25** | 7.84 | **6.60** | 7.32 |
| htdemucs_6s | mix | 9.71 | 8.07 | 5.64 | 7.36 |
| htdemucs_6s | **Kim instrumental** | **10.03** | **8.24** | **6.60** | 7.36 |
| average | mix | 10.15 | 8.26 | 6.03 | 8.97 |
| average | **Kim instrumental** | **10.44** | 8.26 | **6.94** | 8.97 |

### Paired Δ, chain vs mix

| Demucs | drums | bass | other |
|---|---|---|---|
| htdemucs | +0.27 [−0.04, +0.46] 11/12, median +0.44 | −0.03, median +0.07 | +0.95 [+0.30, +1.64] 11/12, median +0.72 |
| htdemucs_6s | +0.32 [+0.17, +0.44] 11/12 | +0.17 [+0.05, +0.31] 10/12 | +0.97 [+0.59, +1.45] **12/12** |
| average | +0.29 [+0.06, +0.44] 11/12, median +0.44 | +0.00, median +0.04 | +0.91 [+0.43, +1.50] 11/12, median +0.67 |

- With vocals already removed, Demucs no longer has to decide what is voice. The vocal energy it used to misplace into "other" (and drums) is gone.
- Adding Demucs' own leftover "vocals" output from the instrumental back into other (`_rv`) changes nothing (+0.00 to +0.02).
- **Guitar** (htdemucs_6s's guitar stem, proxy in §6): the chain cuts the vocals/drums/bass energy in the guitar stem by **2.06 dB on all 9 songs with guitar**, with the same guitar energy.

## 3. Per-stem best-of, and stems that sum to the mix

Per stem, the best scored source is:
- vocals: Kim (13.73)
- drums: chain average (10.44)
- bass: average with polarity TTA (8.47), but in practice chain average or `avg_s0` (8.26), since the TTA edge is LCF
- other: chain average (6.94)

So the best-of *is* `k+chain_avg`: nothing else wins a stem except through LCF.

**Fixing other so the stems sum to the mix** (`_fix`: other := mix − vocals − drums − bass):

| stems | other as separated | other `_fix` | mean Δ | median Δ | residual before fix (dB re mix) |
|---|---|---|---|---|---|
| `avg_s0` (Demucs only) | 6.03 | 5.39 | −0.63 | −0.01 | −25.3 |
| `k+avg_s0` (Kim + Demucs on the mix) | 6.03 | 6.05 | +0.02 (11/12 up) | **+0.50** | **−17.8** |
| `k+chain_avg` (chain) | 6.94 | 6.29 | −0.65 (9/12 up) | +0.05 | −26.4 |

- **Kim + Demucs on the mix leave a residual of −17.8 dB.** The two models disagree about what is voice. On typical tracks, putting that residual in other helps (median +0.50), because it is mostly non-vocal sound Kim removed and Demucs didn't assign.
- **For Demucs alone, and for the chain, the residual is −25 to −26 dB.** Folding it into other is neutral on typical tracks (median −0.01 / +0.05).
- **On LCF every fix costs about −8 dB of other.** The models' bass is badly wrong there, and the fix pushes the error into other.
- The chain therefore **sums to the mix 8.6 dB more closely** than Kim + Demucs on the mix, and needs no fix.
- **For Fretboard's mixer:** today it builds "other" as original − vocals − drums − bass − guitar, which is the `_fix` construction. With the chain, the model's own other is 0.65 dB better on average and equal on the median. Either is fine. If stems must sum exactly, keep the subtraction; otherwise play the model's other.

## 4. Weighted htdemucs / htdemucs_6s averages (weights chosen on 6 tracks, tested on the other 6)

Weight w = share of htdemucs, in {0, 0.25, 0.5, 0.75, 1}. Half A and half B alternate in track order. Mean uSDR at shifts 0:

| stem | A at w = 0 / .25 / .5 / .75 / 1 | picked on A → held-out Δ on B vs 0.5 | B at w = 0 / .25 / .5 / .75 / 1 | picked on B → held-out Δ on A vs 0.5 |
|---|---|---|---|---|
| vocals | 10.21 / 10.36 / **10.41** / 10.34 / 10.17 | 0.50 → 0 | 10.28 / **10.45** / 10.42 / 10.19 / 9.82 | 0.25 → −0.04 |
| drums | 9.91 / 10.15 / **10.27** / 10.25 / 10.10 | 0.50 → 0 | 9.52 / 9.85 / **10.03** / 10.03 / 9.86 | 0.75 → −0.02 |
| bass | 6.23 / 6.51 / 6.70 / **6.78** / 6.73 | 0.75 → **−0.37** | 9.91 / **10.07** / 9.82 / 9.44 / 9.01 | 0.25 → **−0.19** |
| other | 5.85 / 6.17 / 6.37 / **6.43** / 6.35 | 0.75 → −0.29 | 5.42 / **5.72** / 5.68 / 5.39 / 4.96 | 0.25 → −0.20 |

The same holds for the TTA'd stems (`tsw`, `t4` in tables.md): the only held-out gain is +0.07 dB on bass for one `t4` split.

**Verdict:** the optimum is flat around 0.5, and per-stem tuning overfits: it loses 0.2–0.4 dB on held-out bass and other. **Keep the plain mean.**

## 5. Wiener post-filter (norbert EM, 1 iteration, STFT 4096 / hop 1024, 300-frame blocks)

It costs **0.09 CPU-s per audio-s** and hurts every stem of every recipe it was tried on:

| recipe | vocals | drums | bass | other |
|---|---|---|---|---|
| `hd_s0` | −0.12 | −0.89 | −1.27 | −0.35 |
| `avg_s0` | −0.40 | −0.77 | −1.72 | −0.40 |
| `k+avg_s0` | **−2.65** | −0.67 | −1.72 | −0.05 |
| `k+chain_avg` | −2.11 | −1.12 | −1.69 | −1.01 |

HT Demucs' waveform outputs are already better than a mixture-phase spatial filter can make them, and the filter drags Kim's vocals down to the Demucs level. **Rejected.**

## 6. Guitar

No guitar ground truth in MUSDB. The proxy fits the htdemucs_6s guitar estimate, per 1 s window, on the reference vocals, drums, bass and other. It reports the share of energy explained by vocals + drums + bass, i.e. bleed (lower is better). Over the 9 songs whose multitracks contain guitar:

| guitar stem from | bleed | Δ vs `6s_s0` |
|---|---|---|
| 6s on the mix, shifts 0 | −14.61 dB | – |
| 6s, swap TTA | −14.62 dB | −0.01 (6/9) |
| 6s, all-four TTA | −14.58 dB | +0.03 (4/9) |
| **6s on Kim's instrumental** | **−16.67 dB** | **−2.06 (9/9)** |

It can't see keys in the guitar stem or guitar missing from it. A real guitar measurement is proposed below.

## 7. Literature check (2024–2026): what's new since the survey

Only things [../survey.md](../survey.md) doesn't already cover.

| candidate | what | quality (source) | CPU / size | weights licence | verdict |
|---|---|---|---|---|---|
| **MVSep Mega 53-stem BS-RoFormer** (ZFTurbo, MSST release v1.0.21, 2026-04-20) | one BS-RoFormer (dim 256, depth 12, 10 s chunks) with 53 output stems, incl. guitar, electric-guitar, acoustic-guitar, vocals, drums, bass, piano | MVSEP Guitar leaderboard: an entry named `bs_mega_53stem_guitar_mvsep.ckpt` scores **guitar 8.25 dB, against 5.25 for htdemucs_6s**. Rank 18; the best is 9.12. It is not certain that this is the released checkpoint. The release notes say per-stem quality is below MVSep's specialised models. | 1.37 GB fp32 checkpoint (≈342 M params, nearly all in the 53 mask heads). The backbone is about Kim-sized, so a pruned "guitar (+ vocals/drums/bass) only" run should cost roughly what Kim costs (≈5–7 CPU-s per audio-s at overlap 2). That is an estimate from the config, not measured. | **MIT**, stated by the trainer: "I have released the model weights for 53-stem model under the MIT License" ([MSST #245](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/245), 2026-09-25). He also says he doesn't hold the training audio's copyright, and the training data is not disclosed. | **The most important lead for guitar.** Not run here: there's no guitar ground truth to score it on, and its MUSDB scores could be inflated if MUSDB test was in its training data. Run it first once the guitar set below exists. |
| **BS-RoFormer 4-stem, MUSDB-only** (`model_bs_roformer_ep_17_sdr_9.6568`, MSST v1.0.12) | 4-stem BS-RoFormer | Multisong: vocals 9.19, drums 11.29, bass 11.08, other 5.96, against htdemucs at 8.24 / 10.88 / 11.76 / 5.74 ([MSST list](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)). Its bass is worse than htdemucs. | 527 MB fp32 (≈132 M params, dim 384, depth 8): several times htdemucs' CPU | **MIT**, stated by the trainer ([MSST #254](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/254), 2026-09-30); trained on MUSDB18-HQ | Not cheap, and no better than Kim + `avg_s0` on any stem we'd use it for. Skip. |
| **SCNet XL IHF** (MSST v1.0.15) | 4-stem SCNet, wider high-frequency bands | Multisong avg **9.92**: vocals 9.68, drums 11.58, bass 11.94, other 6.48. That beats htdemucs on all four stems: +1.44 / +0.70 / +0.18 / +0.74. | 214 MB fp32 (≈53 M params). SCNet's paper says its base model's CPU time is 48 % of HT Demucs; XL is bigger (not measured). | **Not declared.** A request to confirm MIT, like the two above, has been open since 2026-10-12 without an answer (#245). Trained on MUSDB18 train only. Its checkpoint was picked by MUSDB test SDR, so our MUSDB test tracks would flatter it. | **Best 4-stem candidate for a cheap drums/bass/other upgrade.** Not run: the licence isn't clear yet. Next step: wait for, or ask for, ZFTurbo's answer in #245; then measure CPU and Multisong-style uSDR. |
| **Multichannel Wiener filter post-processing** (norbert, MIT; Open-Unmix's default) | re-filter the 4 stems' STFTs with an EM-fitted spatial model, so they sum to the mix | no Demucs number found | 0.09 CPU-s per audio-s (measured) | MIT code, no weights | **Run here: −0.4 to −2.7 dB, rejected** (§5). |
| **TTA: channel swap + polarity inversion** (MSST paper, [arXiv 2607.23395](https://arxiv.org/html/2607.23395v1), Table 3) | average the plain pass with swapped / inverted passes | +0.00 to +0.06 dB on RoFormer, SCNet and MDX23C | one extra pass each | – | **Run here:** +0.01 to +0.06 dB on typical tracks, the same as shifts (§1). |
| **Time-domain weighted ensembling** (same paper, Table 4) | weighted mean of models | +0.20 to +0.46 dB over the best member | sum of members | – | **Run here:** the 50/50 mean is optimal; tuned weights lose on held-out tracks (§4). |
| Lightweight real-time separators: RT-STT ([2511.13146](https://arxiv.org/abs/2511.13146)), Band-SCNet (Interspeech 2025) | causal, < 3 M params | MUSDB 5.2 and 7.8 dB | tiny | not found | Far below htdemucs (8.8). Not useful when quality comes first. |
| MSS+ GAN post-processing (2026) | GAN refiner for light separators' outputs | perceptual | extra network | not checked | Targets light models; skip. |


## 8. Measuring the guitar stem without MoisesDB or MVSEP

MUSDB has no guitar stem. The 6-stem model's guitar is what Fretboard's mixer mutes, and the strongest lead (Mega 53, above) is a guitar model. So guitar needs ground truth of its own.

### Proposal, in order of cost

1. **Now, no new data: a guitar "bleed" proxy on MUSDB** (computed this round; results in §6: the chain cuts guitar bleed by 2.1 dB on 9/9 songs).
   - Guitar lives inside MUSDB's "other". A guitar estimate should be explained by the other reference and not by vocals, drums or bass.
   - Per 1 s window and channel, fit the guitar estimate by least squares on the four reference stems. Report the energy explained by vocals + drums + bass (bleed: lower is better) and the unexplained rest (artefacts plus anything else).
   - It can't see keys leaking into guitar, or guitar missing from the stem. It only ranks variants of one model against each other, e.g. 6s on the mix vs 6s on Kim's instrumental. It is not an SDR.
2. **Real guitar ground truth from the MUSDB songs' own multitracks (Cambridge-MT): needs the owner's OK.**
   - The 12 test songs come from the Cambridge-MT "Mixing Secrets" library, whose raw multitracks keep every guitar on its own track. Using the 2017 instrument metadata ([SiddGururani/mixing_secrets](https://github.com/SiddGururani/mixing_secrets)), **9 of the 12 have guitar tracks**:
     - Al James: 7
     - Angels In Amplifiers: 3
     - BKS: 23, but no longer listed
     - Ben Carrigan: 3 acoustic
     - Enda Reilly: 1
     - Forkupines: 21
     - Juliet's Rescue: 12
     - Signe Jakobsen: 12
     - The Long Wait: 9
   - "Edited Excerpt" downloads are 33–124 MB each, so the 8 still listed come to **≈0.6 GB**. Add 10–20 more rock / alt-rock projects from the library's genre pages at 30–125 MB each.
   - Build our own mixture as the sum of the raw tracks at fixed gains. Ground truth is the sum of the guitar tracks at the same gains. The official MUSDB mix is processed, so its guitar can't be recovered exactly.
   - Score guitar uSDR exactly as for the other stems, over windows where the guitar is active.
   - **Licence:** "provided free of charge for educational purposes only". The FAQ says research use "hasn't been specifically agreed with contributors". That is the same footing as MUSDB, which is built from these tracks and already in use here. It is not an open licence, so **the owner decides**. The site is also behind a Cloudflare bot check now, so a person would have to download the ZIPs.
   - Caveat: htdemucs was trained on MUSDB train, which is also Cambridge-MT, so pick projects outside MUSDB train. htdemucs_6s was trained on Meta data only. Mega 53's data is unknown.
3. **Strictly CC BY, real guitar, synthetic songs** (no approval issue, ≈1–2 GB):
   - Guitar sources:
     - [Guitar-TECHS](https://zenodo.org/records/14963133): CC BY 4.0, real electric guitar, DI and amp-miked; the musical part P3 is 130 MB.
     - [GuitarSet](https://zenodo.org/records/3371780): CC BY 4.0, real acoustic; mic / pickup mixes ≈0.7 GB each.
     - [EGDB-PG](https://zenodo.org/records/12674910): CC BY 4.0, 1 GB, short clips.
   - Mix them over CC BY backing tracks. Slakh2100's non-guitar stems are CC BY 4.0, but only as one 104 GB tar, so above the 5 GB line and not downloaded. BabySlakh is 16 kHz, so unusable.
   - Weakness: the guitar and the backing don't share key or tempo, so the model can't use musical context. Use it only to rank models against each other.
4. **Downstream, already licensed:** Fretboard's own tab-onset check on its songs, reading the 6s guitar stem instead of the 4-stem other, plus a blind listen of "no guitar" mixes on 5–6 songs. This needs no ground truth, and it is what users hear.

Excluded:
- MedleyDB and GOAT need access requests.
- IDMT-SMT-Guitar is CC BY-NC-ND.
- The Open Multitrack Testbed is offline.
- Weathervane needs a membership.
- MoisesDB and MVSEP are excluded by the owner.

**Recommendation:** ask the owner to approve route 2 for about 20 songs (≈2 GB). It is the only route with real full-band guitar, and its licence is no worse than MUSDB's. Then score htdemucs_6s guitar, chain-6s guitar and Mega 53's guitar.

## Recommendation (quality, then CPU, then size)

Δ uSDR vs today's two splits. The first number is vs `6s_s2`, which feeds the mixer; the second is vs `hd_s2`, the 4-stem split. Mean over 12 tracks, with tracks improved in brackets. CPU uses round 2's CPU rates; today is 2 × 1.61 + 2 × 1.65 = 6.52 CPU-s per audio-s.

| rank | recipe | vocals | drums | bass | other | CPU-s / audio-s | weights (fp16) |
|---|---|---|---|---|---|---|---|
| **1** | **Kim vocals → htdemucs + htdemucs_6s on the instrumental, shifts 0, drums / bass / other averaged; guitar and piano from 6s on the instrumental** (`k+chain_avg`) | **+3.44 / +3.68 (12/12)** | **+0.65 (12/12) / +0.39 (11/12)** | +0.17 / +0.31 | **+1.24 (11/12) / +1.19 (12/12)** | **8.97** (+38 %) | 456 + 84 + 55 MB |
| 2 | Kim → htdemucs_6s only, on the instrumental (`k+chain_6s`) | +3.44 / +3.68 | +0.25 (11/12) / −0.01 | +0.15 / +0.30 | +0.90 (12/12) / +0.86 (10/12) | **7.36** (+13 %) | 456 + 55 MB |
| 3 | htdemucs + htdemucs_6s on the mix, shifts 0, averaged (`avg_s0`, round 2) | +0.12 / +0.37 | +0.36 / +0.11 | +0.17 / +0.31 | +0.32 / +0.28 | **3.26** (−50 %) | 84 + 55 MB |
| 4 | today (`hd_s2` + `6s_s2`) | 0 | 0 | 0 | 0 | 6.52 | 84 + 55 MB |

- **Rank 1 is the quality pick.**
  - Every stem is better than today, and it has the cleanest guitar stem (−2.1 dB bleed).
  - Its stems sum to the mix within −26 dB.
  - Its "other" is +0.91 dB better than Kim + the round 2 average on the mix, at the same CPU.
  - Compared with rank 2, the extra htdemucs pass on the instrumental buys about +0.4 dB on drums, bass and other.
- **Rank 2 costs about today's CPU** and still beats today on every stem the mixer uses. One exception: drums are level with the 4-stem split. It is the choice if the +38 % of rank 1 matters.
- **Rank 3 is the no-Kim fallback** at half today's CPU.
- **In all of them: shifts 0, no TTA, no Wiener filter, plain 50/50 averaging, overlap 0.25, fp16 weights** (round 2).

### Caveats

- **Round 1's downstream checks didn't move** with Kim vocals or with the average (chords, onset sync, sung-pitch match). The mixer is what these SDR gains are for.
- **The two-model average lost one song's tab-sync verdict in round 1**, still under investigation. Rank 1 averages too, so check that song's verdict with rank 1's stems before switching the tab aligner's inputs.
- **Kim weights are MIT.** The Demucs weights have no clear grant, and htdemucs_6s is likely CC BY-NC (round 1).
- **CPU numbers come from a loaded M1 Pro.** The Rust port's own Kim timing (being built by another agent) and the container's CPU will set the real cost.

## What to try next

1. **Port the chain in htdemucs-rs.** It's plumbing: separate Kim → subtract → run the Demucs model(s) on the instrumental. Then A/B it in Fretboard's own checks, including the song whose tab-sync verdict the average lost.
2. **Get guitar ground truth** (proposal below), then score:
   - 6s guitar on the mix vs on Kim's instrumental;
   - **MVSep Mega 53's guitar** (MIT, MVSEP Guitar 8.25 vs 5.25 for 6s);
   - Mega 53's guitar on Kim's instrumental.
3. **SCNet XL IHF** (Multisong 9.92 vs 9.16 for htdemucs, beats it on all 4 stems, ≈53 M params). Ask, or wait for, ZFTurbo's licence answer in MSST #245. If MIT, time it on the CPU and run it in the chain in place of htdemucs.
4. **Kim at overlap 1 or a shorter context**, to cut its 5.7 CPU-s per audio-s, which is two thirds of rank 1's cost. Measure vocals uSDR against overlap 2 on these 12 tracks.
5. **Full 50-track MUSDB test** for the final pick (7.9 GB more to fetch), to firm up bass, which LCF dominates on 12 tracks.

## How to rerun

```bash
cd /Users/divyansh/htdemucs-rs/research/round3            # data/ is git-ignored
MODE=mps ./r3_kim.sh > data/kim_mps.log 2>&1 &              # Kim vocals (MPS transformer + CPU STFT)
MODE=cpu THREADS=2 ./r3_kim.sh > data/kim_cpu.log 2>&1 &    # optional second Kim worker, reverse order
./r3_queue.sh > data/queue.log 2>&1 &                       # 10 Demucs passes per track on MPS (DEV=cpu for the CPU)
OMP_NUM_THREADS=1 uv run --python 3.11.17 --with-requirements r3_env.txt python -I r3_score.py /Users/divyansh/datasets/musdb18hq/test &
touch data/stop                                             # scorer exits when nothing is left; it deletes passes once scored
uv run --python 3.11.17 --with-requirements r3_env.txt python -I r3_summary.py > tables.md
```
