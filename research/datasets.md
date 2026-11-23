# Evaluation data on this Mac

| Set | Where | Source | Licence | On disk |
|---|---|---|---|---|
| MUSDB18-HQ, test subset of 12 tracks | `/Users/divyansh/datasets/musdb18hq/test/<track>/{mixture,vocals,drums,bass,other}.wav` (outside both git repos) | Zenodo record 3338373, file `musdb18hq.zip` (22,656,664,047 bytes): https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1. Rafii, Liutkus, Stöter, Mimilakis, Bittner (2019) | **Research / educational, non-commercial.** Zenodo lists "Other (Non-Commercial)"; the dataset notice says it is for educational purposes and must not be used commercially without the copyright holders' permission; some tracks are CC BY-NC-SA 3.0/4.0 | **2.3 GB** (60 WAV files). The 22.7 GB zip was never stored: the files were read straight out of the remote zip with HTTP range requests (`research/round1/musdb_fetch.py`, Python `remotezip` 0.12.3) |

The 12 tracks (picked to lean towards guitar-band music, like Fretboard's songs): AM Contra – Heart Peripheral; Al James – Schoolboy Facination; Angels In Amplifiers – I'm Alright; BKS – Too Much; Ben Carrigan – We'll Talk About It All Tonight; Cristina Vane – So Easy; Enda Reilly – Cur An Long Ag Seol; Forkupines – Semantics; Juliet's Rescue – Heartbeats; Little Chicago's Finest – My Own; Signe Jakobsen – What Have You Done To Me; The Long Wait – Dark Horses.

The whole test set is 50 tracks, 11.0 GB uncompressed (7.87 GB to transfer). Fetch the rest only when a result is close:

```bash
cd /Users/divyansh/htdemucs-rs/research/round1
uv run --python 3.11.17 --with remotezip==0.12.3 python -I musdb_fetch.py list                        # sizes
uv run --python 3.11.17 --with remotezip==0.12.3 python -I musdb_fetch.py get /Users/divyansh/datasets/musdb18hq   # all 50; stops if < 10 GB would stay free
```

Not approved yet: MVSEP uploads (results become public), MoisesDB access requests.

## Round 4: guitar ground truth

None of these files are in git. Mixes built from them live in `/Users/divyansh/datasets/r4/<set>/<item>/` (float32 WAV: `mixture`, `guitar` and the other parts, which sum to the mixture exactly, plus `meta.json`).

| Set | Where | Source | Licence | Used as |
|---|---|---|---|---|
| **Guitar-TECHS** P3_music (12 musical excerpts, electric, amp-miked `micamp_*.wav`, 48 kHz) | `/Users/divyansh/datasets/guitar-techs/P3_music/` (130 MB zip, kept) | Zenodo [14963133](https://zenodo.org/records/14963133), file `P3_music.zip` (129,505,089 bytes). Pedroza et al., ICASSP 2025 | **CC BY 4.0** (Zenodo licence field) | `gtechs`: 12 items, 21–73 s, over MUSDB test vocals + drums + bass (`r4_mixopen.py`) |
| **GuitarSet** mono mic recordings, 72 files (players 00–05; progressions BN1, Funk2, Jazz3, Rock1, Rock3, SS2; comp + solo) | `/Users/divyansh/datasets/guitarset/` (≈180 MB) | Zenodo [3371780](https://zenodo.org/records/3371780), members of `audio_mono-mic.zip` read with HTTP range requests (`round4/zen_fetch.py`). Xi et al., ISMIR 2018 | **CC BY 4.0** | `gset`: 12 items (a player's comp then solo take of one progression), 39–87 s, acoustic |
| **EGDB-PG** test split (25 clips: EGDB's DI guitar rendered through random BIAS FX2 amp presets, no delay / reverb / modulation) | `/Users/divyansh/datasets/egdb/` (≈110 MB) | Zenodo [12674910](https://zenodo.org/records/12674910), `test/*.wav` members of `EGDB_BIAS_FX2_random.zip` (range requests) | **CC BY 4.0** | `egdb`: 12 items of 1–3 consecutive clips, 47–68 s, electric with amp tone |
| **Cambridge-MT "Mixing Secrets" multitracks**: Edited Excerpt zips of 18 rock / alt-rock projects not in MUSDB18 (by song or artist) | `/Users/divyansh/datasets/cambridge-mt/excerpts/` (≈0.9 GB) | `mtkdata.cambridgemusictechnology.co.uk` / `multitracks.cambridge-mt.com` direct download links from the library manifest (`/Users/divyansh/datasets/cambridge-mt/manifest.json`). Track lists and groups: [round4/cmt_manifest.json](round4/cmt_manifest.json) | **"Provided free of charge for educational purposes only"**; research use not explicitly agreed with contributors (FAQ). Not an open licence. The owner accepted it for internal evaluation: never redistribute, never commit | `cmtx`: 18 items, 18–39 s, mixed with `r4_mixcmt.py` (fixed gains, no effects) |
| **Cambridge-MT Full multitracks** of 8 of the 12 MUSDB test songs (BKS isn't listed any more; Cristina Vane, AM Contra and Little Chicago's Finest have no guitar tracks) | `/Users/divyansh/datasets/cambridge-mt/zips/` (≈3.6 GB, downloaded by the coordinator) | same library | same as above | `cmt8`: our own exact-sum remix of the songs MUSDB was built from |
| **Backing for the open sets:** MUSDB18-HQ test, 12 tracks, vocals + drums + bass stems only | as round 1 | as round 1 | Research / non-commercial (as above) | backing for `gtechs`, `gset`, `egdb` |

Not used:
- **Slakh2100** (CC BY 4.0) is one 104 GB `tar.gz` on Zenodo ([4599666](https://zenodo.org/records/4599666)). Listing the first 150 MB of the stream shows the members are not stored track by track: one track's stems, MIDI and mix are spread across the whole archive (train, validation, test and omitted interleaved). A prefix of the stream therefore holds no complete track, and a gzip stream can't be range-read from the middle. Skipped.
- **IDMT-SMT-Guitar** (CC BY-NC-ND), MedleyDB / GOAT (access requests), MoisesDB / MVSEP (excluded by the owner).
