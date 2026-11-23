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
