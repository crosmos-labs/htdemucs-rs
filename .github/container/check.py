"""
Parity of the container's stems against PyTorch references (tools/ref_separate.py), SNR per stem.

    python check.py MIN_DB <stem dir> <ref.safetensors>                    one model
    python check.py MIN_DB <ensemble dir> <htdemucs ref> <htdemucs_6s ref>  -n ensemble

<stem dir> holds <stem>.wav (float32, --clip-mode none). For the ensemble, <ensemble dir> holds
htdemucs/ and htdemucs_6s/, and the reference is built from the two models' references exactly as
src/ensemble.rs combines them (each model's part of the ensemble is its own seeded run). Prints
one line per set and exits 1 if any stem fails.

A stem passes at SNR >= MIN_DB. SNR is relative to the stem's own level, so a near-silent stem
(reference below -60 dBFS RMS, e.g. htdemucs' "vocals" on the synthetic clip, at -72 dBFS) is held
to the same absolute standard instead: error at or below -(MIN_DB + 40) dBFS RMS, i.e. MIN_DB under
a -40 dBFS stem. Level and error are printed for those.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import st  # noqa: E402


def load_ref(path):
    t, meta = st.load(path)
    return dict(zip(meta["sources"].split(","), t["stems"].astype(np.float32)))


def compare(label, out_dir, ref, min_db):
    worst, row, ok = np.inf, [], True
    for name, r in ref.items():
        x, rate = sf.read(Path(out_dir) / f"{name}.wav", dtype="float32", always_2d=True)
        x = x.T
        assert x.shape == r.shape, f"{name}: shape {x.shape} vs {r.shape}"
        err = ((x.astype(np.float64) - r) ** 2).sum()
        sig = (r.astype(np.float64) ** 2).sum()
        snr = 10 * np.log10(sig / max(err, 1e-300))
        level, err_db = (10 * np.log10(max(v / r.size, 1e-30)) for v in (sig, err))
        if level < -60:
            good = err_db <= -(min_db + 40)
            row.append(f"{name} {snr:.1f} (near-silent: level {level:.0f} dBFS, error {err_db:.0f} dBFS)")
        else:
            good = snr >= min_db
            worst = min(worst, snr)
            row.append(f"{name} {snr:.1f}")
        ok &= good
    print(f"{label}: {', '.join(row)} | min {worst:.1f} dB {'PASS' if ok else 'FAIL'} (>= {min_db})")
    return ok


min_db, out = float(sys.argv[1]), sys.argv[2]
if len(sys.argv) == 4:
    ok = compare(Path(out).name, out, load_ref(sys.argv[3]), min_db)
else:
    four, six = load_ref(sys.argv[3]), load_ref(sys.argv[4])
    e4, e6 = dict(four), dict(six)
    for s in ["vocals", "drums", "bass"]:
        e4[s] = e6[s] = (four[s] + six[s]) * np.float32(0.5)
    other4 = (four["other"] + (six["guitar"] + six["piano"] + six["other"])) * np.float32(0.5)
    e4["other"], e6["other"] = other4, other4 - six["guitar"] - six["piano"]
    ok = compare("ensemble 4-stem", Path(out) / "htdemucs", e4, min_db)
    ok &= compare("ensemble 6-stem", Path(out) / "htdemucs_6s", e6, min_db)
sys.exit(0 if ok else 1)
