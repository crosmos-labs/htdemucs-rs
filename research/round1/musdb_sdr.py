"""Global SDR (MDX / MVSEP definition: 10·log10(Σ s² / Σ (s − ŝ)²) over the whole track, both channels)
and SI-SDR per stem, for separations of MUSDB18-HQ test tracks.
    musdb_sdr.py <musdb test dir> <estimates dir> [stem…]
<estimates dir>/<track name>/<stem>.{wav,flac,mp3} (decoded with ffmpeg, trimmed/padded to the reference).
Prints one JSON line per track, then the per-stem median and mean.
"""
import glob, json, os, subprocess, sys
import numpy as np

ref_root, est_root, *stems = sys.argv[1:]
stems = stems or ['vocals', 'drums', 'bass', 'other']
def dec(p):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-ac', '2', '-ar', '44100', '-f', 'f32le', '-'], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)
def sdr(s, e):
    return 10 * np.log10((s ** 2).sum() + 1e-8) - 10 * np.log10(((s - e) ** 2).sum() + 1e-8)
def sisdr(s, e):
    s, e = s.ravel(), e.ravel(); a = (e @ s) / (s @ s + 1e-12); t = a * s
    return 10 * np.log10((t @ t + 1e-12) / (((e - t) ** 2).sum() + 1e-12))
rows = []
for d in sorted(glob.glob(os.path.join(est_root, '*'))):
    name = os.path.basename(d); r = {'track': name}
    for st in stems:
        est = next((p for p in (f'{d}/{st}.{x}' for x in ('wav', 'flac', 'mp3')) if os.path.exists(p)), None)
        ref = f'{ref_root}/{name}/{st}.wav'
        if not est or not os.path.exists(ref): continue
        s = dec(ref); e = dec(est); e = np.pad(e, ((0, max(0, len(s) - len(e))), (0, 0)))[:len(s)]
        r[st] = {'sdr': round(float(sdr(s, e)), 3), 'sisdr': round(float(sisdr(s, e)), 3)}
    rows.append(r); print(json.dumps(r), flush=True)
for st in stems:
    v = [r[st]['sdr'] for r in rows if st in r]
    if v: print(f'{st}: median SDR {np.median(v):.3f}, mean {np.mean(v):.3f} (n={len(v)})')
