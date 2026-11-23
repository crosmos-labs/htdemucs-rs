"""Score one variant on one MUSDB18-HQ track.
    r2_score.py <musdb test dir> <variant dir> <track> <out.json> [--no-museval]
Targets: vocals, drums, bass, other (those the variant has) and 'accompaniment' = mixture − estimated vocals
(what Fretboard plays with vocals muted), against drums + bass + other.
uSDR: 10·log10(Σs² / Σ(s − ŝ)²) over the whole track, both channels (MDX / MVSEP "SDR").
cSDR: museval (BSSEval v4) SDR per 1 s window (win = hop = 44100), median over windows (NaN windows, i.e. silent
reference, skipped), each target evaluated on its own.
"""
import json, os, sys, time
import numpy as np
import soundfile as sf

musdb, vdir, track, outp = sys.argv[1:5]
do_museval = '--no-museval' not in sys.argv
R = {s: sf.read(f'{musdb}/{track}/{s}.wav', dtype='float64', always_2d=True)[0] for s in ('mixture', 'vocals', 'drums', 'bass', 'other')}
n = len(R['mixture'])
def est(s):
    p = f'{vdir}/{track}/{s}.wav'
    if not os.path.exists(p): return None
    e = sf.read(p, dtype='float64', always_2d=True)[0]
    return np.pad(e, ((0, max(0, n - len(e))), (0, 0)))[:n]
E = {s: est(s) for s in ('vocals', 'drums', 'bass', 'other')}
E = {s: e for s, e in E.items() if e is not None}
pairs = {s: (R[s], e) for s, e in E.items()}
if 'vocals' in E:
    pairs['accompaniment'] = (R['drums'] + R['bass'] + R['other'], R['mixture'] - E['vocals'])

def usdr(s, e):
    return float(10 * np.log10(((s ** 2).sum() + 1e-8) / (((s - e) ** 2).sum() + 1e-8)))
out = {'track': track, 'variant': os.path.basename(vdir.rstrip('/')), 'seconds': round(n / 44100, 2)}
t0 = time.time()
for t, (s, e) in pairs.items():
    out.setdefault(t, {})['uSDR'] = round(usdr(s, e), 4)
if do_museval:
    import museval
    for t, (s, e) in pairs.items():
        sdr, isr, sir, sar = museval.evaluate(s[None], e[None], win=44100, hop=44100, mode='v4')
        out[t]['cSDR'] = round(float(np.nanmedian(sdr[0])), 4)
out['score_seconds'] = round(time.time() - t0, 1)
os.makedirs(os.path.dirname(outp), exist_ok=True)
json.dump(out, open(outp, 'w'))
print(json.dumps(out))
