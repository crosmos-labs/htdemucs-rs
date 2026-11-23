"""Build round 4's openly licensed guitar test mixes: a real guitar recording over an unrelated backing.
    r4_mixopen.py <set: gtechs|gset|egdb> <out root> [--musdb DIR] [--gdir DIR] [--n 12] [--rel -3]
Guitar (CC BY 4.0):
  gtechs  Guitar-TECHS P3_music/audio/micamp/micamp_NN.wav (electric, amp-miked, 48 kHz dual-mono → mono, resampled to 44.1 kHz)
  gset    GuitarSet audio_mono-mic: one player's comp take then solo take of the same progression, concatenated
  egdb    EGDB-PG (EGDB DI rendered through BIAS FX2) test/*.wav, consecutive clips concatenated up to >= 45 s
Backing: a MUSDB18-HQ test track's vocals + drums + bass (its "other", where its own guitars live, left out), the segment
starting 30 s in, wrapping if needed. Item i uses track (i + offset[set]) mod 12, so the three sets pair differently.
Mixing: guitar mono, centred (same signal both channels), scaled so its RMS over the item is `rel` dB below the backing's RMS;
mixture = guitar + vocals + drums + bass exactly (float32 sums, no effects). If the mixture peaks above 0.99, every part is
scaled by the same factor. Writes <out>/<set>/<item>/{mixture,guitar,vocals,drums,bass}.wav (float32) and meta.json.
"""
import argparse, glob, json, os, re
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

ap = argparse.ArgumentParser()
ap.add_argument('set'); ap.add_argument('out')
ap.add_argument('--musdb', default='/Users/divyansh/datasets/musdb18hq/test'); ap.add_argument('--gdir', default=None)
ap.add_argument('--n', type=int, default=12); ap.add_argument('--rel', type=float, default=-3.0)
a = ap.parse_args()
SR = 44100
OFF = {'gtechs': 0, 'gset': 4, 'egdb': 8}[a.set]
tracks = sorted(os.listdir(a.musdb))

def mono(path):
    x, sr = sf.read(path, dtype='float64', always_2d=True); x = x.mean(1)
    if sr != SR:
        assert sr == 48000, sr; x = resample_poly(x, 147, 160)
    return x

items = []
if a.set == 'gtechs':
    d = a.gdir or '/Users/divyansh/datasets/guitar-techs/P3_music/audio/micamp'
    for f in sorted(glob.glob(f'{d}/micamp_*.wav'))[:a.n]:
        items.append((os.path.basename(f)[:-4], [f]))
elif a.set == 'gset':
    d = a.gdir or '/Users/divyansh/datasets/guitarset'
    comps = sorted(glob.glob(f'{d}/*_comp_mic.wav'))
    # one item per (player, progression), cycling genres so each item has a different player/genre pair
    by = {}
    for c in comps:
        p, prog = re.match(r'.*/(\d\d)_([A-Za-z]+\d)-', c).groups()
        if os.path.exists(c.replace('_comp_', '_solo_')): by.setdefault(prog, []).append((p, c))
    progs = sorted(by); k = 0
    while len(items) < a.n:
        prog = progs[k % len(progs)]; lst = by[prog]; p, c = lst[(k // len(progs)) % len(lst)]
        items.append((os.path.basename(c).replace('_comp_mic.wav', ''), [c, c.replace('_comp_', '_solo_')])); k += 1
elif a.set == 'egdb':
    d = a.gdir or '/Users/divyansh/datasets/egdb'
    fs = sorted(glob.glob(f'{d}/*test__*.wav'), key=lambda f: int(re.search(r'__(\d+)\.wav$', f).group(1)))
    i = 0
    while len(items) < a.n and i < len(fs):
        grp = []; dur = 0
        while i < len(fs) and dur < 45:
            grp.append(fs[i]); dur += sf.info(fs[i]).duration; i += 1
        items.append(('egdb_' + '+'.join(re.search(r'__(\d+)\.wav$', f).group(1) for f in grp), grp))

for idx, (name, files) in enumerate(items):
    g = np.concatenate([mono(f) for f in files]); n = len(g)
    tr = tracks[(idx + OFF) % len(tracks)]
    st = {}
    for s in ('vocals', 'drums', 'bass'):
        x = sf.read(f'{a.musdb}/{tr}/{s}.wav', dtype='float64', always_2d=True)[0]
        x = np.concatenate([x[30 * SR:], x[:30 * SR]])
        st[s] = np.tile(x.T, (1, -(-n // len(x))))[:, :n]               # wraps when the track is shorter than the guitar
    back = st['vocals'] + st['drums'] + st['bass']
    rb = np.sqrt((back ** 2).mean()); rg = np.sqrt((g ** 2).mean())
    gain = rb / rg * 10 ** (a.rel / 20)
    st['guitar'] = np.stack([g, g]) * gain
    mix = sum(st.values()); pk = np.abs(mix).max(); sc = min(1.0, 0.99 / pk)
    od = f'{a.out}/{a.set}/{idx:02d}_{name}'; os.makedirs(od, exist_ok=True)
    for s, x in list(st.items()) + [('mixture', mix)]:
        sf.write(f'{od}/{s}.wav', (x * sc).T.astype(np.float32), SR, subtype='FLOAT')
    meta = {'set': a.set, 'guitar_files': [os.path.basename(f) for f in files], 'backing': tr, 'backing_start_s': 30,
            'seconds': round(n / SR, 2), 'guitar_rel_db': a.rel, 'guitar_gain': gain * sc, 'global_scale': sc}
    json.dump(meta, open(f'{od}/meta.json', 'w'), indent=1)
    print(od, meta['seconds'], tr, flush=True)
