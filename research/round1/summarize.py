"""Tables for round 1 from data/results/<set>/<song>/<variant>.json.
    summarize.py [--md]
Noise = the same variant across Demucs seeds (seed0 = pipeline cache, seed1, seed2): per song, max − min.
Effect = variant − V0 on the same seed, averaged over the seeds available; reported with the songs' noise.
"""
import glob, json, os, statistics as st, sys
import numpy as np

R = 'data/results'
res = {}
for f in glob.glob(f'{R}/*/*/*.json'):
    s, song, v = f.split('/')[-3], f.split('/')[-2], f.split('/')[-1][:-5]
    res[(s, song, v)] = json.load(open(f))
sets = sorted({k[0] for k in res}); songs = sorted({k[1] for k in res}); vars_ = sorted({k[2] for k in res}, key=lambda v: ['V0', 'V0L', 'V1a', 'V1b', 'K'].index(v) if v in ['V0', 'V0L', 'V1a', 'V1b', 'K'] else 9)
M = [('chords', 'fitOverEven'), ('chords', 'spotRank'), ('chords', 'spotBass'), ('tab', 'onsetWithin60ms'), ('tab', 'sungPitchMatch'), ('vocalPart', 'sungPitchMatch'), ('vocalPart', 'onSungNote')]
def get(s, song, v, sec, m):
    r = res.get((s, song, v), {}).get(sec) or {}
    x = r.get(m)
    if sec == 'tab' and m == 'onsetWithin60ms' and r.get('melody') != 'instrument': return None
    if sec == 'tab' and m == 'sungPitchMatch' and r.get('melody') != 'voice': return None
    return x
def tshift(s, song, v, ref=('seed0', 'V0')):
    a = (res.get((s, song, v), {}).get('tab') or {}).get('times'); b = (res.get((ref[0], song, ref[1]), {}).get('tab') or {}).get('times')
    if not a or not b or len(a) != len(b): return None
    return float(np.median(np.abs(np.array(a) - np.array(b))))
md = '--md' in sys.argv
def row(cells): return '| ' + ' | '.join(str(c) for c in cells) + ' |' if md else '  '.join(f'{str(c):>12}' for c in cells)
fmt = lambda x: '–' if x is None else (f'{x:.3f}' if isinstance(x, float) else str(x))

print('## Per song (seed0 = pipeline cache), with the V0 range over seeds\n')
for sec, m in M:
    print(f'\n### {sec}.{m}\n')
    print(row(['song'] + vars_ + ['V0 seeds', 'V0 range']))
    if md: print(row(['---'] * (len(vars_) + 3)))
    for song in songs:
        vals = [get('seed0', song, v, sec, m) for v in vars_]
        if all(x is None for x in vals): continue
        v0s = [get(s, song, 'V0', sec, m) for s in sets]; v0s = [x for x in v0s if x is not None]
        print(row([song[:28]] + [fmt(x) for x in vals] + [' / '.join(fmt(x) for x in v0s), fmt(max(v0s) - min(v0s)) if len(v0s) > 1 else '–']))

for BASE in ('V0', 'V0L'):
    print(f'\n## Summary vs {BASE}: mean over songs of (variant − {BASE}, same seed), mean over seeds; noise = mean V0 range over seeds\n')
    print(row(['metric', 'songs', 'V0 noise (mean range)', 'V0 noise (max range)'] + [f'{v} − {BASE}' for v in vars_ if v != BASE] + [f'{v} wins/ties/losses' for v in vars_ if v != BASE]))
    if md: print(row(['---'] * (4 + 2 * (len(vars_) - 1))))
    for sec, m in M:
        ss = [song for song in songs if get('seed0', song, BASE, sec, m) is not None]
        if not ss: continue
        ranges = []
        for song in ss:
            v0s = [get(s, song, 'V0', sec, m) for s in sets]; v0s = [x for x in v0s if x is not None]
            if len(v0s) > 1: ranges.append(max(v0s) - min(v0s))
        noise = st.mean(ranges) if ranges else None; noisemax = max(ranges) if ranges else None
        cells, wtl = [], []
        for v in vars_:
            if v == BASE: continue
            per = []
            for song in ss:
                d = [get(s, song, v, sec, m) - get(s, song, BASE, sec, m) for s in sets if get(s, song, v, sec, m) is not None and get(s, song, BASE, sec, m) is not None]
                if d: per.append(st.mean(d))
            cells.append(f'{st.mean(per):+.3f} (n={len(per)})' if per else '–')
            thr = noisemax if noisemax is not None else 0.0
            wtl.append(f'{sum(x > thr for x in per)}/{sum(abs(x) <= thr for x in per)}/{sum(x < -thr for x in per)}' if per else '–')
        print(row([f'{sec}.{m}', len(ss), fmt(noise), fmt(noisemax)] + cells + wtl))


print('\n## Tab placement: median |Δ note time| vs seed0 V0 (s)\n')
print(row(['song'] + [f'{s}:{v}' for s in sets for v in vars_]))
if md: print(row(['---'] * (1 + len(sets) * len(vars_))))
for song in songs:
    print(row([song[:28]] + [fmt(tshift(s, song, v)) for s in sets for v in vars_]))

print('\n## Sung-pitch match at a fixed placement (seed0 V0L note times), per seed: at the note time / 100 ms later\n')
F = {}
for f in glob.glob('data/fixed/*/*.json'):
    s, song = f.split('/')[-2], f.split('/')[-1][:-5]
    for p, r in json.load(open(f)).items():
        F[(s, song, p.split('/')[-2])] = r
fsongs = sorted({k[1] for k in F}); fvars = [v for v in ['V0', 'V0L', 'V1a', 'V1b', 'K'] if any(k[2] == v for k in F)]
print(row(['song', 'reading'] + fvars))
if md: print(row(['---'] * (2 + len(fvars))))
for song in fsongs:
    for rd in ('atOnset', 'plus100ms'):
        cells = []
        for v in fvars:
            xs = [F[(s, song, v)][rd]['sungPitchMatch'] for s in sets if (s, song, v) in F]
            cells.append(' / '.join(f'{x:.3f}' for x in xs) if xs else '–')
        print(row([song[:28], rd] + cells))
