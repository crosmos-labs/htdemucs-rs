"""Score round 4's guitar estimates against exact ground truth.
    r4_score.py <items root>/<set> <pass root>/<set> <score dir>/<set> [--csdr] [--items a,b]
Per item (a dir with mixture.wav, guitar.wav and the other true parts, which sum to the mixture) and per variant:
  usdr      guitar uSDR = 10·log10(Σg² / Σ(g − ĝ)²), whole item, both channels
  nog_usdr  "guitar removed": the mixer's mute plays mix − ĝ; true no-guitar mix is mix − g, so its error is g − ĝ too:
            10·log10(Σ(mix − g)² / Σ(g − ĝ)²)
  bleed_*   per 1 s window and channel, least-squares fit of ĝ on all true parts (guitar, vocals, drums, bass, keys, other);
            energy of each part's fitted share, summed over windows, in dB relative to Σĝ² (lower = less bleed).
            bleed_vdbk = vocals + drums + bass + keys; resid = what no part explains (artefacts).
  csdr      (--csdr) museval BSSEval v4 median over 1 s windows, two sources: guitar vs the rest (mix − g), with estimates
            ĝ and mix − ĝ; csdr_nog is the second source's. Two sources keep museval affordable.
Variants come from the pass files: 6s_s0, 6s_s2, 6s_inst (round 4 r4_sep.py) and mega_mix / mega_inst (r4_mega.py:
stems guitar, electric-guitar, acoustic-guitar), plus averages of pairs.
"""
import argparse, glob, json, os
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument('items'); ap.add_argument('passes'); ap.add_argument('scores'); ap.add_argument('--csdr', action='store_true')
ap.add_argument('--items-only', default=None)
a = ap.parse_args()
os.makedirs(a.scores, exist_ok=True)
PARTS = ('guitar', 'vocals', 'drums', 'bass', 'keys', 'other')

def variants(pd):
    v = {}
    for p in ('s0', 's2', 'inst'):
        f = f'{pd}/6s_{p}.npy'
        if os.path.exists(f): v[f'6s_{p}'] = np.load(f)[0]
    for p in ('mix', 'inst'):
        f = f'{pd}/mega_{p}.npy'
        if os.path.exists(f):
            m = np.load(f); v[f'mega_{p}'] = m[0]; v[f'megaEA_{p}'] = m[1] + m[2]
    for x, y in (('6s_inst', 'mega_inst'), ('6s_s0', 'mega_mix'), ('6s_inst', 'mega_mix')):
        if x in v and y in v: v[f'avg({x},{y})'] = 0.5 * (v[x] + v[y])
    return v

def db(n, d): return float(10 * np.log10((n + 1e-12) / (d + 1e-12)))

def bleed(est, parts, sr=44100):
    names = [p for p in PARTS if p in parts]; E = dict.fromkeys(names + ['resid'], 0.0); tot = 0.0
    for c in range(est.shape[0]):
        for s in range(0, est.shape[1], sr):
            y = est[c, s:s + sr].astype(np.float64)
            A = np.stack([parts[p][c, s:s + sr] for p in names], 1).astype(np.float64)
            coef = np.linalg.lstsq(A, y, rcond=1e-6)[0]
            for j, p in enumerate(names): E[p] += float(((A[:, j] * coef[j]) ** 2).sum())
            E['resid'] += float(((y - A @ coef) ** 2).sum()); tot += float((y ** 2).sum())
    out = {f'bleed_{p}': db(e, tot) for p, e in E.items()}
    out['bleed_vdbk'] = db(sum(E[p] for p in ('vocals', 'drums', 'bass', 'keys') if p in E), tot)
    return out

for it in sorted(os.listdir(a.items)):
    if a.items_only and it not in a.items_only.split(','): continue
    pd = f'{a.passes}/{it}'; dst = f'{a.scores}/{it}.json'
    if not os.path.isdir(pd): continue
    v = variants(pd)
    old = json.load(open(dst)) if os.path.exists(dst) else {}
    todo = [k for k in v if k not in old or (a.csdr and 'csdr' not in old[k])]
    if not todo: continue
    rd = lambda n: sf.read(f'{a.items}/{it}/{n}.wav', dtype='float32', always_2d=True)[0].T
    mix = rd('mixture'); parts = {p: rd(p) for p in PARTS if os.path.exists(f'{a.items}/{it}/{p}.wav')}
    g = parts['guitar']; nog = mix - g
    for k in todo:
        e = v[k][:, :g.shape[1]]; err = ((g - e) ** 2).sum()
        r = old.get(k, {}); r.update({'usdr': db((g ** 2).sum(), err), 'nog_usdr': db((nog ** 2).sum(), err)})
        if 'bleed_resid' not in r: r.update(bleed(e, parts))
        if a.csdr and 'csdr' not in r:
            import museval
            sdr = museval.evaluate(np.stack([g.T, nog.T]), np.stack([e.T, (mix - e).T]), win=44100, hop=44100)[0]
            r['csdr'] = float(np.nanmedian(sdr[0])); r['csdr_nog'] = float(np.nanmedian(sdr[1]))
        old[k] = r
    old['_meta'] = {'seconds': g.shape[1] / 44100, 'parts': list(parts), 'guitar_db_re_mix': db((g ** 2).sum(), (mix ** 2).sum())}
    json.dump(old, open(dst, 'w'), indent=1)
    print(it, {k: round(old[k]['usdr'], 2) for k in v}, flush=True)
