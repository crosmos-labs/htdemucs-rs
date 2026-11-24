"""Follow-up: does Mega's drum stem lose on some songs because percussion lives in other heads?
Mega 53 has separate heads for drums, percussion, tambourine, congas (and kit pieces kick/snare/toms/hh); the Cambridge-MT
drum group includes percussion, tambourine, claps and brushes tracks. On Kim's instrumental (as kim_mega), score:
  drums   the drums head (= kim_mega's drums)
  dp      drums + percussion + tambourine + congas
  kit     kick + snare + toms + hh + percussion + tambourine + congas
against the true drum group, and the paired chain drums for reference. NOT RUN YET (round 5 ended before the pass).
    r5_drums.py <items root> <pass root> <out.json>      (needs <pass>/<set>/<item>/mega_drums.npy, heads DRUM_HEADS)
"""
import json, os, sys
import numpy as np, soundfile as sf

DRUM_HEADS = ['drums', 'percussion', 'tambourine', 'congas', 'kick', 'snare', 'toms', 'hh']
items, passes, out = sys.argv[1:4]
rd = lambda f: sf.read(f, dtype='float32', always_2d=True)[0].T.astype(np.float64)
db = lambda n, d: float(10 * np.log10((n + 1e-20) / (d + 1e-20)))
res = {}
for s in ('cmtx', 'cmt8'):
    for it in sorted(os.listdir(f'{items}/{s}')):
        f = f'{passes}/{s}/{it}/mega_drums.npy'
        if not os.path.exists(f): continue
        t = rd(f'{items}/{s}/{it}/drums.wav'); n = t.shape[1]
        H = {k: np.load(f)[i, :, :n].astype(np.float64) for i, k in enumerate(DRUM_HEADS)}
        ch = rd(f'{passes}/{s}/{it}/chain/htdemucs_6s/mixture/drums.wav')
        v = {'drums': H['drums'], 'dp': H['drums'] + H['percussion'] + H['tambourine'] + H['congas'],
             'kit': sum(H[k] for k in ('kick', 'snare', 'toms', 'hh', 'percussion', 'tambourine', 'congas')), 'chain': ch}
        res[f'{s}/{it}'] = {k: db((t ** 2).sum(), ((t - e) ** 2).sum()) for k, e in v.items()}
        print(s, it, {k: round(x, 2) for k, x in res[f'{s}/{it}'].items()}, flush=True)
json.dump(res, open(out, 'w'), indent=1)
rng = np.random.default_rng(0)
for k in ('dp', 'kit', 'chain'):
    d = np.array([r[k] - r['drums'] for r in res.values()])
    m = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    print(f'{k} - drums head: {d.mean():+.2f} [{np.percentile(m, 2.5):+.2f}, {np.percentile(m, 97.5):+.2f}] {int((d > 0).sum())}/{len(d)}')
