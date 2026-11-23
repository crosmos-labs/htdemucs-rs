"""Round 3 scoring worker (1 thread). For each track whose passes are all on disk (data/pass/<track>/{hd,6s}_{id,sw,po,swpo,inst}.npy
from r3_sep.py, data/kim/<track>/vocals.wav), builds every variant in memory, scores it, writes
data/scores/<variant>/<track>.json (round 2's format), marks the track scored and deletes its passes and Kim vocals.
    r3_score.py <musdb test dir> [--keep] [--once]
Variants (all shifts 0, overlap 0.25):
  <m>_<c>        m = hd (htdemucs), 6s (htdemucs_6s, other = guitar+piano+other), avg (mean of the two);
                 c = s0 (plain pass), tsw (mean of plain + L/R-swapped pass), tpo (plain + polarity-inverted),
                 tswpo (plain + swapped-and-inverted), t4 (mean of all four)
  w25_<c>, w75_<c>   0.25·hd + 0.75·6s and 0.75·hd + 0.25·6s (w0 = 6s, w50 = avg, w100 = hd)
  kim            Kim Mel-RoFormer vocals (accompaniment = mix − vocals)
  k+<base>       vocals from Kim, drums / bass / other from <base> run on the full mix (any <m>_<c> above)
  k+chain_<m>    vocals from Kim, drums / bass / other from Demucs <m> run on Kim's instrumental (mix − Kim vocals);
                 Demucs' own (leftover) vocals output is dropped
  k+chain_<m>_rv the same, with Demucs' leftover vocals added to other
  <name>_fix     <name> with other replaced by mix − vocals − drums − bass, so the four stems sum to the mix exactly
Each score file also records resid_dB = 10·log10(Σr² / Σmix²), r = mix − (vocals + drums + bass + other), before any fix.
  <name>_wf      multichannel Wiener filter (norbert EM, 1 iteration, STFT 4096 / hop 1024, 300-frame blocks as in
                 Open-Unmix) seeded with <name>'s four complex STFTs; outputs sum to the mix. Only for the names in WF.
uSDR only (museval cSDR was too slow on this shared machine; round 2 has cSDR for the base variants). Accumulated over 7 s chunks.
"""
import glob, json, os, shutil, sys, time
import numpy as np
import soundfile as sf

M = sys.argv[1]; KEEP = '--keep' in sys.argv; ONCE = '--once' in sys.argv
PASS, KIM, SC = 'data/pass', 'data/kim', 'data/scores'
MODELS, PASSES = ('hd', '6s'), ('id', 'sw', 'po', 'swpo', 'inst')
COMBOS = {'s0': ['id'], 'tsw': ['id', 'sw'], 'tpo': ['id', 'po'], 'tswpo': ['id', 'swpo'], 't4': ['id', 'sw', 'po', 'swpo']}
ST = ['vocals', 'drums', 'bass', 'other']; SI = {s: i for i, s in enumerate(ST)}
WF = ['hd_s0', 'avg_s0', 'k+avg_s0', 'k+chain_avg']
WF_CPU = {}

def usdr(s, e):
    return float(10 * np.log10(((s ** 2).sum() + 1e-8) / (((s - e) ** 2).sum() + 1e-8)))

def wiener_seg(X, Ys, iters=1, nfft=4096, hop=1024):
    """Multichannel Wiener filter (norbert EM) of one waveform segment: X (2, L) mixture, Ys list of (2, L) estimates."""
    import torch, norbert
    win = torch.hann_window(nfft, dtype=torch.float64)
    st = lambda a: torch.stft(torch.as_tensor(np.asarray(a, dtype=np.float64)), nfft, hop, window=win, return_complex=True).numpy().transpose(2, 1, 0)
    Y = norbert.expectation_maximization(np.stack([st(e) for e in Ys], -1), st(X), iterations=iters)[0]
    return [torch.istft(torch.as_tensor(Y[..., j].transpose(2, 1, 0).copy()), nfft, hop, window=win, length=X.shape[1]).numpy() for j in range(len(Ys))]

def bleed_acc(acc, g, R, win=44100):
    """Guitar proxy (no guitar reference in MUSDB): per 1 s window and channel, least-squares fit of the guitar estimate on the
    reference vocals, drums, bass and other; accumulates energy explained by vocals+drums+bass (bleed), by other, the rest, total."""
    for c in range(2):
        for b in range(0, g.shape[1] - win + 1, win):
            A = np.stack([R[s][c, b:b + win] for s in ST], 1).astype(np.float64); y = g[c, b:b + win].astype(np.float64)
            if (y ** 2).sum() < 1e-10: continue
            co = np.linalg.lstsq(A, y, rcond=None)[0]
            acc[0] += ((A[:, :3] @ co[:3]) ** 2).sum(); acc[1] += ((A[:, 3] * co[3]) ** 2).sum()
            acc[2] += ((y - A @ co) ** 2).sum(); acc[3] += (y ** 2).sum()

def ready(tr):
    return (os.path.exists(f'{KIM}/{tr}/done') and not os.path.exists(f'{PASS}/{tr}/scored')
            and all(os.path.exists(f'{PASS}/{tr}/{m}_{p}.npy') for m in MODELS for p in PASSES))

def score_track(tr, chunk=7 * 44100, pad=4096):
    """Everything is accumulated over time chunks (sums of squares), so only one chunk of every pass is in memory at once."""
    import torch; torch.set_num_threads(1)
    t0 = time.time()
    rd = {s: sf.SoundFile(f'{M}/{tr}/{s}.wav') for s in ['mixture'] + ST}
    n = rd['mixture'].frames
    kf = sf.SoundFile(f'{KIM}/{tr}/vocals.wav'); assert kf.frames == n
    P = {(m, p): np.load(f'{PASS}/{tr}/{m}_{p}.npy', mmap_mode='r') for m in MODELS for p in PASSES}
    assert all(x.shape[1:] == (2, n) for x in P.values())
    names = None; A = {}; RES = {}; GB = {}; wf_cpu = {k: 0.0 for k in WF}
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk); elo, ehi = max(0, lo - pad), min(n, hi + pad); a, b = lo - elo, hi - elo
        def rdw(f):
            f.seek(elo); return f.read(ehi - elo, dtype='float32', always_2d=True).T
        RE = {s: rdw(rd[s]) for s in ['mixture'] + ST}; mixE = RE['mixture']; K = rdw(kf)
        C = {k: np.asarray(v[:, :, elo:ehi], dtype=np.float32) for k, v in P.items()}
        def base(m, c, s):
            if m == 'avg': return (base('hd', c, s) + base('6s', c, s)) / 2
            if m.startswith('w'):
                w = int(m[1:]) / 100; return w * base('hd', c, s) + (1 - w) * base('6s', c, s)
            ps = COMBOS[c]; return sum(C[(m, p)][SI[s]] for p in ps) / len(ps)
        def chain(m, s):
            if m == 'avg': return (chain('hd', s) + chain('6s', s)) / 2
            return C[(m, 'inst')][SI[s]]
        V = {}
        for m in ('hd', '6s', 'avg', 'w25', 'w75'):
            for c in COMBOS: V[f'{m}_{c}'] = {s: (lambda m=m, c=c, s=s: base(m, c, s)) for s in ST}
        V['kim'] = {'vocals': lambda: K}
        for m in ('hd', '6s', 'avg'):
            for c in COMBOS: V[f'k+{m}_{c}'] = {'vocals': lambda: K, **{s: (lambda m=m, c=c, s=s: base(m, c, s)) for s in ST[1:]}}
            V[f'k+chain_{m}'] = {'vocals': lambda: K, **{s: (lambda m=m, s=s: chain(m, s)) for s in ST[1:]}}
            V[f'k+chain_{m}_rv'] = {'vocals': lambda: K, 'drums': lambda m=m: chain(m, 'drums'), 'bass': lambda m=m: chain(m, 'bass'),
                                    'other': lambda m=m: chain(m, 'other') + chain(m, 'vocals')}
        for name in [v for v in V if set(V[v]) == set(ST)]:
            fx = dict(V[name]); fx['other'] = (lambda g=V[name]: mixE - g['vocals']() - g['drums']() - g['bass']()); V[name + '_fix'] = fx
        for name in WF:
            c0 = time.process_time()
            out = wiener_seg(mixE, [V[name][s]() for s in ST])
            wf_cpu[name] += time.process_time() - c0
            V[name + '_wf'] = {s: (lambda e=out[i]: e) for i, s in enumerate(ST)}
        names = list(V)
        R = {s: RE[s][:, a:b].astype(np.float64) for s in ST}; mix = RE['mixture'][:, a:b].astype(np.float64)
        accref = R['drums'] + R['bass'] + R['other']
        for name, g in V.items():
            tot = np.zeros_like(mix) if (set(g) == set(ST) and not name.endswith('_fix')) else None
            for s, f in g.items():
                e = np.asarray(f()[:, a:b], dtype=np.float64)
                x = A.setdefault((name, s), [0.0, 0.0]); x[0] += (R[s] ** 2).sum(); x[1] += ((R[s] - e) ** 2).sum()
                if s == 'vocals':
                    x = A.setdefault((name, 'accompaniment'), [0.0, 0.0]); x[0] += (accref ** 2).sum(); x[1] += ((accref - (mix - e)) ** 2).sum()
                if tot is not None: tot += e
            if tot is not None:
                x = RES.setdefault(name, [0.0, 0.0]); x[0] += ((mix - tot) ** 2).sum(); x[1] += (mix ** 2).sum()
        for c in COMBOS:
            bleed_acc(GB.setdefault(f'6s_{c}', [0.0] * 4), sum(C[('6s', p)][4] for p in COMBOS[c])[:, a:b] / len(COMBOS[c]), R)
        bleed_acc(GB.setdefault('k+chain_6s', [0.0] * 4), C[('6s', 'inst')][4][:, a:b], R)
    oth = A[('hd_s0', 'other')][0]
    for name in names:
        out = {'track': tr, 'variant': name, 'seconds': round(n / 44100, 2)}
        for s in ST + ['accompaniment']:
            if (name, s) in A:
                num, den = A[(name, s)]; out[s] = {'uSDR': round(float(10 * np.log10((num + 1e-8) / (den + 1e-8))), 4)}
        if name in RES: out['resid_dB'] = round(float(10 * np.log10(RES[name][0] / RES[name][1])), 2)
        os.makedirs(f'{SC}/{name}', exist_ok=True); json.dump(out, open(f'{SC}/{name}/{tr}.json', 'w'))
    for name, (ev, eo, er, et) in GB.items():
        db = lambda x: round(float(10 * np.log10(x / et + 1e-12)), 3)
        os.makedirs(f'data/guitar/{name}', exist_ok=True)
        json.dump({'track': tr, 'variant': name, 'vdb_bleed_dB': db(ev), 'other_fit_dB': db(eo), 'unexplained_dB': db(er),
                   'energy_vs_other_dB': round(float(10 * np.log10(et / oth)), 3)}, open(f'data/guitar/{name}/{tr}.json', 'w'))
    for k, v in wf_cpu.items(): WF_CPU.setdefault(k, []).append((v, n / 44100))
    json.dump(WF_CPU, open('data/wf_cpu.json', 'w'))
    del P, C
    open(f'{PASS}/{tr}/scored', 'w').write(str(round(time.time() - t0, 1)))
    if not KEEP:
        for f in glob.glob(f'{PASS}/{tr}/*.npy'): os.remove(f)
        for f in glob.glob(f'{KIM}/{tr}/*.wav'): os.remove(f)
    print('scored', tr, len(names), 'variants', round(time.time() - t0, 1), 's', flush=True)

if os.path.exists('data/wf_cpu.json'): WF_CPU = json.load(open('data/wf_cpu.json'))
while True:
    todo = [tr for tr in sorted(os.listdir(M)) if ready(tr)]
    for tr in todo: score_track(tr)
    if ONCE or (not todo and os.path.exists('data/stop')): break
    if not todo: time.sleep(30)
