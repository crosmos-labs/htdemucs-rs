"""Round 3 tables from data/scores/<variant>/<track>.json (r3_score.py) and round 2's scores / CPU logs. Prints Markdown.
    r3_summary.py
CPU-s per audio-s of a variant = the sum of its passes' CPU rates measured in round 2 on the CPU (htdemucs pass, htdemucs_6s
pass, Kim at 1 thread), since a swapped / inverted / instrumental pass is the same network call on an input of the same length
(round 3 ran its passes on MPS to keep within the CPU budget; outputs match the CPU's, see the reproduction check), plus the
Wiener filter's own measured CPU (data/wf_cpu.json) for _wf variants.
"""
import glob, json, os, random
import numpy as np

def load(root):
    S = {}
    for f in glob.glob(f'{root}/*/*.json'):
        d = json.load(open(f)); S.setdefault(d['variant'], {})[d['track']] = d
    return S
S = load('data/scores'); S2 = load('../round2/data/scores')
for v in ('hd_s2', '6s_s2', 'avg_s2', 'hd_s1', 'ftv_s0'):
    S['r2:' + v] = S2.get(v, {})
tracks = sorted({t for t in S.get('hd_s0', {})})
ST = ['vocals', 'drums', 'bass', 'other']; STA = ST + ['accompaniment']
COMBOS = {'s0': 1, 'tsw': 2, 'tpo': 2, 'tswpo': 2, 't4': 4}

# ---- CPU model
cpu = {}
for l in open('../round2/data/sep/cpu.jsonl'):
    r = json.loads(l)
    if r['weights'] == 'f32' and r['overlap'] == 0.25 and 's0' in r['passes']:
        x = cpu.setdefault(r['tag'], [0.0, 0.0]); x[0] += r['passes']['s0']['cpu_s']; x[1] += r['audio_s']
for l in open('../round2/data/sep/kim_cpu.jsonl'):
    r = json.loads(l); x = cpu.setdefault('kim', [0.0, 0.0]); x[0] += r['cpu_s']; x[1] += r['audio_s']
RATE = {k: c / a for k, (c, a) in cpu.items()}
WFR = {}
if os.path.exists('data/wf_cpu.json'):
    for k, xs in json.load(open('data/wf_cpu.json')).items(): WFR[k] = sum(c for c, a in xs) / sum(a for c, a in xs)
def cost(v):
    if v.startswith('r2:'):
        b = v[3:]; return {'hd_s2': 2 * RATE['hd'], '6s_s2': 2 * RATE['6s'], 'avg_s2': 2 * (RATE['hd'] + RATE['6s']), 'hd_s1': RATE['hd'], 'ftv_s0': None}[b]
    extra = 0.0
    if v.endswith('_wf'): v = v[:-3]; extra = WFR.get(v, float('nan'))
    if v.endswith('_fix'): v = v[:-4]
    if v.endswith('_rv'): v = v[:-3]
    k = 0.0
    if v.startswith('k+'): k = RATE['kim']; v = v[2:]
    if v == 'kim': return RATE['kim']
    if v.startswith('chain_'):
        m = v[6:]; return k + extra + {'hd': RATE['hd'], '6s': RATE['6s'], 'avg': RATE['hd'] + RATE['6s']}[m]
    m, c = v.split('_'); n = COMBOS[c]
    return k + extra + n * {'hd': RATE['hd'], '6s': RATE['6s']}.get(m, RATE['hd'] + RATE['6s'])

def val(v, t, s, m='uSDR'): return S.get(v, {}).get(t, {}).get(s, {}).get(m)
def mean(v, s, ts=None, m='uSDR'):
    xs = [val(v, t, s, m) for t in (ts or tracks)]; xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs and len(xs) == len(ts or tracks) else None
def paired(v, b, s, ts=None, boot=2000):
    ts = [t for t in (ts or tracks) if val(v, t, s) is not None and val(b, t, s) is not None]
    if not ts: return None
    d = np.array([val(v, t, s) - val(b, t, s) for t in ts]); rng = random.Random(0)
    bs = sorted(np.mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(boot))
    return d.mean(), bs[int(0.025 * boot)], bs[int(0.975 * boot)], int((d > 0).sum()), len(d)
f = lambda x: '–' if x is None else f'{x:.2f}'
def pc(r): return '–' if r is None else f'{r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] {r[3]}/{r[4]}'
def row(v, label=None):
    m = [mean(v, s) for s in ST]
    m4 = f(np.mean(m)) if all(x is not None for x in m) else '–'
    rd = [S[v][t].get('resid_dB') for t in tracks if v in S and t in S[v] and 'resid_dB' in S[v][t]]
    c = cost(v)
    return f'| `{label or v}` | ' + ' | '.join(f(x) for x in m) + f' | {m4} | {f(mean(v, "accompaniment"))} | {f(np.mean(rd)) if rd else "–"} | {f(c)} |'
HDR = '| variant | vocals | drums | bass | other | mean of 4 | accomp. | resid dB | CPU-s / audio-s |\n|---|---|---|---|---|---|---|---|---|'

print(f'Tracks scored: {len(tracks)} of 12. Mean over tracks of per-track uSDR (dB). CPU rates (round 2, CPU): '
      f'htdemucs pass {RATE["hd"]:.2f}, htdemucs_6s pass {RATE["6s"]:.2f}, Kim {RATE["kim"]:.2f} (1 thread) CPU-s per audio-s; '
      + ', '.join(f'Wiener on {k} {r:.3f}' for k, r in WFR.items()) + '.\n')

print('## Reproduction check (round 3 on MPS vs round 2 on CPU, same tracks): largest |Δ uSDR| over tracks and stems\n')
for a, b in [('hd_s0', 'hd_s0'), ('6s_s0', '6s_s0'), ('avg_s0', 'avg_s0'), ('kim', 'kim')]:
    ds = [abs(val(a, t, s) - S2[b][t][s]['uSDR']) for t in tracks for s in STA if val(a, t, s) is not None and s in S2.get(b, {}).get(t, {})]
    print(f'- `{a}`: {max(ds):.4f} dB over {len(ds)} pairs' if ds else f'- `{a}`: –')

print('\n## 1. Test-time augmentation vs random shifts\n')
print(HDR)
for v in ['hd_s0', 'r2:hd_s2', 'hd_tsw', 'hd_tpo', 'hd_tswpo', 'hd_t4', '6s_s0', 'r2:6s_s2', '6s_tsw', '6s_tpo', '6s_tswpo', '6s_t4',
          'avg_s0', 'r2:avg_s2', 'avg_tsw', 'avg_tpo', 'avg_tswpo', 'avg_t4']:
    print(row(v))
print('\nPaired Δ uSDR vs the same model at shifts 0 (2-pass variants cost one extra pass; `s2` = round 2\'s two random shifts, CPU run):\n')
print('| variant | vs | ' + ' | '.join(ST) + ' |\n|---|---|' + '---|' * 4)
for m in ('hd', '6s', 'avg'):
    for c in ('r2', 'tsw', 'tpo', 'tswpo', 't4'):
        v = f'r2:{m}_s2' if c == 'r2' else f'{m}_{c}'
        print(f'| `{v}` | `{m}_s0` | ' + ' | '.join(pc(paired(v, f'{m}_s0', s)) for s in ST) + ' |')

BASES = [('r2:hd_s2', "today's 4-stem (htdemucs, shifts 2)"), ('r2:6s_s2', "today's mixer (htdemucs_6s, shifts 2)"), ('avg_s0', 'avg_s0')]
def vs_table(vs):
    for b, lab in BASES:
        print(f'\nPaired Δ uSDR vs `{b}` ({lab}):\n')
        print('| variant | ' + ' | '.join(STA) + ' |\n|---|' + '---|' * 5)
        for v in vs:
            if v != b: print(f'| `{v}` | ' + ' | '.join(pc(paired(v, b, s)) for s in STA) + ' |')

print('\n## 2. Kim vocals with Demucs on the full mix vs on Kim\'s instrumental (UVR chain)\n')
KV = ['kim', 'k+hd_s0', 'k+chain_hd', 'k+chain_hd_rv', 'k+chain_hd_fix', 'k+6s_s0', 'k+chain_6s', 'k+chain_6s_rv', 'k+chain_6s_fix',
      'k+avg_s0', 'k+avg_s0_fix', 'k+chain_avg', 'k+chain_avg_rv', 'k+chain_avg_fix', 'k+avg_tsw', 'k+avg_t4']
print(HDR)
for v in KV: print(row(v))
print('\nPaired Δ uSDR, chain vs the same Demucs on the full mix:\n')
print('| chain | direct | ' + ' | '.join(ST[1:]) + ' |\n|---|---|' + '---|' * 3)
for m in ('hd', '6s', 'avg'):
    for sfx in ('', '_rv', '_fix'):
        print(f'| `k+chain_{m}{sfx}` | `k+{m}_s0{sfx if sfx == "_fix" else ""}` | ' + ' | '.join(pc(paired(f'k+chain_{m}{sfx}', f'k+{m}_s0' + (sfx if sfx == '_fix' else ''), s)) for s in ST[1:]) + ' |')

print('\n## 3. Per-stem best-of, and forcing the stems to sum to the mix\n')
print('`_fix` replaces other with mix − vocals − drums − bass. resid dB = energy of mix − Σ stems relative to the mix, before the fix.\n')
print('| variant | other | other `_fix` | Δ other | resid dB | other `_wf` (Wiener) | vocals `_wf` − vocals | drums `_wf` − drums | bass `_wf` − bass |\n|---|---|---|---|---|---|---|---|---|')
for v in ['hd_s0', '6s_s0', 'avg_s0', 'avg_t4', 'k+hd_s0', 'k+6s_s0', 'k+avg_s0', 'k+avg_t4', 'k+chain_hd', 'k+chain_6s', 'k+chain_avg']:
    o, ofx = mean(v, 'other'), mean(v + '_fix', 'other')
    rd = [S[v][t].get('resid_dB') for t in tracks if t in S.get(v, {})]
    wf = v + '_wf' in S
    dd = lambda s: f'{mean(v + "_wf", s) - mean(v, s):+.2f}' if wf and mean(v + '_wf', s) is not None else '–'
    print(f'| `{v}` | {f(o)} | {f(ofx)} | {(ofx - o) if o is not None and ofx is not None else 0:+.2f} | {f(np.mean(rd)) if rd else "–"} | '
          f'{f(mean(v + "_wf", "other")) if wf else "–"} | {dd("vocals")} | {dd("drums")} | {dd("bass")} |')
print('\nPaired Δ uSDR of `_fix` and `_wf` vs the plain variant:\n')
print('| variant | vs | ' + ' | '.join(ST) + ' |\n|---|---|' + '---|' * 4)
for v in ['hd_s0', 'avg_s0', 'k+avg_s0', 'k+chain_avg']:
    for sfx in ('_fix', '_wf'):
        if v + sfx in S: print(f'| `{v}{sfx}` | `{v}` | ' + ' | '.join(pc(paired(v + sfx, v, s)) for s in ST) + ' |')

print('\n## 4. Weighted htdemucs / htdemucs_6s average, weights chosen on one half and tested on the other\n')
A = tracks[0::2]; B = tracks[1::2]
print(f'Half A (fit): {", ".join(t.split(" - ")[0] for t in A)}. Half B: {", ".join(t.split(" - ")[0] for t in B)}.\n')
W = [0, 25, 50, 75, 100]
def wname(w, c): return {0: f'6s_{c}', 50: f'avg_{c}', 100: f'hd_{c}'}.get(w, f'w{w}_{c}')
for c in ('s0', 'tsw', 't4'):
    print(f'\n### `{c}` (weight = share of htdemucs)\n')
    print('| stem | half | ' + ' | '.join(f'w={w / 100:.2f}' for w in W) + ' | picked on this half | held-out Δ vs w=0.50 |\n|---|---|' + '---|' * (len(W) + 2))
    for s in STA:
        if s == 'accompaniment': continue
        for half, other_half, hn in ((A, B, 'A'), (B, A, 'B')):
            ms = [mean(wname(w, c), s, half) for w in W]
            if any(x is None for x in ms): continue
            wb = W[int(np.argmax(ms))]
            ho = mean(wname(wb, c), s, other_half) - mean(wname(50, c), s, other_half)
            print(f'| {s} | {hn} | ' + ' | '.join(f(x) for x in ms) + f' | {wb / 100:.2f} | {ho:+.3f} on {"B" if hn == "A" else "A"} |')

KEY = ['r2:hd_s2', 'r2:6s_s2', 'hd_s0', 'avg_s0', 'avg_tsw', 'avg_t4', 'kim', 'k+avg_s0', 'k+avg_s0_fix', 'k+avg_s0_wf', 'k+chain_avg',
       'k+chain_avg_fix', 'k+chain_avg_wf', 'k+avg_t4', 'avg_s0_wf']
print('\n## 5. Key variants\n')
print(HDR)
for v in KEY:
    if v in S: print(row(v))
vs_table([v for v in KEY if v in S and v not in ('r2:hd_s2',)])
print('\n### cSDR (museval v4, median of 1 s windows), where computed\n')
print('| variant | ' + ' | '.join(STA) + ' |\n|---|' + '---|' * 5)
for v in sorted(S):
    if any(val(v, t, 'vocals', 'cSDR') is not None or val(v, t, 'drums', 'cSDR') is not None for t in tracks) and not v.startswith('r2:'):
        print(f'| `{v}` | ' + ' | '.join(f(mean(v, s, m='cSDR')) for s in STA) + ' |')

print('\n## All variants\n')
print(HDR)
for v in sorted(S):
    if not v.startswith('r2:'): print(row(v))
print('\n## Per track uSDR (dB), key variants\n')
for s in ST:
    cols = [v for v in KEY if any(val(v, t, s) is not None for t in tracks)]
    print(f'\n### {s}\n\n| track | ' + ' | '.join(f'`{v}`' for v in cols) + ' |\n|---|' + '---|' * len(cols))
    for t in tracks: print(f'| {t[:30]} | ' + ' | '.join(f(val(v, t, s)) for v in cols) + ' |')
