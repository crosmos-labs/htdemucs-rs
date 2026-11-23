"""Round 2 tables from data/scores/<variant>/<track>.json and the CPU logs. Prints Markdown.
    r2_summary.py
"""
import glob, json, os, random
import numpy as np

S = {}
for f in glob.glob('data/scores/*/*.json'):
    d = json.load(open(f)); S.setdefault(d['variant'], {})[d['track']] = d
tracks = sorted({t for v in S.values() for t in v})
STEMS = ['vocals', 'drums', 'bass', 'other', 'accompaniment']

# CPU seconds per audio second, per variant
cpu = {}
def add(v, c, a):
    x = cpu.setdefault(v, [0.0, 0.0]); x[0] += c; x[1] += a
if os.path.exists('data/sep/cpu.jsonl'):
    for l in open('data/sep/cpu.jsonl'):
        r = json.loads(l); p = r['passes']; a = r['audio_s']; t = r['tag']
        if 's0' in p: add(f'{t}_s0', p['s0']['cpu_s'], a)
        if 'A' in p: add(f'{t}_s1', p['A']['cpu_s'], a)
        if 'A' in p and 'B' in p: add(f'{t}_s2', p['A']['cpu_s'] + p['B']['cpu_s'], a)
if os.path.exists('data/sep/kim_cpu.jsonl'):
    for l in open('data/sep/kim_cpu.jsonl'):
        r = json.loads(l); add('kim', r['cpu_s'], r['audio_s'])
for avg, (a, b) in {'avg_s0': ('hd_s0', '6s_s0'), 'avg_s2': ('hd_s2', '6s_s2')}.items():
    if a in cpu and b in cpu: cpu[avg] = [cpu[a][0] + cpu[b][0], cpu[a][1]]
rate = lambda v: (cpu[v][0] / cpu[v][1]) if v in cpu and cpu[v][1] else None

ORDER = ['hd_s0', 'hd_s1', 'hd_s2', 'hdo10_s0', 'hdo50_s0', 'hdo10_s1', 'hdo10_s2', 'hdo50_s1', 'hdo50_s2', 'hdbf16_s0', 'hdint8_s0',
         '6s_s0', '6s_s1', '6s_s2', 'avg_s0', 'avg_s2', 'ftv_s0', 'kim']
LABEL = {'hd_s0': 'htdemucs, shifts 0, overlap 0.25', 'hd_s1': 'htdemucs, shifts 1', 'hd_s2': 'htdemucs, shifts 2 (today, 4-stem)',
         'hdo10_s0': 'htdemucs, shifts 0, overlap 0.1', 'hdo50_s0': 'htdemucs, shifts 0, overlap 0.5',
         'hdo10_s1': 'htdemucs, shifts 1, overlap 0.1', 'hdo10_s2': 'htdemucs, shifts 2, overlap 0.1',
         'hdo50_s1': 'htdemucs, shifts 1, overlap 0.5', 'hdo50_s2': 'htdemucs, shifts 2, overlap 0.5',
         'hdbf16_s0': 'htdemucs, shifts 0, weights rounded to bf16', 'hdint8_s0': 'htdemucs, shifts 0, int8 per-channel weights',
         '6s_s0': 'htdemucs_6s, shifts 0 (other = gtr+pno+other)', '6s_s1': 'htdemucs_6s, shifts 1', '6s_s2': 'htdemucs_6s, shifts 2 (today, mixer)',
         'avg_s0': 'mean of htdemucs + 6s, shifts 0', 'avg_s2': 'mean of htdemucs + 6s, shifts 2', 'ftv_s0': 'htdemucs_ft vocals model, shifts 0',
         'kim': 'Kim Mel-RoFormer (vocals; accompaniment = mix − vocals)'}
vars_ = [v for v in ORDER if v in S] + sorted(v for v in S if v not in ORDER)
def val(v, t, s, m):
    return S.get(v, {}).get(t, {}).get(s, {}).get(m)
def mean(v, s, m, ts=None):
    xs = [val(v, t, s, m) for t in (ts or tracks)]; xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), len(xs)) if xs else (None, 0)
f = lambda x: '–' if x is None else f'{x:.2f}'

print(f'Tracks scored: {len(tracks)} (MUSDB18-HQ test subset, full length). Mean over tracks of per-track uSDR / cSDR (dB).\n')
print('## All variants\n')
print('| variant | n | vocals | drums | bass | other | mean of 4 | accompaniment | CPU-s per audio-s |')
print('|---|---|---|---|---|---|---|---|---|')
for v in vars_:
    cells = []
    for s in STEMS[:4]:
        u, n = mean(v, s, 'uSDR'); c, _ = mean(v, s, 'cSDR'); cells.append(f'{f(u)} / {f(c)}' if u is not None else '–')
    m4u = [mean(v, s, 'uSDR')[0] for s in STEMS[:4]]; m4c = [mean(v, s, 'cSDR')[0] for s in STEMS[:4]]
    m4 = f'{np.mean(m4u):.2f} / {np.mean(m4c):.2f}' if all(x is not None for x in m4u + m4c) else '–'
    au, n = mean(v, 'accompaniment', 'uSDR'); ac, _ = mean(v, 'accompaniment', 'cSDR')
    nn = max(mean(v, s, 'uSDR')[1] for s in STEMS)
    print(f'| {LABEL.get(v, v)} (`{v}`) | {nn} | ' + ' | '.join(cells) + f' | {m4} | {f(au)} / {f(ac)} | {f(rate(v))} |')

def paired(v, base, s, m='uSDR', boot=2000):
    ts = [t for t in tracks if val(v, t, s, m) is not None and val(base, t, s, m) is not None]
    d = np.array([val(v, t, s, m) - val(base, t, s, m) for t in ts])
    if not len(d): return None
    rng = random.Random(0); bs = [np.mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(boot)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((d > 0).sum()), len(d)
for base in ['hd_s2', 'hd_s0']:
    print(f'\n## Paired Δ uSDR vs `{base}` (mean over tracks, 95% bootstrap CI, tracks improved / n)\n')
    print('| variant | ' + ' | '.join(STEMS) + ' |'); print('|---|' + '---|' * len(STEMS))
    for v in vars_:
        if v == base: continue
        cells = []
        for s in STEMS:
            r = paired(v, base, s)
            cells.append('–' if r is None else f'{r[0]:+.3f} [{r[1]:+.2f}, {r[2]:+.2f}] {r[3]}/{r[4]}')
        print(f'| `{v}` | ' + ' | '.join(cells) + ' |')

print('\n## Weight precision vs f32 (shifts 0): Δ uSDR and Δ cSDR per stem, mean and largest |Δ| over tracks\n')
print('| weights | ' + ' | '.join(STEMS) + ' |'); print('|---|' + '---|' * len(STEMS))
for v in ['hdbf16_s0', 'hdint8_s0']:
    if v not in S: continue
    cells = []
    for s in STEMS:
        du = [val(v, t, s, 'uSDR') - val('hd_s0', t, s, 'uSDR') for t in tracks if val(v, t, s, 'uSDR') is not None and val('hd_s0', t, s, 'uSDR') is not None]
        dc = [val(v, t, s, 'cSDR') - val('hd_s0', t, s, 'cSDR') for t in tracks if val(v, t, s, 'cSDR') is not None and val('hd_s0', t, s, 'cSDR') is not None]
        cells.append(f'{np.mean(du):+.4f} (max {np.max(np.abs(du)):.4f}) / {np.mean(dc):+.4f}' if du else '–')
    print(f'| `{v}` | ' + ' | '.join(cells) + ' |')
for w in glob.glob('data/sep/*_weights.json'):
    print('\n', json.dumps(json.load(open(w))))

print('\n## Per track uSDR (dB)\n')
for s in STEMS:
    cols = [v for v in vars_ if any(val(v, t, s, 'uSDR') is not None for t in tracks)]
    print(f'\n### {s}\n'); print('| track | ' + ' | '.join(f'`{v}`' for v in cols) + ' |'); print('|---|' + '---|' * len(cols))
    for t in tracks:
        print(f'| {t[:34]} | ' + ' | '.join(f(val(v, t, s, 'uSDR')) for v in cols) + ' |')
