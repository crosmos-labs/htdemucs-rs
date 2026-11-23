"""Sung-pitch match at a FIXED placement, so the vocals stem is judged without the aligner's offset search
moving the notes (that search adds its own noise: the tab can land elsewhere). Same pitch tracker and
same rule as guitar-tabs scripts/sync.py (pYIN C3–C6, 100 ms frames, ±1 semitone pitch class).
    fixed_pitch.py <requests.json> <song> <tab|vocalPart> <reference result.json> <vocals path>...
Prints JSON per vocals path: sungPitchMatch (hits / notes while singing) and notesWhileSinging, read at the note
time (the pipeline's rule) and 100 ms later (less sensitive to a few ms of stem delay at note onsets).
"""
import json, sys
import numpy as np
sys.path.insert(0, '/Users/divyansh/guitar-tabs/scripts')
import sync as S  # noqa: E402

reqs, song, which, refp, *paths = sys.argv[1:]
R = json.load(open(reqs))[song]
req = R['tab'] if which == 'tab' else next(p['req'] for p in R['vocalParts'] if 'req' in p)
notes = req['notes']
times = json.load(open(refp))[which]['times']
assert len(times) == len(notes), (len(times), len(notes))
FR = 0.1
out = {}
for p in paths:
    v = S.load(p)
    m, dt = S.pitch_track(v)
    n_a = int(len(v) / S.SR / FR)
    am = np.full(n_a, np.nan)
    for k in range(n_a):
        seg = m[int(k * FR / dt):int((k + 1) * FR / dt)]
        voiced = seg[~np.isnan(seg)]
        if len(seg) and len(voiced) >= len(seg) * 0.5:
            am[k] = np.median(voiced)
    r = {}
    for name, off in (('atOnset', 0.0), ('plus100ms', 0.1)):
        hit = tot = 0
        for n, t in zip(notes, times):
            if not n['midi'] or n.get('tied'):
                continue
            f = int((t + off) / FR)
            if 0 <= f < n_a and not np.isnan(am[f]):
                tot += 1
                hit += abs(((am[f] - max(n['midi'])) + 6) % 12 - 6) <= 1
        r[name] = {'sungPitchMatch': round(hit / max(1, tot), 4), 'notesWhileSinging': tot}
    out[p] = r
print(json.dumps(out))
