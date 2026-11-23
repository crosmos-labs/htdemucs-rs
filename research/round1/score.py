"""Run Fretboard's own aligner checks (guitar-tabs scripts/sync.py and probe.py chordscan, read-only) on one
variant's stems for one song, and write the numbers the pipeline gates on.
    score.py <requests.json> <variant dir> <out.json>
The chords request is the chords stage's (already at the sounding shift, so shift 0 here = the declared
shift); 'chroma' points at the variant's harmonic. The pinned tab part reads the variant's vocals/other/bass,
as runSync does with the 4-stem split. 'vocalPart' is extra: the Songsterr vocal part aligned on its own.
"""
import json, os, subprocess, sys, time

GT = '/Users/divyansh/guitar-tabs'
PY = ['uv', 'run', '--quiet', '--python', '3.11.17', '--with-requirements', 'scripts/requirements-audio.txt', 'python']
reqs, vdir, outp = sys.argv[1:4]
song = os.path.basename(os.path.dirname(os.path.abspath(vdir)))
R = json.load(open(reqs))[song]
audio = f'{GT}/.cache/audio/{song}.m4a'
def stem(k):
    for ext in ('wav', 'mp3'):
        p = os.path.join(os.path.abspath(vdir), f'{k}.{ext}')
        if os.path.exists(p): return p
def run(script, req):
    t = time.time()
    r = subprocess.run(PY + [script], input=json.dumps(req), capture_output=True, text=True, cwd=GT)
    if r.returncode: return {'error': r.stderr.strip()[-300:]}, time.time() - t
    return json.loads(r.stdout), time.time() - t

out = {'song': song, 'variant': os.path.basename(os.path.abspath(vdir))}
c = dict(R['chords']); c.pop('track', None); c.pop('kind', None)
scan, dt = run('scripts/probe.py', {'kind': 'chordscan', 'audio': audio, 'beatsPerBar': c.get('beatsPerBar'), 'chords': c['chords'],
                                    'chroma': stem('harmonic'), 'beatBpm': c.get('beatBpm'), 'spotShifts': [0], 'spotCount': 999})
if 'error' in scan: out['chords'] = scan
else:
    q = scan['fitByShift'][0]; sp = scan['spots']['0']['summary']
    out['chords'] = {'fitOverEven': round(q['fit'] - q['evenlySpread'], 4), 'fit': q['fit'], 'spotRank': round(sp['rank1'] / max(1, sp['n']), 4),
                     'spotBass': round(sp['bassMatch'] / max(1, sp['n']), 4), 'nSpots': sp['n'], 'bestShiftIsDeclared': scan['bestShift'] == 0,
                     'margin': scan['margin'], 'seconds': round(dt, 1)}
def notes(req):
    req = {**req, 'kind': 'notes', 'audio': audio, 'vocals': stem('vocals'), 'other': stem('other'), 'bass': stem('bass')}
    req.pop('track', None)
    s, dt = run('scripts/sync.py', req)
    if 'error' in s: return s
    q = s['quality']
    return {**{k: q.get(k) for k in ['sungPitchMatch', 'onSungNote', 'notesWhileSinging', 'onsetWithin60ms', 'snappedToOnset', 'warpCost']},
            'firstTime': s['times'][0] if s['times'] else None, 'times': s['times'], 'seconds': round(dt, 1)}
out['tab'] = notes(R['tab']); out['tab']['melody'] = R['tabPart']['melody']
vp = [p for p in R.get('vocalParts', []) if 'req' in p]
if vp: out['vocalPart'] = notes(vp[0]['req'])
os.makedirs(os.path.dirname(outp), exist_ok=True)
json.dump(out, open(outp, 'w'))
print(outp, json.dumps({k: (v if k != 'tab' and k != 'vocalPart' else {kk: vv for kk, vv in v.items() if kk != 'times'}) for k, v in out.items()}))
