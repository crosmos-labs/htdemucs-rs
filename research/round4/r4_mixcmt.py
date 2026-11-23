"""Build an exact-ground-truth guitar test mix from one Cambridge-MT multitrack zip (local file).
    r4_mixcmt.py <zip> <out root>/<set> [--name NAME] [--manifest research/round4/cmt_manifest.json] [--max-s 0]
Steps (no effects anywhere; every operation is a fixed gain, so the parts sum to the mixture exactly):
 1. Read every audio file in the zip (resampled to 44.1 kHz with resample_poly if needed; zero-padded to the longest).
 2. Group each track by its file name (classify(); MUSDB's grouping: vocals / drums incl. percussion / bass / other, with
    "other" split further into guitar / keys / other). Tracks named Scratch, Click, Guide, Ref or Mix are left out, and a
    guitar track named DI is left out when the project has a non-DI guitar track (the mic'd / amp track of the part).
    The manifest's "overrides" (per project, track → group or "skip") win over the regex.
 3. Pan: mono tracks constant-power; guitar parts alternate left/right at ±0.6 when there are two or more (two mics of
    one part, e.g. ElecGtr1Mic1/Mic2, share a side), everything else
    centred; stereo files kept as they are.
 4. Within a group, tracks sum at unity (as recorded). Each group is then scaled so its "active RMS" (mean power over the
    1 s windows within 40 dB of its loudest) sits at a fixed level re the vocals: vocals 0, drums 0, bass −3, guitar −3,
    keys −7, other −7 dB. Vocals/drums/bass/other medians come from MUSDB18-HQ's 12 test songs (drums +0.1, bass −3.3,
    other −0.8 re vocals); guitar, keys and other together sum to about −0.5 dB. Without vocals, drums is the reference.
 5. The whole mix is scaled so its peak is 0.9. With --max-s, only the first max-s seconds are kept.
Writes <out>/<name>/{mixture,guitar,vocals,drums,bass,keys,other}.wav (float32; a group with no tracks is not written)
and meta.json (every track: group, pan, gain), and records the same in the manifest under "projects".
"""
import argparse, io, json, os, re, zipfile
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from math import gcd

ap = argparse.ArgumentParser()
ap.add_argument('zip'); ap.add_argument('out'); ap.add_argument('--name'); ap.add_argument('--max-s', type=float, default=0)
ap.add_argument('--manifest', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cmt_manifest.json'))
a = ap.parse_args()
SR = 44100
TARGET = {'vocals': 0, 'drums': 0, 'bass': -3, 'guitar': -3, 'keys': -7, 'other': -7}
RX = [('skip', r'scratch|click|guide|\bref\b|reference|_mix\b|^mix'),
      ('vocals', r'vox|vocal|voc\b|voice|choir|harmon(y|ies)|adlib|speech|spoken|\brap\b|shout|gang'),
      ('bass', r'bass(?!.*drum)|\bsub\b'),
      ('drums', r'kick|snare|\btom|hat|\bhh\b|overhead|\boh\b|room|ambience|cymbal|ride|crash|drum|\bkit\b|brush|tamb|'
                r'shaker|conga|bongo|clap|cowbell|perc|stomp|cajon|timbale|snap|rim|china|splash|stick|sleigh'),
      ('guitar', r'gtr|guitar|gtar|acoustic'),
      ('keys', r'piano|keys|keyboard|organ|hammond|rhodes|wurl|clav|leslie|mellotron|harpsichord|celest'),
      ('other', r'.')]

def classify(fn):
    s = re.sub(r'^\d+[_ -]*', '', os.path.splitext(os.path.basename(fn))[0]).lower()
    s = re.sub(r'([a-z])([0-9])', r'\1 \2', s)
    for g, rx in RX:
        if re.search(rx, s): return g

def active_power(x):
    m = x.mean(0) if x.ndim == 2 else x
    e = np.array([(x[..., i:i + SR] ** 2).mean() for i in range(0, x.shape[-1] - SR + 1, SR)] or [(x ** 2).mean()])
    return e[e >= e.max() * 1e-4].mean()

man = json.load(open(a.manifest)) if os.path.exists(a.manifest) else {'overrides': {}, 'projects': {}}
name = a.name or re.sub(r'(_Full)?\.zip$', '', os.path.basename(a.zip))
ov = man.get('overrides', {}).get(name, {})
z = zipfile.ZipFile(a.zip)
files = sorted(i.filename for i in z.infolist()
               if not i.is_dir() and '__MACOSX' not in i.filename and i.filename.lower().endswith(('.wav', '.aif', '.aiff', '.flac')))
tracks = []
for f in files:
    b = os.path.basename(f); g = ov.get(b, classify(b))
    tracks.append({'file': b, 'group': g, 'path': f})
gtr = [t for t in tracks if t['group'] == 'guitar']
if any(not re.search(r'\bdi\b|di$|di\d|_di', os.path.splitext(t['file'])[0].lower()) for t in gtr):
    for t in gtr:
        if t['file'] not in ov and re.search(r'(?<![a-z])di(?![a-z])|di$', re.sub(r'^\d+_', '', os.path.splitext(t['file'])[0]).lower()):
            t['group'] = 'skip'; t['why'] = 'guitar DI with a non-DI guitar present'
audio = {}
for t in tracks:
    if t['group'] == 'skip': continue
    x, sr = sf.read(io.BytesIO(z.read(t['path'])), dtype='float64', always_2d=True); x = x.T
    if sr != SR: g_ = gcd(SR, sr); x = resample_poly(x, SR // g_, sr // g_, axis=1)
    t['sr'] = sr; t['channels'] = x.shape[0]; audio[t['file']] = x
n = max(x.shape[1] for x in audio.values())
if a.max_s: n = min(n, int(a.max_s * SR))
gi = 0; parts = {}; pans = {}
part_key = lambda f: re.sub(r'mic ?\d+|\.\w+$|^\d+_', '', f.lower())   # two mics of one guitar part share a pan
for t in tracks:
    if t['group'] == 'skip': continue
    x = audio.pop(t['file'])[:, :n]; x = np.pad(x, ((0, 0), (0, n - x.shape[1])))
    if x.shape[0] == 1:
        p = 0.0
        if t['group'] == 'guitar' and len([u for u in tracks if u['group'] == 'guitar']) >= 2:
            k = part_key(t['file'])
            if k not in pans: pans[k] = 0.6 if gi % 2 else -0.6; gi += 1
            p = pans[k]
        th = (p + 1) * np.pi / 4; x = np.stack([x[0] * np.cos(th), x[0] * np.sin(th)]); t['pan'] = p
    elif x.shape[0] > 2:
        x = x[:2]
    parts[t['group']] = parts.get(t['group'], 0) + x
ref = 'vocals' if 'vocals' in parts else 'drums'
P_ref = active_power(parts[ref]); gains = {}
for g, x in parts.items():
    gains[g] = np.sqrt(P_ref / active_power(x) * 10 ** ((TARGET[g] - TARGET[ref]) / 10)); parts[g] = x * gains[g]
mix = sum(parts.values()); sc = 0.9 / np.abs(mix).max()
od = f'{a.out}/{name}'; os.makedirs(od, exist_ok=True)
for g, x in list(parts.items()) + [('mixture', mix)]:
    sf.write(f'{od}/{g}.wav', (x * sc).T.astype(np.float32), SR, subtype='FLOAT')
for t in tracks:
    t.pop('path')
    if t['group'] != 'skip': t['gain'] = float(gains[t['group']] * sc)
meta = {'project': name, 'zip': os.path.basename(a.zip), 'seconds': round(n / SR, 2), 'ref': ref, 'targets_db': TARGET,
        'group_gain': {g: float(v * sc) for g, v in gains.items()}, 'tracks': tracks}
json.dump(meta, open(f'{od}/meta.json', 'w'), indent=1)
man.setdefault('projects', {})[name] = {k: meta[k] for k in ('zip', 'seconds', 'group_gain')} | {
    'tracks': {t['file']: (t['group'] if t['group'] == 'skip' else [t['group'], round(t['gain'], 6), t.get('pan')]) for t in tracks}}
json.dump(man, open(a.manifest, 'w'), indent=1)
print(name, meta['seconds'], {g: sum(1 for t in tracks if t['group'] == g) for g in list(TARGET) + ['skip']})
