"""Score round 5's six-stem recipes against the exact Cambridge-MT ground truth.
    r5_score.py <items root>/<set> <pass root>/<set> <score dir>/<set> [--items a,b]
Per item (dir with mixture.wav and vocals/drums/bass/guitar[/keys][/other].wav, which sum to the mixture):
  truth: vocals, drums, bass, guitar, keys (zeros when the item has none), other = mix - vocals - drums - bass - guitar - keys
         (checked against other.wav, or against silence when the item has none).
  usdr     10*log10(sum t^2 / sum (t - e)^2) over the item, both channels; null when the item has no such part (t = 0)
  removed  "stem removed" (what the mixer's mute plays, mix - e, vs the truth mix - t; error t - e again):
           10*log10(sum (mix - t)^2 / sum (t - e)^2); defined on every item
  est_db   10*log10(sum e^2 / sum mix^2): the estimate's level (the leak, when the item has no such part)
  resid    per recipe, 10*log10(sum (mix - sum of its six stems)^2 / sum mix^2): how far the stems are from summing to the mix
Pass files in <pass dir>/<item>/:
  hyb6/htdemucs_6s/mixture/*.wav   htdemucs-rs -n ensemble (htdemucs + htdemucs_6s, shifts 0, 6-stem view)
  chain/htdemucs_6s/mixture/*.wav  htdemucs-rs -n chain (Kim vocals, ensemble on Kim's instrumental); its vocals are Kim's
  mega_mix.npy   r4_mega.py heads MIX_HEADS on the mix, (n_heads, 2, n)
  mega_inst.npy  r4_mega.py heads INST_HEADS on Kim's instrumental (mix - chain vocals)
Mega recipes: vocals / drums / bass / guitar / keys from heads, other = mix - the five (residual, so resid = 0 by
construction). Keys head choices: piano (as asked), keys, piano+keys, kb5 = piano+keys+digital-piano+organ+synth.
Vocal choices on the mix: vocal head, or lead-vocal + back-vocal (lb).
"""
import argparse, json, os
import numpy as np, soundfile as sf

MIX_HEADS = ['vocal', 'lead-vocal', 'back-vocal', 'drums', 'bass', 'guitar', 'piano', 'keys', 'digital-piano', 'organ', 'synth']
INST_HEADS = ['drums', 'bass', 'guitar', 'piano', 'keys', 'digital-piano', 'organ', 'synth']
STEMS = ('vocals', 'drums', 'bass', 'guitar', 'keys', 'other')
KEYS = {'piano': ['piano'], 'keys': ['keys'], 'pk': ['piano', 'keys'], 'kb5': ['piano', 'keys', 'digital-piano', 'organ', 'synth']}

ap = argparse.ArgumentParser()
ap.add_argument('items'); ap.add_argument('passes'); ap.add_argument('scores'); ap.add_argument('--items', dest='only', default=None)
a = ap.parse_args()
os.makedirs(a.scores, exist_ok=True)

def db(n, d): return float(10 * np.log10((n + 1e-20) / (d + 1e-20)))
def rd(f): return sf.read(f, dtype='float32', always_2d=True)[0].T.astype(np.float64)

def demucs_dir(d):
    s = {k: rd(f'{d}/{"piano" if k == "keys" else k}.wav') for k in STEMS}
    return s

def recipes(pd, mix):
    R = {}
    for name in ('hyb6', 'chain'):
        d = f'{pd}/{name}/htdemucs_6s/mixture'
        if os.path.isdir(d):
            s = demucs_dir(d); R[name] = s
            fx = dict(s); fx['other'] = mix - sum(s[k] for k in STEMS[:5]); R[name + '_fix'] = fx   # other := mix - the rest
    def mega(h, heads, vocals, tag):
        H = {n: h[i].astype(np.float64) for i, n in enumerate(heads)}
        for kn, kh in KEYS.items():
            s = {'vocals': vocals, 'drums': H['drums'], 'bass': H['bass'], 'guitar': H['guitar'], 'keys': sum(H[x] for x in kh)}
            s['other'] = mix - sum(s[k] for k in STEMS[:5])
            R[f'{tag}' + ('' if kn == 'piano' else f'_{kn}')] = s
        return H
    f = f'{pd}/mega_mix.npy'
    if os.path.exists(f):
        h = np.load(f); assert h.shape[0] == len(MIX_HEADS), h.shape
        n = mix.shape[1]; h = h[:, :, :n]
        H = {nm: h[i].astype(np.float64) for i, nm in enumerate(MIX_HEADS)}
        mega(h, MIX_HEADS, H['vocal'], 'mega_mix')
        lb = H['lead-vocal'] + H['back-vocal']
        mega(h, MIX_HEADS, lb, 'mega_mix_lb')
    f = f'{pd}/mega_inst.npy'; kv = f'{pd}/chain/htdemucs_6s/mixture/vocals.wav'
    if os.path.exists(f) and os.path.exists(kv):
        h = np.load(f)[:, :, :mix.shape[1]]; assert h.shape[0] == len(INST_HEADS), h.shape
        mega(h, INST_HEADS, rd(kv), 'kim_mega')
        if 'chain' in R:   # round 4's pick: the chain with Mega's guitar (and keys) on Kim's instrumental, other = residual
            for tag, sw in (('chain+mg', ('guitar',)), ('chain+mgk', ('guitar', 'keys'))):
                s = dict(R['chain']); s.update({k: R['kim_mega'][k] for k in sw})
                s['other'] = mix - sum(s[k] for k in STEMS[:5]); R[tag] = s
    return R

for it in sorted(os.listdir(a.items)):
    if a.only and it not in a.only.split(','): continue
    pd = f'{a.passes}/{it}'
    if not os.path.isdir(pd): continue
    idir = f'{a.items}/{it}'
    mix = rd(f'{idir}/mixture.wav')
    z = np.zeros_like(mix)
    T = {k: (rd(f'{idir}/{k}.wav') if os.path.exists(f'{idir}/{k}.wav') else z) for k in ('vocals', 'drums', 'bass', 'guitar', 'keys')}
    T['other'] = mix - sum(T[k] for k in ('vocals', 'drums', 'bass', 'guitar', 'keys'))
    o_file = rd(f'{idir}/other.wav') if os.path.exists(f'{idir}/other.wav') else z
    me = (mix ** 2).sum()
    meta = {'seconds': mix.shape[1] / 44100, 'has': [k for k in STEMS if (T[k] ** 2).sum() > 1e-9 * me],
            'parts_sum_err_db': db(((T['other'] - o_file) ** 2).sum(), me),
            'stem_db_re_mix': {k: db((T[k] ** 2).sum(), me) for k in STEMS}}
    out = {'_meta': meta, '_resid': {}}
    for name, s in recipes(pd, mix).items():
        r = {}
        for k in STEMS:
            t, e = T[k], s[k]; err = ((t - e) ** 2).sum(); te = (t ** 2).sum()
            r[k] = {'usdr': db(te, err) if k in meta['has'] else None, 'removed': db(((mix - t) ** 2).sum(), err),
                    'est_db': db((e ** 2).sum(), me)}
        out[name] = r
        out['_resid'][name] = db(((mix - sum(s[k] for k in STEMS)) ** 2).sum(), me)
    json.dump(out, open(f'{a.scores}/{it}.json', 'w'), indent=1)
    show = {n: [None if out[n][k]['usdr'] is None else round(out[n][k]['usdr'], 2) for k in STEMS]
            for n in ('hyb6', 'chain', 'mega_mix', 'kim_mega') if n in out}
    print(it, f"partsum {meta['parts_sum_err_db']:.0f} dB", show, flush=True)
