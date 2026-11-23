"""MVSep Mega 53-stem BS-RoFormer (MSST release v1.0.21, mvsep_mega_model_bs_roformer_53_stems_v1.ckpt) on one input.
    r4_mega.py <msst repo> <ckpt> <yaml> <in.wav> <out.npy> [--stems guitar,electric-guitar,acoustic-guitar]
               [--device mps|cpu] [--threads 1] [--overlap 2] [--chunk 882000] [--minus kim_vocals.wav]
Inference as MSST's utils.model_utils.demix in 'generic' mode: chunk_size = inference.chunk_size (20 s), step = chunk/overlap,
linear fades of chunk/10, reflect padding of `border` at both ends, overlap-add divided by the summed window, fp32 (use_amp only
affects CUDA). Only the mask heads for --stems are kept (the backbone is shared, so the other heads don't change these outputs).
Attention runs in slices of 2 along the batch (bands) axis to bound memory; this is the same arithmetic.
--minus subtracts a vocals file from the input first (Kim's instrumental). Writes float32 (n_stems, 2, n) and prints a JSON line.
"""
import argparse, json, sys, time, types
import numpy as np, soundfile as sf, torch, yaml
import torch.nn.functional as F

ap = argparse.ArgumentParser()
for k in ('repo', 'ckpt', 'cfg', 'inp', 'out'): ap.add_argument(k)
ap.add_argument('--stems', default='guitar,electric-guitar,acoustic-guitar'); ap.add_argument('--device', default='mps')
ap.add_argument('--threads', type=int, default=1); ap.add_argument('--overlap', type=int, default=2)
ap.add_argument('--chunk', type=int, default=0); ap.add_argument('--minus', default=None)
a = ap.parse_args()
torch.set_num_threads(a.threads)
# import models.bs_roformer.bs_roformer without the package __init__ (which pulls in the conformer variants)
for name, path in (('models', f'{a.repo}/models'), ('models.bs_roformer', f'{a.repo}/models/bs_roformer')):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
sys.path.insert(0, a.repo)
from models.bs_roformer.bs_roformer import BSRoformer
from models.bs_roformer import attend as att

_fwd = att.Attend.forward
def sliced(self, q, k, v, S=2):
    if q.shape[0] <= S: return _fwd(self, q, k, v)
    return torch.cat([_fwd(self, q[i:i + S], k[i:i + S], v[i:i + S]) for i in range(0, q.shape[0], S)])
att.Attend.forward = sliced

class L(yaml.SafeLoader): pass
L.add_constructor('tag:yaml.org,2002:python/tuple', lambda l, n: tuple(l.construct_sequence(n)))
cfg = yaml.load(open(a.cfg), Loader=L)
inst = cfg['training']['instruments']; want = a.stems.split(',')
ids = [inst.index(s) for s in want]
mk = dict(cfg['model']); model = BSRoformer(**mk)
sd = torch.load(a.ckpt, map_location='cpu', weights_only=True)
if 'state_dict' in sd: sd = sd['state_dict']
model.load_state_dict(sd)
model.mask_estimators = torch.nn.ModuleList([model.mask_estimators[i] for i in ids])
del sd; import gc; gc.collect()
dev = torch.device(a.device); model = model.to(dev).eval()

mix, sr = sf.read(a.inp, dtype='float32', always_2d=True); assert sr == 44100
mix = mix.T.copy()
if a.minus:
    v = sf.read(a.minus, dtype='float32', always_2d=True)[0].T; assert v.shape == mix.shape; mix = mix - v
chunk = a.chunk or cfg['inference']['chunk_size']; step = chunk // a.overlap; border = chunk - step; fade = chunk // 10
win = torch.ones(chunk); win[:fade] = torch.linspace(0, 1, fade); win[-fade:] = torch.linspace(1, 0, fade)
x = torch.from_numpy(mix); n0 = x.shape[-1]
if n0 > 2 * border and border > 0: x = F.pad(x[None], (border, border), mode='reflect')[0]
N = x.shape[-1]; res = torch.zeros(len(ids), 2, N); cnt = torch.zeros(N)
c0, w0 = time.process_time(), time.time()
with torch.no_grad():
    i = 0
    while i < N:
        part = x[:, i:i + chunk]; L_ = part.shape[-1]
        part = F.pad(part[None], (0, chunk - L_), mode='reflect' if L_ > chunk // 2 else 'constant')[0]
        y = model(part[None].to(dev))[0].float().cpu()
        w = win.clone()
        if i == 0: w[:fade] = 1
        if i + step >= N: w[-fade:] = 1
        res[..., i:i + L_] += y[..., :L_] * w[:L_]; cnt[i:i + L_] += w[:L_]
        i += step
out = res / cnt
if n0 > 2 * border and border > 0: out = out[..., border:-border]
out = torch.nan_to_num(out).numpy().astype(np.float32)
cpu, wall = time.process_time() - c0, time.time() - w0
np.save(a.out, out)
print(json.dumps({'stems': want, 'audio_s': round(n0 / sr, 2), 'device': a.device, 'threads': a.threads, 'overlap': a.overlap,
                  'cpu_s': round(cpu, 2), 'wall_s': round(wall, 2), 'minus': bool(a.minus)}))
