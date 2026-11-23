"""Round 3 Demucs passes on MUSDB18-HQ test tracks (PyTorch Demucs 4.0.1, f32, shifts 0, overlap 0.25).
    r3_sep.py <model: htdemucs|htdemucs_6s> <tag> <musdb test dir> <pass dir> [track…]
              [--passes id,sw,po,swpo,inst] [--device cpu|mps] [--threads 1] [--kim data/kim] [--wait]
Passes (each one plain apply_model call, the output mapped back so it estimates the original mix's stems):
  id   = the mix as is (= round 2's _s0)
  sw   = left/right swapped in, swapped back out
  po   = polarity inverted in (−mix), negated out
  swpo = both
  inst = Kim's instrumental, mix − <kim>/<track>/vocals.wav (the UVR chain); --wait waits for that file.
Normalisation as demucs.separate (subtract the mean, divide by the std of the mono input; undone after).
Writes <pass dir>/<track>/<tag>_<pass>.npy, float32 (4, 2, n) in the order vocals, drums, bass, other
(for htdemucs_6s, other = guitar + piano + other, then guitar and piano: shape (6, 2, n)), and appends CPU / wall seconds per pass to <pass dir>/cpu.jsonl.
"""
import argparse, json, os, time
import numpy as np
import soundfile as sf
import torch

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('tag'); ap.add_argument('musdb'); ap.add_argument('out'); ap.add_argument('tracks', nargs='*')
ap.add_argument('--passes', default='id,sw,po,swpo,inst'); ap.add_argument('--device', default='cpu')
ap.add_argument('--threads', type=int, default=1); ap.add_argument('--kim', default='data/kim'); ap.add_argument('--wait', action='store_true')
a = ap.parse_args()
torch.set_num_threads(a.threads)
from demucs.pretrained import get_model
from demucs.apply import apply_model

model = get_model(a.model); model.eval()
src = list(model.sources)

def run(x):
    ref = x.mean(0); mu, sd = ref.mean(), ref.std()
    y = apply_model(model, ((x - mu) / sd)[None], shifts=0, split=True, overlap=0.25, device=a.device)[0]
    return (y * sd + mu).cpu()

def four(y):
    e = {s: y[i] for i, s in enumerate(src)}
    if 'guitar' in e: e['other'] = e['other'] + e['guitar'] + e['piano']
    keep = ('vocals', 'drums', 'bass', 'other') + (('guitar', 'piano') if 'guitar' in e else ())
    return torch.stack([e[s] for s in keep]).numpy().astype(np.float32)

for tr in (a.tracks or sorted(os.listdir(a.musdb))):
    wav, sr = sf.read(f'{a.musdb}/{tr}/mixture.wav', dtype='float32', always_2d=True)
    mix = torch.from_numpy(wav.T.copy())
    d = f'{a.out}/{tr}'; os.makedirs(d, exist_ok=True)
    for p in a.passes.split(','):
        dst = f'{d}/{a.tag}_{p}.npy'
        if os.path.exists(dst) or os.path.exists(f'{d}/scored'): continue
        if p == 'inst':
            kp = f'{a.kim}/{tr}/vocals.wav'
            while a.wait and not os.path.exists(f'{a.kim}/{tr}/done'): time.sleep(20)
            k = sf.read(kp, dtype='float32', always_2d=True)[0]
            assert k.shape == wav.shape, (k.shape, wav.shape)
            x = mix - torch.from_numpy(k.T.copy())
        else:
            x = {'id': mix, 'sw': mix[[1, 0]], 'po': -mix, 'swpo': -mix[[1, 0]]}[p]
        c0, w0 = time.process_time(), time.time()
        with torch.no_grad(): y = run(x)
        cpu, wall = time.process_time() - c0, time.time() - w0
        if p in ('sw', 'swpo'): y = y[:, [1, 0]]
        if p in ('po', 'swpo'): y = -y
        np.save(dst + '.tmp.npy', four(y)); os.replace(dst + '.tmp.npy', dst)
        with open(f'{a.out}/cpu.jsonl', 'a') as f:
            f.write(json.dumps({'tag': a.tag, 'model': a.model, 'pass': p, 'track': tr, 'audio_s': wav.shape[0] / sr, 'device': a.device,
                                'threads': a.threads, 'cpu_s': round(cpu, 2), 'wall_s': round(wall, 2)}) + '\n')
        print(tr, a.tag, p, round(cpu, 1), round(wall, 1), flush=True)
