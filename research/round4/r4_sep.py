"""Round 4 htdemucs_6s guitar passes (PyTorch Demucs 4.0.1, f32, overlap 0.25) on item dirs holding mixture.wav.
    r4_sep.py <item dir> <pass dir> [--passes s0,s2,inst] [--kim <kim vocals.wav>] [--device mps|cpu] [--threads 1]
  s0   = shifts 0 on the mix
  s2   = shifts 2 on the mix, random.seed(0) before apply_model (as round 2's _s2)
  inst = shifts 0 on Kim's instrumental (mix − Kim vocals; the round 3 chain)
Normalisation as demucs.separate. Writes <pass dir>/6s_<pass>.npy, float32 (2, 2, n): guitar, piano; appends CPU / wall
seconds to <pass dir>/cpu.jsonl.
"""
import argparse, json, os, random, time
import numpy as np, soundfile as sf, torch

ap = argparse.ArgumentParser()
ap.add_argument('item'); ap.add_argument('out'); ap.add_argument('--passes', default='s0,s2,inst'); ap.add_argument('--kim')
ap.add_argument('--device', default='mps'); ap.add_argument('--threads', type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)
from demucs.pretrained import get_model
from demucs.apply import apply_model
model = get_model('htdemucs_6s'); model.eval(); src = list(model.sources)
wav, sr = sf.read(f'{a.item}/mixture.wav', dtype='float32', always_2d=True); mix = torch.from_numpy(wav.T.copy())
os.makedirs(a.out, exist_ok=True)
for p in a.passes.split(','):
    dst = f'{a.out}/6s_{p}.npy'
    if os.path.exists(dst): continue
    x = mix
    if p == 'inst':
        k = sf.read(a.kim, dtype='float32', always_2d=True)[0]; assert k.shape == wav.shape
        x = mix - torch.from_numpy(k.T.copy())
    ref = x.mean(0); mu, sd = ref.mean(), ref.std()
    if p == 's2': random.seed(0)
    c0, w0 = time.process_time(), time.time()
    with torch.no_grad():
        y = apply_model(model, ((x - mu) / sd)[None], shifts=2 if p == 's2' else 0, split=True, overlap=0.25, device=a.device)[0]
    y = (y * sd + mu).cpu()
    cpu, wall = time.process_time() - c0, time.time() - w0
    np.save(dst + '.tmp.npy', torch.stack([y[src.index('guitar')], y[src.index('piano')]]).numpy().astype(np.float32))
    os.replace(dst + '.tmp.npy', dst)
    with open(f'{a.out}/cpu.jsonl', 'a') as f:
        f.write(json.dumps({'pass': '6s_' + p, 'audio_s': wav.shape[0] / sr, 'device': a.device, 'threads': a.threads,
                            'cpu_s': round(cpu, 2), 'wall_s': round(wall, 2)}) + '\n')
