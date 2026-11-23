"""Kim's Mel-Band RoFormer vocals (github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model commit 25f44ff, its demix_track:
8 s chunks, num_overlap 2, linear fades), as round 1's kim_sep.py, but reading the WAV with soundfile and, with --fp32, with
the repo's torch.cuda.amp.autocast turned off (so CUDA computes in fp32 like rounds 2-3's CPU / MPS runs).
    r4_kim_sep.py <kim repo dir> <ckpt> <in.wav> <out vocals.wav> [--device cuda|cpu] [--threads N] [--fp32] [--dur S]
"""
import argparse, contextlib, json, os, sys, time
import numpy as np, soundfile as sf, torch

ap = argparse.ArgumentParser()
for k in ('repo', 'ckpt', 'inp', 'out'): ap.add_argument(k)
ap.add_argument('--device', default='cuda'); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--fp32', action='store_true')
ap.add_argument('--dur', type=float, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads)
if a.fp32: torch.cuda.amp.autocast = lambda *x, **k: contextlib.nullcontext()
sys.path.insert(0, a.repo)
import yaml
from ml_collections import ConfigDict
from utils import demix_track, get_model_from_config

cfg = ConfigDict(yaml.load(open(os.path.join(a.repo, 'configs/config_vocals_mel_band_roformer.yaml')), Loader=yaml.FullLoader))
mix, sr = sf.read(a.inp, dtype='float32', always_2d=True); assert sr == 44100
if a.dur: mix = mix[:int(a.dur * sr)]
mix = mix.T.copy()
model = get_model_from_config('mel_band_roformer', cfg)
model.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=True))
model = model.to(a.device).eval()
t1 = time.time(); c1 = time.process_time()
res, _ = demix_track(cfg, model, torch.tensor(mix), a.device)
if a.device == 'cuda': torch.cuda.synchronize()
wall = time.time() - t1; cpu = time.process_time() - c1
sf.write(a.out, res['vocals'].T.astype(np.float32), 44100, subtype='FLOAT')
print(json.dumps({'audio_s': mix.shape[1] / 44100, 'wall_s': round(wall, 2), 'cpu_s': round(cpu, 2), 'device': a.device,
                  'threads': a.threads, 'fp32': a.fp32, 'torch': torch.__version__}))
