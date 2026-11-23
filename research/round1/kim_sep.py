"""Kim's Mel-Band RoFormer vocals, with the inference code of github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model
(commit 25f44ff, cloned to data/ext/kim; its demix_track: 8 s chunks, num_overlap 2, linear fades).
    kim_sep.py <kim repo dir> <ckpt> <input audio> <out vocals.wav> [--device cpu|mps] [--threads N]
              [--start S --dur D] [--overlap N]
Prints JSON timing. The checkpoint is loaded with weights_only=True (it is a pickle from the internet).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('repo'); ap.add_argument('ckpt'); ap.add_argument('inp'); ap.add_argument('out')
ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--start', type=float, default=None); ap.add_argument('--dur', type=float, default=None)
ap.add_argument('--overlap', type=int, default=None)
a = ap.parse_args()

import torch
torch.set_num_threads(a.threads)
sys.path.insert(0, a.repo)
import yaml
from ml_collections import ConfigDict
from utils import demix_track, get_model_from_config  # noqa: E402

cfg = ConfigDict(yaml.load(open(os.path.join(a.repo, 'configs/config_vocals_mel_band_roformer.yaml')), Loader=yaml.FullLoader))
if a.overlap: cfg.inference.num_overlap = a.overlap
cmd = ['ffmpeg', '-v', 'error']
if a.start is not None: cmd += ['-ss', str(a.start)]
if a.dur is not None: cmd += ['-t', str(a.dur)]
cmd += ['-i', a.inp, '-ac', '2', '-ar', '44100', '-f', 'f32le', '-']
mix = np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, dtype=np.float32).reshape(-1, 2).T.copy()

t0 = time.time()
model = get_model_from_config('mel_band_roformer', cfg)
sd = torch.load(a.ckpt, map_location='cpu', weights_only=True)
model.load_state_dict(sd)
model = model.to(a.device).eval()
t_load = time.time() - t0
nparams = sum(p.numel() for p in model.parameters())
t1 = time.time(); c1 = time.process_time()
res, _ = demix_track(cfg, model, torch.tensor(mix), a.device)
if a.device == 'mps': torch.mps.synchronize()
wall = time.time() - t1; cpu = time.process_time() - c1
voc = res['vocals'].T.astype(np.float32)
import soundfile as sf
sf.write(a.out, voc, 44100, subtype='FLOAT')
print(json.dumps({'audio_s': mix.shape[1] / 44100, 'wall_s': round(wall, 2), 'cpu_s': round(cpu, 2), 'load_s': round(t_load, 2), 'params': nparams,
                  'device': a.device, 'threads': a.threads, 'overlap': cfg.inference.num_overlap, 'chunk': cfg.inference.chunk_size, 'torch': torch.__version__}))
