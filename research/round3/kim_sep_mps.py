"""Kim Mel-RoFormer vocals with the transformer on the GPU (MPS) and the STFT, band gather, complex mask average and iSTFT
on the CPU (MPS has no complex scatter_add). Otherwise round 1's kim_sep.py: upstream demix_track, 8 s chunks, overlap 2.
    kim_sep_mps.py <kim repo dir> <ckpt> <input wav> <out vocals.wav> [--threads 1]
Used only to make round 3's reference vocals quickly; checked against the CPU run's uSDR (see round3.md).
"""
import argparse, json, os, sys, time
import numpy as np, soundfile as sf, torch

ap = argparse.ArgumentParser()
ap.add_argument('repo'); ap.add_argument('ckpt'); ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--threads', type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)
sys.path.insert(0, a.repo)
import yaml
from ml_collections import ConfigDict
from utils import demix_track, get_model_from_config

cfg = ConfigDict(yaml.load(open(os.path.join(a.repo, 'configs/config_vocals_mel_band_roformer.yaml')), Loader=yaml.FullLoader))
mix, sr = sf.read(a.inp, dtype='float32', always_2d=True); assert sr == 44100
mix = mix.T.copy()
model = get_model_from_config('mel_band_roformer', cfg)
model.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=True))
model.eval()
G = torch.device('mps')

class In(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m.to(G)
    def forward(self, x): return self.m(x.to(G))
class Out(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m.to(G)
    def forward(self, x): return self.m(x).cpu()
model.band_split = In(model.band_split)
model.layers = model.layers.to(G)
model.mask_estimators = torch.nn.ModuleList([Out(m) for m in model.mask_estimators])
t1 = time.time(); c1 = time.process_time()
with torch.no_grad():
    res, _ = demix_track(cfg, model, torch.tensor(mix), 'cpu')
torch.mps.synchronize()
sf.write(a.out, res['vocals'].T.astype(np.float32), 44100, subtype='FLOAT')
print(json.dumps({'audio_s': mix.shape[1] / 44100, 'wall_s': round(time.time() - t1, 2), 'cpu_s': round(time.process_time() - c1, 2), 'device': 'mps+cpu'}))
