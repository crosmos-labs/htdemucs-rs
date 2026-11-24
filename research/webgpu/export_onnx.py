"""Export the network part of the round-5 recipe (Kim Mel-Band RoFormer, Mega 53 BS-RoFormer with 4 heads) to ONNX
for onnxruntime-web, with STFT / complex mask / iSTFT left outside the graph (ORT's WebGPU EP has no STFT/DFT or complex
types). Also writes one real chunk's input and the PyTorch reference so the browser's output can be checked.

    PY=/Users/divyansh/work/mac-testing/webgpu/venv/bin/python   (torch 2.6.0, onnx 1.17, onnxruntime 1.20.1)
    $PY -I research/webgpu/export_onnx.py mega <msst repo> <mega53.ckpt> <mega53.yaml> <mix.wav> <outdir> [--chunk-s 20] [--budget-mb 96]
    $PY -I research/webgpu/export_onnx.py kim  <kim repo> <MelBandRoformer.ckpt> - <mix.wav> <outdir> [--budget-mb 96]

The graph: X (1, T, F*2*2) float32, the STFT laid out as the models do ('b (f s) t c -> b t (f s c)': frequency outer,
then channel, then re/im) -> masks (1, n, T, F*2*2), same layout. The caller multiplies X by the mask as complex numbers
and runs the iSTFT (Mega also zeroes the DC bin, as BSRoformer's zero_dc does).
Kim's overlapping mel bands: the gather of band frequencies happens inside the graph and the per-frequency average of
overlapping band masks (upstream: scatter_add / num_bands_per_freq) is rewritten as a gather over a padded index table
(ScatterElements/ScatterND are poorly supported on WebGPU), so its output is the averaged mask, same layout as Mega's.

Memory: WebGPU's default maxStorageBufferBindingSize is 128 MiB and a WebGPU Softmax over > 128 MiB has been reported to
go wrong, so the time and frequency transformers run on slices of their batch axis (bands for the time transformer, frames
for the frequency transformer) small enough that the largest intermediate (the attention matrix, or the FF hidden state)
stays under --budget-mb. Same arithmetic, more nodes.

Outputs in <outdir>: <name>.onnx (fp32), <name>_in.bin (X, float32 LE), <name>_ref.bin (PyTorch masks), <name>.json
(shapes, slices, the STFT settings, SNR of the PyTorch core vs the full model on this chunk, ORT-CPU vs PyTorch SNR).
"""
import argparse, json, math, os, sys, time, types
import numpy as np, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('which', choices=['mega', 'kim'])
for k in ('repo', 'ckpt', 'cfg', 'wav', 'outdir'): ap.add_argument(k)
ap.add_argument('--stems', default='drums,bass,guitar,keys')
ap.add_argument('--chunk-s', type=float, default=0, help='chunk length in s (default: the model config: Mega 20, Kim 8)')
ap.add_argument('--start-s', type=float, default=60.0, help='where the test chunk starts in the wav')
ap.add_argument('--budget-mb', type=float, default=96)
ap.add_argument('--opset', type=int, default=17)
ap.add_argument('--name', default=None)
ap.add_argument('--no-ort', action='store_true')
a = ap.parse_args()
torch.set_num_threads(4)
os.makedirs(a.outdir, exist_ok=True)


def snr(ref, x):
    ref = np.asarray(ref, np.float64); x = np.asarray(x, np.float64)
    return float(10 * np.log10((ref ** 2).sum() / max(((ref - x) ** 2).sum(), 1e-300)))


def sliced(mod, x, s):
    """mod over batch slices of x of at most s rows (cat of the results): same arithmetic, bounded intermediates."""
    if x.shape[0] <= s: return mod(x)
    return torch.cat([mod(x[i:i + s]) for i in range(0, x.shape[0], s)], 0)


def slice_rows(n_seq, dim, heads, ff_mult, budget):
    """Largest batch slice whose attention matrix (heads x n x n) and FF hidden (n x dim*ff_mult) fit in budget bytes."""
    per = max(heads * n_seq * n_seq, n_seq * dim * ff_mult, n_seq * heads * 64 * 3) * 4
    return max(1, int(budget // per))


# ---------------------------------------------------------------- models
if a.which == 'mega':
    import yaml
    for name, path in (('models', f'{a.repo}/models'), ('models.bs_roformer', f'{a.repo}/models/bs_roformer')):
        m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
    sys.path.insert(0, a.repo)
    from models.bs_roformer.bs_roformer import BSRoformer

    class L(yaml.SafeLoader): pass
    L.add_constructor('tag:yaml.org,2002:python/tuple', lambda l, n: tuple(l.construct_sequence(n)))
    cfg = yaml.load(open(a.cfg), Loader=L)
    inst = cfg['training']['instruments']; want = a.stems.split(','); ids = [inst.index(s) for s in want]
    mk = dict(cfg['model']); mk['num_stems'] = len(ids); model = BSRoformer(**mk)
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=True, mmap=True)
    if 'state_dict' in sd: sd = sd['state_dict']
    pre = 'mask_estimators.'; keep = {}
    for k, t in sd.items():
        if k.startswith(pre):
            i, rest = k[len(pre):].split('.', 1); i = int(i)
            if i in ids: keep[f'{pre}{ids.index(i)}.{rest}'] = t
        else: keep[k] = t
    model.load_state_dict(keep); del sd, keep
    model.eval()
    chunk = int(round((a.chunk_s or cfg['inference']['chunk_size'] / 44100) * 44100))
    dim, heads, ffm = mk['dim'], mk['heads'], 4
    name = a.name or f'mega4_{int(round(chunk / 44100))}s'
    stems = want
else:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
    sys.path.insert(0, a.repo)
    import yaml
    from ml_collections import ConfigDict
    from utils import get_model_from_config
    kcfg = ConfigDict(yaml.load(open(os.path.join(a.repo, 'configs/config_vocals_mel_band_roformer.yaml')), Loader=yaml.FullLoader))
    model = get_model_from_config('mel_band_roformer', kcfg)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=True)); model.eval()
    chunk = int(round((a.chunk_s or kcfg.inference.chunk_size / 44100) * 44100))
    dim, heads, ffm = kcfg.model.dim, kcfg.model.heads, 4
    name = a.name or f'kim_{int(round(chunk / 44100))}s'
    stems = ['vocals']

hop = model.stft_kwargs['hop_length']; n_fft = model.stft_kwargs['n_fft']
T = chunk // hop + 1; Fq = n_fft // 2 + 1
n_bands = len(model.band_split.to_features)
S_time = slice_rows(T, dim, heads, ffm, a.budget_mb * 2 ** 20)
S_freq = slice_rows(n_bands, dim, heads, ffm, a.budget_mb * 2 ** 20)
print(f'{name}: chunk {chunk} samples, T={T} frames, F={Fq}, bands={n_bands}, slices: time {S_time} bands, freq {S_freq} frames', flush=True)


class MegaCore(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, X):                                    # X (1, T, Fq*2*2)
        m = s.m; x = m.band_split(X)                      # (1, T, bands, d)
        for blk in m.layers:
            tt, ft = blk[-2], blk[-1]
            b, t, f, d = x.shape
            x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
            x = sliced(tt, x, S_time)
            x = x.reshape(b, f, t, d).permute(0, 2, 1, 3).reshape(b * t, f, d)
            x = sliced(ft, x, S_freq)
            x = x.reshape(b, t, f, d)
        x = m.final_norm(x)
        return torch.stack([fn(x) for fn in m.mask_estimators], dim=1)   # (1, n, T, Fq*2*2)


class KimCore(nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
        fi = m.freq_indices.long()                        # indices into (f s) = 2*Fq rows, with repeats (overlapping bands)
        rows = 2 * Fq; pos = [[] for _ in range(rows)]
        for j, r in enumerate(fi.tolist()): pos[r].append(j)
        K = max(len(p) for p in pos)
        idx = torch.zeros(rows, K, dtype=torch.long); w = torch.zeros(rows, K)
        for r, p in enumerate(pos):
            for k, j in enumerate(p): idx[r, k] = j; w[r, k] = 1.0 / len(p)
        assert torch.equal(torch.tensor([len(p) for p in pos]).float(), m.num_bands_per_freq.float().repeat_interleave(2))
        s.register_buffer('fi', fi); s.register_buffer('idx', idx.reshape(-1)); s.register_buffer('w', w.reshape(1, 1, rows, K, 1))
        s.K = K; s.rows = rows
    def forward(s, X):                                    # X (1, T, rows*2)
        m = s.m; b, t, _ = X.shape
        x = X.reshape(b, t, s.rows, 2).index_select(2, s.fi).reshape(b, t, -1)
        x = m.band_split(x)
        for tt, ft in m.layers:
            _, _, f, d = x.shape
            x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
            x = sliced(tt, x, S_time)
            x = x.reshape(b, f, t, d).permute(0, 2, 1, 3).reshape(b * t, f, d)
            x = sliced(ft, x, S_freq)
            x = x.reshape(b, t, f, d)
        mk = m.mask_estimators[0](x).reshape(b, t, -1, 2)            # per band-frequency masks
        avg = (mk.index_select(2, s.idx).reshape(b, t, s.rows, s.K, 2) * s.w).sum(3)
        return avg.reshape(b, 1, t, s.rows * 2)


core = (MegaCore if a.which == 'mega' else KimCore)(model).eval()

# ---------------------------------------------------------------- one real chunk, and the reference
mix, sr = sf.read(a.wav, dtype='float32', always_2d=True); assert sr == 44100
s0 = int(a.start_s * sr); audio = torch.from_numpy(mix[s0:s0 + chunk, :2].T.copy())[None]      # (1, 2, chunk)
assert audio.shape[-1] == chunk
win = torch.hann_window(n_fft)
spec = torch.stft(audio.reshape(2, -1), n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win,
                  normalized=False, center=True, return_complex=True)                         # (2, Fq, T)
X = torch.view_as_real(spec).permute(2, 1, 0, 3).reshape(1, T, Fq * 2 * 2).contiguous()   # b t (f s c)


def apply_masks(X, M):
    """complex X * M, then iSTFT, as the models do -> (n, 2, chunk)."""
    n = M.shape[1]
    Xc = torch.view_as_complex(X.reshape(1, 1, T, Fq, 2, 2))
    Mc = torch.view_as_complex(M.reshape(1, n, T, Fq, 2, 2).contiguous())
    Y = (Xc * Mc)[0].permute(0, 3, 2, 1)                                                     # (n, s, Fq, T)
    if a.which == 'mega' and getattr(model, 'zero_dc', False): Y = Y.clone(); Y[:, :, 0] = 0
    y = torch.istft(Y.reshape(n * 2, Fq, T), n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, length=chunk)
    return y.reshape(n, 2, chunk)


with torch.no_grad():
    t0 = time.time(); full = model(audio)[0]; t_full = time.time() - t0
    if full.ndim == 2: full = full[None]
    t0 = time.time(); M = core(X); t_core = time.time() - t0
    rec = apply_masks(X, M)
core_vs_full = [snr(full[i].numpy(), rec[i].numpy()) for i in range(len(stems))]
print('PyTorch core + outside STFT vs full model, SNR dB per stem:', [round(v, 1) for v in core_vs_full],
      f'(full {t_full:.1f} s, core {t_core:.1f} s on CPU)', flush=True)

X.numpy().astype('<f4').tofile(f'{a.outdir}/{name}_in.bin')
M.numpy().astype('<f4').tofile(f'{a.outdir}/{name}_ref.bin')
torch.save({'audio': audio, 'full': full}, f'{a.outdir}/{name}_ref_audio.pt')

# ---------------------------------------------------------------- export
onnx_path = f'{a.outdir}/{name}.onnx'
t0 = time.time()
with torch.no_grad():
    torch.onnx.export(core, (X,), onnx_path, input_names=['X'], output_names=['masks'], opset_version=a.opset,
                      do_constant_folding=True, dynamo=False)
t_exp = time.time() - t0
import onnx
om = onnx.load(onnx_path, load_external_data=False)
info = dict(name=name, which=a.which, stems=stems, chunk=chunk, chunk_s=chunk / 44100, hop=hop, n_fft=n_fft, T=T, F=Fq,
            in_shape=list(X.shape), out_shape=list(M.shape), bands=n_bands, slice_time=S_time, slice_freq=S_freq,
            budget_mb=a.budget_mb, zero_dc=bool(a.which == 'mega' and getattr(model, 'zero_dc', False)),
            params_m=round(sum(p.numel() for p in core.parameters()) / 1e6, 2),
            onnx_mb=round(os.path.getsize(onnx_path) / 1e6, 1), nodes=len(om.graph.node), export_s=round(t_exp, 1),
            core_vs_full_snr_db=core_vs_full, wav=os.path.basename(os.path.dirname(a.wav)), start_s=a.start_s,
            torch_cpu4_core_s=round(t_core, 2))
del om
print(f'exported {onnx_path}: {info["onnx_mb"]} MB, {info["nodes"]} nodes, {t_exp:.0f} s', flush=True)

if not a.no_ort:
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
    t0 = time.time(); Mo = sess.run(None, {'X': X.numpy()})[0]; t_ort = time.time() - t0
    info['ort_cpu_vs_torch_mask_snr_db'] = snr(M.numpy(), Mo)
    reco = apply_masks(X, torch.from_numpy(Mo))
    info['ort_cpu_vs_full_audio_snr_db'] = [snr(full[i].numpy(), reco[i].numpy()) for i in range(len(stems))]
    info['ort_cpu4_s'] = round(t_ort, 2)
    print('ORT CPU vs PyTorch: mask SNR', round(info['ort_cpu_vs_torch_mask_snr_db'], 1), 'dB; audio SNR vs full model',
          [round(v, 1) for v in info['ort_cpu_vs_full_audio_snr_db']], f'; ORT CPU 4 threads {t_ort:.1f} s', flush=True)
json.dump(info, open(f'{a.outdir}/{name}.json', 'w'), indent=1)
