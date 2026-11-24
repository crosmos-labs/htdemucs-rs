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


def slice_rows(n_seq, dim, heads, ff_mult, budget):
    """Largest batch slice whose attention matrix (heads x n x n), FF hidden (n x dim*ff_mult) and qkv fit in budget bytes."""
    per = max(heads * n_seq * n_seq, n_seq * dim * ff_mult, n_seq * heads * 64 * 3) * 4
    return max(1, int(budget // per))


def cat_tree(xs, dim, k=6):
    """torch.cat in groups of at most k inputs: ORT's WebGPU kernels bind one storage buffer per input/output and the
    default per-stage limit is 8 (a 63-way Split failed validation)."""
    while len(xs) > k: xs = [torch.cat(xs[i:i + k], dim) if len(xs[i:i + k]) > 1 else xs[i] for i in range(0, len(xs), k)]
    return torch.cat(xs, dim) if len(xs) > 1 else xs[0]


def band_split(bs, x):
    """BandSplit.forward with Slice ops instead of one many-output Split."""
    outs, s0 = [], 0
    for d_in, net in zip(bs.dim_inputs, bs.to_features):
        outs.append(net(x[..., s0:s0 + d_in]).unsqueeze(-2)); s0 += d_in
    return cat_tree(outs, -2)


def mask_head(me, x):
    """MaskEstimator.forward with a Gather per band instead of unbind (a many-output Split)."""
    return cat_tree([mlp(x[:, :, i]) for i, mlp in enumerate(me.to_freqs)], -1)


class FastTransformer(nn.Module):
    """One time or frequency Transformer of the RoFormers (pre-norm attention with rotary embedding and per-head sigmoid
    gates, then a GELU feed-forward, then an output RMSNorm), written with few, static-shape ops so the exported graph
    stays small: weights pre-transposed once, rotary cos/sin precomputed for the sequence length, rotate_half as a 64x64
    matmul, and the whole block run on batch slices of at most S rows (bounded intermediates)."""
    def __init__(s, tr, n_seq, S):
        super().__init__(); s.S = S; s.blocks = []
        attn0 = tr.layers[0][0]; rot = attn0.rotary_embed; s.H = attn0.heads; s.scale = attn0.scale
        f = rot.forward(lambda: rot.get_seq_pos(n_seq, device='cpu', dtype=torch.float32, offset=0))   # (n, Dh), as the module
        s.register_buffer('cos', f.cos().contiguous()); s.register_buffer('sin', f.sin().contiguous())
        Dh = f.shape[-1]; R = torch.zeros(Dh, Dh)
        for i in range(Dh // 2): R[2 * i + 1, 2 * i] = -1.0; R[2 * i, 2 * i + 1] = 1.0   # x @ R == rotate_half(x)
        s.register_buffer('R', R); s.Dh = Dh
        s.ps = nn.ParameterList(); idx = []
        def P(t): s.ps.append(nn.Parameter(t.detach().clone().contiguous(), requires_grad=False)); return len(s.ps) - 1
        for attn, ff in tr.layers:
            lin1, lin2 = ff.net[1], ff.net[4]
            idx.append(dict(an=P(attn.norm.gamma * attn.norm.scale), qkv=P(attn.to_qkv.weight.t()), gw=P(attn.to_gates.weight.t()),
                            gb=P(attn.to_gates.bias), o=P(attn.to_out[0].weight.t()), fn=P(ff.net[0].gamma * ff.net[0].scale),
                            w1=P(lin1.weight.t()), b1=P(lin1.bias), w2=P(lin2.weight.t()), b2=P(lin2.bias)))
        s.idx = idx; s.on = P(tr.norm.gamma * tr.norm.scale) if not isinstance(tr.norm, nn.Identity) else None

    @staticmethod
    def rms(x, g): return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12) * g

    def block(s, x):
        b, n, d = x.shape; H, Dh, p = s.H, s.Dh, s.ps
        for L in s.idx:
            h = s.rms(x, p[L['an']])
            qkv = (h @ p[L['qkv']]).reshape(b, n, 3, H, Dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * s.cos + (q @ s.R) * s.sin; k = k * s.cos + (k @ s.R) * s.sin
            a = torch.softmax((q * s.scale) @ k.transpose(-1, -2), dim=-1) @ v               # (b, H, n, Dh)
            g = torch.sigmoid(h @ p[L['gw']] + p[L['gb']]).permute(0, 2, 1).unsqueeze(-1)     # (b, H, n, 1)
            x = x + ((a * g).permute(0, 2, 1, 3).reshape(b, n, H * Dh) @ p[L['o']])
            h = s.rms(x, p[L['fn']])
            x = x + (F.gelu(h @ p[L['w1']] + p[L['b1']]) @ p[L['w2']] + p[L['b2']])
        return s.rms(x, p[s.on]) if s.on is not None else x

    def forward(s, x):
        if x.shape[0] <= s.S: return s.block(x)
        return cat_tree([s.block(x[i:i + s.S]) for i in range(0, x.shape[0], s.S)], 0)


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
    def __init__(s, m):
        super().__init__(); s.m = m
        s.tt = nn.ModuleList([FastTransformer(blk[-2], T, S_time) for blk in m.layers])
        s.ft = nn.ModuleList([FastTransformer(blk[-1], n_bands, S_freq) for blk in m.layers])
    def forward(s, X):                                    # X (1, T, Fq*2*2)
        m = s.m; x = band_split(m.band_split, X)                      # (1, T, bands, d)
        for tt, ft in zip(s.tt, s.ft):
            b, t, f, d = x.shape
            x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
            x = tt(x)
            x = x.reshape(b, f, t, d).permute(0, 2, 1, 3).reshape(b * t, f, d)
            x = ft(x)
            x = x.reshape(b, t, f, d)
        x = m.final_norm(x)
        return torch.stack([mask_head(fn, x) for fn in m.mask_estimators], dim=1)   # (1, n, T, Fq*2*2)


class KimCore(nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
        s.tt = nn.ModuleList([FastTransformer(tt, T, S_time) for tt, _ in m.layers])
        s.ft = nn.ModuleList([FastTransformer(ft, n_bands, S_freq) for _, ft in m.layers])
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
        x = band_split(m.band_split, x)
        for tt, ft in zip(s.tt, s.ft):
            _, _, f, d = x.shape
            x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
            x = tt(x)
            x = x.reshape(b, f, t, d).permute(0, 2, 1, 3).reshape(b * t, f, d)
            x = ft(x)
            x = x.reshape(b, t, f, d)
        mk = mask_head(m.mask_estimators[0], x).reshape(b, t, -1, 2)            # per band-frequency masks
        avg = (mk.index_select(2, s.idx).reshape(b, t, s.rows, s.K, 2) * s.w).sum(3)
        return avg.reshape(b, 1, t, s.rows * 2)


core = (MegaCore if a.which == 'mega' else KimCore)(model).eval()
for p_ in core.parameters(): p_.requires_grad_(False)

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
wide = max(((len(n.input) + len(n.output)), n.op_type) for n in om.graph.node)
print('widest node (inputs + outputs):', wide, flush=True)
params_m = round(sum(int(np.prod(i.dims)) for i in om.graph.initializer if i.data_type == 1) / 1e6, 2)
info = dict(name=name, which=a.which, stems=stems, chunk=chunk, chunk_s=chunk / 44100, hop=hop, n_fft=n_fft, T=T, F=Fq,
            in_shape=list(X.shape), out_shape=list(M.shape), bands=n_bands, slice_time=S_time, slice_freq=S_freq,
            budget_mb=a.budget_mb, params_m=params_m, widest_node=list(wide), zero_dc=bool(a.which == 'mega' and getattr(model, 'zero_dc', False)),
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
