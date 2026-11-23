"""Round 2 separations of MUSDB18-HQ test tracks with PyTorch Demucs 4.0.1 on the CPU (f32 compute).
    r2_sep.py <model: htdemucs|htdemucs_6s|htdemucs_ft_vocals> <variant tag> <musdb test dir> <out dir> [track…]
      [--passes s0,A,B] [--overlap 0.25] [--weights f32|fp16|bf16|int8] [--threads 3]
Pass s0 = no time shift; A and B = the first and second random time shifts of `--shifts 2` with
random.seed(0), exactly as demucs.apply.apply_model draws them. From those passes it writes:
  <tag>_s0 (pass s0), <tag>_s1 (= A, what --shifts 1 gives), <tag>_s2 (= mean of A and B, what --shifts 2 gives).
Normalisation as demucs.separate (subtract mean, divide by std of the mono mix; undone afterwards).
For htdemucs_6s, 'other' is written as guitar + piano + other (4-stem comparison); guitar/piano also kept.
Per track and pass it records CPU seconds (process_time, all threads) in <out>/cpu.jsonl.
"""
import argparse, json, os, random, time
import numpy as np
import soundfile as sf
import torch

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('tag'); ap.add_argument('musdb'); ap.add_argument('out'); ap.add_argument('tracks', nargs='*')
ap.add_argument('--passes', default='s0'); ap.add_argument('--overlap', type=float, default=0.25)
ap.add_argument('--weights', default='f32'); ap.add_argument('--threads', type=int, default=3)
a = ap.parse_args()
torch.set_num_threads(a.threads)
from demucs.pretrained import get_model
from demucs.apply import apply_model, TensorChunk, tensor_chunk

name = 'htdemucs_ft' if a.model == 'htdemucs_ft_vocals' else a.model
bag = get_model(name)
model = bag
if a.model == 'htdemucs_ft_vocals':
    vi = bag.sources.index('vocals')
    k = [i for i, w in enumerate(bag.weights) if w[vi] > 0]
    assert len(k) == 1, bag.weights
    model = bag.models[k[0]]
model.eval()

def quantize(m, how):
    """Round weights: fp16/bf16 = every parameter; int8 = symmetric per-output-channel on conv / linear /
    attention projection weights (biases, norms and scales stay f32). Returns stats."""
    st = {'params': 0, 'changed': 0, 'quantized_params': 0}
    if how in ('fp16', 'bf16'):
        dt = torch.float16 if how == 'fp16' else torch.bfloat16
        for p in m.parameters():
            q = p.data.to(dt).to(torch.float32)
            st['params'] += p.numel(); st['changed'] += int((q != p.data).sum()); p.data.copy_(q)
        return st
    def q8(w, axis):
        dims = [d for d in range(w.dim()) if d != axis]
        s = w.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12) / 127
        return (w / s).round().clamp(-127, 127) * s
    for mod in m.modules():
        targets = []
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)): targets = [('weight', 0)]
        elif isinstance(mod, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)): targets = [('weight', 1)]
        elif isinstance(mod, torch.nn.MultiheadAttention) and mod.in_proj_weight is not None: targets = [('in_proj_weight', 0)]
        for attr, axis in targets:
            w = getattr(mod, attr)
            q = q8(w.data, axis); st['quantized_params'] += w.numel(); st['changed'] += int((q != w.data).sum()); w.data.copy_(q)
    st['params'] = sum(p.numel() for p in m.parameters())
    return st

qstats = quantize(model, a.weights) if a.weights != 'f32' else None
sources = list(model.sources)
os.makedirs(a.out, exist_ok=True)
if qstats: json.dump({'tag': a.tag, 'weights': a.weights, **qstats}, open(f'{a.out}/{a.tag}_weights.json', 'w'))

def one_pass(mix, offset):
    """mix: (2, n) normalised. offset None = no shift; else demucs' shift trick for one pass."""
    if offset is None:
        return apply_model(model, mix[None], shifts=0, split=True, overlap=a.overlap)[0]
    length = mix.shape[-1]; max_shift = int(0.5 * model.samplerate)
    padded = tensor_chunk(mix[None]).padded(length + 2 * max_shift)
    shifted = TensorChunk(padded, offset, length + max_shift - offset)
    return apply_model(model, shifted, shifts=0, split=True, overlap=a.overlap)[0][..., max_shift - offset:]

passes = a.passes.split(',')
tracks = a.tracks or sorted(os.listdir(a.musdb))
for tr in tracks:
    wav, sr = sf.read(f'{a.musdb}/{tr}/mixture.wav', dtype='float32', always_2d=True)
    assert sr == 44100
    mix = torch.from_numpy(wav.T.copy())
    ref = mix.mean(0); mu, sd = ref.mean(), ref.std()
    x = (mix - mu) / sd
    random.seed(0); torch.manual_seed(0)
    max_shift = int(0.5 * model.samplerate)
    offsets = {}
    for p in ('A', 'B'):
        offsets[p] = random.randint(0, max_shift)  # same draws, in the same order, as apply_model(shifts=2)
    res, cpu = {}, {}
    with torch.no_grad():
        for p in passes:
            c0, w0 = time.process_time(), time.time()
            res[p] = one_pass(x, None if p == 's0' else offsets[p]) * sd + mu
            cpu[p] = (time.process_time() - c0, time.time() - w0)
    outs = {}
    if 's0' in res: outs[f'{a.tag}_s0'] = res['s0']
    if 'A' in res: outs[f'{a.tag}_s1'] = res['A']
    if 'A' in res and 'B' in res: outs[f'{a.tag}_s2'] = (res['A'] + res['B']) / 2
    for v, est in outs.items():
        d = f'{a.out}/{v}/{tr}'; os.makedirs(d, exist_ok=True)
        e = {s: est[i].numpy().T for i, s in enumerate(sources)}
        if 'guitar' in e:
            for s in ('guitar', 'piano'): sf.write(f'{d}/{s}.wav', e[s], sr, subtype='FLOAT')
            e['other'] = e['other'] + e['guitar'] + e['piano']
        for s in ('vocals', 'drums', 'bass', 'other'):
            if s in e and (a.model != 'htdemucs_ft_vocals' or s == 'vocals'):
                sf.write(f'{d}/{s}.wav', e[s].astype(np.float32), sr, subtype='FLOAT')
        open(f'{d}/done', 'w').write(json.dumps({p: round(c[0], 2) for p, c in cpu.items()}))
    with open(f'{a.out}/cpu.jsonl', 'a') as f:
        f.write(json.dumps({'tag': a.tag, 'model': a.model, 'weights': a.weights, 'overlap': a.overlap, 'track': tr, 'audio_s': wav.shape[0] / sr,
                            'threads': a.threads, 'passes': {p: {'cpu_s': round(c, 2), 'wall_s': round(w, 2)} for p, (c, w) in cpu.items()}}) + '\n')
    print(tr, {p: round(c[0], 1) for p, c in cpu.items()}, flush=True)
