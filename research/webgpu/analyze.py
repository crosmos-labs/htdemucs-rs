"""Quality of browser outputs: <results>/<tag>_out.bin (masks from bench.html?save=1) -> complex mask + iSTFT (torch) ->
per stem: SNR vs the PyTorch fp32 full model on the same chunk, and uSDR vs the MUSDB ground truth where MUSDB has the stem
(Mega: drums, bass; Kim: vocals), next to the PyTorch model's own uSDR. Prints one JSON line per tag.
    analyze.py <models dir> <results dir> <musdb song dir> <name> <tag> [<tag> ...]
"""
import json, sys, numpy as np, soundfile as sf, torch

d, res, song, name = sys.argv[1:5]; tags = sys.argv[5:]
info = json.load(open(f'{d}/{name}.json')); T, Fq, L, hop, nfft = info['T'], info['F'], info['chunk'], info['hop'], info['n_fft']
n = info['out_shape'][1]; stems = info['stems']
X = torch.from_numpy(np.fromfile(f'{d}/{name}_in.bin', '<f4').reshape(1, T, Fq * 4))
full = torch.load(f'{d}/{name}_ref_audio.pt')['full'].numpy().astype(np.float64)
win = torch.hann_window(nfft)


def istft(M):
    Xc = torch.view_as_complex(X.reshape(1, 1, T, Fq, 2, 2))
    Mc = torch.view_as_complex(torch.from_numpy(M).reshape(1, n, T, Fq, 2, 2).contiguous())
    Y = (Xc * Mc)[0].permute(0, 3, 2, 1).clone()
    if info['zero_dc']: Y[:, :, 0] = 0
    return torch.istft(Y.reshape(n * 2, Fq, T), n_fft=nfft, hop_length=hop, win_length=nfft, window=win, length=L).reshape(n, 2, L).numpy().astype(np.float64)


def db(s, e): return float(10 * np.log10((s ** 2).sum() / max((e ** 2).sum(), 1e-300)))


truth = {}
s0 = int(round(info['start_s'] * 44100))
for st, f in (('drums', 'drums'), ('bass', 'bass'), ('vocals', 'vocals')):
    if st in stems:
        truth[st] = sf.read(f'{song}/{f}.wav', dtype='float64', always_2d=True)[0][s0:s0 + L].T
ref_usdr = {st: db(truth[st], truth[st] - full[stems.index(st)]) for st in truth}
for tag in tags:
    M = np.fromfile(f'{res}/{tag}_out.bin', '<f4')
    y = istft(M)
    out = dict(tag=tag, snr_vs_pytorch_db={st: round(db(full[i], full[i] - y[i]), 2) for i, st in enumerate(stems)},
               usdr={st: round(db(truth[st], truth[st] - y[stems.index(st)]), 3) for st in truth},
               usdr_pytorch_fp32={k: round(v, 3) for k, v in ref_usdr.items()})
    print(json.dumps(out), flush=True)
