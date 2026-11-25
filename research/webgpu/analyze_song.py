"""uSDR of song.html's vocals / drums / bass (<results>/<tag>_stems.bin, int16, 6 x n) against a MUSDB song's stems.
    analyze_song.py <results dir> <tag> <musdb song dir>"""
import json, sys, numpy as np, soundfile as sf
res, tag, song = sys.argv[1:4]
x = np.fromfile(f'{res}/{tag}_stems.bin', '<i2').astype(np.float64) / 32767
n = x.size // 6; x = x.reshape(3, 2, n)
out = {'tag': tag}
for k, st in enumerate(('vocals', 'drums', 'bass')):
    s = sf.read(f'{song}/{st}.wav', dtype='float64', always_2d=True)[0].T[:, :n]
    out[st] = round(float(10 * np.log10((s ** 2).sum() / ((s - x[k]) ** 2).sum())), 3)
print(json.dumps(out))
