"""Disk pressure: keep only items 00-07 of an open set, cropped to their first 30 s, along with any 6s passes and Kim vocals
already computed on the full item (their first 30 s are kept; those estimates saw the rest of the item as context).
    r4_trim_open.py <items root>/<set> <round4 data dir> <set>"""
import glob, json, os, shutil, sys
import numpy as np, soundfile as sf
root, data, s = sys.argv[1:4]; N = 30 * 44100
for it in sorted(os.listdir(root)):
    d = f'{root}/{it}'
    if int(it[:2]) >= 8:
        for x in (d, f'{data}/pass/{s}/{it}', f'{data}/kim/{s}/{it}'): shutil.rmtree(x, ignore_errors=True)
        continue
    for f in glob.glob(f'{d}/*.wav') + glob.glob(f'{data}/kim/{s}/{it}/vocals.wav'):
        x, sr = sf.read(f, dtype='float32', always_2d=True)
        if len(x) > N: sf.write(f, x[:N], sr, subtype='FLOAT')
    for f in glob.glob(f'{data}/pass/{s}/{it}/*.npy'):
        y = np.load(f)
        if y.shape[-1] > N: np.save(f, np.ascontiguousarray(y[..., :N]))
    m = json.load(open(f'{d}/meta.json')); m['crop'] = {'start_s': 0, 'dur_s': min(30, m['seconds'])}; json.dump(m, open(f'{d}/meta.json', 'w'), indent=1)
    print(s, it, flush=True)
