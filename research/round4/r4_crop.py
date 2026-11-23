"""Crop every WAV of an item dir to [start, start+dur) s into a new item dir (float32), and optionally keep the full-length
parts as 24-bit FLAC (−146 dB quantisation) in an archive dir.   r4_crop.py <src item> <dst item> <start s> <dur s> [--flac DIR]"""
import argparse, glob, json, os, shutil
import soundfile as sf
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('start', type=float); ap.add_argument('dur', type=float)
ap.add_argument('--flac'); a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)
for f in glob.glob(f'{a.src}/*.wav'):
    x, sr = sf.read(f, dtype='float32', always_2d=True)
    s0 = int(a.start * sr); sf.write(f'{a.dst}/{os.path.basename(f)}', x[s0:s0 + int(a.dur * sr)], sr, subtype='FLOAT')
    if a.flac:
        os.makedirs(a.flac, exist_ok=True); sf.write(f'{a.flac}/{os.path.basename(f)[:-4]}.flac', x, sr, subtype='PCM_24')
m = json.load(open(f'{a.src}/meta.json')); m['crop'] = {'start_s': a.start, 'dur_s': a.dur}
json.dump(m, open(f'{a.dst}/meta.json', 'w'), indent=1)
if a.flac: shutil.copy(f'{a.src}/meta.json', f'{a.flac}/meta.json')
