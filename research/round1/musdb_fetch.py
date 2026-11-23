"""Fetch only MUSDB18-HQ test/ (or a list of tracks) out of Zenodo's zip with HTTP range requests (remotezip),
never storing the 22.7 GB zip.
    musdb_fetch.py list                      → prints test tracks and their uncompressed sizes
    musdb_fetch.py get <dest> [track…]       → extracts test/<track>/*.wav (all test tracks if none given)
Source: https://zenodo.org/records/3338373 (MUSDB18-HQ, Rafii et al. 2019), file musdb18hq.zip, 22,656,664,047 bytes.
Licence: research / educational, non-commercial (Zenodo: "Other (Non-Commercial)"; some tracks CC BY-NC-SA).
"""
import os, shutil, sys
from remotezip import RemoteZip

URL = 'https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1'
with RemoteZip(URL) as z:
    test = [i for i in z.infolist() if i.filename.startswith('test/') and not i.is_dir()]
    if sys.argv[1] == 'list':
        tracks = {}
        for i in test:
            t = i.filename.split('/')[1]; tracks[t] = tracks.get(t, 0) + i.file_size
        for t, s in sorted(tracks.items()): print(f'{s/1e6:8.1f} MB  {t}')
        print(f'{len(tracks)} tracks, {sum(tracks.values())/1e9:.2f} GB uncompressed, {sum(i.compress_size for i in test)/1e9:.2f} GB to transfer')
    else:
        dest = sys.argv[2]; want = set(sys.argv[3:])
        for i in test:
            t = i.filename.split('/')[1]
            if want and t not in want: continue
            out = os.path.join(dest, i.filename)
            if os.path.exists(out) and os.path.getsize(out) == i.file_size: continue
            if shutil.disk_usage(dest).free < 10e9 + i.file_size: sys.exit('stopping: would leave < 10 GB free')
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(i) as src, open(out + '.part', 'wb') as dst: shutil.copyfileobj(src, dst, 1 << 20)
            os.rename(out + '.part', out); print('got', i.filename, flush=True)
