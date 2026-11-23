"""Fetch selected members of a Zenodo zip with HTTP range requests (remotezip), never storing the whole zip.
    zen_fetch.py <zip url> <dest dir> <regex>     → extracts members whose name matches the regex (flattened)"""
import os, re, shutil, sys
from remotezip import RemoteZip
url, dest, rx = sys.argv[1:4]
with RemoteZip(url) as z:
    for i in z.infolist():
        if i.is_dir() or not re.search(rx, i.filename) or '__MACOSX' in i.filename: continue
        out = os.path.join(dest, i.filename.replace('/', '__'))
        if os.path.exists(out) and os.path.getsize(out) == i.file_size: continue
        with z.open(i) as s, open(out + '.part', 'wb') as d: shutil.copyfileobj(s, d, 1 << 20)
        os.rename(out + '.part', out); print('got', i.filename, flush=True)
