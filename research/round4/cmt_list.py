"""List Cambridge-MT candidate projects' zip contents with HTTP range requests (no download), excluding MUSDB18 songs.
    cmt_list.py <manifest.json> <out.json>
MUSDB18's 150 track names come from the MUSDB18-HQ zip's directory listing (Zenodo 3338373), matched to Cambridge-MT
file names by artist+title letters only. Writes {project: {"url", "size_mb", "genre", "files": [...]}} for rock / alt-rock
projects that have an Excerpt zip; zips the server won't range-read are recorded with files = null.
"""
import json, re, sys
from remotezip import RemoteZip

norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
with RemoteZip('https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1') as z:
    musdb = sorted({i.filename.split('/')[1] for i in z.infolist() if i.filename.count('/') >= 2})
mk = {norm(t.replace(' - ', '')): t for t in musdb}
man = json.load(open(sys.argv[1])); out = {'musdb': musdb, 'projects': {}}
for e in man:
    if not (e['genre'].startswith('Rock') or e['genre'].startswith('Alt')) or 'excerpt' not in e: continue
    url = e['excerpt']; name = url.rsplit('/', 1)[1].split('?')[0].removesuffix('.zip')
    n = norm(name.split('_', 1)[-1] if name.startswith('MR') else name)
    hit = [t for k, t in mk.items() if n and (n in k or k in n)]
    rec = {'url': url, 'full': e.get('full'), 'size_mb': e.get('excerptSize'), 'genre': e['genre'], 'title': e['title'], 'musdb': hit}
    if not hit:
        try:
            with RemoteZip(url, headers={'User-Agent': 'Mozilla/5.0'}, support_suffix_range=False) as z:
                rec['files'] = [i.filename for i in z.infolist() if not i.is_dir()]
        except Exception as ex:
            rec['files'] = None; rec['err'] = str(ex)[:200]
    out['projects'][name] = rec
    print(name, hit or len(rec.get('files') or []), flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
