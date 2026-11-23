"""Print the file list of remote zips (Cambridge-MT) by URL, reading only the central directory.  cmt_ls.py <url>..."""
import sys, zipfile
from httpzip import HttpFile
for u in sys.argv[1:]:
    try:
        z = zipfile.ZipFile(HttpFile(u)); L = [i for i in z.infolist() if not i.is_dir() and '__MACOSX' not in i.filename]; err = ''
    except Exception as e:
        L, err = [], f'FAILED {e}'
    print('==', u.rsplit('/', 1)[1], err, flush=True)
    for i in L: print(f'  {i.file_size/1e6:7.1f} MB  {i.filename}', flush=True)
