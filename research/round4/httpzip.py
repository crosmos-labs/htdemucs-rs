"""A seekable read-only file over HTTP range requests, so zipfile can list or extract members of a remote zip without
downloading it. Sends Accept-Encoding: identity (Cloudflare drops Content-Length on compressed HEAD replies)."""
import io, time, requests

class HttpFile(io.RawIOBase):
    def __init__(self, url, ua='Mozilla/5.0'):
        self.url, self.pos, self.s = url, 0, requests.Session()
        self.s.headers.update({'User-Agent': ua, 'Accept-Encoding': 'identity'})
        r = self._get(0, 0); self.size = int(r.headers['Content-Range'].rsplit('/', 1)[1])
    def _get(self, a, b):
        for k in range(8):
            try:
                r = self.s.get(self.url, headers={'Range': f'bytes={a}-{b}'}, timeout=60)
                if r.status_code == 206: return r
            except requests.RequestException: pass
            time.sleep(2 + 3 * k)
        raise IOError(f'range {a}-{b} failed for {self.url}')
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b''
        d = self._get(self.pos, self.pos + n - 1).content; self.pos += len(d); return d
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
