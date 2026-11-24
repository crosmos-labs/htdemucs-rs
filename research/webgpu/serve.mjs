// Local-only static server for the WebGPU prototype (binds 127.0.0.1). Serves the page from research/webgpu/page,
// onnxruntime-web from <scratch>/node_modules/onnxruntime-web/dist at /ort/, models and test chunks from <scratch>/models at
// /models/, with COOP/COEP so WASM threads (SharedArrayBuffer) work. POST /result?name=x saves the body to <scratch>/results/x.
//   node research/webgpu/serve.mjs <scratch dir> [port]
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import url from 'node:url';
const here = path.dirname(url.fileURLToPath(import.meta.url));
const scratch = path.resolve(process.argv[2] || '.'); const port = +(process.argv[3] || 8765);
const roots = { '/ort/': path.join(scratch, 'node_modules/onnxruntime-web/dist'), '/models/': path.join(scratch, 'models'), '/': path.join(here, 'page') };
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' };
fs.mkdirSync(path.join(scratch, 'results'), { recursive: true });
http.createServer((req, res) => {
  const u = new URL(req.url, 'http://x');
  const h = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp', 'Cache-Control': 'no-store' };
  if (req.method === 'POST' && u.pathname === '/result') {
    const name = path.basename(u.searchParams.get('name') || 'result.json');
    const out = fs.createWriteStream(path.join(scratch, 'results', name));
    req.pipe(out); out.on('finish', () => { res.writeHead(200, h); res.end('ok'); }); return;
  }
  const pre = Object.keys(roots).find(p => u.pathname.startsWith(p));
  const f = path.join(roots[pre], path.normalize(decodeURIComponent(u.pathname.slice(pre.length)) || 'index.html'));
  if (!f.startsWith(roots[pre])) { res.writeHead(403); return res.end(); }
  fs.stat(f, (e, st) => {
    if (e || !st.isFile()) { res.writeHead(404, h); return res.end('404'); }
    res.writeHead(200, { ...h, 'Content-Type': types[path.extname(f)] || 'application/octet-stream', 'Content-Length': st.size });
    fs.createReadStream(f).pipe(res);
  });
}).listen(port, '127.0.0.1', () => console.log(`serving on http://127.0.0.1:${port}/ (scratch ${scratch})`));
