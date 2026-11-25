// Drive the bench page in the installed Chrome (separate throwaway profile) and print its result JSON, plus the peak
// physical footprint (macOS `footprint`, includes Metal/IOAccelerator memory) of the browser's GPU and renderer processes.
//   NODE_PATH=<scratch>/node_modules node research/webgpu/run.mjs '<query string>' [--headful] [--port 8765]
import { createRequire } from 'node:module'; import { execFileSync } from 'node:child_process'; import fs from 'node:fs';
const require = createRequire(process.env.NODE_PATH + '/'); const puppeteer = require('puppeteer-core');
const qs = process.argv[2]; const headful = process.argv.includes('--headful');
const pageName = process.argv.includes("--page") ? process.argv[process.argv.indexOf("--page") + 1] : "bench";
const port = process.argv.includes('--port') ? process.argv[process.argv.indexOf('--port') + 1] : '8765';
const prof = fs.mkdtempSync('/Users/divyansh/work/mac-testing/webgpu/profile-');
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: !headful, userDataDir: prof, protocolTimeout: 0,
  args: ['--enable-unsafe-webgpu', '--no-first-run', '--no-default-browser-check', '--disable-background-timer-throttling',
         '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] });
const root = browser.process().pid; let peak = {};
function tree(pid) { let k = []; try { k = execFileSync('pgrep', ['-P', String(pid)]).toString().trim().split('\n').filter(Boolean); } catch {} return [pid, ...k.flatMap(c => tree(+c))]; }
function sample() {
  for (const pid of tree(root)) {
    let cmd = ''; try { cmd = execFileSync('ps', ['-o', 'command=', '-p', String(pid)]).toString(); } catch { continue; }
    const kind = /--type=gpu-process/.test(cmd) ? 'gpu' : /--type=renderer/.test(cmd) ? 'renderer' : null; if (!kind) continue;
    try { const o = execFileSync('footprint', ['-p', String(pid), '-f', 'bytes'], { stdio: ['ignore', 'pipe', 'ignore'] }).toString();
      const m = o.match(/Footprint:\s*([\d,]+)\s*B/i) || o.match(/phys_footprint:\s*([\d,]+)/i);
      if (m) { const v = +m[1].replace(/,/g, '') / 2 ** 20; peak[kind] = Math.max(peak[kind] || 0, v); } } catch {}
  }
}
const timer = setInterval(sample, 1500);
const page = await browser.newPage(); page.setDefaultTimeout(0);
page.on('console', m => process.stderr.write(`[page] ${m.text()}\n`));
await page.goto(`http://127.0.0.1:${port}/${pageName}.html?${qs}`);
await page.waitForFunction('window.__result', { timeout: 0, polling: 1000 });
const r = await page.evaluate('window.__result'); clearInterval(timer); sample();
r.peak_footprint_mb = Object.fromEntries(Object.entries(peak).map(([k, v]) => [k, Math.round(v)]));
console.log(JSON.stringify(r));
await browser.close(); fs.rmSync(prof, { recursive: true, force: true });
