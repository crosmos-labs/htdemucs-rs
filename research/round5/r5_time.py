"""Time one pass for r5_cost.sh: run the command, then print one JSON line with the child's user + sys CPU seconds, wall
seconds, peak RSS, and the rates per audio second (r5_mega.py's own JSON adds its long-track rate from per-chunk CPU).
    r5_time.py <pass> <threads> <input name> <input wav> <device> <host> -- <cmd ...>
"""
import json, resource, subprocess, sys, time
import soundfile as sf

i = sys.argv.index('--'); p, th, iname, inp, dev, host = sys.argv[1:i]; cmd = sys.argv[i + 1:]
d = sf.info(inp).duration
w0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True); wall = time.time() - w0
if r.returncode: sys.stderr.write(r.stderr[-2000:]); sys.exit(r.returncode)
u = resource.getrusage(resource.RUSAGE_CHILDREN); cpu = u.ru_utime + u.ru_stime
rss = u.ru_maxrss / (2 ** 30 if sys.platform == 'darwin' else 2 ** 20)
o = {'pass': p, 'threads': int(th), 'device': dev, 'host': host, 'input': iname, 'audio_s': round(d, 2), 'cpu_s': round(cpu, 2),
     'wall_s': round(wall, 2), 'cpu_per_audio_s': round(cpu / d, 3), 'wall_per_audio_s': round(wall / d, 3), 'peak_rss_gb': round(rss, 2)}
for line in r.stdout.splitlines():
    if line.startswith('{'):
        j = json.loads(line)
        o.update({'mega_rate_long': j.get('rate_long'), 'mega_chunks': j.get('chunks'), 'gpu_mem_gb': j.get('gpu_mem_gb')})
print(json.dumps(o))
