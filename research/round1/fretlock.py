"""Hold Fretboard's Demucs lock (same file and format as guitar-tabs server/audio.ts withDemucsLock)
around a command, so our splits never run on the Apple GPU at the same time as the app's:
    python3 fretlock.py <command…>
"""
import json, os, subprocess, sys, tempfile, time

LOCK = os.path.join(tempfile.gettempdir(), 'fretboard-demucs.lock')
STALE = 2 * 3600 * 1000

def alive(pid):
    try:
        os.kill(pid, 0); return True
    except PermissionError:
        return True
    except OSError:
        return False

while True:
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({'pid': os.getpid(), 'at': int(time.time() * 1000)}).encode()); os.close(fd)
        break
    except FileExistsError:
        try:
            held = json.load(open(LOCK))
        except Exception:
            held = {}
        if (held.get('pid') and not alive(held['pid'])) or (held.get('at') and time.time() * 1000 - held['at'] > STALE):
            try: os.rename(LOCK, f'{LOCK}.stale-{os.getpid()}-{int(time.time()*1000)}')
            except OSError: pass
        else:
            time.sleep(2)
try:
    p = subprocess.Popen(sys.argv[1:], start_new_session=True)
    held = json.load(open(LOCK)); held['child'] = p.pid; open(LOCK, 'w').write(json.dumps(held))
    sys.exit(p.wait())
finally:
    try:
        if json.load(open(LOCK)).get('pid') == os.getpid(): os.remove(LOCK)
    except Exception:
        pass
