"""Scoring worker (1 thread): scores every finished separation in data/sep/<variant>/<track>/ (a 'done' marker),
builds the per-stem averages of htdemucs and htdemucs_6s (avg_s0 = mean of hd_s0 and 6s_s0; avg_s2 likewise),
and deletes each separation's WAVs once nothing else needs them. Exits when data/stop exists and nothing is left.
    r2_loop.py <musdb test dir>
"""
import glob, json, os, shutil, subprocess, sys, time
import numpy as np
import soundfile as sf

M = sys.argv[1]; SEP = 'data/sep'; SC = 'data/scores'
PY = ['uv', 'run', '--quiet', '--python', '3.11.17', '--with-requirements', 'r2_env.txt', 'python', '-I', 'r2_score.py']
AVG = {'avg_s0': ('hd_s0', '6s_s0'), 'avg_s2': ('hd_s2', '6s_s2')}
NEEDED_BY_AVG = {v for pair in AVG.values() for v in pair}
env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}

def done(v, tr): return os.path.exists(f'{SEP}/{v}/{tr}/done')
def scored(v, tr): return os.path.exists(f'{SC}/{v}/{tr}.json')
while True:
    work = False
    for avg, (a, b) in AVG.items():
        for tr in sorted(os.listdir(M)):
            if done(a, tr) and done(b, tr) and not done(avg, tr) and not scored(avg, tr):
                d = f'{SEP}/{avg}/{tr}'; os.makedirs(d, exist_ok=True)
                for s in ('vocals', 'drums', 'bass', 'other'):
                    x = sf.read(f'{SEP}/{a}/{tr}/{s}.wav', dtype='float32')[0]; y = sf.read(f'{SEP}/{b}/{tr}/{s}.wav', dtype='float32')[0]
                    sf.write(f'{d}/{s}.wav', (x + y) / 2, 44100, subtype='FLOAT')
                open(f'{d}/done', 'w').write('avg'); work = True
    for dn in sorted(glob.glob(f'{SEP}/*/*/done')):
        d = os.path.dirname(dn); v, tr = d.split('/')[-2], d.split('/')[-1]
        if not scored(v, tr):
            subprocess.run(PY + [M, f'{SEP}/{v}', tr, f'{SC}/{v}/{tr}.json'], env=env, stdout=subprocess.DEVNULL)
            print('scored', v, tr, flush=True); work = True
        if scored(v, tr):
            avg_pending = v in NEEDED_BY_AVG and not any(scored(avg, tr) for avg, pair in AVG.items() if v in pair)
            if not avg_pending:
                for f in glob.glob(f'{d}/*.wav'): os.remove(f)
    if not work:
        if os.path.exists('data/stop'): break
        time.sleep(20)
