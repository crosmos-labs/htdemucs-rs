"""Build the stem sets each variant feeds the Fretboard aligner, from Demucs splits already on disk.
    make_variants.py <set tag> <dir4> <dir6> <mix.m4a> <outdir> [--kim vocals.wav]
Writes <outdir>/<variant>/{vocals,other,bass,harmonic}.wav (float32, 44.1 kHz stereo) and lags.json.
Variants (what the aligner reads; the chords always read 'harmonic'):
  V0L V0 with the 4-stem mp3s' encoder delay removed (sample-aligned to the recording), as wav
  V0  today: 4-stem vocals/other/bass (the mp3s, as the pipeline passes them); harmonic = mix − 6s vocals − 6s drums
  V1a one 6s pass: 6s vocals; other = 6s guitar + piano + other; 6s bass; harmonic as V0
  V1b two-model average: vocals/drums/bass = mean of 4s and 6s; other = mean(4s other, 6s guitar+piano+other);
      harmonic = mix − mean vocals − mean drums
  K   Kim Mel-RoFormer vocals (if --kim): vocals = Kim; other/bass = 4s delay-fixed (as V0L); harmonic = mix − Kim vocals − 6s drums
Every file is decoded with ffmpeg, as scripts/sync.py does.
"""
import json, os, subprocess, sys
import numpy as np
import soundfile as sf

tag, d4, d6, mixp, out = sys.argv[1:6]
kim = sys.argv[sys.argv.index('--kim') + 1] if '--kim' in sys.argv else None

def dec(p):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-ac', '2', '-ar', '44100', '-f', 'f32le', '-'], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()

mix = dec(mixp); n = len(mix)
def fit(x):
    return np.pad(x, ((0, max(0, n - len(x))), (0, 0)))[:n]
s4 = {k: fit(dec(f'{d4}/{k}.mp3')) for k in ['vocals', 'drums', 'bass', 'other']}
s6 = {k: fit(dec(f'{d6}/{k}.flac')) for k in ['vocals', 'drums', 'bass', 'other', 'guitar', 'piano']}

def lag(a, b, max_lag=4410):
    """Lag of b vs a (samples) on mono 60 s from the middle, by FFT cross-correlation."""
    m = len(a) // 2; a = a[m:m + 44100 * 60].mean(1); b = b[m:m + 44100 * 60].mean(1)
    L = 1 << int(np.ceil(np.log2(len(a) * 2)))
    c = np.fft.irfft(np.fft.rfft(a, L) * np.conj(np.fft.rfft(b, L)), L)
    c = np.concatenate([c[-max_lag:], c[:max_lag + 1]])
    return int(np.argmax(c) - max_lag)

lags = {'4s_vs_6s_vocals': lag(s6['vocals'], s4['vocals']), 'mix_vs_6s_sum': lag(mix, sum(s6.values())), 'mix_vs_4s_sum': lag(mix, sum(s4.values()))}
# The 4-stem mp3s come out delayed vs the recording (mp3 encoder delay, no gapless trim on decode).
# Measure the delay against the recording itself and undo it for every variant except V0.
D = -lags['mix_vs_4s_sum']
lags['4s_delay_samples'] = D
s4_raw = s4
s4 = {k: np.pad(v[D:], ((0, D), (0, 0))) if D > 0 else v for k, v in s4.items()}
lags['mix_vs_4s_sum_after_fix'] = lag(mix, sum(s4.values()))
def snr(ref, est):
    return float(10 * np.log10((ref ** 2).sum() / (((ref - est) ** 2).sum() + 1e-12)))
lags['residual_6s_dB'] = -snr(mix, sum(s6.values()))
lags['residual_4s_dB'] = -snr(mix, sum(s4.values()))
lags['residual_4s_unfixed_dB'] = -snr(mix, sum(s4_raw.values()))
lags['vocals_4s_vs_6s_snr_dB'] = snr(s4['vocals'], s6['vocals'])

def write(v, name, x):
    os.makedirs(f'{out}/{v}', exist_ok=True)
    sf.write(f'{out}/{v}/{name}.wav', x.astype(np.float32), 44100, subtype='FLOAT')

oth6 = s6['guitar'] + s6['piano'] + s6['other']
harm6 = mix - s6['vocals'] - s6['drums']
write('V0', 'harmonic', harm6)
for k in ['vocals', 'other', 'bass']:
    os.makedirs(f'{out}/V0', exist_ok=True)
    dst = f'{out}/V0/{k}.mp3'
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.abspath(f'{d4}/{k}.mp3'), dst)
write('V0L', 'vocals', s4['vocals']); write('V0L', 'other', s4['other']); write('V0L', 'bass', s4['bass']); write('V0L', 'harmonic', harm6)
write('V1a', 'vocals', s6['vocals']); write('V1a', 'other', oth6); write('V1a', 'bass', s6['bass']); write('V1a', 'harmonic', harm6)
av = (s4['vocals'] + s6['vocals']) / 2; ad = (s4['drums'] + s6['drums']) / 2
write('V1b', 'vocals', av); write('V1b', 'other', (s4['other'] + oth6) / 2); write('V1b', 'bass', (s4['bass'] + s6['bass']) / 2); write('V1b', 'harmonic', mix - av - ad)
if kim:
    kv = fit(dec(kim)); lags['kim_vs_6s_vocals'] = lag(s6['vocals'], kv)
    lags['kim_vs_6s_vocals_snr_dB'] = snr(kv, s6['vocals']); lags['kim_vs_4s_vocals_snr_dB'] = snr(kv, s4['vocals'])
    write('K', 'vocals', kv); write('K', 'harmonic', mix - kv - s6['drums'])
    write('K', 'other', s4['other']); write('K', 'bass', s4['bass'])
json.dump(lags, open(f'{out}/lags.json', 'w'), indent=1)
print(tag, json.dumps(lags))
