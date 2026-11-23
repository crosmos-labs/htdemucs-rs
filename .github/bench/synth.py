"""
Deterministic synthetic test music for CI (no copyrighted audio): drums (kick, snare, hats), a
bass line, a chord pad and a vibrato "voice", panned apart, stereo float32 WAV at 44.1 kHz.

    python .github/bench/synth.py out.wav [seconds]
"""
import struct
import sys

import numpy as np

SR = 44100
out = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
n = int(SR * secs)
t = np.arange(n) / SR
rng = np.random.default_rng(1234)
bpm = 120.0
beat = 60.0 / bpm


def env(start, attack, decay, length=None):
    """Exponential-decay envelope starting at `start` seconds."""
    x = t - start
    e = np.where(x < 0, 0.0, np.minimum(x / attack, 1.0) * np.exp(-np.maximum(x - attack, 0) / decay))
    if length is not None:
        e = np.where(x > length, 0.0, e)
    return e


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


drums = np.zeros(n)
bass = np.zeros(n)
pad = np.zeros((2, n))
voice = np.zeros(n)
nbeats = int(secs / beat)
noise = rng.standard_normal(n)
hat_noise = np.diff(rng.standard_normal(n + 1))  # crude high-pass
for b in range(nbeats):
    s = b * beat
    if b % 2 == 0:  # kick: falling sine sweep
        x = np.maximum(t - s, 0)
        drums += 0.9 * env(s, 0.002, 0.12, 0.5) * np.sin(2 * np.pi * (50 * x + 60 * (1 - np.exp(-x * 30)) / 30))
    else:  # snare: noise + tone
        drums += 0.5 * env(s, 0.001, 0.08, 0.4) * (noise + 0.5 * np.sin(2 * np.pi * 190 * t))
    for h in range(2):  # eighth-note hats
        drums += 0.15 * env(s + h * beat / 2, 0.0005, 0.02, 0.1) * hat_noise

roots = [40, 45, 43, 38]  # E2 A2 G2 D2
chords = [[64, 67, 71], [69, 72, 76], [67, 71, 74], [62, 66, 69]]
melody = [76, 74, 72, 71, 72, 74, 76, 79]
for bar in range(int(secs / (4 * beat)) + 1):
    i = bar % 4
    s0 = bar * 4 * beat
    for q in range(4):  # bass: eighth notes, a few harmonics
        for e in range(2):
            s = s0 + (q * 2 + e) * beat / 2
            f = hz(roots[i] + (12 if e else 0))
            g = env(s, 0.005, 0.25, beat / 2)
            bass += g * sum(np.sin(2 * np.pi * k * f * t) / k ** 1.5 for k in range(1, 6))
    g = env(s0, 0.3, 3.0, 4 * beat)  # pad: detuned stacks, wide
    for j, m in enumerate(chords[i]):
        for ch, det in enumerate((-0.003, 0.003)):
            f = hz(m) * (1 + det)
            pad[ch] += g * sum(np.sin(2 * np.pi * k * f * t + j) / k ** 2 for k in range(1, 8))
    for q in range(2):  # voice: two half-bar notes with vibrato and a vowel-ish spectrum
        s = s0 + q * 2 * beat
        f = hz(melody[(2 * bar + q) % len(melody)])
        ph = 2 * np.pi * np.cumsum(f * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t))) / SR
        g = env(s, 0.05, 1.5, 2 * beat)
        voice += g * sum(np.sin(k * ph) * np.exp(-((k * f - 700) / 600) ** 2) for k in range(1, 12))

mix = np.zeros((2, n))
for sig, pan, lvl in [(drums, 0.0, 0.5), (bass, -0.2, 0.35), (voice, 0.1, 0.3)]:
    sig = sig / (np.abs(sig).max() + 1e-9)
    mix[0] += lvl * np.sqrt(0.5 - pan / 2) * sig
    mix[1] += lvl * np.sqrt(0.5 + pan / 2) * sig
pad = pad / (np.abs(pad).max() + 1e-9)
mix += 0.2 * pad
mix *= 0.8 / np.abs(mix).max()
data = mix.T.astype("<f4").tobytes()

# WAVE_FORMAT_IEEE_FLOAT, 2 channels, 32-bit.
with open(out, "wb") as f:
    f.write(b"RIFF" + struct.pack("<I", 4 + 26 + 12 + 8 + len(data)) + b"WAVE")
    f.write(b"fmt " + struct.pack("<IHHIIHHH", 18, 3, 2, SR, SR * 8, 8, 32, 0))
    f.write(b"fact" + struct.pack("<II", 4, n))
    f.write(b"data" + struct.pack("<I", len(data)) + data)
print(f"{out}: {secs:.1f}s stereo float32, peak {np.abs(mix).max():.2f}")
