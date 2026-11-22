"""
Full-song PyTorch reference: demucs.separate's own code path (load_track, whole-track
normalisation, apply_model with shifts / split / overlap 0.25, de-normalisation), seeded exactly
like Fretboard's scripts/demucs_seeded.py (random.seed + torch.manual_seed before loading), on CPU
float32. Saves the raw float stems BEFORE any clipping / encoding, as safetensors
{"stems": (S, 2, L)} with the source names in the metadata.

    uv run ... python tools/ref_separate.py --model htdemucs --shifts 2 --seed 0 in.wav out.safetensors
"""
import argparse
import random
import resource
import sys
import time

import torch

import st

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--shifts", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--threads", type=int, default=0)
p.add_argument("input")
p.add_argument("out")
args = p.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.threads:
    torch.set_num_threads(args.threads)
from demucs.apply import apply_model  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402
from demucs.separate import load_track  # noqa: E402

torch.set_grad_enabled(False)
model = get_model(args.model)
model.cpu()
model.eval()
t0 = time.perf_counter()
wav = load_track(args.input, model.audio_channels, model.samplerate)
ref = wav.mean(0)
wav -= ref.mean()
wav /= ref.std()
sources = apply_model(model, wav[None], device="cpu", shifts=args.shifts, split=True,
                      overlap=0.25, progress=False, num_workers=0, segment=None)[0]
sources *= ref.std()
sources += ref.mean()
secs = time.perf_counter() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
st.save(args.out, {"stems": sources.contiguous()},
        metadata=dict(model=args.model, shifts=args.shifts, seed=args.seed, sources=",".join(model.sources),
                      seconds=secs, threads=torch.get_num_threads(), peak_rss_mb=rss, input=args.input))
print(f"{args.model} shifts={args.shifts} seed={args.seed}: {sources.shape[-1] / 44100:.1f}s of audio in "
      f"{secs:.1f}s on {torch.get_num_threads()} threads, peak RSS {rss:.0f} MB -> {args.out}")
