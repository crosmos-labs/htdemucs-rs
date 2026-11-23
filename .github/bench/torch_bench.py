"""
PyTorch CPU baseline: one HTDemucs segment forward (float32), wall and process CPU time.

    python .github/bench/torch_bench.py <model> <threads> [runs]
"""
import sys
import time

import torch

from demucs.pretrained import get_model

name, threads = sys.argv[1], int(sys.argv[2])
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
torch.set_num_threads(threads)
torch.set_grad_enabled(False)
model = get_model(name).models[0].cpu().eval()
length = int(model.segment * model.samplerate)
x = torch.randn(1, 2, length, generator=torch.Generator().manual_seed(0))
model(x)
best = float("inf")
c0, w0 = time.process_time(), time.perf_counter()
for _ in range(runs):
    t = time.perf_counter()
    model(x)
    best = min(best, time.perf_counter() - t)
cpu = (time.process_time() - c0) / runs
mean = (time.perf_counter() - w0) / runs
seg = length / model.samplerate
print(f"{name}: torch {torch.__version__} {threads} threads: wall best {best:.3f}s mean {mean:.3f}s, "
      f"CPU {cpu:.3f}s per segment ({cpu / seg:.3f} CPU-s per audio-s of segment)")
print(f"RESULT model={name} threads={threads} wall_best={best:.4f} wall_mean={mean:.4f} cpu={cpu:.4f} "
      f"seg_secs={seg:.4f} impl=torch")
