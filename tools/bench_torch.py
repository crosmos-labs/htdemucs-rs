"""Time PyTorch's HTDemucs forward on one segment (CPU float32), best of N runs, to compare with
`cargo run --release --bin bench -- weights <model>`.

    uv run ... python tools/bench_torch.py htdemucs [runs]
"""
import sys
import time

import torch

from demucs.pretrained import get_model

name = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
torch.set_grad_enabled(False)
model = get_model(name).models[0].cpu().eval()
x = torch.randn(1, 2, int(model.segment * model.samplerate), generator=torch.Generator().manual_seed(0))
model(x)
best = float("inf")
for _ in range(runs):
    t = time.perf_counter()
    model(x)
    best = min(best, time.perf_counter() - t)
print(f"{name}: torch {torch.__version__} best of {runs}: {best:.3f}s per segment ({torch.get_num_threads()} threads)")
