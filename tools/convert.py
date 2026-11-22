"""
Convert a pretrained Demucs HTDemucs checkpoint (.th, from torch hub's cache) into
  weights/<name>.safetensors  - the model's state_dict, PyTorch names unchanged, float16 (the
                                checkpoint's own precision: lossless);
  weights/<name>.json         - the model's constructor kwargs plus a few derived values.

    uv run --quiet --python 3.11.17 --with-requirements <fretboard>/scripts/requirements-demucs.txt \
        python tools/convert.py htdemucs htdemucs_6s

The checkpoint file comes from demucs' own remote model list (demucs/remote/<name>.yaml names the
signature, files.txt the file), so which .th is which model is never guessed:
  htdemucs    -> 955717e8-8726e21a.th
  htdemucs_6s -> 5c90dfd2-34c22ccb.th
"""
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import torch

from demucs.apply import BagOfModels
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model, REMOTE_ROOT
import yaml

import st

ROOT = Path(__file__).resolve().parent.parent


def checkpoint_file(name):
    sigs = yaml.safe_load((REMOTE_ROOT / f"{name}.yaml").read_text())["models"]
    files = (REMOTE_ROOT / "files.txt").read_text().split()
    return [f for s in sigs for f in files if f.startswith(s + "-")]


def jsonable(v):
    if isinstance(v, Fraction):
        return {"num": v.numerator, "den": v.denominator}
    return v


def convert(name):
    files = checkpoint_file(name)
    bag = get_model(name)
    assert isinstance(bag, BagOfModels) and len(bag.models) == 1, "expected a bag of one model"
    assert all(w == 1.0 for ws in bag.weights for w in ws)
    model = bag.models[0]
    assert isinstance(model, HTDemucs)
    _, kwargs = model._init_args_kwargs
    config = {k: jsonable(v) for k, v in kwargs.items()}
    config["name"] = name
    config["checkpoint"] = files[0]
    config["training_length"] = int(model.segment * model.samplerate)
    # Demucs ships its checkpoints in float16 and upcasts to float32 on load, so storing float16
    # is lossless (checked here) and halves the file.
    state = {}
    for k, v in model.state_dict().items():
        h = v.half()
        assert torch.equal(h.float(), v.float()), f"{k} is not exactly float16"
        state[k] = h.contiguous()

    out = ROOT / "weights"
    out.mkdir(exist_ok=True)
    st.save(out / f"{name}.safetensors", state, metadata={"model": name, "checkpoint": files[0]})
    (out / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
    th = Path(torch.hub.get_dir()) / "checkpoints" / files[0]
    n = sum(v.numel() for v in state.values())
    print(f"{name}: {files[0]} ({os.path.getsize(th) / 1e6:.1f} MB) -> "
          f"weights/{name}.safetensors ({os.path.getsize(out / f'{name}.safetensors') / 1e6:.1f} MB, "
          f"{n / 1e6:.2f}M params, {len(state)} tensors)")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    for name in sys.argv[1:] or ["htdemucs", "htdemucs_6s"]:
        convert(name)
