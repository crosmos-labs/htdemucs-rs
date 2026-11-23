"""
Kim's Mel-Band RoFormer vocal model (huggingface.co/KimberleyJSN/melbandroformer, MIT since
2026-04-22): weight conversion and PyTorch references for the Rust port (src/roformer.rs).

Uses the upstream inference code, github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model at
25f44ff (model definition, config, demix_track), passed as <repo>, with research/round1/kim_env.txt
(torch 2.2.2, rotary_embedding_torch 0.3.5, librosa 0.10.2):

    PY="uv run --quiet --python 3.11.17 --with-requirements research/round1/kim_env.txt python"
    $PY tools/melroformer.py convert  <repo> <MelBandRoformer.ckpt> [--half]
        -> weights/melroformer.safetensors (state_dict, PyTorch names, float32 - the checkpoint's
           own precision; --half: float16, lossy, written as melroformer-f16.safetensors)
           + weights/melroformer.json (constructor kwargs, inference settings, the mel bands)
    $PY tools/melroformer.py segment  <repo> <ckpt> <in.wav> <out.safetensors> [--chunk K]
        -> one chunk exactly as demix_track feeds it (input, STFT, band split, every transformer,
           masks, output)
    $PY tools/melroformer.py segment ... --double   (float64 model: the reference for both float32s)
    $PY tools/melroformer.py compare-dumps <a.safetensors> <b.safetensors>   (SNR of a vs b per tensor)
    $PY tools/melroformer.py bench    <repo> <ckpt> [--threads N] [--runs R]
        -> CPU and wall time of one chunk forward (PyTorch's baseline), RESULT line
    $PY tools/melroformer.py separate <repo> <ckpt> <in.wav> <out.safetensors> [--threads N]
        -> {"stems": (2, 2, L)} = vocals, instrumental (mix - vocals, as upstream inference.py),
           the format src/bin/compare.rs reads

The checkpoint is a pickle from the internet: it is loaded with weights_only=True.
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = "configs/config_vocals_mel_band_roformer.yaml"


def load(repo, ckpt):
    sys.path.insert(0, repo)
    import yaml
    from ml_collections import ConfigDict
    from utils import get_model_from_config

    cfg = ConfigDict(yaml.load(open(os.path.join(repo, CONFIG)), Loader=yaml.FullLoader))
    model = get_model_from_config("mel_band_roformer", cfg)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return cfg, model.eval()


def read_wav(path):
    import soundfile as sf

    mix, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 44100, f"{path}: {sr} Hz"
    if mix.shape[1] == 1:
        mix = np.concatenate([mix, mix], 1)
    return torch.tensor(mix[:, :2].T.copy())


def convert(a):
    cfg, model = load(a.repo, a.ckpt)
    sd = model.state_dict()
    state, exact = {}, 0
    for k, v in sd.items():
        v = v.float().contiguous()
        h = v.half()
        exact += int(torch.equal(h.float(), v))
        # --half: the weights in float16; the rotary frequencies stay float32 (they are constants,
        # not learned, and rounding them would move every rotation angle).
        state[k] = h if a.half and not k.endswith("rotary_embed.freqs") else v
    m = cfg.model
    # The bands: freqs_per_band (60 x 1025, bool) from librosa's mel filter bank, with the
    # upstream fix-ups ([0][0] and [-1][-1] forced on). Band b's features are its frequencies in
    # ascending order, each (channel 0, channel 1) x (re, im).
    fpb = model.freqs_per_band.numpy()
    bands = [np.nonzero(r)[0].tolist() for r in fpb]
    rot = sd["layers.0.0.layers.0.0.rotary_embed.freqs"]
    config = {
        "name": "melroformer",
        "checkpoint": os.path.basename(a.ckpt),
        "upstream": "github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model@25f44ff",
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(m).items()},
        "chunk_size": cfg.inference.chunk_size,
        "num_overlap": cfg.inference.num_overlap,
        "target_instrument": cfg.training.target_instrument,
        "mask_estimator_hidden": 4 * m.dim,
        "ff_mult": 4,
        "bands": bands,
        "num_bands_per_freq": model.num_bands_per_freq.tolist(),
    }
    assert torch.equal(rot, 1.0 / (10000 ** (torch.arange(0, m.dim_head, 2).float() / m.dim_head)))
    out = ROOT / "weights"
    out.mkdir(exist_ok=True)
    name = "melroformer-f16" if a.half else "melroformer"
    st.save(out / f"{name}.safetensors", state, metadata={"model": "melroformer", "checkpoint": config["checkpoint"]})
    (out / "melroformer.json").write_text(json.dumps(config) + "\n")
    n = sum(v.numel() for v in state.values())
    print(f"{a.ckpt} ({os.path.getsize(a.ckpt) / 1e6:.1f} MB) -> weights/{name}.safetensors "
          f"({os.path.getsize(out / f'{name}.safetensors') / 1e6:.1f} MB, {n / 1e6:.2f}M params, {len(state)} tensors; "
          f"{exact} of them exactly float16)")


def chunk_input(cfg, mix, k):
    """The k-th chunk demix_track feeds the model (border reflect padding, last-chunk padding)."""
    C = cfg.inference.chunk_size
    step = C // cfg.inference.num_overlap
    border = C - step
    if mix.shape[1] > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")
    i = k * step
    assert i < mix.shape[1], f"chunk {k} past the end"
    part = mix[:, i:i + C]
    length = part.shape[-1]
    if length < C:
        if length > C // 2 + 1:
            part = nn.functional.pad(input=part, pad=(0, C - length), mode="reflect")
        else:
            part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode="constant", value=0)
    return part


def segment(a):
    torch.set_num_threads(a.threads)
    cfg, model = load(a.repo, a.ckpt)
    x = chunk_input(cfg, read_wav(a.inp), a.chunk)
    dump = {"input": x.clone()}
    if a.double:  # float64 everywhere (weights, input, STFT): the reference for both float32s
        model = model.double()
        x = x.double()
        w = torch.hann_window(cfg.model.stft_win_length).double()  # float32 Hann values, as float64
        model.stft_window_fn = lambda device=None, dtype=None: w
    hooks = []

    def keep(name):
        def h(_m, _i, o):
            dump[name] = o.detach().clone()
        return h

    hooks.append(model.band_split.register_forward_hook(keep("band_split")))
    for i, (tt, ft) in enumerate(model.layers):
        hooks.append(tt.register_forward_hook(keep(f"layers.{i}.time")))
        hooks.append(ft.register_forward_hook(keep(f"layers.{i}.freq")))
        if i == 0:
            hooks.append(tt.layers[0][0].register_forward_hook(keep("layers.0.time.attn")))
            hooks.append(tt.layers[0][1].register_forward_hook(keep("layers.0.time.ff")))
    hooks.append(model.mask_estimators[0].register_forward_hook(keep("mask")))
    with torch.no_grad():
        stft = torch.stft(x, **model.stft_kwargs, window=model.stft_window_fn(device="cpu"), return_complex=True)
        dump["stft"] = torch.view_as_real(stft)
        t0 = time.process_time()
        y = model(x[None])[0]
        cpu = time.process_time() - t0
    dump["output"] = y
    # Batch dimension dropped; transformer outputs are (sequences, length, dim).
    out = {k: (v[0] if k in ("band_split", "mask") else v).float().contiguous() for k, v in dump.items()}
    if a.double:
        out["input"] = dump["input"].float()
    st.save(a.out, out, metadata={"chunk": a.chunk, "input": a.inp, "cpu_s": cpu, "threads": torch.get_num_threads()})
    print(f"chunk {a.chunk}: forward {cpu:.1f} CPU s on {torch.get_num_threads()} threads; "
          f"{len(out)} tensors -> {a.out}")


def separate(a):
    torch.set_num_threads(a.threads)
    cfg, model = load(a.repo, a.ckpt)
    from utils import demix_track  # the repo is on the path now
    mix = read_wav(a.inp)
    t0 = time.perf_counter()
    c0 = time.process_time()
    devnull = open(os.devnull, "w")
    so, sys.stdout = sys.stdout, devnull  # demix_track prints its own progress
    try:
        res, _ = demix_track(cfg, model, mix, "cpu")
    finally:
        sys.stdout = so
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    voc = torch.from_numpy(res["vocals"]).float()
    # inference.py: instrumental = original_mix - vocals (float64 mix from soundfile, written as
    # float32): the correctly rounded float32 difference.
    inst = (mix.double() - voc.double()).float()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
    st.save(a.out, {"stems": torch.stack([voc, inst]).contiguous()},
            metadata=dict(model="melroformer", sources="vocals,instrumental", seconds=wall, cpu_s=cpu,
                          threads=torch.get_num_threads(), peak_rss_mb=rss, input=a.inp))
    audio = mix.shape[1] / 44100
    print(json.dumps({"input": a.inp, "audio_s": round(audio, 2), "wall_s": round(wall, 1), "cpu_s": round(cpu, 1),
                      "cpu_per_audio_s": round(cpu / audio, 3), "threads": torch.get_num_threads(), "peak_rss_mb": round(rss)}))


def compare_dumps(a):
    """SNR (dB) and max abs error of dump a against dump b, per shared tensor."""
    x, _ = st.load(a.a)
    y, _ = st.load(a.b)
    for k in x:
        if k in y and k != "input":
            u, v = x[k].astype(np.float64), y[k].astype(np.float64)
            err = ((u - v) ** 2).sum()
            snr = 10 * np.log10((v ** 2).sum() / err) if err > 0 else float("inf")
            print(f"| {k} | {snr:.1f} | {np.abs(u - v).max():.2e} |")


def bench(a):
    torch.set_num_threads(a.threads)
    cfg, model = load(a.repo, a.ckpt)
    x = torch.randn(1, 2, cfg.inference.chunk_size, generator=torch.Generator().manual_seed(0)) * 0.1
    model(x)
    best = float("inf")
    c0 = time.process_time()
    for _ in range(a.runs):
        t = time.perf_counter()
        model(x)
        best = min(best, time.perf_counter() - t)
    cpu = (time.process_time() - c0) / a.runs
    audio = cfg.inference.chunk_size / 44100
    print(f"torch {torch.__version__} {a.threads} threads: wall best {best:.2f}s, CPU {cpu:.2f}s per chunk "
          f"({cpu / audio:.3f} CPU-s per audio-s of chunk)")
    print(f"RESULT model=melroformer threads={a.threads} wall_best={best:.3f} cpu={cpu:.3f} "
          f"cpu_per_audio_s={cpu / audio:.4f} impl=torch")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("repo"); c.add_argument("ckpt"); c.add_argument("--half", action="store_true")
    s = sub.add_parser("segment")
    s.add_argument("repo"); s.add_argument("ckpt"); s.add_argument("inp"); s.add_argument("out")
    s.add_argument("--chunk", type=int, default=0); s.add_argument("--threads", type=int, default=4)
    s.add_argument("--double", action="store_true")
    f = sub.add_parser("separate")
    f.add_argument("repo"); f.add_argument("ckpt"); f.add_argument("inp"); f.add_argument("out")
    f.add_argument("--threads", type=int, default=4)
    b = sub.add_parser("bench")
    b.add_argument("repo"); b.add_argument("ckpt")
    b.add_argument("--threads", type=int, default=1); b.add_argument("--runs", type=int, default=2)
    d = sub.add_parser("compare-dumps")
    d.add_argument("a"); d.add_argument("b")
    a = p.parse_args()
    {"compare-dumps": compare_dumps, "convert": convert, "segment": segment, "separate": separate, "bench": bench}[a.cmd](a)
