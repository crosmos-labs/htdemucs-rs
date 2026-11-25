"""
MVSEP's "Mega 53" BS-RoFormer (53 stems; MSST release v1.0.21,
mvsep_mega_model_bs_roformer_53_stems_v1.ckpt, sha256 c6282089...3519f; MIT per the trainer in
MSST issue #245): weight conversion of the shared backbone plus one stem's mask head or a list of
heads, and PyTorch references for the Rust port (src/roformer.rs, `-n megaguitar`, `-n mega`,
`-n kimmega`).

Uses Music-Source-Separation-Training (github.com/ZFTurbo/Music-Source-Separation-Training,
MIT) at 1523206 as <repo> (only models/bs_roformer is imported), with
research/round4/r4_msst_env.txt (torch 2.6.0, rotary_embedding_torch 0.3.5, einops, beartype):

    PY="uv run --quiet --python 3.11 --with-requirements research/round4/r4_msst_env.txt python"
    $PY tools/bsroformer.py convert  <ckpt> <yaml> [--stem guitar] [--name megaguitar] [--float32]
        -> weights/megaguitar.safetensors (the backbone + the guitar head as mask_estimators.0,
           PyTorch names, float16 - the checkpoint's own precision, so lossless; --float32: the same
           values as float32, megaguitar-f32.safetensors) + weights/megaguitar.json (constructor
           kwargs, inference settings, bands)
    $PY tools/bsroformer.py convert  <ckpt> <yaml> --name mega [--stems drums,bass,guitar,keys]
        -> weights/mega.safetensors + weights/mega.json: the backbone + the listed heads as
           mask_estimators.0..N-1 in that order (default: -n kimmega's four), `instruments` in the
           JSON; sources = the heads + "rest" (the input minus all of them)

Every command below takes --stems (several heads, num_stems = N; the default with --name mega is
drums,bass,guitar,keys) or --stem (one head, the megaguitar format).
    $PY tools/bsroformer.py segment  <repo> <ckpt> <yaml> <in.wav> <out.safetensors> [--chunk K]
        -> one chunk exactly as r4_mega.py feeds it (input, STFT, band split, every transformer,
           final norm, mask, output)
    $PY tools/bsroformer.py segment ... --double   (float64 model: the reference for both float32s)
    $PY tools/bsroformer.py separate <repo> <ckpt> <yaml> <in.wav> <out.safetensors> [--threads N]
           [--overlap N] [--chunk C] [--device cpu|cuda] [--minus vocals] [--recipe]
        -> {"stems": (2, 2, L)} = guitar, no_guitar (mix - guitar), the format src/bin/compare.rs
           reads; inference exactly as research/round4/r4_mega.py (MSST's generic demix: 20 s
           chunks, overlap 2, linear fades of chunk/10, reflect padding, float32). Several heads:
           (N + 1, 2, L) = the heads + rest (input - their sum, in float64), as research/round5/
           r5_mega.py runs them (one N-head model). --minus <vocals.wav | melroformer.py separate
           output>: the input is mix - vocals (float32, r5_mega.py's --minus: Kim's instrumental).
           --recipe (with --minus): -n kimmega's six stems, vocals, the heads (keys named piano),
           other = the instrumental - the heads
    $PY tools/bsroformer.py bench    <repo> <ckpt> <yaml> [--threads N] [--runs R]
        -> CPU and wall time of one chunk forward (PyTorch's baseline), RESULT line

The model is built with one mask head (num_stems=1) and loaded with the backbone and the chosen
stem's head: the 53 heads share the backbone and are independent of each other, so this is the
output r4_mega.py gets from the full model with its other heads dropped. The checkpoint is a
pickle from the internet: it is loaded with weights_only=True.
"""
import argparse
import json
import os
import resource
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = "github.com/ZFTurbo/Music-Source-Separation-Training@1523206 (models/bs_roformer)"


def read_cfg(path):
    import yaml

    class L(yaml.SafeLoader):
        pass

    L.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: tuple(l.construct_sequence(n)))
    return yaml.load(open(path), Loader=L)


DEFAULT_HEADS = ["drums", "bass", "guitar", "keys"]  # -n kimmega (research/round5: kim_mega_keys)


def heads_of(a):
    """The heads a command works on: --stems, else the four with --name mega, else --stem."""
    if a.stems:
        return a.stems.split(",")
    if a.name == "mega":
        return list(DEFAULT_HEADS)
    return [a.stem]


def multi(a):
    """The several-heads format (instruments list, sources heads + rest) rather than megaguitar's."""
    return bool(a.stems) or a.name == "mega"


def head_state(ckpt, cfg, stems):
    """The checkpoint's backbone tensors plus the mask heads of `stems` (a name or a list), renamed
    mask_estimators.0, 1, ... in that order."""
    if isinstance(stems, str):
        stems = [stems]
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    inst = cfg["training"]["instruments"]
    pos = {inst.index(s): j for j, s in enumerate(stems)}
    assert len(pos) == len(stems), f"repeated heads: {stems}"
    out = {}
    for k, v in sd.items():
        if k.startswith("mask_estimators."):
            parts = k.split(".")
            if int(parts[1]) not in pos:
                continue
            k = ".".join(["mask_estimators", str(pos[int(parts[1])])] + parts[2:])
        out[k] = v
    return out


def build(repo, ckpt, cfg_path, stems):
    # import models.bs_roformer.bs_roformer without the package __init__ (which pulls in the other
    # architectures' dependencies), as research/round4/r4_mega.py does
    for name, path in (("models", f"{repo}/models"), ("models.bs_roformer", f"{repo}/models/bs_roformer")):
        m = types.ModuleType(name)
        m.__path__ = [path]
        sys.modules[name] = m
    sys.path.insert(0, repo)
    from models.bs_roformer.bs_roformer import BSRoformer

    if isinstance(stems, str):
        stems = [stems]
    cfg = read_cfg(cfg_path)
    mk = dict(cfg["model"])
    mk["num_stems"] = len(stems)
    model = BSRoformer(**mk)
    model.load_state_dict(head_state(ckpt, cfg, stems))  # strict: every tensor present and used
    return cfg, model.eval()


def convert(a):
    cfg = read_cfg(a.cfg)
    m = cfg["model"]
    # The configuration src/roformer.rs computes; refuse others.
    assert m.get("linear_transformer_depth", 0) == 0, "linear transformers"
    assert not m.get("skip_connection", False), "skip connections"
    assert not m.get("use_pope", False), "PoPE"
    assert m.get("zero_dc", True), "zero_dc False"
    assert m["stereo"] and m["time_transformer_depth"] == 1 and m["freq_transformer_depth"] == 1
    heads = heads_of(a)
    sd = head_state(a.ckpt, cfg, heads)
    fpb = list(m["freqs_per_bands"])
    edges = np.cumsum([0] + fpb).tolist()
    bands = [list(range(edges[i], edges[i + 1])) for i in range(len(fpb))]
    assert edges[-1] == m["dim_freqs_in"] == m["stft_n_fft"] // 2 + 1
    state, exact = {}, 0
    for k, v in sd.items():
        exact += int(v.dtype == torch.float16)
        # float16 as stored (lossless: the checkpoint is float16); --float32 the same values widened.
        state[k] = v.float().contiguous() if a.float32 else v.contiguous()
    rot = [v for k, v in sd.items() if k.endswith("rotary_embed.freqs")]
    # The rotary frequencies are float16-rounded in the checkpoint and the model uses them as
    # loaded (into float32 parameters), so they are kept exactly as stored.
    assert all(torch.equal(r, rot[0]) for r in rot)
    mult = m.get("mlp_expansion_factor", 4)
    config = {
        "name": a.name,
        "checkpoint": os.path.basename(a.ckpt),
        "upstream": UPSTREAM,
        "architecture": "bs_roformer",
        "dim": m["dim"], "depth": m["depth"], "heads": m.get("heads", 8), "dim_head": m.get("dim_head", 64),
        "num_bands": len(fpb), "stereo": True, "num_stems": 1,
        "time_transformer_depth": 1, "freq_transformer_depth": 1,
        "dim_freqs_in": m["dim_freqs_in"], "sample_rate": cfg["audio"]["sample_rate"],
        "stft_n_fft": m["stft_n_fft"], "stft_hop_length": m["stft_hop_length"],
        "stft_win_length": m["stft_win_length"], "stft_normalized": m["stft_normalized"],
        "mask_estimator_depth": m["mask_estimator_depth"],
        "mask_estimator_hidden": m["dim"] * mult,
        "mask_estimator_linears": m["mask_estimator_depth"],  # MSST's MLP: depth linear layers
        "ff_mult": 4,
        "chunk_size": cfg["inference"]["chunk_size"],
        "num_overlap": cfg["inference"]["num_overlap"],
        "target_instrument": a.stem,
        **({"num_stems": len(heads), "target_instrument": ",".join(heads), "instruments": heads} if multi(a) else {}),
        "bands": bands,
        "num_bands_per_freq": [1] * edges[-1],
        "transformer_norm_output": False,
        "final_norm": True,
        "zero_dc": True,
        "demix": "msst",
        "sources": heads + ["rest"] if multi(a) else [a.stem, f"no_{a.stem}"],
    }
    out = ROOT / "weights"
    out.mkdir(exist_ok=True)
    name = f"{a.name}-f32" if a.float32 else a.name
    st.save(out / f"{name}.safetensors", state, metadata={"model": a.name, "checkpoint": config["checkpoint"], "stem": ",".join(heads) if multi(a) else a.stem})
    (out / f"{a.name}.json").write_text(json.dumps(config) + "\n")
    n = sum(v.numel() for v in state.values())
    print(f"{a.ckpt} ({os.path.getsize(a.ckpt) / 1e6:.1f} MB) -> weights/{name}.safetensors "
          f"({os.path.getsize(out / f'{name}.safetensors') / 1e6:.1f} MB, {n / 1e6:.2f}M params, {len(state)} tensors, "
          f"{exact} float16 in the checkpoint) + weights/{a.name}.json")


def read_wav(path):
    import soundfile as sf

    mix, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 44100, f"{path}: {sr} Hz"
    if mix.shape[1] == 1:
        mix = np.concatenate([mix, mix], 1)
    return torch.from_numpy(np.ascontiguousarray(mix[:, :2].T))


def window(chunk, fade):
    w = torch.ones(chunk)
    w[:fade] = torch.linspace(0, 1, fade)
    w[-fade:] = torch.linspace(1, 0, fade)
    return w


def chunks(mix, chunk, overlap):
    """research/round4/r4_mega.py's walk: yields (i, L, part, window) for the padded track x."""
    step = chunk // overlap
    border = chunk - step
    fade = chunk // 10
    win = window(chunk, fade)
    x = mix
    n0 = x.shape[-1]
    if n0 > 2 * border and border > 0:
        x = F.pad(x[None], (border, border), mode="reflect")[0]
    N = x.shape[-1]
    i = 0
    while i < N:
        part = x[:, i:i + chunk]
        L_ = part.shape[-1]
        part = F.pad(part[None], (0, chunk - L_), mode="reflect" if L_ > chunk // 2 else "constant")[0]
        w = win.clone()
        if i == 0:
            w[:fade] = 1
        if i + step >= N:
            w[-fade:] = 1
        yield i, L_, part, w
        i += step


def demix(model, mix, chunk, overlap, dev, n=1):
    """research/round4/r4_mega.py's (r5_mega.py's) inference, `n` heads, float32: (n, 2, L), or
    (2, L) for one head."""
    step = chunk // overlap
    border = chunk - step
    n0 = mix.shape[-1]
    padded = n0 > 2 * border and border > 0
    N = n0 + 2 * border if padded else n0
    res = torch.zeros(n, 2, N)
    cnt = torch.zeros(N)
    per = []
    with torch.no_grad():
        for i, L_, part, w in chunks(mix, chunk, overlap):
            t0 = time.process_time()
            y = model(part[None].to(dev))[0].float().cpu().reshape(n, 2, chunk)
            per.append(time.process_time() - t0)
            res[..., i:i + L_] += y[..., :L_] * w[:L_]
            cnt[i:i + L_] += w[:L_]
    out = res / cnt
    if padded:
        out = out[..., border:-border]
    out = torch.nan_to_num(out)
    return (out[0] if n == 1 else out), per


def segment(a):
    torch.set_num_threads(a.threads)
    heads = heads_of(a)
    cfg, model = build(a.repo, a.ckpt, a.cfg, heads)
    nh = len(heads)
    chunk = cfg["inference"]["chunk_size"]
    walk = list(chunks(input_mix(a), chunk, cfg["inference"]["num_overlap"]))
    assert a.chunk < len(walk), f"chunk {a.chunk} past the end ({len(walk)} chunks)"
    x = walk[a.chunk][2]
    dump = {"input": x.clone()}
    if a.double:  # float64 everywhere (weights, input, STFT): the reference for both float32s
        model = model.double()
        x = x.double()
        w = torch.hann_window(model.stft_kwargs["win_length"]).double()  # float32 Hann values, as float64
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
    hooks.append(model.final_norm.register_forward_hook(keep("final_norm")))
    for h in range(nh):
        hooks.append(model.mask_estimators[h].register_forward_hook(keep("mask" if nh == 1 else f"mask.{h}")))
    with torch.no_grad():
        stft = torch.stft(x, **model.stft_kwargs, window=model.stft_window_fn(device="cpu"), return_complex=True)
        dump["stft"] = torch.view_as_real(stft)
        t0 = time.process_time()
        y = model(x[None])[0]
        cpu = time.process_time() - t0
    if nh == 1:
        dump["output"] = y[0]
    else:
        for h in range(nh):
            dump[f"output.{h}"] = y[h]
    # Batch dimension dropped; transformer outputs are (sequences, length, dim).
    batched = lambda k: k in ("band_split", "final_norm") or k == "mask" or k.startswith("mask.")
    out = {k: (v[0] if batched(k) else v).float().contiguous() for k, v in dump.items()}
    if a.double:
        out["input"] = dump["input"].float()
    st.save(a.out, out, metadata={"chunk": a.chunk, "input": a.inp, "cpu_s": cpu, "threads": torch.get_num_threads(),
                                  "heads": ",".join(heads), "minus": a.minus or ""})
    print(f"chunk {a.chunk} of {len(walk)}: forward {cpu:.1f} CPU s on {torch.get_num_threads()} threads; "
          f"{len(out)} tensors -> {a.out}")


def read_vocals(path):
    """Vocals to subtract: a WAV, or tools/melroformer.py separate's output (stems[0])."""
    if path.endswith(".safetensors"):
        return torch.from_numpy(st.load(path)[0]["stems"][0].astype(np.float32))
    return read_wav(path)


def input_mix(a):
    """The model's input: the WAV, or (--minus) the WAV minus vocals in float32 (r5_mega.py)."""
    mix = read_wav(a.inp)
    if a.minus:
        v = read_vocals(a.minus)
        assert v.shape == mix.shape, f"--minus: {tuple(v.shape)} vs the mix's {tuple(mix.shape)}"
        mix = mix - v
    return mix


def separate(a):
    torch.set_num_threads(a.threads)
    heads = heads_of(a)
    nh = len(heads)
    cfg, model = build(a.repo, a.ckpt, a.cfg, heads)
    dev = torch.device(a.device)
    model = model.to(dev)
    mix = input_mix(a)
    chunk = a.chunk or cfg["inference"]["chunk_size"]
    overlap = a.overlap or cfg["inference"]["num_overlap"]
    t0 = time.perf_counter()
    c0 = time.process_time()
    g, per = demix(model, mix, chunk, overlap, dev, nh)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    if not multi(a):
        stems = [g, (mix.double() - g.double()).float()]  # mix - guitar, correctly rounded
        sources = f"{a.stem},no_{a.stem}"
    else:
        rest = (mix.double() - g.double().sum(0)).float()  # input - the heads, summed in float64
        stems = list(g) + [rest]
        sources = ",".join(heads + ["rest"])
        if a.recipe:  # -n kimmega's layout: vocals, the heads (keys -> piano), other
            assert a.minus, "--recipe needs --minus (the vocals)"
            stems = [read_vocals(a.minus)] + stems
            sources = ",".join(["vocals"] + ["piano" if h == "keys" else h for h in heads] + ["other"])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
    st.save(a.out, {"stems": torch.stack(stems).contiguous()},
            metadata=dict(model=a.name, sources=sources, seconds=wall, cpu_s=cpu, chunk=chunk, heads=",".join(heads),
                          minus=a.minus or "", overlap=overlap, threads=torch.get_num_threads(), device=a.device,
                          peak_rss_mb=rss, input=a.inp))
    audio = mix.shape[1] / 44100
    print(json.dumps({"input": a.inp, "audio_s": round(audio, 2), "device": a.device, "chunk": chunk, "overlap": overlap,
                      "wall_s": round(wall, 1), "cpu_s": round(cpu, 1), "cpu_per_audio_s": round(cpu / audio, 3),
                      "chunks": len(per), "cpu_per_chunk": round(float(np.median(per)), 2),
                      "rate_long": round(float(np.median(per)) * overlap / (chunk / 44100), 3),
                      "threads": torch.get_num_threads(), "peak_rss_mb": round(rss)}))


def bench(a):
    torch.set_num_threads(a.threads)
    cfg, model = build(a.repo, a.ckpt, a.cfg, heads_of(a))
    chunk = cfg["inference"]["chunk_size"]
    x = torch.randn(1, 2, chunk, generator=torch.Generator().manual_seed(0)) * 0.1
    with torch.no_grad():
        model(x)
        best = float("inf")
        c0 = time.process_time()
        for _ in range(a.runs):
            t = time.perf_counter()
            model(x)
            best = min(best, time.perf_counter() - t)
    cpu = (time.process_time() - c0) / a.runs
    audio = chunk / 44100
    print(f"torch {torch.__version__} {a.threads} threads: wall best {best:.2f}s, CPU {cpu:.2f}s per chunk "
          f"({cpu / audio:.3f} CPU-s per audio-s of chunk)")
    print(f"RESULT model={a.name} threads={a.threads} wall_best={best:.3f} cpu={cpu:.3f} "
          f"cpu_per_audio_s={cpu / audio:.4f} impl=torch")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, repo=True):
        if repo:
            s.add_argument("repo")
        s.add_argument("ckpt")
        s.add_argument("cfg")
        s.add_argument("--stem", default="guitar")
        s.add_argument("--stems", default=None, help="comma-separated heads (several-heads format)")
        s.add_argument("--name", default=None, help="megaguitar (one head), or mega (several; default heads "
                       + ",".join(DEFAULT_HEADS) + ")")

    c = sub.add_parser("convert")
    common(c, repo=False)
    c.add_argument("--float32", action="store_true")
    s = sub.add_parser("segment")
    common(s)
    s.add_argument("inp"); s.add_argument("out")
    s.add_argument("--chunk", type=int, default=0); s.add_argument("--threads", type=int, default=4)
    s.add_argument("--double", action="store_true"); s.add_argument("--minus", default=None)
    f = sub.add_parser("separate")
    common(f)
    f.add_argument("inp"); f.add_argument("out")
    f.add_argument("--threads", type=int, default=4); f.add_argument("--device", default="cpu")
    f.add_argument("--overlap", type=int, default=0); f.add_argument("--chunk", type=int, default=0)
    f.add_argument("--minus", default=None); f.add_argument("--recipe", action="store_true")
    b = sub.add_parser("bench")
    common(b)
    b.add_argument("--threads", type=int, default=1); b.add_argument("--runs", type=int, default=2)
    a = p.parse_args()
    if a.name is None:
        a.name = "mega" if a.stems else "megaguitar"
    {"convert": convert, "segment": segment, "separate": separate, "bench": bench}[a.cmd](a)
