"""
PyTorch reference for `htdemucs-rs separate -n chain`: round 3's `k+chain_avg`
(research/round3/r3_sep.py + r3_score.py), on CPU in float32.

  vocals        Kim Mel-RoFormer (upstream demix_track, as tools/melroformer.py separate)
  instrumental  mix - vocals
  Demucs        htdemucs and htdemucs_6s on the instrumental, as r3_sep.py's run(): normalised by
                the instrumental's mono mean / std, apply_model(shifts=0, split=True, overlap=0.25)
  4-stem view   vocals = Kim; drums, bass = mean of the two; other = mean(htdemucs other,
                6s other + guitar + piano)
  6-stem set    vocals, drums, bass as above; guitar, piano = 6s'; other = 4-stem other - guitar - piano

Writes <out>.4.safetensors and <out>.6.safetensors ({"stems": (S, 2, L)}, sources in each model's
order, as src/bin/compare.rs reads) and prints CPU time per stage.

    PY="uv run --quiet --python 3.11.17 --with-requirements research/round1/kim_env.txt \\
        --with-requirements research/round3/r3_env.txt python"
    $PY tools/chain_ref.py <kim repo> <MelBandRoformer.ckpt> in.wav out [--threads 4]
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st  # noqa: E402
from melroformer import load, read_wav  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("repo"); ap.add_argument("ckpt"); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.set_grad_enabled(False)

mix = read_wav(a.inp)
cpu = {}
c0 = time.process_time()
cfg, kim = load(a.repo, a.ckpt)
from utils import demix_track  # noqa: E402  (the repo is on the path now)
so, sys.stdout = sys.stdout, open(os.devnull, "w")
try:
    res, _ = demix_track(cfg, kim, mix, "cpu")
finally:
    sys.stdout = so
voc = torch.from_numpy(res["vocals"]).float()
del kim
cpu["kim"] = time.process_time() - c0
inst = mix - voc

from demucs.apply import apply_model  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402


def run(name, x):
    model = get_model(name)
    model.eval()
    ref = x.mean(0)
    mu, sd = ref.mean(), ref.std()
    y = apply_model(model, ((x - mu) / sd)[None], shifts=0, split=True, overlap=0.25, device="cpu")[0]
    return {s: t for s, t in zip(model.sources, y * sd + mu)}, list(model.sources)


c0 = time.process_time()
hd, src4 = run("htdemucs", inst)
cpu["htdemucs"] = time.process_time() - c0
c0 = time.process_time()
six, src6 = run("htdemucs_6s", inst)
cpu["htdemucs_6s"] = time.process_time() - c0

v4 = {"vocals": voc, "drums": (hd["drums"] + six["drums"]) / 2, "bass": (hd["bass"] + six["bass"]) / 2,
      "other": (hd["other"] + (six["other"] + six["guitar"] + six["piano"])) / 2}
v6 = dict(v4, guitar=six["guitar"], piano=six["piano"], other=v4["other"] - six["guitar"] - six["piano"])
resid = mix - (v4["vocals"] + v4["drums"] + v4["bass"] + v4["other"])
resid_db = float(10 * torch.log10((resid.double() ** 2).sum() / (mix.double() ** 2).sum()))
meta = dict(model="chain", input=a.inp, threads=a.threads, cpu=json.dumps({k: round(v, 1) for k, v in cpu.items()}), resid_dB=resid_db)
st.save(a.out + ".4.safetensors", {"stems": torch.stack([v4[s] for s in src4]).contiguous()}, metadata=dict(meta, sources=",".join(src4)))
st.save(a.out + ".6.safetensors", {"stems": torch.stack([v6[s] for s in src6]).contiguous()}, metadata=dict(meta, sources=",".join(src6)))
audio = mix.shape[1] / 44100
print(json.dumps({"input": a.inp, "audio_s": round(audio, 2), "cpu_s": {k: round(v, 1) for k, v in cpu.items()},
                  "cpu_per_audio_s": round(sum(cpu.values()) / audio, 3), "resid_dB": round(resid_db, 2), "threads": a.threads}))
