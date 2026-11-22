"""
Reference harness: run ONE HTDemucs segment on CPU float32 exactly as `demucs.separate` +
`demucs.apply.apply_model` would feed it, and dump the input, the intermediate activations and the
output to a safetensors file (the ground truth for the Rust parity test).

    uv run --quiet --python 3.11.17 --with-requirements <fretboard>/scripts/requirements-demucs.txt \
        python tools/harness.py --wav ref/aerials.wav --model htdemucs --chunk 10 --out ref/aerials-htdemucs-c10.safetensors

How the segment is built (demucs 4.0.1, `separate.main` -> `apply_model`):
  * the track is decoded by `load_track` (ffmpeg -> f32le at 44.1 kHz, stereo);
  * separate.py normalises the WHOLE track before splitting: ref = wav.mean(0) (mono),
    wav = (wav - ref.mean()) / ref.std()   (torch std: unbiased, N-1);
  * the pretrained model is a BagOfModels of one HTDemucs, so apply_model recurses into the
    sub-model (bag weights are all 1.0: no numerical effect);
  * with shifts > 0 the mix is zero-padded by max_shift = 22050 on both sides and viewed from a
    random offset (`--shift-offset` emulates one such pass; without it we take the shifts=0 path);
  * split: segment_length = int(44100 * segment) = 343980 for segment 39/5 s, stride =
    int(0.75 * segment_length) = 257985, chunks at offsets range(0, length, stride);
  * each chunk is padded to the model's training length (valid_length = 343980) by
    TensorChunk.padded, which CENTERS the chunk and reads real samples from the underlying tensor
    on both sides where they exist (zeros only past the tensor's ends) - so the short last chunk is
    preceded by audio from the previous chunk, not zeros;
  * the model output (343980 samples) is center-trimmed back to the chunk length.
We use demucs' own TensorChunk for all of this and check the result against apply_model itself.
"""
import argparse
import re
import time

import torch
import torch.nn.functional as F

from demucs.apply import TensorChunk, apply_model
from demucs.pretrained import get_model
from demucs.separate import load_track
from demucs.transformer import create_2d_sin_embedding, create_sin_embedding
from demucs.utils import center_trim

import st

# Module outputs to dump (by module name in the HTDemucs model).
KEEP = [
    r"(encoder|tencoder|decoder|tdecoder)\.\d+",
    r"(encoder|tencoder)\.\d+\.(conv|rewrite|dconv|dconv\.layers\.\d)",
    r"(decoder|tdecoder)\.\d+\.(conv_tr|rewrite|dconv|dconv\.layers\.\d)",
    r"channel_(up|down)sampler(_t)?",
    r"crosstransformer(\.(norm_in|norm_in_t))?",
    r"crosstransformer\.layers(_t)?\.\d+(\.(self_attn|cross_attn|linear2|norm_out))?",
]
KEEP_RE = re.compile("^(" + "|".join(KEEP) + ")$")


def first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def real(t):
    return torch.view_as_real(t) if t.is_complex() else t


def build_segment(model, wav_path, chunk, shift_offset):
    wav = load_track(wav_path, model.audio_channels, model.samplerate)
    # demucs/separate.py, verbatim.
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()
    mix = wav[None]
    length = mix.shape[-1]

    if shift_offset is not None:
        # apply_model's shift branch: pad by max_shift each side, then view from `offset`.
        max_shift = int(0.5 * model.samplerate)
        assert 0 <= shift_offset <= max_shift
        padded_mix = TensorChunk(mix).padded(length + 2 * max_shift)
        src = TensorChunk(padded_mix, shift_offset, length + max_shift - shift_offset)
    else:
        src = TensorChunk(mix)

    segment_length = int(model.samplerate * model.segment)
    stride = int((1 - 0.25) * segment_length)
    offsets = list(range(0, src.length, stride))
    if chunk < 0:
        chunk += len(offsets)
    offset = offsets[chunk]
    piece = TensorChunk(src, offset, segment_length)
    valid_length = model.valid_length(piece.length)
    padded = piece.padded(valid_length)
    info = dict(track_length=length, n_chunks=len(offsets), chunk=chunk, offset=offset,
                chunk_length=piece.length, valid_length=valid_length,
                ref_mean=float(ref.mean()), ref_std=float(ref.std()),
                shift_offset=-1 if shift_offset is None else shift_offset)
    return piece, padded, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--model", required=True, choices=["htdemucs", "htdemucs_6s"])
    p.add_argument("--chunk", type=int, default=1, help="chunk index (negative: from the end)")
    p.add_argument("--shift-offset", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.set_grad_enabled(False)
    bag = get_model(args.model)
    model = bag.models[0]
    model.cpu().eval()

    piece, padded, info = build_segment(model, args.wav, args.chunk, args.shift_offset)
    print("segment", info)

    # 1. Plain forward (no hooks), timed; and demucs' own apply_model on the same chunk.
    out = model(padded)
    t0 = time.perf_counter()
    out = model(padded)
    info["torch_seconds"] = time.perf_counter() - t0
    info["torch_threads"] = torch.get_num_threads()
    ref_apply = apply_model(model, piece, shifts=0, split=False)
    assert torch.equal(center_trim(out, piece.length), ref_apply), "harness != apply_model"

    # 2. Hooked forward: module outputs plus the non-module steps of HTDemucs.forward.
    dump = {"input": padded}
    hooks = []
    for name, mod in model.named_modules():
        if KEEP_RE.match(name):
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, name=name: dump.__setitem__(name, real(first(o)).clone())))
    pre = {
        "encoder.0": "x_norm",        # normalised CaC spectrogram (freq branch input)
        "tencoder.0": "xt_norm",      # normalised waveform (time branch input)
        "encoder.1": "encoder.0+emb", # after the frequency embedding
    }
    for name, key in pre.items():
        hooks.append(model.get_submodule(name).register_forward_pre_hook(
            lambda m, i, key=key: dump.__setitem__(key, i[0].clone())))

    spec, mag, mask, ispec = model._spec, model._magnitude, model._mask, model._ispec

    def w_spec(x):
        z = spec(x)
        dump["spec"] = real(z).clone()
        return z

    def w_mag(z):
        m = mag(z)
        dump["mag"] = m.clone()
        return m

    def w_mask(z, m):
        dump["mask_in"] = m.clone()  # decoder output * std + mean, (B, S, 2*C, F, T)
        zo = mask(z, m)
        dump["zout"] = real(zo).clone()
        return zo

    def w_ispec(z, length=None, scale=0):
        x = ispec(z, length, scale)
        dump["ispec"] = x.clone()
        return x

    model._spec, model._magnitude, model._mask, model._ispec = w_spec, w_mag, w_mask, w_ispec
    out2 = model(padded)
    for h in hooks:
        h.remove()
    del model._spec, model._magnitude, model._mask, model._ispec
    assert torch.equal(out, out2), "hooked forward differs"
    dump["output"] = out

    # Positional embeddings of the cross-transformer, from demucs' own functions with the
    # shapes the forward used (B, C, Fr, T1 of the upsampled freq input; T2 of the time input).
    Fr, T1 = dump["encoder.%d" % (model.depth - 1)].shape[-2:]
    _, C, T2 = dump["channel_upsampler_t"].shape
    dump["pos_emb_2d"] = create_2d_sin_embedding(C, Fr, T1, "cpu", model.crosstransformer.max_period)
    dump["pos_emb_1d"] = create_sin_embedding(T2, C, shift=0, device="cpu",
                                              max_period=model.crosstransformer.max_period)

    meta = dict(info, model=args.model, wav=args.wav, sources=",".join(model.sources),
                torch=torch.__version__)
    st.save(args.out, {k: v.float() if v.dtype == torch.float64 else v for k, v in dump.items()},
            metadata=meta)
    total = sum(v.numel() for v in dump.values()) * 4 / 1e6
    print(f"wrote {args.out}: {len(dump)} tensors, {total:.0f} MB; torch forward "
          f"{info['torch_seconds']:.2f}s on {info['torch_threads']} threads")


if __name__ == "__main__":
    main()
