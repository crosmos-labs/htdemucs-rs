"""
Minimal safetensors writer (no extra Python dependency beyond numpy): an 8-byte little-endian
header length, a JSON header mapping each name to dtype/shape/byte offsets, then the raw
little-endian tensor bytes. https://github.com/huggingface/safetensors#format
"""
import json
import struct

import numpy as np

_DTYPES = {np.dtype("float16"): "F16", np.dtype("float32"): "F32", np.dtype("float64"): "F64",
           np.dtype("int64"): "I64", np.dtype("int32"): "I32"}


def save(path, tensors, metadata=None):
    """tensors: dict name -> numpy array (or torch tensor); written in insertion order."""
    header = {}
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().contiguous().numpy()
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in _DTYPES:
            raise ValueError(f"{name}: unsupported dtype {arr.dtype}")
        data = arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {"dtype": _DTYPES[arr.dtype], "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)  # keep the data 8-byte aligned
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for b in blobs:
            f.write(b)


def load(path):
    """Returns (dict name -> numpy array, metadata dict)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        data = f.read()
    meta = header.pop("__metadata__", {})
    inv = {v: k for k, v in _DTYPES.items()}
    out = {}
    for name, h in header.items():
        a, b = h["data_offsets"]
        out[name] = np.frombuffer(data[a:b], dtype=inv[h["dtype"]]).reshape(h["shape"])
    return out, meta
