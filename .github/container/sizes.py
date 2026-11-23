"""
Image size from `docker save` output, layer by layer: uncompressed (the layer tar as unpacked on
disk) and gzip -6 compressed (what a registry stores and an instance pulls). Works whatever form
the layer blobs are saved in (containerd's image store saves them gzip-compressed, the classic
store as plain tars).

    docker save IMAGE | python sizes.py IMAGE
"""
import gzip
import json
import sys
import tarfile
import zlib

name = sys.argv[1]
tf = tarfile.open(fileobj=sys.stdin.buffer, mode="r|")
blobs = {}
manifest = None
for m in tf:
    if not m.isfile():
        continue
    data = tf.extractfile(m).read()
    if m.name == "manifest.json":
        manifest = json.loads(data)
    else:
        blobs[m.name] = data
layers = []
for path in manifest[0]["Layers"]:
    raw = blobs[path]
    if raw[:2] == b"\x1f\x8b":
        raw = zlib.decompress(raw, 16 + zlib.MAX_WBITS)
    layers.append((len(raw), len(gzip.compress(raw, 6))))
u, c = sum(x for x, _ in layers), sum(y for _, y in layers)
big = ", ".join(f"{x / 1e6:.1f}/{y / 1e6:.1f}" for x, y in layers if x > 1e6)
print(f"| {name} | {len(layers)} | {u / 1e6:.1f} | {c / 1e6:.1f} | {big} |")
