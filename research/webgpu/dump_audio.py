"""<name>_ref_audio.pt (from export_onnx.py) -> <name>_audio.bin (input chunk, planar (2, L) float32) and <name>_full.bin
(the PyTorch model's own output for it, (n, 2, L)), for the page's dsp=1 check.   dump_audio.py <models dir> <name>"""
import sys, torch
d, n = sys.argv[1:3]; r = torch.load(f'{d}/{n}_ref_audio.pt')
r['audio'][0].float().numpy().astype('<f4').tofile(f'{d}/{n}_audio.bin'); r['full'].float().numpy().astype('<f4').tofile(f'{d}/{n}_full.bin')
print(n, tuple(r['audio'].shape), tuple(r['full'].shape))
