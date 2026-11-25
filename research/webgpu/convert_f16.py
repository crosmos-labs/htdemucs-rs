"""Smaller variants of an exported core (inputs and outputs stay float32):
  <name>_fp16n.onnx as fp16 but ReduceL2 (the RMSNorm's norm) in float32
  <name>_fp16.onnx  every op in float16 (onnxconverter_common, keep_io_types); the GPU computes in f16 (needs shader-f16)
  <name>_w16.onnx   weights stored in float16, each followed by a Cast to float32: half the download, float32 compute
  <name>_q8.onnx    weight-only int8 (onnxruntime.quantization.quantize_dynamic would quantize activations too; this
                    uses MatMul weights stored as int8 with a per-column scale + DequantizeLinear), float32 compute
    convert_f16.py <models dir> <name> [fp16] [w16] [q8]
"""
import sys, time, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto

d, name = sys.argv[1:3]; kinds = sys.argv[3:] or ['fp16', 'w16']
src = f'{d}/{name}.onnx'


def big_float_inits(g, min_elems=1024):
    return [i for i in g.initializer if i.data_type == TensorProto.FLOAT and np.prod(i.dims) >= min_elems]


for kind in kinds:
    t0 = time.time(); m = onnx.load(src); g = m.graph
    if kind in ('fp16', 'fp16n'):
        # fp16n: as fp16 but the RMSNorm's ReduceL2 stays float32 (a sum of 384 squares overflows float16's 65504:
        # plain fp16 Kim measured 14 dB SNR vs float32 / -1 dB uSDR)
        from onnxconverter_common import float16
        m = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=False,
                                             op_block_list=['ReduceL2'] if kind == 'fp16n' else None)
    elif kind in ('w16', 'q8'):
        new_nodes = []
        for init in big_float_inits(g):
            w = numpy_helper.to_array(init); nm = init.name
            if kind == 'w16':
                g.initializer.remove(init)
                g.initializer.append(numpy_helper.from_array(w.astype(np.float16), nm + '_h'))
                new_nodes.append(helper.make_node('Cast', [nm + '_h'], [nm], to=TensorProto.FLOAT))
            else:   # symmetric int8 with one scale per output column (last axis) for 2-D weights, per tensor otherwise
                if w.ndim == 2:
                    sc = np.maximum(np.abs(w).max(0), 1e-12) / 127.0; ax = 1
                else:
                    sc = np.array(max(np.abs(w).max(), 1e-12) / 127.0, np.float32); ax = None
                qw = np.clip(np.round(w / sc), -127, 127).astype(np.int8)
                g.initializer.remove(init)
                g.initializer.extend([numpy_helper.from_array(qw, nm + '_q'), numpy_helper.from_array(sc.astype(np.float32), nm + '_s')])
                kw = {'axis': ax} if ax is not None else {}
                new_nodes.append(helper.make_node('DequantizeLinear', [nm + '_q', nm + '_s'], [nm], **kw))
        nodes = list(g.node); del g.node[:]; g.node.extend(new_nodes + nodes)
    out = f'{d}/{name}_{kind}.onnx'
    onnx.save(m, out)
    import os
    print(f'{out}: {os.path.getsize(out) / 1e6:.1f} MB ({time.time() - t0:.0f} s)', flush=True)
