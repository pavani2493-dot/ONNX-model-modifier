"""Build the small public demo model used by the app (examples/demo_resnet.onnx).

A tiny ResNet-style CNN written directly with the ONNX graph API:
Conv -> BatchNorm -> Relu -> MaxPool -> 2 residual blocks -> GlobalAveragePool -> Gemm -> Softmax.
Weights are random (seeded), so the model contains nothing private.

Usage:
    python tools/make_demo_model.py                       # writes examples/demo_resnet.onnx
    python tools/make_demo_model.py path/to/output.onnx
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto
from onnx import helper as h
from onnx import numpy_helper as nh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GraphBuilder:
    """Collects nodes and initializers while the network is described layer by layer."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.nodes, self.inits = [], []

    def weight(self, name, shape, value=None):
        arr = value if value is not None else (self.rng.standard_normal(shape) * 0.1)
        self.inits.append(nh.from_array(arr.astype(np.float32), name))
        return name

    def conv(self, x, out, cin, cout, k, stride=1):
        pad = k // 2
        self.nodes.append(h.make_node(
            "Conv", [x, self.weight(f"{out}.weight", (cout, cin, k, k)), self.weight(f"{out}.bias", (cout,))],
            [out], name=out, kernel_shape=[k, k], pads=[pad] * 4, strides=[stride, stride]))
        return out

    def batchnorm(self, x, out, c):
        ins = [x,
               self.weight(f"{out}.scale", (c,)), self.weight(f"{out}.B", (c,)),
               self.weight(f"{out}.mean", (c,)), self.weight(f"{out}.var", (c,), np.ones(c))]
        self.nodes.append(h.make_node("BatchNormalization", ins, [out], name=out, epsilon=1e-5))
        return out

    def op(self, op_type, inputs, out, **attrs):
        self.nodes.append(h.make_node(op_type, inputs, [out], name=out, **attrs))
        return out


def build_model():
    b = GraphBuilder(seed=0)
    x = b.conv("input", "conv1", 3, 16, 3, stride=2)
    x = b.batchnorm(x, "bn1", 16)
    x = b.op("Relu", [x], "relu1")
    x = b.op("MaxPool", [x], "pool1", kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])

    for i in (1, 2):  # two residual blocks
        y = b.conv(x, f"block{i}.conv1", 16, 16, 3)
        y = b.batchnorm(y, f"block{i}.bn1", 16)
        y = b.op("Relu", [y], f"block{i}.relu1")
        y = b.conv(y, f"block{i}.conv2", 16, 16, 3)
        y = b.batchnorm(y, f"block{i}.bn2", 16)
        y = b.op("Add", [y, x], f"block{i}.add")
        x = b.op("Relu", [y], f"block{i}.out")

    x = b.op("GlobalAveragePool", [x], "gap")
    x = b.op("Flatten", [x], "flat", axis=1)
    x = b.op("Gemm", [x, b.weight("fc.weight", (10, 16)), b.weight("fc.bias", (10,))], "logits", transB=1)
    x = b.op("Dropout", [x], "drop")
    b.op("Softmax", [x], "output", axis=1)

    graph = h.make_graph(
        b.nodes, "demo_cnn",
        [h.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])],
        [h.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
        b.inits)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], producer_name="onnx-modifier-demo")
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "examples", "demo_resnet.onnx")
    model = build_model()
    onnx.save(model, out)
    print(f"Saved {out}: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers")
