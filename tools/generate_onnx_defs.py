"""Generate docs/lib/onnx-defs.js for the browser version of the app.

The web version can't import the `onnx` Python package, so this script exports what it needs:
  * ONNX_OP_INPUTS: the formal input names of every ONNX operator, per opset version
    (e.g. Conv -> X, W, B; DequantizeLinear -> x, x_scale, x_zero_point), used to label
    the rows inside each node card the way Netron does;
  * ONNX_PROTO: the onnx.proto schema text, which protobuf.js uses to decode/encode .onnx files.

Run it again after upgrading the onnx package:
    python tools/generate_onnx_defs.py
"""
import json
import os

import onnx
from onnx import defs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "lib", "onnx-defs.js")


def operator_inputs():
    """{"domain::OpType": [[since_version, [input names], variadic_flag], ...]}"""
    table = {}
    for schema in defs.get_all_schemas_with_history():
        key = f"{schema.domain or ''}::{schema.name}"
        variadic = bool(schema.inputs) and \
            schema.inputs[-1].option == defs.OpSchema.FormalParameterOption.Variadic
        table.setdefault(key, []).append(
            [schema.since_version, [i.name for i in schema.inputs], int(variadic)])

    # keep only the versions where the input list actually changed
    for key, versions in table.items():
        versions.sort()
        compact = []
        for v in versions:
            if not compact or compact[-1][1:] != v[1:]:
                compact.append(v)
        table[key] = compact
    return table


def main():
    proto_path = os.path.join(os.path.dirname(onnx.__file__), "onnx.proto")
    with open(proto_path, encoding="utf-8") as f:
        proto_text = f.read()

    ops = operator_inputs()
    js = (f"// Generated from the ONNX Python package (onnx {onnx.__version__}). Do not edit by hand.\n"
          f"window.ONNX_OP_INPUTS = {json.dumps(ops, separators=(',', ':'))};\n"
          f"window.ONNX_PROTO = {json.dumps(proto_text)};\n")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(js)
    print(f"Wrote {OUT}: {len(ops)} operators, {len(js) // 1024} KB")


if __name__ == "__main__":
    main()
