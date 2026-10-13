"""ONNX Model Modifier - Netron-style viewer + simple editor (Gradio).

Keep `dagre.min.js` in the same folder as this script so the graph works offline.
If it is missing, the layout library is loaded from the jsDelivr CDN instead.
"""
import html
import json
import os
import tempfile

import gradio as gr
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

HERE = os.path.dirname(os.path.abspath(__file__))
DAGRE_CDN = "https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"

# Header colours taken from Netron's light theme
CATEGORY_COLORS = {
    "layer": "#335588", "activation": "#702921", "pool": "#335533",
    "normalization": "#335544", "dropout": "#454770", "shape": "#6c4f47",
    "tensor": "#59423b", "transform": "#335544", "data": "#555555",
}
OP_CATEGORY = {
    **dict.fromkeys(["Conv", "ConvTranspose", "Gemm", "MatMul", "ConvInteger", "QLinearConv",
                     "QLinearMatMul", "MatMulInteger", "LSTM", "GRU", "RNN", "Einsum"], "layer"),
    **dict.fromkeys(["Relu", "Sigmoid", "Tanh", "LeakyRelu", "PRelu", "Elu", "Selu", "Softmax",
                     "LogSoftmax", "HardSigmoid", "HardSwish", "Softplus", "Softsign", "Clip",
                     "Gelu", "Mish", "ThresholdedRelu", "Celu"], "activation"),
    **dict.fromkeys(["MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "LpPool",
                     "MaxRoiPool", "GlobalLpPool"], "pool"),
    **dict.fromkeys(["BatchNormalization", "InstanceNormalization", "LayerNormalization",
                     "LRN", "GroupNormalization", "LpNormalization"], "normalization"),
    **dict.fromkeys(["Dropout"], "dropout"),
    **dict.fromkeys(["Reshape", "Flatten", "Squeeze", "Unsqueeze", "Expand", "Tile"], "shape"),
    **dict.fromkeys(["Concat", "Split", "Slice", "Gather", "GatherElements", "GatherND",
                     "Pad", "Transpose"], "tensor"),
}


# ================================================================ model helpers
def load_onnx_model(file_obj):
    """Gradio 4+ gives a path string; older versions give a tempfile object."""
    path = file_obj if isinstance(file_obj, str) else file_obj.name
    model = onnx.load(path)
    ensure_unique_node_names(model)
    return model


def ensure_unique_node_names(model):
    """Many exported models have empty/duplicate node names, which breaks the graph."""
    seen = set()
    for i, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}_{i}"
        while name in seen:
            name = f"{name}_{i}"
        node.name = name
        seen.add(name)


def _dims(shape_proto):
    return [d.dim_param if d.dim_param else (d.dim_value if d.HasField("dim_value") else "?")
            for d in shape_proto.dim]


def tensor_shapes(model, infer=False):
    """Map tensor name -> list of dims. infer=True runs shape inference on a copy."""
    if infer:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass
    g = model.graph
    shapes = {}
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.type.tensor_type.HasField("shape"):
            shapes[vi.name] = _dims(vi.type.tensor_type.shape)
    for init in g.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def constant_names(model):
    return {init.name for init in model.graph.initializer}


def fmt_shape(dims):
    return "×".join(str(d) for d in dims)


def fmt_value(arr, limit=10):
    """Format a scalar the way Netron does: full float32 precision, truncated with '…'."""
    v = arr.item()
    if isinstance(v, float):
        s = repr(float(np.float32(v))) if arr.dtype == np.float32 else repr(v)
    else:
        s = str(v)
    return s if len(s) <= limit else s[:limit] + "…"


def schema_input_names(op_type, domain, opset, count):
    """Formal input names from the ONNX op schema (x, x_scale, W, B, ...)."""
    try:
        schema = onnx.defs.get_schema(op_type, opset, domain)
        formal = [i.name for i in schema.inputs]
        variadic = schema.inputs and schema.inputs[-1].option == \
            onnx.defs.OpSchema.FormalParameterOption.Variadic
    except Exception:
        return [None] * count
    names = []
    for i in range(count):
        if i < len(formal):
            names.append(formal[i])
        elif variadic and formal:
            names.append(formal[-1])
        else:
            names.append(None)
    return names


def attr_to_str(a):
    val = onnx.helper.get_attribute_value(a)
    if isinstance(val, bytes):
        return val.decode(errors="replace")
    if isinstance(val, onnx.TensorProto):
        return f"tensor ⟨{fmt_shape(val.dims)}⟩"
    if isinstance(val, onnx.GraphProto):
        return f"graph '{val.name}'"
    if isinstance(val, (list, tuple)):
        return ", ".join(str(x) for x in val) if len(val) < 20 else f"[{len(val)} values]"
    return str(val)


def node_summary(node, shapes, consts):
    lines = [f"{node.op_type}  ({node.name})", "", "Inputs:"]
    for t in node.input:
        if t:
            tag = " [initializer]" if t in consts else ""
            lines.append(f"  {t}  {fmt_shape(shapes.get(t, []))}{tag}")
    lines.append("Outputs:")
    for t in node.output:
        lines.append(f"  {t}  {fmt_shape(shapes.get(t, []))}")
    if node.attribute:
        lines.append("Attributes:")
        for a in node.attribute:
            lines.append(f"  {a.name} = {attr_to_str(a)[:80]}")
    return "\n".join(lines)


# ================================================================ graph data
def build_graph_data(model):
    """Turn the ONNX graph into Netron-style cards: initializer inputs are shown as
    rows inside the node; tensors produced by other nodes become edges."""
    g = model.graph
    raw_shapes = tensor_shapes(model)            # what the file itself declares
    full_shapes = tensor_shapes(model, infer=True)  # for the properties panel
    inits = {i.name: i for i in g.initializer}
    opsets = {o.domain: o.version for o in model.opset_import}

    nodes, edges, producer = [], [], {}

    for inp in g.input:
        if inp.name in inits:
            continue
        nid = f"in::{inp.name}"
        nodes.append({"id": nid, "kind": "io", "label": inp.name,
                      "props": {"type": "input", "name": inp.name,
                                "shape": fmt_shape(full_shapes.get(inp.name, []))}})
        producer[inp.name] = nid

    for node in g.node:
        for t in node.output:
            producer[t] = node.name

    for node in g.node:
        formal = schema_input_names(node.op_type, node.domain or "",
                                    opsets.get(node.domain or "", 1), len(node.input))
        rows, prop_inputs = [], []
        for t, fname in zip(node.input, formal):
            label = fname or t
            if not t:
                continue
            prop_inputs.append({"name": label, "tensor": t,
                                "shape": fmt_shape(full_shapes.get(t, []))})
            if t in inits:
                arr = numpy_helper.to_array(inits[t])
                if arr.ndim == 0:
                    rows.append({"name": label, "value": fmt_value(arr)})
                else:
                    rows.append({"name": label, "shape": fmt_shape(arr.shape), "bold": True})
            elif t in producer:
                dims = raw_shapes.get(t)
                # Netron only labels edges whose shape is fully known
                lbl = fmt_shape(dims) if dims and all(isinstance(d, int) for d in dims) else ""
                edges.append({"from": producer[t], "to": node.name, "label": lbl})

        cat = OP_CATEGORY.get(node.op_type)
        nodes.append({
            "id": node.name, "kind": "op", "label": node.op_type, "rows": rows,
            "color": CATEGORY_COLORS.get(cat, "#333333"),
            "props": {
                "type": node.op_type, "name": node.name,
                "domain": node.domain or "ai.onnx",
                "attributes": [{"name": a.name, "value": attr_to_str(a)} for a in node.attribute],
                "inputs": prop_inputs,
                "outputs": [{"tensor": t, "shape": fmt_shape(full_shapes.get(t, []))}
                            for t in node.output],
            },
        })

    for out in g.output:
        nid = f"out::{out.name}"
        nodes.append({"id": nid, "kind": "io", "label": out.name,
                      "props": {"type": "output", "name": out.name,
                                "shape": fmt_shape(full_shapes.get(out.name, []))}})
        if out.name in producer:
            dims = raw_shapes.get(out.name)
            lbl = fmt_shape(dims) if dims and all(isinstance(d, int) for d in dims) else ""
            edges.append({"from": producer[out.name], "to": nid, "label": lbl})

    return {"nodes": nodes, "edges": edges}


# ================================================================ HTML renderer
VIEWER_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
__DAGRE__
<style>
  html, body { margin:0; height:100%; font-family:-apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background:#fff; }
  #wrap { position:absolute; inset:0; overflow:auto; }
  #svg { display:block; margin:0 auto; }
  #side { position:absolute; top:0; right:0; bottom:0; width:340px; background:#fafafa; border-left:1px solid #ddd;
          overflow:auto; font-size:12px; padding:12px 14px; box-sizing:border-box; display:none; }
  #side h3 { margin:2px 0 10px; font-size:14px; }
  #side .sec { margin:12px 0 4px; font-weight:600; color:#444; text-transform:uppercase; font-size:11px; letter-spacing:.04em; }
  #side .kv { display:flex; gap:8px; padding:3px 0; border-bottom:1px solid #eee; }
  #side .k { width:110px; color:#555; flex:none; word-break:break-all; }
  #side .v { font-family:Consolas, Menlo, monospace; word-break:break-all; }
  #side .close { position:absolute; top:6px; right:10px; cursor:pointer; font-size:18px; color:#888; }
  #tools { position:absolute; left:10px; bottom:10px; display:flex; gap:6px; }
  #tools button { width:30px; height:30px; border:1px solid #bbb; border-radius:4px; background:#fff; cursor:pointer; font-size:15px; }
  #msg { position:absolute; top:40%; width:100%; text-align:center; color:#888; }
  .node { cursor:pointer; }
  .node .body { fill:#fff; stroke:#333; stroke-width:1; }
  .node .io { fill:#eee; stroke:#333; stroke-width:1; }
  .node.sel .body, .node.sel .io, .node.sel .hdr { stroke:#e05a00; stroke-width:2.5; }
  .node text { font-size:11px; dominant-baseline:central; }
  .edge path { fill:none; stroke:#000; stroke-width:1; }
  .edge text { font-size:10px; fill:#000; text-anchor:middle; dominant-baseline:central; }
</style></head>
<body>
<div id="wrap"><svg id="svg" xmlns="http://www.w3.org/2000/svg"></svg></div>
<div id="tools"><button id="zin" title="Zoom in (Ctrl+wheel)">+</button><button id="zout" title="Zoom out">−</button><button id="zfit" title="Fit">⤢</button><button id="z1" title="100%">1:1</button></div>
<div id="side"><span class="close" id="close">×</span><div id="sidec"></div></div>
<div id="msg"></div>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const FONT = '11px -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
const BOLD = 'bold ' + FONT;
const ctx = document.createElement("canvas").getContext("2d");
function tw(t, bold){ ctx.font = bold ? BOLD : FONT; return ctx.measureText(t).width; }
function el(tag, attrs, parent){ const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

if (typeof dagre === "undefined") {
  document.getElementById("msg").textContent = "Could not load dagre.min.js (put it next to ONNX_modifier.py).";
  throw new Error("dagre missing");
}

// ---- measure nodes (same geometry as Netron: header + one line per initializer)
const HDR = 21, ROW = 15, PADX = 8, PADY = 5;
for (const n of DATA.nodes) {
  if (n.kind === "io") { n.w = Math.max(tw(n.label) + 2*PADX + 4, 40); n.h = 26; continue; }
  let w = tw(n.label) + 2*PADX + 2;
  for (const r of n.rows) {
    r.text1 = r.name; r.text2 = r.shape !== undefined ? " ⟨" + r.shape + "⟩" : " = " + r.value;
    w = Math.max(w, tw(r.text1, r.bold) + tw(r.text2) + 2*PADX);
  }
  n.w = Math.ceil(w);
  n.h = HDR + (n.rows.length ? n.rows.length * ROW + 2*PADY : 0);
}

// ---- layout
const g = new dagre.graphlib.Graph({ multigraph:true });
g.setGraph({ rankdir:"TB", nodesep:20, ranksep:20, edgesep:10, marginx:40, marginy:40 });
g.setDefaultEdgeLabel(() => ({}));
const byId = {};
for (const n of DATA.nodes) { byId[n.id] = n; g.setNode(n.id, { width:n.w, height:n.h }); }
DATA.edges.forEach((e, i) => {
  const lw = e.label ? tw(e.label) + 6 : 0;
  g.setEdge(e.from, e.to, { width:lw, height:e.label ? 14 : 0, labelpos:"c", labeloffset:0 }, "e" + i);
});
dagre.layout(g);

// ---- d3.curveBasis equivalent
function basis(pts){
  if (pts.length < 3) return "M" + pts.map(p => p.x + "," + p.y).join("L");
  let d = "", x0, y0, x1, y1, state = 0;
  const bez = (x, y) => { d += "C" + (2*x0+x1)/3 + "," + (2*y0+y1)/3 + " " + (x0+2*x1)/3 + "," + (y0+2*y1)/3 + " " + (x0+4*x1+x)/6 + "," + (y0+4*y1+y)/6; };
  for (const p of pts) {
    const x = p.x, y = p.y;
    if (state === 0) { state = 1; d += "M" + x + "," + y; }
    else if (state === 1) { state = 2; }
    else { if (state === 2) { state = 3; d += "L" + (5*x0+x1)/6 + "," + (5*y0+y1)/6; } bez(x, y); }
    x0 = x1; x1 = x; y0 = y1; y1 = y;
  }
  bez(x1, y1); d += "L" + x1 + "," + y1;
  return d;
}

// ---- draw
const svg = document.getElementById("svg");
const gw = g.graph().width, gh = g.graph().height;
const defs = el("defs", {}, svg);
const mk = el("marker", { id:"arrow", viewBox:"0 0 10 10", refX:9, refY:5, markerWidth:7, markerHeight:7, orient:"auto" }, defs);
el("path", { d:"M0,0 L10,5 L0,10 L3,5 z", fill:"#000" }, mk);
const root = el("g", {}, svg);
const edgeLayer = el("g", {}, root), nodeLayer = el("g", {}, root);

g.edges().forEach(eo => {
  const e = g.edge(eo), lbl = DATA.edges[+eo.name.slice(1)].label;
  const grp = el("g", { class:"edge" }, edgeLayer);
  el("path", { d:basis(e.points), "marker-end":"url(#arrow)" }, grp);
  if (lbl) {
    el("rect", { x:e.x - e.width/2, y:e.y - 7, width:e.width, height:14, fill:"#fff", opacity:0.85 }, grp);
    el("text", { x:e.x, y:e.y }, grp).textContent = lbl;
  }
});

const nodeEls = {};
for (const n of DATA.nodes) {
  const p = g.node(n.id), x = p.x - n.w/2, y = p.y - n.h/2;
  const grp = el("g", { class:"node", transform:`translate(${x},${y})` }, nodeLayer);
  nodeEls[n.id] = grp;
  if (n.kind === "io") {
    el("rect", { class:"io", width:n.w, height:n.h, rx:6 }, grp);
    el("text", { x:n.w/2, y:n.h/2, "text-anchor":"middle" }, grp).textContent = n.label;
  } else if (!n.rows.length) {
    el("rect", { class:"hdr", width:n.w, height:HDR, rx:5, fill:n.color, stroke:"#333" }, grp);
    el("text", { x:n.w/2, y:HDR/2, "text-anchor":"middle", fill:"#fff" }, grp).textContent = n.label;
  } else {
    el("rect", { class:"body", width:n.w, height:n.h, rx:5 }, grp);
    el("path", { class:"hdr", d:`M0,${HDR} V5 Q0,0 5,0 H${n.w-5} Q${n.w},0 ${n.w},5 V${HDR} Z`, fill:n.color, stroke:"#333" }, grp);
    el("text", { x:n.w/2, y:HDR/2, "text-anchor":"middle", fill:"#fff" }, grp).textContent = n.label;
    n.rows.forEach((r, i) => {
      const t = el("text", { x:PADX, y:HDR + PADY + ROW*i + ROW/2 }, grp);
      const a = el("tspan", { "font-weight": r.bold ? "bold" : "normal" }, t); a.textContent = r.text1;
      const b = el("tspan", {}, t); b.textContent = r.text2;
    });
  }
  grp.addEventListener("click", ev => { ev.stopPropagation(); select(n); });
}

// ---- properties panel (like Netron's sidebar)
const side = document.getElementById("side"), sidec = document.getElementById("sidec");
let selected = null;
function kv(k, v){ return `<div class="kv"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`; }
function select(n){
  if (selected) nodeEls[selected].classList.remove("sel");
  selected = n.id; nodeEls[n.id].classList.add("sel");
  const p = n.props;
  let h = `<h3>${esc(p.type)}</h3>` + kv("name", p.name);
  if (n.kind === "io") h += kv("shape", p.shape || "?");
  else {
    h += kv("domain", p.domain);
    if (p.attributes.length) { h += `<div class="sec">Attributes</div>`; p.attributes.forEach(a => h += kv(a.name, a.value)); }
    h += `<div class="sec">Inputs</div>`; p.inputs.forEach(i => h += kv(i.name, i.tensor + (i.shape ? "  ⟨" + i.shape + "⟩" : "")));
    h += `<div class="sec">Outputs</div>`; p.outputs.forEach(o => h += kv("output", o.tensor + (o.shape ? "  ⟨" + o.shape + "⟩" : "")));
  }
  sidec.innerHTML = h; side.style.display = "block";
}
document.getElementById("close").onclick = () => { side.style.display = "none"; if (selected) nodeEls[selected].classList.remove("sel"); selected = null; };
svg.addEventListener("click", () => document.getElementById("close").onclick());

// ---- zoom / pan (scrollbars like Netron, Ctrl+wheel to zoom, drag to pan)
const wrap = document.getElementById("wrap");
let scale = 1;
function setScale(s, cx, cy){
  s = Math.max(0.05, Math.min(4, s));
  cx = cx ?? wrap.clientWidth/2; cy = cy ?? wrap.clientHeight/2;
  const gx = (wrap.scrollLeft + cx) / scale, gy = (wrap.scrollTop + cy) / scale;
  scale = s;
  svg.setAttribute("width", gw*scale); svg.setAttribute("height", gh*scale);
  svg.setAttribute("viewBox", `0 0 ${gw} ${gh}`);
  wrap.scrollLeft = gx*scale - cx; wrap.scrollTop = gy*scale - cy;
}
wrap.addEventListener("wheel", ev => { if (ev.ctrlKey || ev.metaKey) { ev.preventDefault(); const r = wrap.getBoundingClientRect(); setScale(scale * (ev.deltaY < 0 ? 1.1 : 1/1.1), ev.clientX - r.left, ev.clientY - r.top); } }, { passive:false });
let drag = null;
wrap.addEventListener("mousedown", ev => { if (ev.button === 0 && !ev.target.closest(".node")) drag = { x:ev.clientX, y:ev.clientY, l:wrap.scrollLeft, t:wrap.scrollTop, moved:false }; });
window.addEventListener("mousemove", ev => { if (drag) { wrap.scrollLeft = drag.l - (ev.clientX - drag.x); wrap.scrollTop = drag.t - (ev.clientY - drag.y); wrap.style.cursor = "grabbing"; } });
window.addEventListener("mouseup", () => { drag = null; wrap.style.cursor = ""; });
document.getElementById("zin").onclick = () => setScale(scale * 1.25);
document.getElementById("zout").onclick = () => setScale(scale / 1.25);
document.getElementById("z1").onclick = () => setScale(1);
document.getElementById("zfit").onclick = () => setScale(Math.min(wrap.clientWidth/gw, wrap.clientHeight/gh));

// start at 100% with the graph input at the top, like Netron
setScale(1);
const first = DATA.nodes.find(n => n.kind === "io");
if (first) { const p = g.node(first.id); wrap.scrollLeft = p.x - wrap.clientWidth/2; wrap.scrollTop = Math.max(0, p.y - 60); }
</script></body></html>"""


def _dagre_tag():
    local = os.path.join(HERE, "dagre.min.js")
    if os.path.exists(local):
        with open(local, "r", encoding="utf-8") as f:
            return "<script>" + f.read() + "</script>"
    return f'<script src="{DAGRE_CDN}"></script>'


def visualize_onnx(model):
    data = json.dumps(build_graph_data(model)).replace("</", "<\\/")
    page = VIEWER_TEMPLATE.replace("__DAGRE__", _dagre_tag()).replace("__DATA__", data)
    # gr.HTML does not execute <script> tags, so render the page inside an iframe.
    return (f'<iframe srcdoc="{html.escape(page, quote=True)}" '
            f'style="width:100%;height:800px;border:1px solid #ddd;border-radius:6px;"></iframe>')


def node_choices(model):
    return [n.name for n in model.graph.node]


# ================================================================ edits
def rename_node(model, old_name, new_name):
    if new_name in node_choices(model):
        raise ValueError(f"A node named '{new_name}' already exists.")
    for node in model.graph.node:
        if node.name == old_name:
            node.name = new_name
            return model
    raise ValueError(f"Node '{old_name}' not found.")


def delete_node(model, node_name):
    """Remove a node and reconnect its consumers to its main input (bypass),
    so the model keeps valid connections instead of dangling tensors."""
    g = model.graph
    target = next((n for n in g.node if n.name == node_name), None)
    if target is None:
        raise ValueError(f"Node '{node_name}' not found.")

    consts = constant_names(model)
    data_inputs = [t for t in target.input if t and t not in consts]
    if len(target.output) != 1 or len(data_inputs) != 1:
        raise ValueError(
            f"'{node_name}' ({target.op_type}) has {len(data_inputs)} data input(s) and "
            f"{len(target.output)} output(s). Only single-input/single-output nodes "
            f"(Relu, Dropout, Identity, Q/DQ, ...) can be bypassed safely.")

    old_out, new_src = target.output[0], data_inputs[0]
    for n in g.node:
        for i, t in enumerate(n.input):
            if t == old_out:
                n.input[i] = new_src
    for o in g.output:
        if o.name == old_out:  # deleting the last node: graph output becomes its input
            o.name = new_src
    g.node.remove(target)
    for vi in list(g.value_info):
        if vi.name == old_out:
            g.value_info.remove(vi)
    return model


# ================================================================ UI
with gr.Blocks(title="ONNX Modifier") as app:
    gr.Markdown("# ONNX Model Modifier")
    model_state = gr.State(None)  # the working copy - edits accumulate here

    with gr.Row():
        model_input = gr.File(label="Upload ONNX model", file_types=[".onnx"], scale=1)
        with gr.Column():
            info = gr.Markdown("No model loaded.")
            examples_box = gr.Column()  # demo model button is placed here below

    with gr.Tab("Graph"):
        gr.Markdown("Click a node to see its properties · drag to pan · Ctrl + mouse wheel to zoom")
        graph_html = gr.HTML("<p>Upload a model to see its graph.</p>")

    with gr.Tab("Modify"):
        node_dd = gr.Dropdown(label="Node", choices=[], interactive=True)
        node_details = gr.Code(label="Node details", language=None, interactive=False)
        new_name = gr.Textbox(label="New name")
        with gr.Row():
            rename_btn = gr.Button("Rename node")
            delete_btn = gr.Button("Delete node (bypass)", variant="stop")
        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            save_btn = gr.Button("Validate & export model", variant="primary")
            download = gr.File(label="Modified model")

    def describe(model):
        g = model.graph
        consts = constant_names(model)
        return (f"**Nodes:** {len(g.node)} · **Inputs:** "
                f"{', '.join(i.name for i in g.input if i.name not in consts)} · **Outputs:** "
                f"{', '.join(o.name for o in g.output)} · **Opset:** "
                f"{next((o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')), '?')}")

    def refresh(model):
        return (visualize_onnx(model), describe(model),
                gr.update(choices=node_choices(model), value=None))

    def on_upload(file_obj):
        if not file_obj:
            return None, "<p>Upload a model to see its graph.</p>", "No model loaded.", gr.update(choices=[], value=None)
        try:
            model = load_onnx_model(file_obj)
        except Exception as e:
            return None, f"<p style='color:red'>Failed to load model: {html.escape(str(e))}</p>", "Load failed.", gr.update(choices=[])
        return (model, *refresh(model))

    def on_select(model, name):
        if model is None or not name:
            return ""
        node = next((n for n in model.graph.node if n.name == name), None)
        if node is None:
            return ""
        return node_summary(node, tensor_shapes(model, infer=True), constant_names(model))

    def on_rename(model, old, new):
        if model is None or not old or not new.strip():
            return model, "Pick a node and enter a new name.", gr.update(), gr.update(), gr.update()
        try:
            model = rename_node(model, old, new.strip())
        except Exception as e:
            return model, f"Error: {e}", gr.update(), gr.update(), gr.update()
        return (model, f"Renamed '{old}' → '{new.strip()}'.", *refresh(model))

    def on_delete(model, name):
        if model is None or not name:
            return model, "Pick a node first.", gr.update(), gr.update(), gr.update()
        try:
            model = delete_node(model, name)
        except Exception as e:
            return model, f"Error: {e}", gr.update(), gr.update(), gr.update()
        return (model, f"Deleted '{name}' and reconnected its consumers.", *refresh(model))

    def on_save(model):
        if model is None:
            return "No model loaded.", None
        out_path = os.path.join(tempfile.mkdtemp(), "modified_model.onnx")
        onnx.save(model, out_path)
        try:
            onnx.checker.check_model(out_path, full_check=True)  # strict type/shape check
            msg = "Model is valid."
            try:
                import onnxruntime as ort
                ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
                msg += " ONNX Runtime loads it OK."
            except ImportError:
                pass
        except Exception as e:
            msg = f"WARNING - the model is broken by your edits: {str(e)[:400]}"
        return f"{msg} Saved – use the download box.", out_path

    demo_path = os.path.join(HERE, "examples", "demo_resnet.onnx")
    if os.path.exists(demo_path):
        with examples_box:
            gr.Examples(examples=[[demo_path]], inputs=model_input,
                        outputs=[model_state, graph_html, info, node_dd], fn=on_upload,
                        run_on_click=True, cache_examples=False,
                        label="No ONNX file? Click to try a demo model (small ResNet)")

    model_input.upload(on_upload, model_input, [model_state, graph_html, info, node_dd])
    model_input.clear(on_upload, model_input, [model_state, graph_html, info, node_dd])
    node_dd.change(on_select, [model_state, node_dd], node_details)
    rename_btn.click(on_rename, [model_state, node_dd, new_name],
                     [model_state, status, graph_html, info, node_dd])
    delete_btn.click(on_delete, [model_state, node_dd],
                     [model_state, status, graph_html, info, node_dd])
    save_btn.click(on_save, model_state, [status, download])

if __name__ == "__main__":
    app.launch()
