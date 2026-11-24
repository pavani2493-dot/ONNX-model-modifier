/* ONNX Model Modifier: runs entirely in the browser.
 *  - protobuf.js decodes / encodes the .onnx file (schema: lib/onnx-defs.js)
 *  - dagre computes the Netron-style top-to-bottom layout
 *  - the graph is drawn as SVG; edits change the model in memory; Export downloads it
 */
"use strict";

// ------------------------------------------------------------------ setup
const ModelProto = protobuf.parse(window.ONNX_PROTO, { keepCase: true }).root.lookupType("onnx.ModelProto");
const ORT_VERSION = "1.20.1";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const CATEGORY_COLORS = {
  layer: "#335588", activation: "#702921", pool: "#335533", normalization: "#335544",
  dropout: "#454770", shape: "#6c4f47", tensor: "#59423b", data: "#555555",
};
const OP_CATEGORY = {};
const cat = (c, ops) => ops.forEach(o => (OP_CATEGORY[o] = c));
cat("layer", ["Conv", "ConvTranspose", "Gemm", "MatMul", "ConvInteger", "QLinearConv", "QLinearMatMul", "MatMulInteger", "LSTM", "GRU", "RNN", "Einsum"]);
cat("activation", ["Relu", "Sigmoid", "Tanh", "LeakyRelu", "PRelu", "Elu", "Selu", "Softmax", "LogSoftmax", "HardSigmoid", "HardSwish", "Softplus", "Softsign", "Clip", "Gelu", "Mish", "ThresholdedRelu", "Celu"]);
cat("pool", ["MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "LpPool", "MaxRoiPool", "GlobalLpPool"]);
cat("normalization", ["BatchNormalization", "InstanceNormalization", "LayerNormalization", "LRN", "GroupNormalization", "LpNormalization"]);
cat("dropout", ["Dropout"]);
cat("shape", ["Reshape", "Flatten", "Squeeze", "Unsqueeze", "Expand", "Tile"]);
cat("tensor", ["Concat", "Split", "Slice", "Gather", "GatherElements", "GatherND", "Pad", "Transpose"]);

const $ = id => document.getElementById(id);
const state = {
  model: null, fileName: "", data: null, layout: null, nodeEls: {}, selected: null, scale: 1,
  tab: "graph",      // "graph" | "modify"
  graphDirty: false, // model changed while the Graph tab was hidden: redraw when it is shown
  modSel: null,      // node picked in the Modify tab
};

// ------------------------------------------------------------------ small helpers
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
let toastTimer;
function toast(msg, isErr = false, ms = 4000) {
  const t = $("toast");
  t.textContent = msg; t.classList.toggle("err", isErr); t.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), ms);
}
function busy(on, text = "Loading…") { $("busy").hidden = !on; $("busyText").textContent = text; }
const nextFrame = () => new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));

const fmtShape = dims => (dims || []).join("×");
function dimsOf(typeProto) {
  const tt = typeProto && typeProto.tensor_type;
  if (!tt || !tt.shape) return null;
  return tt.shape.dim.map(d => d.dim_param ? d.dim_param
    : Object.prototype.hasOwnProperty.call(d, "dim_value") ? Number(d.dim_value) : "?");
}

// float16 bits -> number
function f16(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

// Read the single value of a scalar initializer
function scalarValue(t) {
  const dt = t.data_type, raw = t.raw_data;
  if (raw && raw.length) {
    const v = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    switch (dt) {
      case 1: return { v: v.getFloat32(0, true), f: true };
      case 2: return { v: v.getUint8(0) };
      case 3: return { v: v.getInt8(0) };
      case 4: return { v: v.getUint16(0, true) };
      case 5: return { v: v.getInt16(0, true) };
      case 6: return { v: v.getInt32(0, true) };
      case 7: return { v: v.getBigInt64(0, true).toString() };
      case 9: return { v: v.getUint8(0) ? "true" : "false" };
      case 10: return { v: f16(v.getUint16(0, true)), f: true };
      case 11: return { v: v.getFloat64(0, true), f: true };
      case 12: return { v: v.getUint32(0, true) };
      case 13: return { v: v.getBigUint64(0, true).toString() };
    }
    return null;
  }
  if (t.float_data.length) return { v: t.float_data[0], f: true };
  if (t.double_data.length) return { v: t.double_data[0], f: true };
  if (t.int64_data.length) return { v: t.int64_data[0] };
  if (t.uint64_data.length) return { v: t.uint64_data[0] };
  if (t.int32_data.length) return dt === 10 ? { v: f16(t.int32_data[0]), f: true } : { v: t.int32_data[0] };
  return null;
}
function fmtScalar(t) {
  const r = scalarValue(t);
  if (!r) return "?";
  const s = r.f && t.data_type === 1 ? String(Math.fround(r.v)) : String(r.v);
  return s.length > 10 ? s.slice(0, 10) + "…" : s;
}

function attrToString(a) {
  // AttributeType: 1 FLOAT 2 INT 3 STRING 4 TENSOR 5 GRAPH 6 FLOATS 7 INTS 8 STRINGS
  const dec = b => new TextDecoder().decode(b);
  switch (a.type) {
    case 1: return String(Math.fround(a.f));
    case 2: return String(a.i);
    case 3: return dec(a.s);
    case 4: return `tensor ⟨${fmtShape(a.t ? a.t.dims.map(Number) : [])}⟩`;
    case 5: return `graph '${a.g ? a.g.name : ""}'`;
    case 6: return a.floats.length < 20 ? a.floats.map(x => Math.fround(x)).join(", ") : `[${a.floats.length} values]`;
    case 7: return a.ints.length < 20 ? a.ints.join(", ") : `[${a.ints.length} values]`;
    case 8: return a.strings.map(dec).join(", ");
    default: return "";
  }
}

// Formal input names (x, x_scale, W, B ...) from the ONNX op schema for the model's opset
function formalInputs(op, domain, opset, count) {
  const versions = window.ONNX_OP_INPUTS[(domain === "ai.onnx" ? "" : domain) + "::" + op];
  if (!versions) return new Array(count).fill(null);
  let pick = versions[0];
  for (const v of versions) if (v[0] <= opset) pick = v;
  const [, names, variadic] = pick;
  return Array.from({ length: count }, (_, i) => i < names.length ? names[i] : (variadic && names.length ? names[names.length - 1] : null));
}

// ------------------------------------------------------------------ model -> graph data
function ensureUniqueNames(graph) {
  const seen = new Set();
  graph.node.forEach((n, i) => {
    let name = n.name || `${n.op_type}_${i}`;
    while (seen.has(name)) name = `${name}_${i}`;
    n.name = name; seen.add(name);
  });
}

function buildGraphData(model) {
  const g = model.graph;
  const shapes = {};
  for (const vi of [...g.input, ...g.output, ...g.value_info]) { const d = dimsOf(vi.type); if (d) shapes[vi.name] = d; }
  const inits = {};
  for (const t of g.initializer) { inits[t.name] = t; shapes[t.name] = t.dims.map(Number); }
  const opsets = {};
  for (const o of model.opset_import) opsets[o.domain || ""] = Number(o.version);
  const known = d => d && d.length && d.every(x => typeof x === "number");

  const nodes = [], edges = [], producer = {};
  for (const inp of g.input) {
    if (inits[inp.name]) continue;
    const id = "in::" + inp.name;
    nodes.push({ id, kind: "io", label: inp.name, props: { type: "input", name: inp.name, shape: fmtShape(shapes[inp.name]) } });
    producer[inp.name] = id;
  }
  for (const n of g.node) for (const t of n.output) producer[t] = n.name;

  for (const n of g.node) {
    const dom = n.domain || "";
    const formal = formalInputs(n.op_type, dom, opsets[dom] || opsets[""] || 1, n.input.length);
    const rows = [], inputs = [];
    n.input.forEach((t, i) => {
      if (!t) return;
      const label = formal[i] || t;
      inputs.push({ name: label, tensor: t, shape: fmtShape(shapes[t]) });
      if (inits[t]) {
        const init = inits[t];
        if (init.dims.length === 0) rows.push({ name: label, text2: " = " + fmtScalar(init) });
        else rows.push({ name: label, text2: " ⟨" + fmtShape(init.dims.map(Number)) + "⟩", bold: true });
      } else if (producer[t] !== undefined) {
        edges.push({ from: producer[t], to: n.name, label: known(shapes[t]) ? fmtShape(shapes[t]) : "" });
      }
    });
    nodes.push({
      id: n.name, kind: "op", label: n.op_type, rows,
      color: CATEGORY_COLORS[OP_CATEGORY[n.op_type]] || "#333333",
      props: {
        type: n.op_type, name: n.name, domain: dom || "ai.onnx",
        attributes: n.attribute.map(a => ({ name: a.name, value: attrToString(a) })),
        inputs, outputs: n.output.map(t => ({ tensor: t, shape: fmtShape(shapes[t]) })),
      },
    });
  }
  for (const out of g.output) {
    const id = "out::" + out.name;
    nodes.push({ id, kind: "io", label: out.name, props: { type: "output", name: out.name, shape: fmtShape(shapes[out.name]) } });
    if (producer[out.name] !== undefined) edges.push({ from: producer[out.name], to: id, label: known(shapes[out.name]) ? fmtShape(shapes[out.name]) : "" });
  }
  return { nodes, edges };
}

// ------------------------------------------------------------------ layout + drawing
const NS = "http://www.w3.org/2000/svg";
const FONT = '11px -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
const ctx = document.createElement("canvas").getContext("2d");
function tw(t, bold) { ctx.font = (bold ? "bold " : "") + FONT; return ctx.measureText(t).width; }
function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
const HDR = 21, ROW = 15, PADX = 8, PADY = 5;

function layout(data) {
  for (const n of data.nodes) {
    if (n.kind === "io") { n.w = Math.max(tw(n.label) + 2 * PADX + 4, 40); n.h = 26; continue; }
    let w = tw(n.label) + 2 * PADX + 2;
    for (const r of n.rows) w = Math.max(w, tw(r.name, r.bold) + tw(r.text2) + 2 * PADX);
    n.w = Math.ceil(w);
    n.h = HDR + (n.rows.length ? n.rows.length * ROW + 2 * PADY : 0);
  }
  const g = new dagre.graphlib.Graph({ multigraph: true });
  g.setGraph({ rankdir: "TB", nodesep: 20, ranksep: 20, edgesep: 10, marginx: 40, marginy: 40 });
  g.setDefaultEdgeLabel(() => ({}));
  for (const n of data.nodes) g.setNode(n.id, { width: n.w, height: n.h });
  data.edges.forEach((e, i) => g.setEdge(e.from, e.to,
    { width: e.label ? tw(e.label) + 6 : 0, height: e.label ? 14 : 0, labelpos: "c" }, "e" + i));
  dagre.layout(g);
  return g;
}

// same curve as d3.curveBasis (what Netron uses)
function basis(pts) {
  if (pts.length < 3) return "M" + pts.map(p => p.x + "," + p.y).join("L");
  let d = "", x0, y0, x1, y1, st = 0;
  const bez = (x, y) => { d += `C${(2 * x0 + x1) / 3},${(2 * y0 + y1) / 3} ${(x0 + 2 * x1) / 3},${(y0 + 2 * y1) / 3} ${(x0 + 4 * x1 + x) / 6},${(y0 + 4 * y1 + y) / 6}`; };
  for (const { x, y } of pts) {
    if (st === 0) { st = 1; d += `M${x},${y}`; }
    else if (st === 1) st = 2;
    else { if (st === 2) { st = 3; d += `L${(5 * x0 + x1) / 6},${(5 * y0 + y1) / 6}`; } bez(x, y); }
    x0 = x1; x1 = x; y0 = y1; y1 = y;
  }
  bez(x1, y1); d += `L${x1},${y1}`;
  return d;
}

function draw(data, g) {
  const svg = $("svg");
  svg.innerHTML = "";
  state.nodeEls = {};
  const defs = el("defs", {}, svg);
  const mk = el("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs);
  el("path", { d: "M0,0 L10,5 L0,10 L3,5 z", fill: "#000" }, mk);
  const edgeLayer = el("g", {}, svg), nodeLayer = el("g", {}, svg);

  for (const eo of g.edges()) {
    const e = g.edge(eo), lbl = data.edges[+eo.name.slice(1)].label;
    const grp = el("g", { class: "edge" }, edgeLayer);
    el("path", { d: basis(e.points), "marker-end": "url(#arrow)" }, grp);
    if (lbl) {
      el("rect", { x: e.x - e.width / 2, y: e.y - 7, width: e.width, height: 14, fill: "#fff", opacity: 0.85 }, grp);
      el("text", { x: e.x, y: e.y }, grp).textContent = lbl;
    }
  }
  for (const n of data.nodes) {
    const p = g.node(n.id);
    const grp = el("g", { class: "node", transform: `translate(${p.x - n.w / 2},${p.y - n.h / 2})` }, nodeLayer);
    state.nodeEls[n.id] = grp;
    if (n.kind === "io") {
      el("rect", { class: "io", width: n.w, height: n.h, rx: 6 }, grp);
      el("text", { x: n.w / 2, y: n.h / 2, "text-anchor": "middle" }, grp).textContent = n.label;
    } else if (!n.rows.length) {
      el("rect", { class: "hdr", width: n.w, height: HDR, rx: 5, fill: n.color, stroke: "#333" }, grp);
      el("text", { x: n.w / 2, y: HDR / 2, "text-anchor": "middle", fill: "#fff" }, grp).textContent = n.label;
    } else {
      el("rect", { class: "body", width: n.w, height: n.h, rx: 5 }, grp);
      el("path", { class: "hdr", d: `M0,${HDR} V5 Q0,0 5,0 H${n.w - 5} Q${n.w},0 ${n.w},5 V${HDR} Z`, fill: n.color, stroke: "#333" }, grp);
      el("text", { x: n.w / 2, y: HDR / 2, "text-anchor": "middle", fill: "#fff" }, grp).textContent = n.label;
      n.rows.forEach((r, i) => {
        const t = el("text", { x: PADX, y: HDR + PADY + ROW * i + ROW / 2 }, grp);
        el("tspan", { "font-weight": r.bold ? "bold" : "normal" }, t).textContent = r.name;
        el("tspan", {}, t).textContent = r.text2;
      });
    }
    grp.addEventListener("click", ev => { ev.stopPropagation(); select(n.id); });
  }
}

// ------------------------------------------------------------------ zoom / pan
function graphSize() { const gr = state.layout.graph(); return [gr.width, gr.height]; }
function setScale(s, cx, cy) {
  const wrap = $("wrap"), svg = $("svg"), [gw, gh] = graphSize();
  s = Math.max(0.05, Math.min(4, s));
  cx = cx ?? wrap.clientWidth / 2; cy = cy ?? wrap.clientHeight / 2;
  const gx = (wrap.scrollLeft + cx) / state.scale, gy = (wrap.scrollTop + cy) / state.scale;
  state.scale = s;
  svg.setAttribute("width", gw * s); svg.setAttribute("height", gh * s); svg.setAttribute("viewBox", `0 0 ${gw} ${gh}`);
  wrap.scrollLeft = gx * s - cx; wrap.scrollTop = gy * s - cy;
}
function svgOffsetX() { const [gw] = graphSize(); return Math.max(0, ($("wrap").clientWidth - gw * state.scale) / 2); }
function scrollToNode(id, top = false) {
  const p = state.layout.node(id), wrap = $("wrap");
  if (!p) return;
  wrap.scrollLeft = p.x * state.scale + svgOffsetX() - wrap.clientWidth / 2;
  wrap.scrollTop = top ? Math.max(0, p.y * state.scale - 70) : p.y * state.scale - wrap.clientHeight / 2;
}
function initZoom() {
  const wrap = $("wrap");
  wrap.addEventListener("wheel", ev => {
    if (!(ev.ctrlKey || ev.metaKey)) return;
    ev.preventDefault();
    const r = wrap.getBoundingClientRect();
    setScale(state.scale * (ev.deltaY < 0 ? 1.1 : 1 / 1.1), ev.clientX - r.left, ev.clientY - r.top);
  }, { passive: false });
  let drag = null;
  wrap.addEventListener("pointerdown", ev => {
    if (ev.button !== 0 || ev.target.closest(".node")) return;
    drag = { x: ev.clientX, y: ev.clientY, l: wrap.scrollLeft, t: wrap.scrollTop, moved: false };
  });
  window.addEventListener("pointermove", ev => {
    if (!drag) return;
    const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; wrap.classList.add("grab"); }
    wrap.scrollLeft = drag.l - dx; wrap.scrollTop = drag.t - dy;
  });
  window.addEventListener("pointerup", () => {
    if (drag && !drag.moved && state.selected) closeSide();
    drag = null; wrap.classList.remove("grab");
  });
  $("zin").onclick = () => setScale(state.scale * 1.25);
  $("zout").onclick = () => setScale(state.scale / 1.25);
  $("z1").onclick = () => setScale(1);
  $("zfit").onclick = () => { const [gw, gh] = graphSize(); setScale(Math.min(wrap.clientWidth / gw, wrap.clientHeight / gh, 1)); wrap.scrollTop = 0; };
}

// ------------------------------------------------------------------ side panel + editing
const kv = (k, v) => `<div class="kv"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`;
function findNode(id) { return state.data.nodes.find(n => n.id === id); }

function select(id) {
  const n = findNode(id);
  if (!n) return;
  if (state.selected && state.nodeEls[state.selected]) state.nodeEls[state.selected].classList.remove("sel");
  state.selected = id;
  state.nodeEls[id].classList.add("sel");
  if (n.kind === "op") { state.modSel = id; refreshModify(); } // keep the Modify tab on the same node
  const p = n.props;
  let h = `<h2>${esc(p.type)}</h2>` + kv("name", p.name);
  if (n.kind === "io") h += kv("shape", p.shape || "?");
  else {
    h += kv("domain", p.domain);
    if (p.attributes.length) { h += `<div class="sec">Attributes</div>`; p.attributes.forEach(a => (h += kv(a.name, a.value))); }
    h += `<div class="sec">Inputs</div>`; p.inputs.forEach(i => (h += kv(i.name, i.tensor + (i.shape ? `  ⟨${i.shape}⟩` : ""))));
    h += `<div class="sec">Outputs</div>`; p.outputs.forEach(o => (h += kv("output", o.tensor + (o.shape ? `  ⟨${o.shape}⟩` : ""))));
    h += `<div class="sec">Edit</div>
      <div class="edit">
        <label for="renameIn">Node name</label>
        <div class="row"><input type="text" id="renameIn" value="${esc(p.name)}"><button class="btn" id="renameBtn">Rename</button></div>
        <div class="row" style="margin-top:10px"><button class="btn danger" id="deleteBtn">Delete node (bypass)</button></div>
        <p class="note">Delete removes the node and connects whatever came after it to its input. It works for nodes with one data input and one output (Relu, Dropout, Identity, Q/DQ…). In quantized models, remove Quantize/Dequantize nodes in pairs.</p>
      </div>`;
  }
  $("sideContent").innerHTML = h;
  $("side").hidden = false;
  if (n.kind === "op") {
    $("renameBtn").onclick = () => doRename(id, $("renameIn").value.trim());
    $("renameIn").onkeydown = e => { if (e.key === "Enter") doRename(id, $("renameIn").value.trim()); };
    $("deleteBtn").onclick = () => doDelete(id);
  }
}
function closeSide() {
  $("side").hidden = true;
  if (state.selected && state.nodeEls[state.selected]) state.nodeEls[state.selected].classList.remove("sel");
  state.selected = null;
}

// Report the result of an edit in both places: a toast and the Modify tab's Status box
function report(msg, isErr = false) {
  toast(msg, isErr, isErr ? 6000 : 4000);
  setStatus("modStatus", msg, isErr ? "err" : "ok");
}
function setStatus(id, msg, cls = "") {
  const el = $(id);
  el.textContent = msg;
  el.className = "status" + (cls ? " " + cls : "");
}

// After an edit: redraw now if the graph is on screen, otherwise when the Graph tab is opened
function afterEdit(selectId) {
  if (state.tab === "graph") return rerender(selectId);
  state.data = buildGraphData(state.model);
  state.graphDirty = true;
  updateInfo();
  refreshModify();
}

function doRename(oldName, newName) {
  const g = state.model.graph;
  if (!newName) return report("Enter a new name.", true);
  if (newName === oldName) return report("The new name is the same as the current one.", true);
  if (g.node.some(n => n.name === newName)) return report(`A node named “${newName}” already exists.`, true);
  const node = g.node.find(n => n.name === oldName);
  if (!node) return report(`Node “${oldName}” not found.`, true);
  node.name = newName;
  if (state.modSel === oldName) state.modSel = newName;
  afterEdit(state.tab === "graph" ? newName : null);
  report(`Renamed “${oldName}” → “${newName}”.`);
}

function doDelete(name) {
  const g = state.model.graph;
  const idx = g.node.findIndex(n => n.name === name);
  if (idx < 0) return report(`Node “${name}” not found.`, true);
  const target = g.node[idx];
  const inits = new Set(g.initializer.map(t => t.name));
  const dataIn = target.input.filter(t => t && !inits.has(t));
  if (target.output.length !== 1 || dataIn.length !== 1) {
    return report(`“${name}” (${target.op_type}) has ${dataIn.length} data input(s) and ${target.output.length} output(s). Only single-input / single-output nodes (Relu, Dropout, Identity, Q/DQ…) can be bypassed safely.`, true);
  }
  const oldOut = target.output[0], src = dataIn[0];
  for (const n of g.node) n.input.forEach((t, i) => { if (t === oldOut) n.input[i] = src; });
  for (const o of g.output) if (o.name === oldOut) o.name = src;
  g.node.splice(idx, 1);
  const vi = g.value_info.findIndex(v => v.name === oldOut);
  if (vi >= 0) g.value_info.splice(vi, 1);
  if (state.modSel === name) state.modSel = null;
  closeSide();
  afterEdit(null);
  report(`Deleted “${name}” and reconnected its consumers.`);
}

// ------------------------------------------------------------------ Modify tab
function nodeSummary(id) {
  const n = findNode(id);
  if (!n) return "Pick a node to see its details.";
  const p = n.props, inits = new Set(state.model.graph.initializer.map(t => t.name));
  const lines = [`${p.type}  (${p.name})`, "", "Inputs:"];
  p.inputs.forEach(i => lines.push(`  ${i.name}: ${i.tensor}${i.shape ? "  ⟨" + i.shape + "⟩" : ""}${inits.has(i.tensor) ? "  [initializer]" : ""}`));
  lines.push("Outputs:");
  p.outputs.forEach(o => lines.push(`  ${o.tensor}${o.shape ? "  ⟨" + o.shape + "⟩" : ""}`));
  if (p.attributes.length) {
    lines.push("Attributes:");
    p.attributes.forEach(a => lines.push(`  ${a.name} = ${a.value.length > 80 ? a.value.slice(0, 80) + "…" : a.value}`));
  }
  return lines.join("\n");
}

function refreshModify() {
  if (!state.model) return;
  const sel = $("nodeSelect"), nodes = state.model.graph.node;
  if (state.modSel && !nodes.some(n => n.name === state.modSel)) state.modSel = null;
  sel.innerHTML = `<option value="">Choose a node… (${nodes.length} nodes)</option>` +
    nodes.map(n => `<option value="${esc(n.name)}">${esc(n.name)}  —  ${esc(n.op_type)}</option>`).join("");
  sel.value = state.modSel || "";
  const has = !!state.modSel;
  $("nodeDetails").textContent = has ? nodeSummary(state.modSel) : "Pick a node to see its details.";
  $("newName").disabled = $("modRename").disabled = $("modDelete").disabled = !has;
  $("newName").value = has ? state.modSel : "";
  $("newName").placeholder = has ? "" : "Pick a node first";
}

async function showTab(tab) {
  state.tab = tab;
  $("tabGraph").classList.toggle("active", tab === "graph");
  $("tabModify").classList.toggle("active", tab === "modify");
  $("tabGraph").setAttribute("aria-selected", tab === "graph");
  $("tabModify").setAttribute("aria-selected", tab === "modify");
  $("viewer").hidden = tab !== "graph";
  $("modifyView").hidden = tab !== "modify";
  if (tab === "modify") refreshModify();
  if (tab === "graph" && state.graphDirty) {
    state.graphDirty = false;
    await rerender(state.selected && findNode(state.selected) ? state.selected : null);
  }
}

async function modifyExport() {
  busy(true, "Checking the model with ONNX Runtime…");
  let result;
  try { result = await checkModel(); } finally { busy(false); }
  const file = exportModel(false);
  setStatus("exportStatus",
    (result.ok ? result.msg + " " : "WARNING: the model is broken by your edits. " + result.msg + " ") + `Downloaded ${file}.`,
    result.ok ? "ok" : "err");
}

// ------------------------------------------------------------------ validation + export
function structuralCheck(model) {
  const g = model.graph, problems = [];
  const avail = new Set([...g.input.map(i => i.name), ...g.initializer.map(i => i.name)]);
  const produced = new Set();
  for (const n of g.node) for (const t of n.output) { if (produced.has(t)) problems.push(`tensor “${t}” is produced twice`); produced.add(t); }
  for (const n of g.node) for (const t of n.input) if (t && !avail.has(t) && !produced.has(t)) problems.push(`node “${n.name}” uses missing tensor “${t}”`);
  for (const o of g.output) if (!avail.has(o.name) && !produced.has(o.name)) problems.push(`graph output “${o.name}” is not produced`);
  return problems;
}

let ortPromise = null;
function loadOrt() {
  if (window.ort) return Promise.resolve(window.ort);
  if (!ortPromise) ortPromise = new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = ORT_CDN + "ort.min.js";
    s.onload = () => { ort.env.wasm.wasmPaths = ORT_CDN; ort.env.wasm.numThreads = 1; res(ort); };
    s.onerror = () => { ortPromise = null; rej(new Error("Could not download ONNX Runtime Web (are you offline?)")); };
    document.head.appendChild(s);
  });
  return ortPromise;
}

// Structural check, then a real load in ONNX Runtime Web → { ok, msg }
async function checkModel() {
  const problems = structuralCheck(state.model);
  if (problems.length) return { ok: false, msg: "Broken connections: " + problems.slice(0, 3).join("; ") + (problems.length > 3 ? ` (+${problems.length - 3} more)` : "") + "." };
  let ortLib;
  try { ortLib = await loadOrt(); }
  catch (e) { return { ok: true, msg: "Connections look fine (ONNX Runtime check skipped: " + e.message + ")." }; }
  try {
    const bytes = ModelProto.encode(state.model).finish();
    const sess = await ortLib.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    if (sess.release) await sess.release();
    return { ok: true, msg: "✓ Model is valid: ONNX Runtime loads it without errors." };
  } catch (e) {
    return { ok: false, msg: String(e.message || e).slice(0, 300) };
  }
}

async function validate() {
  busy(true, "Checking the model with ONNX Runtime…");
  try {
    const r = await checkModel();
    toast(r.ok ? r.msg : "Model problem: " + r.msg, !r.ok, r.ok ? 4000 : 9000);
  } finally { busy(false); }
}

function exportModel(showToast = true) {
  const problems = structuralCheck(state.model);
  const bytes = ModelProto.encode(state.model).finish();
  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = state.fileName.replace(/\.onnx$/i, "") + "_modified.onnx";
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  if (showToast) toast(problems.length ? "Exported, but the graph has broken connections. Click Validate for details." : "Exported " + a.download, problems.length > 0);
  return a.download;
}

// ------------------------------------------------------------------ load / render
function updateInfo() {
  const m = state.model, g = m.graph, inits = new Set(g.initializer.map(t => t.name));
  const opset = (m.opset_import.find(o => !o.domain || o.domain === "ai.onnx") || {}).version;
  $("modelinfo").innerHTML = `<b>${esc(state.fileName)}</b> · ${g.node.length} nodes · inputs: ${esc(g.input.filter(i => !inits.has(i.name)).map(i => i.name).join(", "))} · outputs: ${esc(g.output.map(o => o.name).join(", "))}` + (opset ? ` · opset ${opset}` : "");
  $("nodenames").innerHTML = g.node.map(n => `<option value="${esc(n.name)}">${esc(n.op_type)}</option>`).join("");
}

async function rerender(selectId, first = false) {
  const wrap = $("wrap");
  // remember what the user was looking at
  const keep = !first && state.layout ? { x: (wrap.scrollLeft + wrap.clientWidth / 2) / state.scale, y: (wrap.scrollTop + wrap.clientHeight / 2) / state.scale } : null;
  busy(true, first ? "Laying out the graph…" : "Updating…");
  await nextFrame();
  try {
    state.data = buildGraphData(state.model);
    state.layout = layout(state.data);
    draw(state.data, state.layout);
    setScale(first ? 1 : state.scale);
    if (first) {
      const firstIn = state.data.nodes.find(n => n.kind === "io") || state.data.nodes[0];
      if (firstIn) scrollToNode(firstIn.id, true);
    } else if (keep) {
      wrap.scrollLeft = keep.x * state.scale - wrap.clientWidth / 2;
      wrap.scrollTop = keep.y * state.scale - wrap.clientHeight / 2;
    }
    updateInfo();
    refreshModify();
    if (selectId) select(selectId);
  } finally { busy(false); }
}

async function loadBytes(buffer, name) {
  busy(true, "Reading model…");
  await nextFrame();
  let model;
  try {
    model = ModelProto.decode(new Uint8Array(buffer));
    if (!model.graph) throw new Error("no graph found in the file");
  } catch (e) {
    busy(false);
    return toast(`Could not read “${name}” as an ONNX model: ${e.message}`, true, 7000);
  }
  ensureUniqueNames(model.graph);
  state.model = model; state.fileName = name; state.selected = null; state.modSel = null; state.graphDirty = false;
  $("welcome").hidden = true; $("side").hidden = true; $("tabs").hidden = false;
  ["exportBtn", "checkBtn", "find"].forEach(id => ($(id).disabled = false));
  setStatus("modStatus", "No changes yet.");
  setStatus("exportStatus", "Checks the model with ONNX Runtime, then downloads it as " + name.replace(/\.onnx$/i, "") + "_modified.onnx.");
  await showTab("graph");
  await rerender(null, true);
  if (model.graph.node.length > 3000) toast("Large model: the layout may take a moment after each edit.");
}

async function loadFile(file) {
  if (!file) return;
  loadBytes(await file.arrayBuffer(), file.name);
}

async function loadDemos() {
  try {
    const res = await fetch("examples/examples.json", { cache: "no-cache" });
    if (!res.ok) return;
    const list = await res.json();
    if (!list.length) return;
    const box = $("demos");
    box.innerHTML = `<div class="label">No model? Try a demo</div>`;
    for (const d of list) {
      const b = document.createElement("button");
      b.className = "btn"; b.textContent = d.title || d.file;
      b.onclick = async () => {
        busy(true, "Downloading demo model…");
        try {
          const r = await fetch("examples/" + d.file);
          if (!r.ok) throw new Error(r.status + " " + r.statusText);
          await loadBytes(await r.arrayBuffer(), d.file);
        } catch (e) { busy(false); toast("Could not load demo: " + e.message, true); }
      };
      box.appendChild(b);
    }
  } catch { /* opened from disk (file://): demos unavailable, that's fine */ }
}

// ------------------------------------------------------------------ wire up UI
function init() {
  const fi = $("fileInput");
  $("openBtn").onclick = $("openBtn2").onclick = () => fi.click();
  fi.onchange = () => { loadFile(fi.files[0]); fi.value = ""; };
  $("exportBtn").onclick = exportModel;
  $("checkBtn").onclick = validate;
  $("closeSide").onclick = closeSide;
  $("find").addEventListener("change", async e => {
    const name = e.target.value.trim();
    if (!name || !state.model) return;
    if (!state.model.graph.node.some(n => n.name === name)) return toast(`No node named “${name}”.`, true);
    if (state.tab === "modify") {  // on the Modify tab, pick it in the dropdown
      state.modSel = name; refreshModify(); return;
    }
    if (state.layout && state.layout.node(name)) { scrollToNode(name); select(name); }
  });

  // Graph / Modify tabs
  $("tabGraph").onclick = () => showTab("graph");
  $("tabModify").onclick = () => showTab("modify");
  $("nodeSelect").onchange = e => { state.modSel = e.target.value || null; refreshModify(); setStatus("modStatus", state.modSel ? "Selected “" + state.modSel + "”." : "No node selected."); };
  const renameFromModify = () => doRename(state.modSel, $("newName").value.trim());
  $("modRename").onclick = renameFromModify;
  $("newName").onkeydown = e => { if (e.key === "Enter") renameFromModify(); };
  $("modDelete").onclick = () => doDelete(state.modSel);
  $("modExport").onclick = modifyExport;

  const drop = $("drop");
  ["dragenter", "dragover"].forEach(t => document.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(t => document.addEventListener(t, e => { e.preventDefault(); if (t === "drop" || e.target === document.documentElement) drop.classList.remove("over"); }));
  document.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

  initZoom();
  loadDemos();
}
init();
