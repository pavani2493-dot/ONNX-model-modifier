---
title: ONNX Model Modifier
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: ONNX_modifier.py
pinned: false
short_description: Netron-style viewer and editor for ONNX models
---

# ONNX Model Modifier

A Netron-style web tool for **viewing and editing ONNX models**, built with Python and Gradio.

**Live demo:** https://huggingface.co/spaces/<your-hf-username>/onnx-modifier

## Features

- **Netron-like graph view**: nodes appear as cards showing their weights, scales and zero-points, with an automatic top-to-bottom layout (dagre)
- Edge labels with tensor shapes
- Click any node to open a properties panel (attributes, inputs, outputs)
- Pan by dragging, zoom with Ctrl + mouse wheel, fit / 1:1 buttons
- **Edit the model**
  - Rename nodes
  - Delete (bypass) single-input / single-output nodes, with connections rewired automatically
- **Validate and export**: strict ONNX check and an ONNX Runtime load test, then download `modified_model.onnx`
- Works offline (the layout library `dagre.min.js` is bundled)

## Installation

```bash
git clone https://github.com/<your-username>/onnx-modifier.git
cd onnx-modifier
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Usage

```bash
python ONNX_modifier.py
```

Then open http://127.0.0.1:7860 in your browser and upload a `.onnx` file, or click the demo model.

## Project structure

| File | Purpose |
|---|---|
| `ONNX_modifier.py` | The app: model loading, graph rendering, editing, export |
| `dagre.min.js` | Graph layout library (must sit next to the script) |
| `requirements.txt` | Python dependencies |
| `examples/demo_resnet.onnx` | Small public demo model for visitors |

## Notes

- In quantized (QDQ) models, delete `QuantizeLinear` / `DequantizeLinear` nodes **in pairs**. Removing only one leaves the model with type errors, which the export check reports.
- Nodes with several data inputs (`Add`, `Concat`, ...) can't be bypassed automatically.

## Credits

- Graph style inspired by [Netron](https://github.com/lutzroeder/netron)
- Layout by [dagre](https://github.com/dagrejs/dagre) (MIT License, bundles lodash under MIT)
