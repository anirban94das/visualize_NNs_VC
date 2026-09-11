# Gemma 3n E4B + visualtorch — task tracker

Goal: test whether `visualtorch` can render the architecture of Gemma 3n E4B
the way it already renders the vizdoom PPO CNN policy
(`E:\Code_Base\vizdoom_vbcd_proj_v2\visualize_PPO_model.py`).

## Status: DONE (all 6 steps run). Summary written back to the user.

- [x] **1. Locate local Gemma download & identify format**
  - Found: `C:\Users\anirb\.lmstudio\models\lmstudio-community\gemma-4-E4B-it-GGUF\`
    - `gemma-4-E4B-it-Q4_K_M.gguf` (5.0 GB)
    - `mmproj-gemma-4-E4B-it-BF16.gguf` (945 MB, vision/audio projector)
  - Format: **GGUF (llama.cpp)** — visualtorch CANNOT trace this (needs a torch nn.Module).
  - Naming note: local files say "gemma-4-E4B-it", not "gemma-3n-E4B". No
    transformers/safetensors Gemma checkpoint anywhere on C:/D:/E:.
  - **Did NOT pull `google/gemma-3n-E4B`** — steps 3/5 used an untrained model
    built from config (no weights, no download), matching visualize_PPO_model.py's
    no-model fallback. Ask user if they want the real weights later.
- [x] **2. Install missing deps** — `transformers 5.17.0` + accelerate + safetensors
  + sentencepiece + psutil into the vizdoom_vbcd_proj_v2 .venv. All Gemma3n*
  classes import. torch 2.12.1, visualtorch 1.2.1 already present.
- [x] **3. Minimal POC (tiny 2-layer untrained Gemma3nTextModel)** — traces
  cleanly, all 3 styles render. Needed `num_kv_shared_layers=0` in the tiny
  config (default −13 → empty `layer_types[:idx]` → `list.index` ValueError).
- [x] **4. input_shape / token-ID limitation** — CONFIRMED. visualtorch 1.2.1
  has NO way to pass an explicit input tensor; `utils/recorder.py:242` hardcodes
  `torch.rand(shape)` (floats). Workaround that works: wrapper does
  `input_ids = (x * 0).long() + 3` — valid ids, still descended from the traced
  input so the graph stays connected.
- [x] **5. Scale up** — full 35-layer E4B backbone (untrained) INSTANTIATES
  (6.84B params, ~5 min, ~7 GB RAM) and forward works, but visualtorch render
  dies with MemoryError (PIL `Image.new` — canvas too big). Single decoder
  layer at full E4B width (112M params) renders fine. Tiny full backbone
  (2 layers) also renders.
- [x] **6. Save PNGs + summary** — see below.

## Artifacts produced

Script: `E:\Code_Base\vizdoom_vbcd_proj_v2\visualize_gemma_model.py`
(companion to visualize_PPO_model.py; `--scope layer|backbone`, `--tiny`, `--style`)

Renders in `E:\Code_Base\vizdoom_vbcd_proj_v2\`:
- `gemma_layer_render_graph.png`  (6070x3710) — one E4B decoder layer, graph style ← most legible
- `gemma_layer_render_flow.png`   (3571x3041) — one E4B decoder layer, flow style
- `gemma_backbone_tiny_render_graph.png` (11470x9510) — full depth, shrunk width (2 layers)

## Key findings

- visualtorch works on Gemma 3n **at the single-decoder-layer level**, not the
  full model. Same play as rendering one residual block of a deep CNN.
- The special sublayers (AltUp, LAuReL, per-layer-input gate/projection, the 4
  RMSNorms) all show up as leaf nodes; the AltUp 4-way split + LAuReL low-rank
  path + residuals produce a big fan of skip-connection edges over the top of
  the layer chain — that's what blows up the canvas at full depth.
- `graph` style >> `flow`/`lenet` for transformers (flow/lenet size boxes off
  spatial dims; a (P,B,S,H) tensor collapses to slivers).
- known visualtorch limits both bite here: (a) tuple output — decoder layer
  returns `(hidden, ...)`, sized off first tensor (fine); (b) data-dependent
  control flow — only the traced attention type (sliding vs full) is shown.

## Qwen3 MoE ("Qwen3.6-35B-A3B") — DONE

Script: `E:\Code_Base\vizdoom_vbcd_proj_v2\visualize_qwen_moe_model.py`
(`--scope layer|backbone`, `--real`, `--hf-config <repo>`, `--style`).

- Not downloaded locally (only DeepSeek-R1-Qwen3-8B GGUF + ollama qwen3). Same
  GGUF caveat. Script uses an untrained config, no weights/download needed.
- `Qwen3MoeConfig()` defaults = real arch: 24 layers, hidden 2048, 32/4 KV heads,
  128 experts, top-8.
- **Surprise:** in transformers 5.17.0 the MoE block is `gate`
  (`Qwen3MoeTopKRouter`) + `experts` (`Qwen3MoeExperts`), and BOTH store weights
  as raw `nn.Parameter` [num_experts,...] tensors with `F.linear` on slices — NO
  per-expert child modules. So visualtorch draws the MoE block as just 2 leaf
  nodes regardless of expert count. The "only routed experts get traced" caveat
  doesn't arise (nothing to miss); you also don't get 128 expert boxes.
- One real decoder layer = 613M untrained params, renders fine (graph 2170x5660).
  ~13 leaf nodes: q/k/v/o_proj, q_norm, k_norm, 2 layernorms, router, experts,
  2 residual arcs. Cleaner than the Gemma layer (no AltUp/LAuReL).
- Renders: `qwen_moe_layer_real_render_graph.png`,
  `qwen_moe_layer_reduced_render_graph.png` / `_flow.png`,
  `qwen_moe_backbone_reduced_render_graph.png` (4870x2810).
- `--hf-config Qwen/Qwen3-30B-A3B` path is coded but untested (needs network).
