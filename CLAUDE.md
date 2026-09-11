# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A personal collection of standalone, drag-and-drop scripts that visualize or
diagnose neural networks — no shared library code, no build system, no
tests, no dependency manifest (each script lazily imports whatever heavy
deps it needs, so it stays runnable with only `torch`/`numpy` until you pass
flags that need more). There are two independent families, both originally
written against `vizdoom_clone_project_vibecoded` (also referenced elsewhere
as `vizdoom_vbcd_proj_v2`)'s PPO `CnnPolicy` and designed to be copied into
that sibling repo, next to its own `visualize_PPO_model.py`, though every
script also runs fully standalone here on a freshly built untrained network.

**1. ZFNet-style diagnostics** (Zeiler & Fergus 2014) — deconvnet
reconstruction, occlusion sensitivity, and saliency/guided backprop for the
NatureCNN + policy/value heads:
- `zf_viz.py` — the portable, `WrappedPolicy`-based version; `README_zf_viz.md`
  is its authoritative usage walkthrough (paper section ↔ technique mapping,
  full flag reference). Keep the two in sync when changing CLI flags/behavior.
- `visualize_cnn_diagnostics.py` — a parallel evolution of the same three
  techniques wired directly to a `DiagnosticPolicy` (cnn/linear/action_net/
  value_net) instead of `WrappedPolicy`. **Flag names differ from zf_viz.py**:
  `--env {basic,deadly_corridor}` instead of `--live`, `--frame` (singular)
  in addition to `--frames`, output dir `cnn_diagnostics_out/` instead of
  `viz_out/`. Don't assume the two scripts' CLIs are interchangeable.

**2. `visualtorch` architecture-diagram renders** — render a model's module
graph as an image, not its activations:
- `visualize_PPO_model.py` — the original; renders the PPO actor branch
  (NatureCNN → `action_net`) with `visualtorch.render()`.
- `visualize_gemma_model.py` — same recipe applied to a Gemma 3n E4B text
  decoder layer (or full backbone with `--tiny`), built from an untrained
  `transformers` config, no download/weights needed.
- `visualize_qwen_moe_model.py` — same recipe for a Qwen3-MoE decoder layer
  (`--real` for true dims, `--hf-config <repo>` to pull a real `config.json`).
- `GEMMA_VISUALTORCH_TODO.md` is the (completed) task log for how the Gemma/
  Qwen scripts were derived — it's the source for the `visualtorch`-specific
  gotchas listed below and worth reading before touching these two scripts.
- `renders/` holds checked-in example PNG/JPG outputs from all five scripts.

## Running

All scripts are plain `python <script>.py [flags]`; none need a venv of
their own for the zero-argument path (only `torch`/`numpy`/`matplotlib`).
Flags that load a real trained model or a live env need `stable_baselines3`
/ `gymnasium` / `vizdoom` (or, for the Gemma/Qwen scripts, `transformers`),
which live in the *sibling* vizdoom project's venv, not one in this repo —
run through that project's `.venv\Scripts\python.exe` for those paths.

```
# ZFNet diagnostics — shape sanity-check only, no deps beyond torch/matplotlib
python zf_viz.py --technique all
python visualize_cnn_diagnostics.py

# ZFNet diagnostics — against a real trained model (run from the sibling repo, its venv active)
.venv\Scripts\python.exe zf_viz.py --model models/latest/ppo_basic.zip --live basic --technique all --out viz_out/basic
.venv\Scripts\python.exe visualize_cnn_diagnostics.py --model models/latest/ppo_basic.zip --env basic --technique all

# visualtorch architecture renders
python visualize_PPO_model.py                     # untrained CnnPolicy, flow style
python visualize_gemma_model.py --style graph      # one E4B decoder layer (default scope)
python visualize_qwen_moe_model.py --real --style graph
```

See `README_zf_viz.md` for the full `zf_viz.py` flag reference (top-k deconv
scans, occlusion patch size/stride, saliency vs. guided) and each script's
module docstring / `--help` for the rest.

## Architecture

### ZFNet diagnostics family (`zf_viz.py`, `visualize_cnn_diagnostics.py`)

Both scripts follow the same pipeline, in file order:

1. **Network wrapping** — everything downstream operates on plain `nn.Module`
   pieces, never on `stable_baselines3` types directly: a `cnn`
   (`nn.Sequential` of alternating `Conv2d`/`ReLU`, **no pooling, no
   flatten**) plus policy/value heads. This is what lets both scripts run on
   a fresh untrained network or on a real loaded model's own live layers
   interchangeably, and is why extending either to a new model format only
   needs a new `build_from_saved_model`-style loader.
2. **Deconvnet** — forward hooks record post-ReLU activations and each
   conv's *input* spatial size (needed as `output_size` for
   `conv_transpose2d` on strided convs). Reconstruction: isolate one channel
   (and its strongest spatial location, or a given position), then walk back
   layer→0 doing rectify + `conv_transpose2d` reusing the forward conv
   weights. **Unpooling is deliberately omitted** — the target NatureCNN
   (three strided convs) has no pooling layers, so the paper's per-layer
   "unpool → rectify → filter" collapses to "rectify → filter".
3. **Occlusion sensitivity** — slides a gray patch over every input
   position in one batched forward pass; reshapes results into per-action +
   value grids. "Class probability" in the paper maps to **action
   probability** (policy head, softmax) and **value estimate** (critic head).
4. **Saliency / guided backprop** — not in the paper; gradient-based
   cross-check. Guided backprop adds full-backward hooks on every `nn.ReLU`
   that also zero negative gradients (Springenberg et al. 2015).
5. **Frame acquisition** — a live frame from the sibling project's `envs/`
   package (lazy-imported), or a saved `.npy` frame/batch, or (with no flags)
   a random frame for a shape-only sanity check.
6. **Plotting** — Agg-backend matplotlib, lazy-imported inside each `plot_*`.

Constants at the top of each file (`N_STACK=4`, `FRAME_SIZE=84`, `N_ACTIONS`)
must match the target project's `NatureCNN`/env; `N_ACTIONS` is overridden
automatically when a real `--model` is loaded.

### `visualtorch` architecture-render family

Common recipe across all three scripts: wrap the target module (or one
representative sub-module) in a thin `nn.Module` whose `forward()` takes a
single traced tensor and returns a single tensor, then call
`visualtorch.render(model, input_shape=..., style=...)`. Known constraints
that shape all three scripts (learned the hard way — see
`GEMMA_VISUALTORCH_TODO.md`):

- **No explicit-input-tensor API.** `visualtorch` 1.2.1 only accepts
  `input_shape` and internally does `torch.rand(shape)` (random floats).
  Token-ID models (Gemma, Qwen) can't take a float tensor as input, so the
  wrapper derives valid ids *from* the traced tensor —
  `input_ids = (x * 0).long() + k` — preserving lineage to the traced input
  node while fixing dtype/range.
- **Full deep backbones OOM the renderer.** Instantiation and a forward pass
  work fine even at full scale (e.g. Gemma 3n E4B's 35 layers, ~6.8B
  untrained params), but `render()` unrolls every layer/skip-connection and
  PIL's `Image.new()` raises `MemoryError` on the resulting canvas. Both
  transformer scripts therefore default `--scope` to `layer` (render one
  decoder layer as the representative repeating unit — the transformer
  analogue of rendering one residual block of a deep CNN) and keep
  `--scope backbone` only for small/`--tiny` configs.
- **Render styles are `graph` / `flow` / `lenet`** (there is no `layered`).
  `graph` is the only legible style for transformers (`flow`/`lenet` size
  boxes off spatial dims, collapsing a `(P,B,S,H)` tensor to slivers); `flow`
  works fine for the CNN case in `visualize_PPO_model.py`.
- **MoE experts don't show up as separate boxes.** In this `transformers`
  version, `Qwen3MoeSparseMoeBlock` stores all experts' weights as raw
  `nn.Parameter` `[num_experts, ...]` tensors and runs them via `F.linear` on
  slices — there are no per-expert child modules, so `visualtorch` always
  draws exactly two leaf nodes (`gate` router + `experts`) regardless of
  expert count.
- Local Gemma/Qwen downloads on this machine are GGUF (LM Studio format),
  which `visualtorch` can't trace at all (needs a `torch.nn.Module`); both
  scripts sidestep this by building an **untrained model from a
  `transformers` config** instead of loading real weights.

## Conventions

- Heavy dependencies (`stable_baselines3`, `gymnasium`, `vizdoom`,
  `matplotlib`, `transformers`, the sibling repo's `envs/`) are **imported
  lazily**, only in the function that needs them, so every script's
  zero-argument path stays dependency-light. Preserve this in any new script
  or edit.
- "Class probability" from the ZF paper maps to **action probability**
  (policy head) and **value estimate** (critic head) throughout the
  diagnostics family.
- New model formats for the diagnostics family: add a loader function
  returning the wrapped policy type; nothing else should need to change as
  long as the conv stack is `Conv2d`/`ReLU`-only. New models for the
  `visualtorch` family: add a `visualize_<model>_model.py` following the
  same wrap-to-single-tensor-io recipe, and default to rendering one
  representative layer rather than the full stack.
