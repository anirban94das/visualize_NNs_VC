# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`zf_viz.py` is a single-file, drag-and-drop implementation of the three diagnostic
tools from Zeiler & Fergus, *Visualizing and Understanding Convolutional Networks*
(ECCV 2014), adapted from an ImageNet softmax classifier to an **RL policy/value
network**. It is written to be copied into the sibling project
`vizdoom_clone_project_vibecoded` (next to its `visualize_PPO_model.py`) and run
from that repo's root, but it also runs fully standalone.

There is no build system, no tests, no git repo here — just `zf_viz.py` and
`README_zf_viz.md`. The README is the authoritative usage walkthrough; keep the
two in sync when changing CLI flags or behavior.

## Running

Standalone shape sanity-check (only needs `torch`, `numpy`, `matplotlib`):
```
python zf_viz.py --technique all
```
Uses a fresh untrained `SyntheticNatureCNNPolicy` and a random-noise frame.
Output PNGs go to `viz_out/zf_*.png` (prefix set by `--out`).

Against a real trained model, from the `vizdoom_clone_project_vibecoded` repo root
with its venv active:
```
.venv\Scripts\python.exe zf_viz.py --model models/latest/ppo_basic.zip \
    --live basic --technique all --out viz_out/basic
```

Key flags: `--technique {deconv,occlusion,saliency,guided,all}`, `--model` (SB3
`PPO.zip`), `--image` (`.npy` frame) / `--live {basic,deadly_corridor}` /
`--frames` (`.npy` `(N,C,H,W)` for the top-k deconv scan), `--layer {0,1,2}` /
`--channel` for deconv, `--patch-size` / `--stride` for occlusion, `--action` for
saliency/guided.

## Architecture

Pipeline in `zf_viz.py`, in file order:

1. **Network wrapping** — everything downstream operates on a `WrappedPolicy`
   dataclass: `cnn` (an `nn.Sequential` of alternating `Conv2d`/`ReLU`, **no
   pooling, no flatten**) plus a `head` callable mapping the flattened CNN
   features to `{"action_logits", "action_probs", "value"}`.
   - `SyntheticNatureCNNPolicy` / `wrap_synthetic` — untrained network with SB3's
     exact `NatureCNN` + PPO head shapes (`net_arch=[]`, no hidden MLP).
   - `load_sb3_policy` — lazy-imports `stable_baselines3`, loads a `PPO.zip`,
     strips the trailing `nn.Flatten` off `pi_features_extractor.cnn`, and wires a
     `head` through `mlp_extractor` + `action_net` / `value_net`.
2. **Deconvnet** (`deconv_reconstruct`, `find_top_k_activations`,
   `ActivationRecorder`) — forward hooks record post-ReLU activations and each
   conv's *input* spatial size (needed as `output_size` for
   `conv_transpose2d` on strided convs). Reconstruction: isolate one
   channel (and its strongest spatial location, or a given `spatial_pos`), then
   walk back layer→0 doing rectify + `conv_transpose2d` reusing the forward conv
   weights. **Unpooling is deliberately omitted** — the target CNN has no pooling.
3. **Occlusion sensitivity** (`occlusion_sensitivity`, `upsample_heatmap`) —
   builds one big batch with a gray patch slid to every position, one forward
   pass, reshapes results into per-action + value grids.
4. **Saliency / guided backprop** (`vanilla_saliency`, `guided_saliency`,
   `guided_backprop_context`) — not in the paper; gradient-based cross-check.
   `guided_backprop_context` adds full-backward hooks on every `nn.ReLU` that also
   zero negative gradients.
5. **Frame acquisition** — `live_frame_from_env` lazy-imports this repo's `envs/`
   package; `load_frame_from_npy` handles `(H,W,C)`→`(C,H,W)` and broadcasts a
   single channel to the 4-frame stack.
6. **Plotting** — Agg-backend matplotlib, lazy-imported inside each `plot_*`.

Constants at the top (`N_STACK=4`, `FRAME_SIZE=84`, `N_ACTIONS=3`) match the
target project's `NatureCNN`; `N_ACTIONS` is overridden automatically when
`--model` is supplied.

## Conventions

- Heavy dependencies (`stable_baselines3`, `gymnasium`, `vizdoom`, `matplotlib`,
  the target repo's `envs/`) are **imported lazily**, only in the function that
  needs them, so the standalone path stays dependency-light. Preserve this.
- "Class probability" from the paper maps to **action probability** (policy head)
  and **value estimate** (critic head) throughout.
- New model formats: add a `load_*_policy()` returning a `WrappedPolicy`; nothing
  else should need to change as long as the conv stack is `Conv2d`/`ReLU`-only.
