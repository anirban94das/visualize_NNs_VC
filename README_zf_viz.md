# zf_viz.py — Zeiler & Fergus visualization for the vizdoom PPO agent

Implements the diagnostic tools from Zeiler & Fergus, *Visualizing and
Understanding Convolutional Networks* (ECCV 2014), applied to
`vizdoom_clone_project_vibecoded`'s CNN policy instead of an ImageNet
classifier.

## What each technique shows, mapped to the paper

| Paper section | Technique | What it answers here |
|---|---|---|
| §2.1 (Fig. 1, 2, 4) | **Deconvnet** | "What input pattern caused this feature-map channel to fire?" Reconstructs a chosen conv-layer channel's activation back to pixel space. |
| §4.2 (Fig. 6) | **Occlusion sensitivity** | "Which part of the screen is the agent actually using?" Slides a gray patch over the frame and tracks how much each action's probability / the value estimate drops. |
| — (not in paper) | **Saliency / guided backprop** | Gradient-based alternative to deconvnet — cheaper, useful as a cross-check on a network this shallow. |

One deliberate deviation from the paper: the original deconvnet inverts
**max-pooling** using stored switches. This project's `NatureCNN` (three
strided convs, no pooling at all — see `visualize_PPO_model.py` in the repo)
has nothing to invert there, so that step is simply skipped; the rectify +
transposed-conv steps are implemented exactly as described.

"Class probability" in the paper becomes **action probability** (from the
policy head) and **value estimate** (from the critic head), since this is
an RL policy/value network, not a softmax classifier.

## Setup

Copy `zf_viz.py` into the repo root, next to `visualize_PPO_model.py`. It
only needs `torch`, `numpy`, and `matplotlib` — all already in
`requirements.txt`. `stable-baselines3` / `vizdoom` / `gymnasium` are
imported lazily, only if you use `--model` / `--live`.

## Quick check (no trained model, no vizdoom needed)

```
.venv\Scripts\python.exe zf_viz.py --technique all
```

Runs everything against a fresh untrained network and a random noise frame
— confirms the shapes line up before you point it at anything real. Output
goes to `viz_out/zf_*.png`.

## Against a trained model

```
.venv\Scripts\python.exe zf_viz.py --model models/latest/ppo_basic.zip --live basic --technique all --out viz_out/basic
```

- `--model` loads the real trained `CnnPolicy` (same loading path as
  `visualize_PPO_model.py`'s `--model` flag) so the action count and
  weights are correct instead of the `basic.wad` defaults.
- `--live basic` grabs one real frame from `envs/basic_env.py` (must be run
  from the repo root with the venv active, same as `train_basic.py`). Use
  `--live deadly_corridor` for that scenario.
- Swap `--technique all` for `deconv`, `occlusion`, `saliency`, or `guided`
  to run just one.

## Deconvnet on a specific channel

```
.venv\Scripts\python.exe zf_viz.py --model models/latest/ppo_basic.zip --live basic \
    --technique deconv --layer 2 --channel 10 --out viz_out/basic
```

`--layer` is 0, 1, or 2 (the three conv layers). `--channel` picks which of
that layer's feature maps (32 / 64 / 64 channels respectively) to
reconstruct. Without `--frames`, it reconstructs from the single `--image`
or `--live` frame's strongest activation location for that channel.

### Top-9 grid (matches paper Fig. 2 exactly)

The paper's headline visualization is the *top 9 activations across many
images* for one feature map, not just one frame. To reproduce that:

1. Capture a stash of frames during a `watch_agent.py` run — add one line
   dumping the raw `(84,84,1)` observation to a growing list, then
   `np.save("frames.npy", np.stack(frames))` on exit (shape `(N, 1, 84,
   84)`; the script will broadcast the single channel to the 4-frame stack
   automatically).
2. Then:

```
.venv\Scripts\python.exe zf_viz.py --model models/latest/ppo_basic.zip \
    --frames frames.npy --technique deconv --layer 2 --channel 10 --top-k 9 \
    --out viz_out/basic
```

This scans every frame for its strongest activation of that channel, keeps
the top 9, and reconstructs each — directly comparable to Fig. 2 in the
paper.

## Occlusion sensitivity

```
.venv\Scripts\python.exe zf_viz.py --model models/latest/ppo_basic.zip --live basic \
    --technique occlusion --patch-size 8 --stride 4 --out viz_out/basic
```

Produces one heatmap panel per action (probability drop when that region is
occluded) plus one for the value estimate, overlaid on the input frame —
this is the direct analogue of Fig. 6's `(d)` and `(e)` panels. Smaller
`--patch-size` / `--stride` gives a finer-grained but slower map.

## Saliency / guided backprop

```
.venv\Scripts\python.exe zf_viz.py --model models/latest/ppo_basic.zip --live basic \
    --technique guided --action 2 --out viz_out/basic
```

`--action` picks which action's logit to backprop from (default: whatever
the policy's current argmax action is). `guided` (guided backprop) usually
gives a visually cleaner map than `saliency` (vanilla gradient) — try both.

## Extending

Everything in `zf_viz.py` operates on plain `nn.Module` pieces
(`WrappedPolicy.cnn` + a `head` callable), not on stable-baselines3 types
directly, so it'll work on any Conv2d/ReLU-only conv stack (no pooling) —
including, per this project's stated end goal, other open-source model
architectures beyond this one, as long as you write a loader function like
`load_sb3_policy()` for the new model's format.
