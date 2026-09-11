"""Zeiler & Fergus (2014) "Visualizing and Understanding Convolutional
Networks" style diagnostics for this project's PPO CnnPolicy
(NatureCNN -> action_net / value_net, see visualize_PPO_model.py).

Three techniques, each a section of the paper:

- Deconvnet reconstruction (paper Sec 2.1 / Sec 3, Fig 2, Fig 4): pick one
  strong activation in one conv layer's feature map, zero everything else in
  that layer, and run the conv stack in reverse -- rectify, then filter with
  the *transposed* learned filters -- back down to input-pixel space. This
  net is three strided convs with no max-pooling (see NatureCNN below), so
  there is no "unpool via stored switches" step to implement; the paper's
  per-layer "unpool -> rectify -> filter" collapses to "rectify -> filter".
- Occlusion sensitivity (paper Sec 4.2, Fig 6): slide a gray patch over the
  input and rewatch how outputs move at every position. Adapted from "class
  probability" to this policy's two heads: per-action probability (from
  action_net, softmax) and the value estimate (from value_net).
- Saliency / guided backprop (paper Sec 3, "as a proxy"): gradient of a
  chosen output (an action's probability, or the value estimate) w.r.t. the
  input pixels. Guided backprop additionally zeros negative gradients at
  every ReLU on the way back (Springenberg et al. 2015, the paper this
  repo's task description cites for the exact hook semantics).

Architecture assumptions (verified against a real loaded model in this
repo's venv -- see README section "CNN diagnostics" for what was checked):
stable_baselines3 ActorCriticCnnPolicy with share_features_extractor=True
(the default), net_arch=[] (also the CnnPolicy default), so
mlp_extractor.policy_net / .value_net are both empty nn.Sequential()s --
i.e. action_net and value_net both read directly off the same 512-d NatureCNN
output, with no hidden MLP in between. pi_features_extractor and
vf_features_extractor are literally the same module when shared.

Usage mirrors visualize_PPO_model.py's conventions:

    python visualize_cnn_diagnostics.py
        # zero extra deps beyond torch/matplotlib: fresh untrained network,
        # random input frame, shape sanity-check only.

    python visualize_cnn_diagnostics.py --model models/latest/ppo_basic.zip --env basic
        # real trained model, a live frame played out of envs/basic_env.py.

Install requirements (same as visualize_PPO_model.py): torch, matplotlib.
--env / --model additionally need this project's own stable_baselines3 /
gymnasium / vizdoom stack (already in requirements.txt), imported lazily so
the zero-argument path never needs them.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_STACK = 4       # matches VecFrameStack(vec_env, n_stack=4) in train_common.py
FRAME_SIZE = 84    # matches envs/common.py's ResizeObservation(env, shape=(84, 84))
N_ACTIONS = 3      # standalone-mode default; basic.wad's real trained model has 4
                    # (MOVE_LEFT, MOVE_RIGHT, ATTACK, plus one more -- see README)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class DiagnosticPolicy(nn.Module):
    """The exact forward path this project's PPO CnnPolicy uses to produce
    both heads: NatureCNN (3 conv+ReLU, no pooling) -> 512-d Linear+ReLU ->
    action_net (policy logits) and value_net (scalar), reading the *same*
    512-d features (see module docstring re: net_arch=[]/shared extractor).
    Either wraps a real trained model's own layers, or a freshly initialized
    stand-in with identical shapes."""

    def __init__(self, cnn: nn.Sequential, linear: nn.Sequential, action_net: nn.Linear, value_net: nn.Linear):
        super().__init__()
        self.cnn = cnn          # Conv2d, ReLU, Conv2d, ReLU, Conv2d, ReLU, Flatten
        self.linear = linear    # Linear(3136, 512), ReLU
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.linear(self.cnn(x))
        return self.action_net(features), self.value_net(features)

    @property
    def conv_layers(self) -> list[nn.Conv2d]:
        return [m for m in self.cnn if isinstance(m, nn.Conv2d)]

    @property
    def relu_modules(self) -> list[nn.ReLU]:
        return [m for m in self.cnn if isinstance(m, nn.ReLU)] + [m for m in self.linear if isinstance(m, nn.ReLU)]


def build_fresh(n_stack: int = N_STACK, frame_size: int = FRAME_SIZE, n_actions: int = N_ACTIONS) -> DiagnosticPolicy:
    cnn = nn.Sequential(
        nn.Conv2d(n_stack, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )
    with torch.no_grad():
        n_flatten = cnn(torch.zeros(1, n_stack, frame_size, frame_size)).shape[1]
    linear = nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU())
    action_net = nn.Linear(512, n_actions)
    value_net = nn.Linear(512, 1)
    diag = DiagnosticPolicy(cnn, linear, action_net, value_net)
    diag.eval()
    return diag


def build_from_saved_model(model_path: str, device: str = "cpu") -> tuple[DiagnosticPolicy, tuple[int, int, int], int]:
    """Loads a trained model.zip and wraps its *actual* live layers (not
    copies) -- eval mode, gradients off on parameters. Returns the wrapped
    policy, its (n_stack, height, width) observation shape, and its action
    count (varies per scenario: 4 for basic, 8 for deadly_corridor, etc. --
    see visualize_PPO_model.py's note on this)."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device=device)
    policy = model.policy
    # pi_features_extractor == vf_features_extractor when share_features_extractor
    # (SB3's CnnPolicy default, confirmed against this repo's real ppo_basic.zip).
    fx = policy.pi_features_extractor
    diag = DiagnosticPolicy(fx.cnn, fx.linear, policy.action_net, policy.value_net).to(device)
    diag.eval()
    for p in diag.parameters():
        p.requires_grad_(False)
    n_stack, height, width = policy.observation_space.shape
    n_actions = int(policy.action_space.n)
    return diag, (n_stack, height, width), n_actions


# --------------------------------------------------------------------------
# Frame acquisition
# --------------------------------------------------------------------------

def to_frame01(raw: np.ndarray) -> torch.Tensor:
    """raw: (n_stack, H, W) or (H, W, n_stack) array, values in [0, 255].
    Returns a (n_stack, H, W) float32 tensor in [0, 1], matching SB3's own
    preprocess_obs (normalize_images=True -> obs.float() / 255.0)."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr / 255.0)


def make_env_factory(env_name: str):
    if env_name == "basic":
        from envs.basic_env import make_basic_env as factory
    elif env_name == "deadly_corridor":
        from envs.deadly_corridor_env import make_deadly_corridor_env as factory
    else:
        raise ValueError(f"Unknown --env {env_name!r}; expected 'basic' or 'deadly_corridor'")
    return factory


def grab_frames_from_env(env_name: str, n_frames: int, n_stack: int = N_STACK, seed: int | None = None) -> np.ndarray:
    """Plays the real scenario with random actions and returns a batch of
    stacked frames exactly as this project's own VecFrameStack pipeline
    would hand them to the policy (channel-first, uint8-range 0-255).
    Reuses envs/*_env.py + VecFrameStack rather than reimplementing the
    frame-stacking order by hand, so this is guaranteed to match training."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    factory = make_env_factory(env_name)
    vec_env = DummyVecEnv([lambda: factory(render_mode=None)])
    vec_env = VecFrameStack(vec_env, n_stack=n_stack)
    if seed is not None:
        try:
            vec_env.seed(seed)
        except AttributeError:
            pass

    obs = vec_env.reset()
    frames = []
    # Warm up n_stack steps so the stack isn't mostly reset-frame padding.
    for _ in range(n_stack):
        obs, _, dones, _ = vec_env.step([vec_env.action_space.sample()])
    frames.append(obs[0])
    while len(frames) < n_frames:
        obs, _, dones, _ = vec_env.step([vec_env.action_space.sample()])
        if dones[0]:
            obs = vec_env.reset()
        frames.append(obs[0])
    vec_env.close()

    batch = np.stack(frames, axis=0)  # (n_frames, H, W, n_stack) channel-last
    return np.transpose(batch, (0, 3, 1, 2)).astype(np.float32)  # -> (n_frames, n_stack, H, W)


# --------------------------------------------------------------------------
# 1. Deconvnet reconstruction (paper Sec 2.1, Sec 3)
# --------------------------------------------------------------------------

def _cnn_forward_with_intermediates(diag: DiagnosticPolicy, x: torch.Tensor):
    """One forward pass through the conv stack, recording each conv layer's
    post-ReLU activations and its *input* shape (needed below: a strided
    conv's output size alone doesn't uniquely determine its input size, so
    conv_transpose2d needs this recorded explicitly rather than inferred)."""
    post_relu = []
    input_shapes = []
    convs_and_relus = [m for m in diag.cnn if not isinstance(m, nn.Flatten)]
    feats = x
    for i in range(0, len(convs_and_relus), 2):
        conv, relu = convs_and_relus[i], convs_and_relus[i + 1]
        input_shapes.append(feats.shape)
        feats = relu(conv(feats))
        post_relu.append(feats)
    return post_relu, input_shapes


def _conv_transpose_like(conv: nn.Conv2d, x: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
    """Adjoint of conv (PyTorch does cross-correlation, not textbook
    convolution) via ConvTranspose2d sharing conv's own weight tensor -- this
    is mathematically exact in PyTorch's convention, so no manual kernel-flip
    is needed. output_size disambiguates the target spatial size (see
    _cnn_forward_with_intermediates); ConvTranspose2d.forward's output_size
    argument resolves the needed output_padding internally rather than us
    computing it by hand."""
    deconv = nn.ConvTranspose2d(
        in_channels=conv.out_channels,
        out_channels=conv.in_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False,
    ).to(x.device)
    with torch.no_grad():
        deconv.weight.copy_(conv.weight)
        return deconv(x, output_size=output_size[-2:])


def deconv_reconstruct(
    diag: DiagnosticPolicy,
    frame01: torch.Tensor,
    layer_idx: int,
    channel_idx: int,
    spatial_idx: tuple[int, int] | None = None,
) -> dict:
    """frame01: (n_stack, H, W) float tensor in [0, 1]. Zeros every
    activation in conv layer `layer_idx` except channel `channel_idx`'s
    strongest spatial location (or `spatial_idx` if given), then runs
    rectify -> transposed-filter back down through layers layer_idx..0 to
    input-pixel space. No unpooling step: this net has no max-pooling to
    invert (see module docstring)."""
    diag.eval()
    with torch.no_grad():
        x = frame01.unsqueeze(0)
        post_relu, input_shapes = _cnn_forward_with_intermediates(diag, x)
        conv_layers = diag.conv_layers

        act = post_relu[layer_idx]
        channel_map = act[0, channel_idx]
        if spatial_idx is None:
            flat_idx = int(torch.argmax(channel_map))
            row, col = divmod(flat_idx, channel_map.shape[-1])
        else:
            row, col = spatial_idx
        activation_value = channel_map[row, col].item()

        recon = torch.zeros_like(act)
        recon[0, channel_idx, row, col] = act[0, channel_idx, row, col]
        for l in range(layer_idx, -1, -1):
            recon = F.relu(recon)  # rectify (the deconvnet's own nonlinearity)
            recon = _conv_transpose_like(conv_layers[l], recon, input_shapes[l])

        return {
            "reconstruction": recon[0].cpu().numpy(),  # (n_stack, H, W), input-pixel space
            "row": row,
            "col": col,
            "activation_value": activation_value,
        }


def strongest_channel(diag: DiagnosticPolicy, frame01: torch.Tensor, layer_idx: int) -> tuple[int, float]:
    """Picks the channel with the largest activation at layer_idx for this
    frame. A fixed default channel (e.g. 0) is frequently dead (max
    activation exactly 0, a legitimate ReLU outcome for that input) -- the
    deconv reconstruction is correctly all-zero for a dead channel, but
    that's a degenerate pick, not a bug, so the CLI defaults to this instead
    of a fixed index."""
    diag.eval()
    with torch.no_grad():
        post_relu, _ = _cnn_forward_with_intermediates(diag, frame01.unsqueeze(0))
        per_channel_max = post_relu[layer_idx][0].flatten(1).amax(dim=1)  # (C,)
        channel_idx = int(torch.argmax(per_channel_max))
        return channel_idx, per_channel_max[channel_idx].item()


def deconv_topk(
    diag: DiagnosticPolicy,
    frames01: torch.Tensor,
    layer_idx: int,
    channel_idx: int,
    k: int,
) -> list[dict]:
    """The paper's signature figure (Fig 4): scan a batch of frames for the
    top-k strongest activations of one channel at one layer, and reconstruct
    each in input-pixel space."""
    diag.eval()
    with torch.no_grad():
        post_relu, _ = _cnn_forward_with_intermediates(diag, frames01)
        channel_maps = post_relu[layer_idx][:, channel_idx]  # (N, h, w)
        per_frame_max = channel_maps.flatten(1).amax(dim=1)  # (N,)
        k = min(k, frames01.shape[0])
        top_values, top_indices = torch.topk(per_frame_max, k)

    results = []
    for value, idx in zip(top_values.tolist(), top_indices.tolist()):
        recon = deconv_reconstruct(diag, frames01[idx], layer_idx, channel_idx)
        results.append({"frame_index": idx, "max_activation": value, **recon})
    return results


# --------------------------------------------------------------------------
# 2. Occlusion sensitivity (paper Sec 4.2, Fig 6)
# --------------------------------------------------------------------------

def occlusion_sensitivity(
    diag: DiagnosticPolicy,
    frame01: torch.Tensor,
    patch_size: int = 12,
    stride: int = 4,
    gray_value: float = 0.5,
    batch_size: int = 256,
) -> dict:
    """Slides a gray patch over frame01 (n_stack, H, W), occluding the same
    spatial region across all stacked frames at once (a diagnostic choice:
    occluding only the newest frame barely moves the output, since the
    stacked frames mostly show the same static scene -- see README). Every
    position is vectorized into one batched forward pass per action_net
    logits and value_net. Output: per-action probability *drop* and value
    *drop* at each position (positive = occluding this patch hurt that
    output -- i.e. the model was relying on that region)."""
    diag.eval()
    n_stack, H, W = frame01.shape
    device = next(diag.parameters()).device

    with torch.no_grad():
        logits0, value0 = diag(frame01.unsqueeze(0).to(device))
        probs0 = F.softmax(logits0, dim=-1)[0].cpu()
        value0 = value0[0, 0].cpu()

    rows = list(range(0, H - patch_size + 1, stride))
    cols = list(range(0, W - patch_size + 1, stride))
    positions = [(r, c) for r in rows for c in cols]
    n_pos = len(positions)

    occluded = frame01.unsqueeze(0).repeat(n_pos, 1, 1, 1).clone()
    for i, (r, c) in enumerate(positions):
        occluded[i, :, r : r + patch_size, c : c + patch_size] = gray_value

    probs, values = [], []
    with torch.no_grad():
        for i in range(0, n_pos, batch_size):
            logits, vals = diag(occluded[i : i + batch_size].to(device))
            probs.append(F.softmax(logits, dim=-1).cpu())
            values.append(vals.cpu())
    probs = torch.cat(probs, dim=0)
    values = torch.cat(values, dim=0)[:, 0]

    grid_h, grid_w = len(rows), len(cols)
    n_actions = probs0.shape[0]
    prob_drop = (probs0.unsqueeze(0) - probs).reshape(grid_h, grid_w, n_actions).numpy()
    value_drop = (value0 - values).reshape(grid_h, grid_w).numpy()

    return {
        "baseline_probs": probs0.numpy(),
        "baseline_value": value0.item(),
        "prob_drop": prob_drop,     # (grid_h, grid_w, n_actions)
        "value_drop": value_drop,   # (grid_h, grid_w)
        "grid_shape": (grid_h, grid_w),
        "patch_size": patch_size,
        "stride": stride,
    }


def upsample_grid(grid: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    t = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


# --------------------------------------------------------------------------
# 3. Saliency / guided backprop (paper Sec 3, as a cheaper proxy)
# --------------------------------------------------------------------------

def _guided_relu_hook(module, grad_input, grad_output):
    """Guided backprop (Springenberg et al. 2015): let through only gradient
    that is both (a) flowing back through a unit that was positive on the
    forward pass -- already encoded in grad_input by autograd's normal ReLU
    backward -- and (b) itself positive coming in. grad_input[0] already
    carries mask (a); multiplying by (grad_output[0] > 0) adds mask (b).
    Do NOT use relu(grad_output) -- that drops mask (a) entirely."""
    return (grad_input[0] * (grad_output[0] > 0).type_as(grad_input[0]),)


def saliency_map(
    diag: DiagnosticPolicy,
    frame01: torch.Tensor,
    target: str = "action",
    action_idx: int | None = None,
    guided: bool = False,
) -> dict:
    """Gradient of a chosen scalar output w.r.t. input pixels. target="action"
    backprops from that action's softmax probability (argmax action if
    action_idx is None); target="value" backprops from the critic's scalar
    value estimate -- the paper's "class probability" doesn't exist for a
    PPO policy, so these are the two substitutes named in the task."""
    diag.eval()
    device = next(diag.parameters()).device

    hooks = []
    if guided:
        hooks = [m.register_full_backward_hook(_guided_relu_hook) for m in diag.relu_modules]

    try:
        x = frame01.clone().unsqueeze(0).to(device)
        x.requires_grad_(True)

        logits, value = diag(x)
        if target == "value":
            scalar = value[0, 0]
            resolved_action_idx = None
        else:
            probs = F.softmax(logits, dim=-1)
            resolved_action_idx = int(torch.argmax(probs[0]).item()) if action_idx is None else action_idx
            scalar = probs[0, resolved_action_idx]

        scalar.backward()
        grad = x.grad[0].detach().cpu()
    finally:
        for h in hooks:
            h.remove()

    heatmap = grad.abs().amax(dim=0).numpy()  # reduce the n_stack channel dim -> (H, W)
    return {"heatmap": heatmap, "action_idx": resolved_action_idx, "target_value": scalar.item()}


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _normalize01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _background(frame01: torch.Tensor) -> np.ndarray:
    return frame01[-1].numpy()  # most recent frame in the stack


def plot_deconv(result: dict, frame01: torch.Tensor, layer_idx: int, channel_idx: int, out_path: Path) -> None:
    recon = np.abs(result["reconstruction"]).sum(axis=0)  # collapse n_stack -> (H, W)
    recon_vis = _normalize01(recon)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(_background(frame01), cmap="gray")
    axes[0].set_title("input frame")
    axes[0].axis("off")
    axes[1].imshow(recon_vis, cmap="inferno")
    axes[1].set_title(
        f"deconv: layer {layer_idx} ch {channel_idx}\n"
        f"act={result['activation_value']:.3f} @ ({result['row']},{result['col']})"
    )
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[deconv] layer={layer_idx} channel={channel_idx} activation={result['activation_value']:.4f} "
          f"recon stats: min={recon.min():.4f} max={recon.max():.4f} mean={recon.mean():.4f}")
    print(f"[deconv] saved: {out_path}")


def plot_deconv_topk(results: list[dict], out_path: Path, layer_idx: int, channel_idx: int) -> None:
    k = len(results)
    cols = min(k, 3)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, r in zip(axes, results):
        recon = _normalize01(np.abs(r["reconstruction"]).sum(axis=0))
        ax.imshow(recon, cmap="inferno")
        ax.set_title(f"frame {r['frame_index']}\nact={r['max_activation']:.3f}", fontsize=9)
        ax.axis("off")
    for ax in axes[k:]:
        ax.axis("off")
    fig.suptitle(f"top-{k} activations: layer {layer_idx} channel {channel_idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[deconv-topk] saved: {out_path}")


def plot_occlusion(result: dict, frame01: torch.Tensor, out_path: Path, action_labels: list[str] | None = None) -> None:
    n_actions = result["prob_drop"].shape[-1]
    labels = action_labels or [f"action {i}" for i in range(n_actions)]
    n_panels = n_actions + 1
    cols = min(n_panels, 4)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    bg = _background(frame01)

    for i in range(n_actions):
        heat = upsample_grid(result["prob_drop"][:, :, i])
        vmax = np.abs(heat).max() or 1.0
        ax = axes[i]
        ax.imshow(bg, cmap="gray")
        im = ax.imshow(heat, cmap="coolwarm", alpha=0.55, vmin=-vmax, vmax=vmax)
        ax.set_title(f"{labels[i]}\nP={result['baseline_probs'][i]:.3f}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        print(f"[occlusion] {labels[i]}: prob_drop min={heat.min():.4f} max={heat.max():.4f} "
              f"mean={heat.mean():.4f} std={heat.std():.4f}")

    value_heat = upsample_grid(result["value_drop"])
    vmax = np.abs(value_heat).max() or 1.0
    ax = axes[n_actions]
    ax.imshow(bg, cmap="gray")
    im = ax.imshow(value_heat, cmap="coolwarm", alpha=0.55, vmin=-vmax, vmax=vmax)
    ax.set_title(f"value\nV={result['baseline_value']:.3f}", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    print(f"[occlusion] value: value_drop min={value_heat.min():.4f} max={value_heat.max():.4f} "
          f"mean={value_heat.mean():.4f} std={value_heat.std():.4f}")

    for ax in axes[n_panels:]:
        ax.axis("off")
    fig.suptitle(f"occlusion sensitivity (patch={result['patch_size']}, stride={result['stride']})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[occlusion] saved: {out_path}")


def plot_saliency(result: dict, frame01: torch.Tensor, out_path: Path, tag: str) -> None:
    heat = result["heatmap"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(_background(frame01), cmap="gray")
    axes[0].set_title("input frame")
    axes[0].axis("off")
    axes[1].imshow(_background(frame01), cmap="gray")
    axes[1].imshow(_normalize01(heat), cmap="inferno", alpha=0.6)
    title = f"{tag}: action {result['action_idx']}" if result["action_idx"] is not None else f"{tag}: value"
    axes[1].set_title(f"{title}\noutput={result['target_value']:.3f}")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{tag}] output={result['target_value']:.4f} heatmap stats: "
          f"min={heat.min():.4f} max={heat.max():.4f} mean={heat.mean():.4f} std={heat.std():.4f}")
    print(f"[{tag}] saved: {out_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None,
                         help="Path to a saved PPO model, e.g. models/latest/ppo_basic.zip. "
                              "If omitted, uses a fresh untrained network with basic.wad's default shapes.")
    parser.add_argument("--env", type=str, default=None, choices=["basic", "deadly_corridor"],
                         help="Grab a live frame (or frame batch) by playing this scenario with random actions.")
    parser.add_argument("--frame", type=str, default=None,
                         help="Path to a saved .npy single frame (n_stack, H, W) or (H, W, n_stack), values 0-255.")
    parser.add_argument("--frames", type=str, default=None,
                         help="Path to a saved .npy batch of frames (N, n_stack, H, W) or (N, H, W, n_stack), "
                              "for the deconv top-k scan.")
    parser.add_argument("--technique", nargs="+", default=["all"],
                         choices=["deconv", "occlusion", "saliency", "guided", "all"],
                         help="Which technique(s) to run.")
    parser.add_argument("--layer", type=int, default=2, choices=[0, 1, 2],
                         help="Conv layer index for deconv (0=first/8x8, 1=second/4x4, 2=third/3x3).")
    parser.add_argument("--channel", type=int, default=None,
                         help="Feature-map channel index for deconv. Default: auto-pick the channel with the "
                              "strongest activation at --layer for the acquired frame (a fixed default channel "
                              "is often dead -- all-zero max activation -- for a given frame; deconv on a dead "
                              "channel legitimately reconstructs to all-zero, which isn't a bug but isn't useful).")
    parser.add_argument("--topk", type=int, default=9,
                         help="If a frame batch is available (--frames, or --env with --n-live-frames > 1), "
                              "run the top-k activation scan instead of a single-frame reconstruction.")
    parser.add_argument("--n-live-frames", type=int, default=1,
                         help="How many frames to collect when --env is given without --frames/--frame. "
                              ">1 enables the deconv top-k scan.")
    parser.add_argument("--action-idx", type=int, default=None,
                         help="Action index to target for saliency/guided (default: argmax action).")
    parser.add_argument("--patch-size", type=int, default=12, help="Occlusion patch size in pixels.")
    parser.add_argument("--patch-stride", type=int, default=4, help="Occlusion sliding stride in pixels.")
    parser.add_argument("--gray-value", type=float, default=0.5, help="Occlusion patch fill value, normalized [0,1].")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the live env and the standalone random frame.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out-dir", type=str, default="cnn_diagnostics_out", help="Output directory for PNGs.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- build the policy -------------------------------------------------
    if args.model:
        print(f"Loading real model from: {args.model}")
        diag, (n_stack, h, w), n_actions = build_from_saved_model(args.model, device=args.device)
    else:
        print(f"No --model given: fresh untrained network ({N_STACK}x{FRAME_SIZE}x{FRAME_SIZE} -> "
              f"Discrete({N_ACTIONS})).")
        diag = build_fresh().to(args.device)
        n_stack, h, w = N_STACK, FRAME_SIZE, FRAME_SIZE
        n_actions = N_ACTIONS

    # --- acquire frame(s) ---------------------------------------------------
    frames_batch = None  # (N, n_stack, H, W) 0-255, only populated for topk scans
    if args.frames:
        raw = np.load(args.frames)
        if raw.shape[-1] in (n_stack,) and raw.shape[1] != n_stack:
            raw = np.transpose(raw, (0, 3, 1, 2))
        frames_batch = raw.astype(np.float32)
        frame01 = to_frame01(frames_batch[0])
        print(f"Loaded frame batch from {args.frames}: shape={frames_batch.shape}")
    elif args.frame:
        raw = np.load(args.frame)
        frame01 = to_frame01(raw)
        print(f"Loaded single frame from {args.frame}: shape={raw.shape}")
    elif args.env:
        n = max(args.n_live_frames, args.topk) if args.n_live_frames > 1 else 1
        raw_batch = grab_frames_from_env(args.env, n_frames=n, n_stack=n_stack, seed=args.seed)
        print(f"Grabbed {raw_batch.shape[0]} live frame(s) from '{args.env}'.")
        if raw_batch.shape[0] > 1:
            frames_batch = raw_batch
        frame01 = to_frame01(raw_batch[0])
    else:
        print("No --frame/--frames/--env given: using a random frame for a shape sanity-check.")
        frame01 = torch.rand(n_stack, h, w)

    frame01 = frame01.to(args.device)
    print(f"Frame stats: shape={tuple(frame01.shape)} min={frame01.min():.3f} max={frame01.max():.3f} "
          f"mean={frame01.mean():.3f}")

    with torch.no_grad():
        logits, value = diag(frame01.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    print(f"Baseline: action_probs={np.round(probs, 4).tolist()} value={value.item():.4f}")

    techniques = set(args.technique)
    if "all" in techniques:
        techniques = {"deconv", "occlusion", "saliency", "guided"}

    # --- deconvnet ----------------------------------------------------------
    if "deconv" in techniques:
        channel = args.channel
        if channel is None:
            channel, best_activation = strongest_channel(diag, frame01, args.layer)
            print(f"[deconv] --channel not given: auto-picked channel {channel} "
                  f"(max activation {best_activation:.4f} at layer {args.layer} for this frame)")
        if frames_batch is not None and frames_batch.shape[0] > 1:
            # Both ingestion paths (--frames, --env) already normalize to
            # (N, n_stack, H, W), so a straight /255.0 is all that's needed.
            batch01 = torch.from_numpy(frames_batch / 255.0).float().to(args.device)
            results = deconv_topk(diag, batch01, args.layer, channel, args.topk)
            plot_deconv_topk(results, out_dir / f"deconv_topk_layer{args.layer}_ch{channel}.png",
                              args.layer, channel)
        else:
            result = deconv_reconstruct(diag, frame01, args.layer, channel)
            plot_deconv(result, frame01.cpu(), args.layer, channel,
                        out_dir / f"deconv_layer{args.layer}_ch{channel}.png")

    # --- occlusion sensitivity ------------------------------------------------
    if "occlusion" in techniques:
        result = occlusion_sensitivity(diag, frame01, patch_size=args.patch_size, stride=args.patch_stride,
                                        gray_value=args.gray_value)
        plot_occlusion(result, frame01.cpu(), out_dir / "occlusion.png")

    # --- saliency / guided backprop -------------------------------------------
    if "saliency" in techniques:
        r_action = saliency_map(diag, frame01, target="action", action_idx=args.action_idx, guided=False)
        plot_saliency(r_action, frame01.cpu(), out_dir / "saliency_action.png", "saliency")
        r_value = saliency_map(diag, frame01, target="value", guided=False)
        plot_saliency(r_value, frame01.cpu(), out_dir / "saliency_value.png", "saliency")

    if "guided" in techniques:
        r_action = saliency_map(diag, frame01, target="action", action_idx=args.action_idx, guided=True)
        plot_saliency(r_action, frame01.cpu(), out_dir / "guided_action.png", "guided")
        r_value = saliency_map(diag, frame01, target="value", guided=True)
        plot_saliency(r_value, frame01.cpu(), out_dir / "guided_value.png", "guided")

    print(f"Done. Outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
