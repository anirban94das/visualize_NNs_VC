"""zf_viz.py -- Zeiler & Fergus (2014) style CNN visualization, drag-and-drop.

Implements the three diagnostic tools from "Visualizing and Understanding
Convolutional Networks" (Zeiler & Fergus, ECCV 2014), adapted to a
policy/value network instead of a softmax classifier:

  1. Deconvnet feature visualization (Section 2.1 / Fig. 2)
     Projects a chosen conv-layer activation back to input-pixel space by
     running it backwards through the network: rectify -> transposed-conv
     (using the *same* filter weights as the forward conv, per layer) ->
     repeat down to layer 0. The paper also unpools using stored max-pool
     switches; the network this file targets (NatureCNN: three strided
     convs, no max-pooling at all) has no pooling layers, so that step is
     simply omitted -- everything else follows the paper exactly.

  2. Occlusion sensitivity (Section 4.2 / Fig. 6)
     Slides a gray patch over the input frame, reruns the network at every
     position, and records how much each action's probability (and the
     value estimate) drops. In the paper this used class probability; here
     "class" is replaced with "action" (policy branch) and "value estimate"
     (critic branch), since this is an RL policy, not a classifier.

  3. Vanilla saliency + guided backprop (not in the original paper, but the
     same "what pixels matter" question via gradients instead of a
     deconvnet -- included because it's ~10 lines on top of what's here
     already and is a useful cross-check when deconv reconstructions look
     noisy on a network this shallow).

Designed against this project's NatureCNN + PPO CnnPolicy shapes
(4-frame stack, 84x84 grayscale, stable-baselines3), but every function
takes plain nn.Module pieces, so it works with any conv-stack + head(s)
network as long as the conv stack is Conv2d/ReLU pairs with no pooling.

Usage
-----
Copy this file into the vizdoom_clone_project_vibecoded repo root (next to
visualize_PPO_model.py) and run, e.g.:

    python zf_viz.py --model models/latest/ppo_basic.zip \
        --live basic --technique all --out viz_out/basic

    python zf_viz.py --technique deconv --layer 2 --channel 5 \
        --out viz_out/synthetic

See the bottom of this file / --help for all options, and README_zf_viz.md
(if present) for a fuller walkthrough.

Requires: torch (always). stable-baselines3, gymnasium, vizdoom, and this
repo's envs/ package are only imported lazily, and only if you pass
--model and/or --live -- the tool runs standalone (on a random or
synthetic frame) without any of them installed, e.g. for a quick shape
sanity-check.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Defaults matching this project's NatureCNN (see visualize_PPO_model.py)
# ---------------------------------------------------------------------------
N_STACK = 4
FRAME_SIZE = 84
N_ACTIONS = 3  # basic.wad default; overridden automatically when --model is given


# ===========================================================================
# 1. Network wrapping: turn "a conv stack + head(s)" into one object that
#    exposes both an inference API (for occlusion) and per-layer hooks (for
#    deconv/saliency), regardless of whether it came from a saved SB3 model
#    or a freshly constructed synthetic network.
# ===========================================================================

class SyntheticNatureCNNPolicy(nn.Module):
    """Untrained network with this project's exact NatureCNN + PPO head shapes.

    Mirrors stable_baselines3.common.torch_layers.NatureCNN followed by PPO's
    action_net / value_net, for when no trained --model is supplied (SB3's
    default net_arch=[] means there's no hidden MLP between the CNN's 512-d
    output and the heads, so this reproduces that path exactly).
    """

    def __init__(self, n_stack: int = N_STACK, frame_size: int = FRAME_SIZE, n_actions: int = N_ACTIONS):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, n_stack, frame_size, frame_size)).flatten(1).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU())
        self.action_net = nn.Linear(512, n_actions)
        self.value_net = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.cnn(x).flatten(1)
        feats = self.linear(feats)
        logits = self.action_net(feats)
        value = self.value_net(feats)
        return {"action_logits": logits, "action_probs": F.softmax(logits, dim=-1), "value": value}


@dataclass
class WrappedPolicy:
    """Uniform view over a policy network, whatever its source.

    cnn: nn.Sequential of alternating Conv2d/ReLU (no pooling, no flatten).
    head: callable mapping the flattened CNN feature vector -> dict with
        "action_logits", "action_probs", "value".
    obs_shape: (C, H, W) expected input shape.
    """

    cnn: nn.Sequential
    head: "callable[[torch.Tensor], dict[str, torch.Tensor]]"
    obs_shape: tuple[int, int, int]

    def conv_layers(self) -> list[nn.Conv2d]:
        return [m for m in self.cnn if isinstance(m, nn.Conv2d)]

    def forward_full(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.cnn(x).flatten(1)
        return self.head(feats)

    def eval(self) -> "WrappedPolicy":
        self.cnn.eval()
        return self


def wrap_synthetic(model: SyntheticNatureCNNPolicy) -> WrappedPolicy:
    def head(feats: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = model.linear(feats)
        logits = model.action_net(feats)
        value = model.value_net(feats)
        return {"action_logits": logits, "action_probs": F.softmax(logits, dim=-1), "value": value}

    obs_shape = (model.cnn[0].in_channels, FRAME_SIZE, FRAME_SIZE)
    return WrappedPolicy(cnn=model.cnn, head=head, obs_shape=obs_shape)


def load_sb3_policy(model_path: str, device: str = "cpu") -> WrappedPolicy:
    """Load a saved stable-baselines3 PPO model.zip and extract its CNN +
    both heads (policy/action and value branches)."""
    from stable_baselines3 import PPO  # lazy import, only needed here

    sb3_model = PPO.load(model_path, device=device)
    policy = sb3_model.policy
    policy.eval()

    fe = policy.pi_features_extractor  # NatureCNN: .cnn (Conv/ReLU x3, Flatten), .linear
    # fe.cnn ends in nn.Flatten(); strip it so WrappedPolicy.cnn is pure Conv/ReLU,
    # and forward_full() does the flatten itself (needed so hooks below only ever
    # see Conv2d/ReLU modules).
    conv_relu_only = nn.Sequential(*[m for m in fe.cnn if not isinstance(m, nn.Flatten)])

    def head(feats: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = fe.linear(feats)
        pi_latent = policy.mlp_extractor.policy_net(latent)
        vf_latent = policy.mlp_extractor.value_net(latent)
        logits = policy.action_net(pi_latent)
        value = policy.value_net(vf_latent)
        return {"action_logits": logits, "action_probs": F.softmax(logits, dim=-1), "value": value}

    obs_shape = tuple(policy.observation_space.shape)  # (C, H, W); SB3 wraps obs channel-first
    return WrappedPolicy(cnn=conv_relu_only, head=head, obs_shape=obs_shape)


def preprocess(image: torch.Tensor) -> torch.Tensor:
    """Match SB3's default image preprocessing (uint8 [0,255] -> float [0,1]).
    No-op if the tensor is already float in [0,1]."""
    if image.dtype == torch.uint8 or image.max() > 1.5:
        image = image.float() / 255.0
    return image.float()


# ===========================================================================
# 2. Deconvnet feature visualization (paper Section 2.1, Fig. 1 top/Fig. 2)
# ===========================================================================

class ActivationRecorder:
    """Forward hooks that record, for every Conv2d in a Conv/ReLU-only
    nn.Sequential: (a) the spatial (H, W) size of its *input* -- needed as
    output_size when inverting that conv with conv_transpose2d, since a
    strided conv's output size alone doesn't uniquely determine it -- and
    (b) the post-ReLU activation map that layer produced.

    No max-pool switches are recorded because this architecture has no
    pooling layers; if you point this at a network that *does* have
    max-pooling, add unpooling here per the paper (store argmax indices in
    the forward pass, scatter back in reverse) -- deconv_reconstruct()
    below only handles the rectify+filter steps.
    """

    def __init__(self, cnn: nn.Sequential):
        self.cnn = cnn
        self.input_sizes: list[tuple[int, int]] = []
        self.activations: list[torch.Tensor] = []
        self._handles = []

    def __enter__(self) -> "ActivationRecorder":
        self.input_sizes = []
        self.activations = []
        convs = [m for m in self.cnn if isinstance(m, nn.Conv2d)]
        relus = [m for m in self.cnn if isinstance(m, nn.ReLU)]
        assert len(convs) == len(relus), "expected alternating Conv2d/ReLU with no pooling"

        for conv in convs:
            self._handles.append(conv.register_forward_pre_hook(
                lambda mod, inp: self.input_sizes.append(tuple(inp[0].shape[-2:]))
            ))
        for relu in relus:
            self._handles.append(relu.register_forward_hook(
                lambda mod, inp, out: self.activations.append(out.detach())
            ))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def deconv_reconstruct(
    policy: WrappedPolicy,
    image: torch.Tensor,
    layer_idx: int,
    channel: int,
    spatial_pos: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Reconstruct, in input-pixel space, the pattern that caused a chosen
    feature map's activation at `layer_idx` / `channel`.

    Steps (paper Section 2.1, minus unpooling -- see ActivationRecorder):
      1. Forward-pass `image` through the conv stack, recording per-layer
         activations and input sizes.
      2. Take the activation map at `layer_idx`, zero every channel except
         `channel`. If `spatial_pos` is None, additionally keep only that
         channel's single strongest spatial location (this is what the
         paper does per-visualization in Fig. 2/4 -- "the strongest
         activation ... within a given feature map"); otherwise keep only
         the given (y, x).
      3. Walk back from `layer_idx` to layer 0: rectify (ReLU), then filter
         with conv_transpose2d reusing that layer's *forward* conv weights
         (PyTorch's conv2d is cross-correlation, so ConvTranspose2d with the
         same weight tensor is exactly its adjoint -- no manual kernel
         flipping needed).

    Returns a (1, C, H, W) tensor in input-pixel space (same shape as
    `image`), not renormalized -- caller decides how to display it (e.g.
    clip to [0,1] and take a per-pixel max/mean across the C=4 stacked
    frames to render as a single grayscale image).
    """
    conv_layers = policy.conv_layers()
    assert 0 <= layer_idx < len(conv_layers)

    with ActivationRecorder(policy.cnn) as rec:
        with torch.no_grad():
            policy.cnn(image)

    a = rec.activations[layer_idx].clone()
    masked = torch.zeros_like(a)
    if spatial_pos is None:
        chan = a[0, channel]
        flat_idx = int(torch.argmax(chan).item())
        y, x = divmod(flat_idx, chan.shape[-1])
    else:
        y, x = spatial_pos
    masked[0, channel, y, x] = a[0, channel, y, x]
    a = masked

    with torch.no_grad():
        for i in range(layer_idx, -1, -1):
            a = F.relu(a)
            conv = conv_layers[i]
            out_size = rec.input_sizes[i]
            a = F.conv_transpose2d(
                a, weight=conv.weight, stride=conv.stride, padding=conv.padding, output_size=out_size
            )
    return a


def find_top_k_activations(
    policy: WrappedPolicy, frames: torch.Tensor, layer_idx: int, channel: int, k: int = 9
) -> list[tuple[int, int, int, float]]:
    """Scan a batch of frames for the top-k strongest activations of one
    feature map (paper Fig. 2: "top 9 activations ... across the validation
    data"). frames: (N, C, H, W). Returns [(frame_idx, y, x, value), ...]
    sorted descending by value -- feed each into deconv_reconstruct(...,
    spatial_pos=(y, x)) on frames[frame_idx:frame_idx+1] to get the actual
    reconstructions.
    """
    results: list[tuple[int, int, int, float]] = []
    with ActivationRecorder(policy.cnn) as rec, torch.no_grad():
        for n in range(frames.shape[0]):
            rec.activations = []
            policy.cnn(frames[n : n + 1])
            act = rec.activations[layer_idx][0, channel]
            flat_idx = int(torch.argmax(act).item())
            y, x = divmod(flat_idx, act.shape[-1])
            results.append((n, y, x, float(act[y, x].item())))
    results.sort(key=lambda t: t[3], reverse=True)
    return results[:k]


# ===========================================================================
# 3. Occlusion sensitivity (paper Section 4.2, Fig. 6)
# ===========================================================================

def occlusion_sensitivity(
    policy: WrappedPolicy,
    image: torch.Tensor,
    patch_size: int = 8,
    stride: int = 4,
    occlusion_value: float = 0.5,
) -> dict[str, np.ndarray]:
    """Slide a gray patch over `image` (1, C, H, W), rerun the network at
    every position, and record how far each action's probability and the
    value estimate drop relative to the unoccluded frame.

    Returns a dict with:
      "action_probs": (n_actions, n_pos_h, n_pos_w) grid of probabilities
      "value": (n_pos_h, n_pos_w) grid of value estimates
      "positions": list of (y, x) top-left patch corners, row-major, matching
                   the grid's flattened order
    All grids are in "sliding-window index" resolution; use
    upsample_heatmap() to stretch to the input's H x W for overlay plotting.
    """
    _, C, H, W = image.shape
    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))

    batch = image.repeat(len(ys) * len(xs), 1, 1, 1).clone()
    positions = []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            idx = i * len(xs) + j
            batch[idx, :, y : y + patch_size, x : x + patch_size] = occlusion_value
            positions.append((y, x))

    with torch.no_grad():
        out = policy.forward_full(batch)

    n_actions = out["action_probs"].shape[-1]
    action_probs = out["action_probs"].reshape(len(ys), len(xs), n_actions).permute(2, 0, 1).numpy()
    value = out["value"].reshape(len(ys), len(xs)).numpy()
    return {"action_probs": action_probs, "value": value, "positions": positions, "ys": ys, "xs": xs}


def upsample_heatmap(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-style upsample of a small (h, w) sensitivity grid to full
    frame resolution for overlaying on the input image."""
    t = torch.from_numpy(grid).float()[None, None]
    up = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


# ===========================================================================
# 4. Saliency / guided backprop (bonus, not in the original paper)
# ===========================================================================

def vanilla_saliency(
    policy: WrappedPolicy, image: torch.Tensor, target: str = "action", action_idx: int | None = None
) -> torch.Tensor:
    """d(output)/d(input), i.e. the classic gradient saliency map. target is
    "action" (uses action_logits[action_idx], or the argmax action if
    action_idx is None) or "value" (the critic's output)."""
    image = image.clone().requires_grad_(True)
    out = policy.forward_full(image)
    if target == "value":
        scalar = out["value"].sum()
    else:
        idx = action_idx if action_idx is not None else int(out["action_logits"].argmax(-1).item())
        scalar = out["action_logits"][0, idx]
    policy.cnn.zero_grad(set_to_none=True)
    scalar.backward()
    return image.grad.detach()


class guided_backprop_context:
    """Context manager: while active, every nn.ReLU in `cnn` also zeroes
    negative *gradients* on the backward pass (not just negative inputs, as
    plain ReLU already does) -- the "guided backprop" trick (Springenberg et
    al. 2014) that tends to give cleaner saliency maps than vanilla gradients
    on small networks. Hooks are removed on exit, so this is safe to use
    around a single backward() call and nothing else.
    """

    def __init__(self, cnn: nn.Sequential):
        self.cnn = cnn
        self._handles = []

    def __enter__(self) -> "guided_backprop_context":
        def hook(module, grad_input, grad_output):
            # Standard ReLU backward already zeroes grad_input where the
            # forward input was <=0 (grad_input[0] == grad_output[0] *
            # (input>0)); guided backprop additionally zeroes it wherever
            # the incoming gradient itself is negative. Multiplying
            # grad_input[0] (which already carries the input>0 mask) by
            # (grad_output[0] > 0) applies both conditions at once.
            return (grad_input[0] * (grad_output[0] > 0).float(),)

        for m in self.cnn:
            if isinstance(m, nn.ReLU):
                self._handles.append(m.register_full_backward_hook(hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def guided_saliency(
    policy: WrappedPolicy, image: torch.Tensor, target: str = "action", action_idx: int | None = None
) -> torch.Tensor:
    with guided_backprop_context(policy.cnn):
        return vanilla_saliency(policy, image, target=target, action_idx=action_idx)


# ===========================================================================
# 5. Frame acquisition helpers
# ===========================================================================

def random_demo_frame() -> torch.Tensor:
    """Shape-correct random noise frame, for running this tool with zero
    dependencies beyond torch (sanity-checking the code, not a real
    visualization)."""
    return torch.rand(1, N_STACK, FRAME_SIZE, FRAME_SIZE)


def live_frame_from_env(scenario: str) -> torch.Tensor:
    """Grab one real frame from this project's actual vizdoom env, stacked
    N_STACK times (a single reset observation, not a true rolling stack --
    good enough to see what the network is looking at; for a real in-episode
    frame stack, capture inside watch_agent.py instead and save with
    np.save). Requires vizdoom/gymnasium and this repo's envs/ package to be
    importable, i.e. run this from the repo root with its venv active.
    """
    if scenario == "basic":
        from envs.basic_env import make_basic_env as make_env
    elif scenario == "deadly_corridor":
        from envs.deadly_corridor_env import make_deadly_corridor_env as make_env
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    env = make_env()
    obs, _ = env.reset()  # (84, 84, 1) uint8
    env.close()
    frame = torch.from_numpy(np.asarray(obs)).permute(2, 0, 1).float()  # (1, 84, 84)
    return frame.repeat(N_STACK, 1, 1).unsqueeze(0)  # (1, N_STACK, 84, 84)


def load_frame_from_npy(path: str) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[-1] in (1, N_STACK):  # (H, W, C) -> (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
    if arr.shape[0] == 1 and N_STACK > 1:
        arr = np.repeat(arr, N_STACK, axis=0)
    t = torch.from_numpy(arr).float()
    return t.unsqueeze(0) if t.ndim == 3 else t


# ===========================================================================
# 6. Plotting
# ===========================================================================

def _to_displayable(img_1chw: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) -> (H, W) grayscale for imshow: max across the stacked
    frames, then min-max normalized to [0, 1]."""
    arr = img_1chw[0].detach().numpy()
    arr = arr.max(axis=0)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def plot_deconv_grid(recons: list[torch.Tensor], titles: list[str], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(recons)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, recon, title in zip(axes, recons, titles):
        ax.imshow(_to_displayable(recon), cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(recons):]:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_occlusion_overlay(
    image: torch.Tensor, occ: dict[str, np.ndarray], out_path: str, action_names: list[str] | None = None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = image.shape[-2:]
    base = _to_displayable(image)
    n_actions = occ["action_probs"].shape[0]
    action_names = action_names or [f"action {i}" for i in range(n_actions)]

    fig, axes = plt.subplots(1, n_actions + 1, figsize=(3.2 * (n_actions + 1), 3.2))
    value_map = upsample_heatmap(occ["value"], H, W)
    axes[0].imshow(base, cmap="gray")
    axes[0].imshow(value_map, cmap="jet", alpha=0.45)
    axes[0].set_title("value estimate", fontsize=9)
    axes[0].axis("off")

    for i in range(n_actions):
        prob_map = upsample_heatmap(occ["action_probs"][i], H, W)
        axes[i + 1].imshow(base, cmap="gray")
        axes[i + 1].imshow(prob_map, cmap="jet", alpha=0.45)
        axes[i + 1].set_title(f"P({action_names[i]})", fontsize=9)
        axes[i + 1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_saliency(image: torch.Tensor, grad: torch.Tensor, title: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = _to_displayable(image)
    sal = grad[0].abs().max(dim=0).values.numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    axes[0].imshow(base, cmap="gray")
    axes[0].set_title("input", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(base, cmap="gray")
    axes[1].imshow(sal, cmap="hot", alpha=0.6)
    axes[1].set_title(title, fontsize=9)
    axes[1].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ===========================================================================
# 7. CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None, help="Path to a saved SB3 PPO model.zip. Omit to use a fresh untrained network with this project's default shapes.")
    parser.add_argument("--image", type=str, default=None, help="Path to a saved .npy frame, e.g. dumped from watch_agent.py.")
    parser.add_argument("--live", type=str, default=None, choices=["basic", "deadly_corridor"], help="Grab one real frame from this repo's env instead of --image.")
    parser.add_argument("--frames", type=str, default=None, help="Path to a .npy of shape (N, C, H, W) for --technique deconv's top-k scan across many frames.")
    parser.add_argument("--technique", type=str, default="all", choices=["deconv", "occlusion", "saliency", "guided", "all"])
    parser.add_argument("--layer", type=int, default=2, help="Conv layer index (0, 1, or 2) for deconv.")
    parser.add_argument("--channel", type=int, default=0, help="Feature map channel for deconv.")
    parser.add_argument("--top-k", type=int, default=9, help="Top-k activations to reconstruct when --frames is given.")
    parser.add_argument("--action", type=int, default=None, help="Action index for saliency/guided (default: argmax action).")
    parser.add_argument("--patch-size", type=int, default=8, help="Occlusion patch size in pixels.")
    parser.add_argument("--stride", type=int, default=4, help="Occlusion sliding-window stride in pixels.")
    parser.add_argument("--out", type=str, default="viz_out/zf", help="Output path prefix.")
    args = parser.parse_args()

    if args.model:
        print(f"Loading trained model: {args.model}")
        policy = load_sb3_policy(args.model)
    else:
        print("No --model given: using a fresh untrained network with this project's default shapes.")
        policy = wrap_synthetic(SyntheticNatureCNNPolicy())
    policy.eval()

    if args.image:
        image = load_frame_from_npy(args.image)
    elif args.live:
        print(f"Grabbing a live frame from scenario: {args.live}")
        image = live_frame_from_env(args.live)
    else:
        print("No --image/--live given: using a random noise frame (shape check only, not a real visualization).")
        image = torch.rand(1, *policy.obs_shape)
    image = preprocess(image)

    run_deconv = args.technique in ("deconv", "all")
    run_occ = args.technique in ("occlusion", "all")
    run_sal = args.technique in ("saliency", "all")
    run_guided = args.technique in ("guided", "all")

    if run_deconv:
        if args.frames:
            frames = preprocess(torch.from_numpy(np.load(args.frames)).float())
            top = find_top_k_activations(policy, frames, args.layer, args.channel, k=args.top_k)
            recons, titles = [], []
            for frame_idx, y, x, val in top:
                r = deconv_reconstruct(policy, frames[frame_idx : frame_idx + 1], args.layer, args.channel, spatial_pos=(y, x))
                recons.append(r)
                titles.append(f"frame {frame_idx} act={val:.2f}")
            plot_deconv_grid(recons, titles, f"{args.out}_deconv_top{args.top_k}_L{args.layer}C{args.channel}.png")
        else:
            r = deconv_reconstruct(policy, image, args.layer, args.channel)
            plot_deconv_grid([r], [f"layer {args.layer} channel {args.channel}"], f"{args.out}_deconv_L{args.layer}C{args.channel}.png")

    if run_occ:
        occ = occlusion_sensitivity(policy, image, patch_size=args.patch_size, stride=args.stride)
        plot_occlusion_overlay(image, occ, f"{args.out}_occlusion.png")

    if run_sal:
        grad = vanilla_saliency(policy, image, action_idx=args.action)
        plot_saliency(image, grad, "vanilla saliency", f"{args.out}_saliency.png")

    if run_guided:
        grad = guided_saliency(policy, image, action_idx=args.action)
        plot_saliency(image, grad, "guided backprop", f"{args.out}_guided.png")


if __name__ == "__main__":
    main()
