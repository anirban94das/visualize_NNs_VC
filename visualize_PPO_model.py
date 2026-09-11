"""Render this project's PPO CnnPolicy architecture with visualtorch.

Two modes:

1. Point it at a saved model (models/latest/ppo_basic.zip, etc.) to render
   the *actual* trained policy's architecture:

       python visualize_ppo_model.py --model models/latest/ppo_basic.zip

2. Run it with no arguments to build a fresh, untrained CnnPolicy with the
   same shapes this project uses (4-frame stack, 84x84, Discrete(3) actions
   for basic.wad) and render that instead — useful if you don't have a
   saved model yet, or don't want to pull in vizdoom/gymnasium just to plot
   a diagram.

Install requirements (in addition to this project's requirements.txt):
    pip install visualtorch matplotlib
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

import visualtorch

N_STACK = 4       # VecFrameStack(vec_env, n_stack=4) in train_basic.py / train_deadly_corridor.py
FRAME_SIZE = 84   # matches envs/common.py's ResizeObservation(env, shape=(84, 84))
N_ACTIONS = 3     # basic.wad's default Discrete(3): MOVE_LEFT, MOVE_RIGHT, ATTACK


class PPOActorNetwork(nn.Module):
    """NatureCNN feature extractor + PPO's action head (the policy branch).

    Mirrors stable_baselines3.common.torch_layers.NatureCNN followed by the
    action_net Linear layer. SB3's default net_arch=[] for CnnPolicy means
    mlp_extractor.policy_net is an empty nn.Sequential() in the real model
    -- i.e. there's no hidden MLP between the CNN's 512-d output and
    action_net, so this class reproduces that path exactly rather than
    approximating it.
    """

    def __init__(self, n_stack: int = N_STACK, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, n_stack, FRAME_SIZE, FRAME_SIZE)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU())
        self.action_net = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.linear(x)
        return self.action_net(x)


def build_from_saved_model(model_path: str) -> tuple[nn.Module, tuple[int, int, int, int]]:
    """Load a trained model.zip and pull out its actual action-branch
    (features_extractor -> action_net) so the render reflects the real
    architecture and action count, not the basic.wad default."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    policy = model.policy

    class ActorBranch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features_extractor = policy.pi_features_extractor
            self.mlp_extractor_policy_net = policy.mlp_extractor.policy_net
            self.action_net = policy.action_net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features_extractor(x)
            x = self.mlp_extractor_policy_net(x)
            return self.action_net(x)

    # SB3 wraps CnnPolicy envs with VecTransposeImage, so by the time the
    # policy sees it, observation_space.shape is already channel-first
    # (C, H, W) - not (H, W, C) as the raw gymnasium env declares.
    n_stack, height, width = policy.observation_space.shape
    input_shape = (1, n_stack, height, width)
    return ActorBranch(), input_shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a saved PPO model, e.g. models/latest/ppo_basic.zip. "
        "If omitted, renders a fresh untrained CnnPolicy with this "
        "project's default basic.wad shapes instead.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="flow",
        choices=["flow", "graph", "layered"],
        help="visualtorch render style.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="ppo_actor_render.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Upscale factor for output resolution (default 3x visualtorch's "
        "native render size).",
    )
    args = parser.parse_args()

    if args.model:
        print(f"Loading real model from: {args.model}")
        model, input_shape = build_from_saved_model(args.model)
    else:
        print("No --model given, using a fresh untrained CnnPolicy "
              f"({N_STACK}x{FRAME_SIZE}x{FRAME_SIZE} -> Discrete({N_ACTIONS})).")
        model = PPOActorNetwork()
        input_shape = (1, N_STACK, FRAME_SIZE, FRAME_SIZE)

    model.eval()
    img = visualtorch.render(model, input_shape=input_shape, style=args.style, legend=True)

    # Figure size (in inches) is fixed to visualtorch's native render size at
    # this base dpi; save-time dpi is scaled up from there so the output
    # pixel dimensions grow by --scale instead of the two dpi's cancelling out.
    base_dpi = 100
    plt.figure(figsize=(img.width / base_dpi, img.height / base_dpi), dpi=base_dpi)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=base_dpi * args.scale, bbox_inches="tight")
    print(f"Saved: {args.out}")

    # Shape trace, printed alongside the image for a quick sanity check.
    x = torch.zeros(*input_shape)
    with torch.no_grad():
        out = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(out.shape)}")
    print(f"params (this branch): {total_params:,}")


if __name__ == "__main__":
    main()