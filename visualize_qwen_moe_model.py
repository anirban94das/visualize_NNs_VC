"""Render a Qwen3 MoE (e.g. Qwen3-*-A3B) decoder layer's architecture with visualtorch.

Companion to visualize_PPO_model.py (CNN) and visualize_gemma_model.py (Gemma 3n).
Same recipe: wrap the module so forward() takes one traced tensor and returns a
single tensor (cf. visualize_PPO_model.py's ActorBranch), then let visualtorch
trace one forward pass. No weights, no network, no download -- everything is an
untrained module built from a config, like visualize_PPO_model.py's no-model path.

Why one decoder layer and not the whole model
---------------------------------------------
A Qwen3-MoE like "35B-A3B" is ~24-48 identical decoder layers; a full unrolled
diagram is unreadable and (as with Gemma) large enough to OOM PIL. One layer is
the representative repeating unit -- same as rendering a single residual block of
a deep CNN.

Two MoE-specific notes
----------------------
1. In this transformers version, the sparse block is `Qwen3MoeSparseMoeBlock` =
   `gate` (`Qwen3MoeTopKRouter`) + `experts` (`Qwen3MoeExperts`). BOTH keep their
   weights as raw `nn.Parameter` 3D tensors ([num_experts, ...]) and run the FFN
   via `F.linear` on parameter slices -- there are NO per-expert child modules.
   So visualtorch draws the MoE block as exactly two leaf nodes (router +
   experts), no matter how many experts the config has. The classic "visualtorch
   only traces the routed experts" caveat therefore doesn't even arise here --
   but neither do you get 128 parallel expert boxes, because the code doesn't
   express them as modules.
2. `--real` uses the true Qwen3-MoE dims (128 experts, top-8, hidden 2048); one
   such layer is ~0.6B untrained params (~2.5 GB fp32) but still renders. The
   default is a reduced config (fast, same structure).

Usage
-----
    python visualize_qwen_moe_model.py                     # reduced 1-layer, flow
    python visualize_qwen_moe_model.py --style graph
    python visualize_qwen_moe_model.py --real --style graph
    python visualize_qwen_moe_model.py --scope backbone    # reduced full stack
    python visualize_qwen_moe_model.py --hf-config Qwen/Qwen3-30B-A3B   # real config.json off HF

Requires (already in this project's venv): transformers>=4.51, visualtorch, torch.
"""

import argparse

import torch
from torch import nn

import visualtorch

SEQ = 16  # enough tokens that every expert in the reduced config gets routed something


def build_config(real: bool, hf_config: str | None):
    from transformers import Qwen3MoeConfig

    if hf_config:
        print(f"Loading real config from HF: {hf_config} (downloads config.json only)")
        return Qwen3MoeConfig.from_pretrained(hf_config)
    if real:
        # Qwen3MoeConfig()'s defaults already are the real Qwen3-MoE text arch:
        # 24 layers, hidden 2048, 32 heads / 4 KV, 128 experts, top-8, moe_ffn 768.
        return Qwen3MoeConfig()
    # Reduced: same structure, small enough to build/trace instantly.
    return Qwen3MoeConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=256,
    )


class OneDecoderLayer(nn.Module):
    """One Qwen3MoeDecoderLayer, driven directly. Its forward() wants
    hidden_states (B, S, H) plus rotary position embeddings; we synthesize the
    rotary part so the one tensor visualtorch traces with is the hidden state."""

    def __init__(self, cfg):
        super().__init__()
        from transformers.models.qwen3_moe import modeling_qwen3_moe as q3

        self.layer = q3.Qwen3MoeDecoderLayer(cfg, layer_idx=0)
        self.hidden_size = cfg.hidden_size
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    @property
    def input_shape(self):
        return (1, SEQ, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos = x.new_ones(1, SEQ, self.head_dim)
        sin = x.new_zeros(1, SEQ, self.head_dim)
        out = self.layer(
            x,
            attention_mask=None,
            position_ids=torch.arange(SEQ).unsqueeze(0),
            position_embeddings=(cos, sin),
        )
        return out[0] if isinstance(out, tuple) else out


class TextBackbone(nn.Module):
    """The whole Qwen3MoeModel. forward() turns the traced float tensor into
    valid token IDs -- visualtorch only ever hands you `torch.rand(shape)`
    (floats), and an nn.Embedding needs a LongTensor index. `(x * 0).long() + k`
    is a valid id tensor that still descends from the traced input, so the
    diagram's input node stays connected."""

    def __init__(self, cfg):
        super().__init__()
        from transformers import Qwen3MoeModel

        self.model = Qwen3MoeModel(cfg)

    @property
    def input_shape(self):
        return (1, SEQ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_ids = (x * 0).long() + 3
        return self.model(input_ids=input_ids).last_hidden_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", choices=["layer", "backbone"], default="layer",
                        help="'layer': one decoder layer (default, always renders). "
                             "'backbone': the full Qwen3MoeModel (use the reduced config unless you have lots of RAM).")
    parser.add_argument("--real", action="store_true",
                        help="Use the true Qwen3-MoE dims (128 experts, top-8, hidden 2048) instead of the reduced config.")
    parser.add_argument("--hf-config", type=str, default=None,
                        help="Load a real config.json from a HF repo, e.g. Qwen/Qwen3-30B-A3B. Downloads config only.")
    parser.add_argument("--style", choices=["flow", "graph", "lenet"], default="flow",
                        help="visualtorch render style. 'graph' is the most legible for a transformer.")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path.")
    args = parser.parse_args()

    cfg = build_config(real=args.real, hf_config=args.hf_config)
    tag = "hf" if args.hf_config else ("real" if args.real else "reduced")

    if args.scope == "layer":
        n_kv = cfg.num_key_value_heads
        print(f"Building one untrained Qwen3-MoE decoder layer "
              f"(hidden={cfg.hidden_size}, heads={cfg.num_attention_heads}/{n_kv} KV, "
              f"{cfg.num_experts} experts, top-{cfg.num_experts_per_tok}).")
        model = OneDecoderLayer(cfg)
    else:
        print(f"Building the full untrained Qwen3MoeModel ({cfg.num_hidden_layers} layers, "
              f"hidden={cfg.hidden_size}, {cfg.num_experts} experts). "
              "The real config here needs a lot of RAM.")
        model = TextBackbone(cfg)

    model.eval()
    input_shape = model.input_shape
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params in this piece: {n_params:,}")

    with torch.no_grad():
        out = model(torch.rand(*input_shape))
    print(f"input:  {tuple(input_shape)}  ->  output: {tuple(out.shape)}")

    out_path = args.out or f"qwen_moe_{args.scope}_{tag}_render_{args.style}.png"
    print(f"Rendering with visualtorch (style={args.style})...")
    style_kwargs = {"legend": True} if args.style == "flow" else {}
    img = visualtorch.render(model, input_shape=input_shape, style=args.style, **style_kwargs)

    # Save the PIL image straight to disk; the graph style produces a canvas big
    # enough that routing it through matplotlib would try to allocate GBs.
    img.save(out_path)
    print(f"Saved: {out_path}  ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
