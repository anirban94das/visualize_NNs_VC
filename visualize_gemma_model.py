"""Render Gemma 3n E4B's text-backbone architecture with visualtorch.

Companion to visualize_PPO_model.py, which does the same for this project's
PPO CnnPolicy. Same idea: wrap the model in a thin nn.Module whose forward()
takes one input and returns a single tensor (see visualize_PPO_model.py's
ActorBranch), then let visualtorch trace one forward pass.

Two important differences from the CNN case, both handled here:

1. Gemma's input is token IDs (a LongTensor of small ints), but visualtorch
   traces with `torch.rand(input_shape)` -- random floats. There is no way to
   hand it an explicit input tensor (checked: visualtorch 1.2.1's
   render()/*_view() only take `input_shape`, and utils/recorder.py hardcodes
   `torch.rand`). Workaround: the wrapper derives valid token IDs *from* the
   traced tensor -- `(x * 0).long() + k` -- so lineage from the input node is
   preserved while the dtype/range become valid.

2. The full 35-layer backbone traces fine but is unrenderable: visualtorch
   unrolls every layer and draws every AltUp / LAuReL / per-layer-embedding
   skip connection, and the resulting canvas is large enough that PIL's
   Image.new() raises MemoryError (measured: ~6.8B untrained params, instantiates
   in ~5 min / ~7 GB RAM, forward OK, render dies). So --scope defaults to
   `layer`: render ONE decoder layer as the representative repeating unit, the
   same thing you'd do for a CNN with many identical residual blocks.
   --scope backbone is kept for completeness / smaller custom configs.

Usage:
    python visualize_gemma_model.py                       # one E4B decoder layer, flow
    python visualize_gemma_model.py --style graph
    python visualize_gemma_model.py --scope backbone --tiny # shrunk config, full depth

Requires (added to this project's venv): transformers>=4.53, plus visualtorch
and matplotlib (already used by visualize_PPO_model.py). No model weights and
no network are needed -- everything is an untrained module built from a config,
matching visualize_PPO_model.py's no-model fallback path.
"""

import argparse

import torch
from torch import nn

import visualtorch

SEQ = 8  # short sequence: diagram structure doesn't depend on it, and it keeps the trace cheap


def build_config(tiny: bool):
    """Default Gemma3nTextConfig() is already the E4B text backbone (35 layers,
    hidden 2048, 8 heads / 2 KV, head_dim 256, AltUp x4, LAuReL rank 64,
    per-layer-input width 256). --tiny shrinks width + depth for a renderable
    full-backbone diagram; the special sublayers are all still present."""
    from transformers import Gemma3nTextConfig

    if tiny:
        cfg = Gemma3nTextConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=64,
            hidden_size_per_layer_input=32,
            vocab_size_per_layer_input=512,
            num_kv_shared_layers=0,
            sliding_window=32,
        )
    else:
        cfg = Gemma3nTextConfig()
        # layer 0 is never a KV-shared layer; zeroing this avoids an index error
        # when a stripped-down single layer is built in isolation.
        cfg.num_kv_shared_layers = 0
    return cfg


class OneDecoderLayer(nn.Module):
    """One Gemma3nTextDecoderLayer, driven directly. Its forward() wants a
    stacked (altup_num_inputs, B, S, H) hidden state plus rotary position
    embeddings and the per-layer input; we synthesize all of them so the one
    input visualtorch traces with is the hidden state."""

    def __init__(self, cfg):
        super().__init__()
        from transformers.models.gemma3n import modeling_gemma3n as g3n

        self.layer = g3n.Gemma3nTextDecoderLayer(cfg, layer_idx=0)
        self.h = cfg.hidden_size
        self.p = cfg.altup_num_inputs
        self.hd = cfg.head_dim
        self.pli = cfg.hidden_size_per_layer_input

    @property
    def input_shape(self):
        return (self.p, 1, SEQ, self.h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos = x.new_ones(1, SEQ, self.hd)
        sin = x.new_zeros(1, SEQ, self.hd)
        per_layer_input = x.new_zeros(1, SEQ, self.pli)
        out = self.layer(
            x,
            position_embeddings=(cos, sin),
            per_layer_input=per_layer_input,
            attention_mask=None,
            position_ids=torch.arange(SEQ).unsqueeze(0),
        )
        return out[0] if isinstance(out, tuple) else out


class TextBackbone(nn.Module):
    """The whole Gemma3nTextModel. forward() takes the traced float tensor and
    turns it into valid token IDs (see module docstring, point 1)."""

    def __init__(self, cfg):
        super().__init__()
        from transformers import Gemma3nTextModel

        self.model = Gemma3nTextModel(cfg)

    @property
    def input_shape(self):
        return (1, SEQ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_ids = (x * 0).long() + 3  # valid ids, still descended from x
        return self.model(input_ids=input_ids).last_hidden_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", choices=["layer", "backbone"], default="layer",
                        help="'layer': one decoder layer (default, always renders). "
                             "'backbone': the full Gemma3nTextModel (use --tiny unless you have lots of RAM).")
    parser.add_argument("--tiny", action="store_true",
                        help="Shrink the config (width + depth) so --scope backbone is renderable.")
    parser.add_argument("--style", choices=["flow", "graph", "lenet"], default="flow",
                        help="visualtorch render style. 'graph' is the most legible for this model.")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path.")
    args = parser.parse_args()

    cfg = build_config(tiny=args.tiny)
    if args.scope == "layer":
        print("Building one untrained Gemma 3n decoder layer "
              f"(hidden={cfg.hidden_size}, heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} KV, "
              f"AltUp x{cfg.altup_num_inputs}, LAuReL rank {cfg.laurel_rank}).")
        model = OneDecoderLayer(cfg)
    else:
        print(f"Building the full untrained Gemma3nTextModel ({cfg.num_hidden_layers} layers, "
              f"hidden={cfg.hidden_size}). This can take minutes and a lot of RAM for the real E4B config.")
        model = TextBackbone(cfg)

    model.eval()
    input_shape = model.input_shape
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params in this piece: {n_params:,}")

    with torch.no_grad():
        out = model(torch.rand(*input_shape))
    print(f"input:  {tuple(input_shape)}  ->  output: {tuple(out.shape)}")

    out_path = args.out or f"gemma_{args.scope}{'_tiny' if args.tiny else ''}_render_{args.style}.png"
    print(f"Rendering with visualtorch (style={args.style})...")
    style_kwargs = {"legend": True} if args.style == "flow" else {}
    img = visualtorch.render(model, input_shape=input_shape, style=args.style, **style_kwargs)

    # visualtorch already returns a large PIL image for this model; save it
    # directly rather than round-tripping through matplotlib (which would try to
    # allocate a multi-GB RGBA array for the graph style).
    img.save(out_path)
    print(f"Saved: {out_path}  ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
