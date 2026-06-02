"""Trainable DeepSeek-V4-Flash MTP head as a single nn.Module, for the FastMTP
fine-tune. Wraps transformers' DeepseekV4DecoderLayer (MLA + 256-expert MoE +
Sinkhorn HC) + the nextn glue (enorm/e_proj, hnorm/h_proj, hc-head, mtp.0.norm)
+ frozen base embed/output. Warm-started from mtp_bf16.pt (full-precision source).

Forward is split into the pieces the K-step recursive unroll needs:
  project(tokens, prev_hc) -> input_hc        # e_proj(enorm(embed)) + h_proj(hnorm(prev))
  decode(input_hc, pos, mask, pe) -> out_hc   # the DecoderLayer (HC stream [.,.,hc,D])
  to_logits(out_hc) -> logits                 # hc_head -> mtp.0.norm -> base output head

freeze_for_finetune() freezes the 256 experts (grad still flows through them) and
the base embed/output; everything else (~165M: attn, projections, norms, the two
HyperConnections, hc_head, router gate) stays trainable. Gradient checkpointing on
the decoder keeps the K-step unroll memory ~1x.
"""

from __future__ import annotations
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/trevor/Projects/ds4/tools/mtp")
from gguf import GGUFReader, GGMLQuantizationType, quants  # noqa: E402
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M  # noqa: E402
from transformers.masking_utils import create_sliding_window_causal_mask  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
import mtp_head as H  # build_config, build_decoder_state, GLUE, MTP_PT, BASE_GGUF, MTP_LAYER  # noqa: E402


def _deq_base(base_gguf: str, names: set[str]) -> dict[str, torch.Tensor]:
    r = GGUFReader(base_gguf)
    out = {}
    for t in r.tensors:
        if t.name in names:
            arr = quants.dequantize(t.data, GGMLQuantizationType(t.tensor_type))
            out[t.name] = torch.from_numpy(np.ascontiguousarray(arr, np.float32))
    return out


class DeepseekV4MtpHead(nn.Module):
    embed: torch.Tensor  # frozen base token embedding (registered buffer)
    output_w: torch.Tensor  # frozen base lm-head weight (registered buffer)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_size
        self.decoder = M.DeepseekV4DecoderLayer(cfg, H.MTP_LAYER)
        self.hc_head = M.DeepseekV4HyperHead(cfg)
        self.enorm = M.DeepseekV4RMSNorm(D, eps=cfg.rms_norm_eps)
        self.hnorm = M.DeepseekV4RMSNorm(D, eps=cfg.rms_norm_eps)
        self.norm = M.DeepseekV4RMSNorm(D, eps=cfg.rms_norm_eps)
        self.e_proj = nn.Linear(D, D, bias=False)
        self.h_proj = nn.Linear(D, D, bias=False)
        self.rotary = M.DeepseekV4RotaryEmbedding(cfg)
        # frozen base bits (large; persistent=False so they're not in the ckpt)
        self.register_buffer("embed", torch.zeros(cfg.vocab_size, D), persistent=False)
        self.register_buffer(
            "output_w", torch.zeros(cfg.vocab_size, D), persistent=False
        )

    @classmethod
    def from_pt(
        cls, mtp_pt: str = H.MTP_PT, base_gguf: str = H.BASE_GGUF, dtype=torch.bfloat16
    ):
        cfg = H.build_config()
        head = cls(cfg)
        sd = torch.load(mtp_pt, map_location="cpu", mmap=True, weights_only=True)
        head.decoder.load_state_dict(H.build_decoder_state(sd, dtype), strict=True)

        def g(k):
            return sd["mtp.0." + k].to(dtype).clone()

        head.enorm.weight.data = g("enorm.weight")
        head.hnorm.weight.data = g("hnorm.weight")
        head.norm.weight.data = g("norm.weight")
        head.e_proj.weight.data = g("e_proj.weight")
        head.h_proj.weight.data = g("h_proj.weight")
        head.hc_head.load_state_dict(
            {
                "hc_fn": g("hc_head_fn"),
                "hc_base": g("hc_head_base"),
                "hc_scale": g("hc_head_scale"),
            },
            strict=False,
        )
        base = _deq_base(base_gguf, {"token_embd.weight", "output.weight"})
        head.embed = base["token_embd.weight"].reshape(-1, cfg.hidden_size).to(dtype)
        head.output_w = base["output.weight"].reshape(-1, cfg.hidden_size).to(dtype)
        return head.to(dtype)

    def freeze_for_finetune(self):
        self.requires_grad_(True)
        self.decoder.mlp.experts.requires_grad_(False)  # freeze the 256 routed experts
        # embed/output_w are buffers (no grad); nothing else to do.

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---- forward pieces ------------------------------------------------------
    def project(self, tokens: torch.Tensor, prev_hc: torch.Tensor) -> torch.Tensor:
        """tokens [B,S] long ; prev_hc [B,S,hc,D] -> input_hc [B,S,hc,D]."""
        e = self.e_proj(self.enorm(self.embed[tokens]))  # [B,S,D]
        h = self.h_proj(self.hnorm(prev_hc))  # [B,S,hc,D]
        return e.unsqueeze(2) + h

    def rope_mask(self, S: int, position_ids: torch.Tensor, device, dtype):
        ref = torch.zeros(1, S, self.cfg.hidden_size, dtype=dtype, device=device)
        pe = {
            "main": self.rotary(ref, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary(
                ref, position_ids=position_ids, layer_type="compress"
            ),
        }
        mask = create_sliding_window_causal_mask(
            config=self.cfg,
            inputs_embeds=ref,
            attention_mask=None,
            past_key_values=DynamicCache(config=self.cfg),
            position_ids=position_ids,
        )
        return pe, mask

    def decode(self, input_hc, position_ids, mask, pe):
        """input_hc [B,S,hc,D] -> out_hc [B,S,hc,D] (the DecoderLayer HC stream)."""
        return self.decoder(
            input_hc,
            input_ids=None,
            position_ids=position_ids,
            position_embeddings=pe,
            attention_mask=mask,
        )

    def to_logits(self, out_hc):
        """out_hc [B,S,hc,D] -> logits [B,S,vocab]."""
        collapsed = self.hc_head(out_hc)  # [B,S,D]
        return self.norm(collapsed) @ self.output_w.t()
