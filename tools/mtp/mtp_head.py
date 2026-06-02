"""DeepSeek-V4-Flash MTP (nextn) head, assembled from transformers' DeepseekV4
components — NOT a from-scratch port. The hard block (MLA + 256-expert MoE +
Sinkhorn hyperconnection) is transformers' DeepseekV4DecoderLayer; this module
adds the nextn glue (enorm/e_proj, hnorm/h_proj, shared_head.norm, hc-head) per
sglang's DeepseekV4ModelNextN, and loads the mtp.0.* weights from mtp_bf16.pt.

The MTP slot is layer 43; transformers' per-layer arrays are length 43, so we
extend them with the MTP block's modes (sliding_attention + moe — the ds4 gguf
shows MLA+sinks with no compressor, and a bias router => moe).

Weight map is ~1:1 (see ATTN_MAP); only the routed experts need stacking
(gate_up_proj[e] = cat([w1,w3]); down_proj[e] = w2 — F.linear convention, no
transpose). Validated: load_state_dict(strict=True) with zero missing/unexpected.
"""
from __future__ import annotations
import json
import sys

import torch

_FORK = "/home/trevor/Projects/llama.cpp-tjs-fork/gguf-py"
if _FORK not in sys.path:
    sys.path.insert(0, _FORK)
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M  # noqa: E402
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402

SNAP = ("/home/trevor/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/"
        "snapshots/6c858e71890b508e4f3fd6491f45b325580ba934")
MTP_PT = "/home/trevor/Projects/spark-vllm/mv0/mtp_bf16.pt"
MTP_LAYER = 43
N_EXPERTS = 256

# DecoderLayer param  <-  mtp.0.<...>  (1:1, no transpose)
ATTN_MAP = {
    "self_attn.sinks": "attn.attn_sink",
    "self_attn.q_a_proj.weight": "attn.wq_a.weight",
    "self_attn.q_a_norm.weight": "attn.q_norm.weight",
    "self_attn.q_b_proj.weight": "attn.wq_b.weight",
    "self_attn.kv_proj.weight": "attn.wkv.weight",
    "self_attn.kv_norm.weight": "attn.kv_norm.weight",
    "self_attn.o_a_proj.weight": "attn.wo_a.weight",
    "self_attn.o_b_proj.weight": "attn.wo_b.weight",
    "mlp.gate.weight": "ffn.gate.weight",
    "mlp.gate.e_score_correction_bias": "ffn.gate.bias",
    "mlp.shared_experts.gate_proj.weight": "ffn.shared_experts.w1.weight",
    "mlp.shared_experts.up_proj.weight": "ffn.shared_experts.w3.weight",
    "mlp.shared_experts.down_proj.weight": "ffn.shared_experts.w2.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "attn_hc.fn": "hc_attn_fn", "attn_hc.base": "hc_attn_base", "attn_hc.scale": "hc_attn_scale",
    "ffn_hc.fn": "hc_ffn_fn", "ffn_hc.base": "hc_ffn_base", "ffn_hc.scale": "hc_ffn_scale",
}
# nextn glue (lives on the wrapper, not the DecoderLayer)
GLUE = ["enorm.weight", "hnorm.weight", "e_proj.weight", "h_proj.weight", "norm.weight",
        "hc_head_fn", "hc_head_base", "hc_head_scale"]


def build_config() -> DeepseekV4Config:
    cfg = DeepseekV4Config(**json.load(open(f"{SNAP}/config.json")))
    cfg.layer_types = list(cfg.layer_types) + ["sliding_attention"]
    cfg.mlp_layer_types = list(cfg.mlp_layer_types) + ["moe"]
    return cfg


def build_decoder_state(sd: dict, dtype=torch.bfloat16) -> dict:
    """mtp_bf16.pt -> DeepseekV4DecoderLayer state_dict (experts stacked)."""
    g = lambda k: sd["mtp.0." + k].to(dtype)
    out = {dk: g(sk) for dk, sk in ATTN_MAP.items()}
    out["mlp.experts.gate_up_proj"] = torch.stack(
        [torch.cat([g(f"ffn.experts.{i}.w1.weight"), g(f"ffn.experts.{i}.w3.weight")], 0)
         for i in range(N_EXPERTS)])
    out["mlp.experts.down_proj"] = torch.stack(
        [g(f"ffn.experts.{i}.w2.weight") for i in range(N_EXPERTS)])
    return out


def load_head(dtype=torch.bfloat16):
    """Returns (decoder_layer, glue_dict, config) with mtp.0.* weights loaded."""
    cfg = build_config()
    sd = torch.load(MTP_PT, map_location="cpu", mmap=True, weights_only=True)
    layer = M.DeepseekV4DecoderLayer(cfg, MTP_LAYER).to(dtype)
    missing, unexpected = layer.load_state_dict(build_decoder_state(sd, dtype), strict=False)
    glue = {k: sd["mtp.0." + k].to(dtype) for k in GLUE}
    return layer, glue, cfg, missing, unexpected


if __name__ == "__main__":
    layer, glue, cfg, missing, unexpected = load_head()
    print(f"DecoderLayer loaded: {sum(p.numel() for p in layer.parameters())/1e9:.2f}B params")
    print(f"  missing   = {list(missing)}")
    print(f"  unexpected= {list(unexpected)}")
    print(f"  glue loaded = {sorted(glue)}")
    ok = not missing and not unexpected
    print("WEIGHT MAP:", "PASS (strict 1:1)" if ok else "MISMATCH — see above")
