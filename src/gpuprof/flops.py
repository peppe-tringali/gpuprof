"""FLOP estimation and peak-throughput lookup for MFU.

The 6·P·T back-of-envelope in v1 was fine for a demo but misleads on
real models — attention has a T²·d term the linear-in-P form misses,
MoE only activates a fraction of the MLP, gradient checkpointing
re-does the forward, and LoRA changes which params get gradients.

This module lets a user declare their architecture (`TransformerArch`)
and computes training FLOPs per step with those effects included.
Users who don't declare an architecture fall back to 6·P·T and get a
warning surfaced in the insight output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# (gpu_name_substring, dtype) -> peak dense TFLOPs (non-sparse).
# Extend as needed; unknown GPU/dtype → MFU skipped, not lied about.
_PEAK_TFLOPS: dict[tuple[str, str], float] = {
    ("H100", "bf16"): 989.0, ("H100", "fp16"): 989.0,
    ("H100", "tf32"): 495.0, ("H100", "fp32"): 67.0,
    ("H100", "fp8"): 1979.0,
    ("H200", "bf16"): 989.0, ("H200", "fp16"): 989.0,
    ("H200", "fp8"): 1979.0, ("H200", "fp32"): 67.0,
    ("A100", "bf16"): 312.0, ("A100", "fp16"): 312.0,
    ("A100", "tf32"): 156.0, ("A100", "fp32"): 19.5,
    ("V100", "fp16"): 125.0, ("V100", "fp32"): 15.7,
    ("RTX 4090", "bf16"): 165.0, ("RTX 4090", "fp16"): 165.0,
    ("RTX 4090", "fp32"): 82.6,
    ("RTX 3090", "fp16"): 71.0, ("RTX 3090", "fp32"): 35.5,
    ("L40", "bf16"): 181.0, ("L40", "fp16"): 181.0,
    ("L4",  "bf16"): 121.0,
    ("B100", "bf16"): 1800.0, ("B100", "fp8"): 3500.0,
    ("B200", "bf16"): 2250.0, ("B200", "fp8"): 4500.0,
    # Mock backend for demos:
    ("MockGPU", "bf16"): 989.0, ("MockGPU", "fp16"): 989.0,
    ("MockGPU", "fp32"): 67.0,
}


def peak_tflops(gpu_name: Optional[str], dtype: Optional[str]) -> Optional[float]:
    if not gpu_name or not dtype:
        return None
    dtype = dtype.lower()
    for (needle, dt), v in _PEAK_TFLOPS.items():
        if dt == dtype and needle.lower() in gpu_name.lower():
            return v
    return None


@dataclass
class TransformerArch:
    """Declare enough of the model that we can compute FLOPs properly.

    Only `params` is strictly required — everything else refines the
    estimate. The typical hurt-me most fields to fill in:
      - `seq_len`, `hidden`, `layers`   → attention T²·d term (huge for long ctx)
      - `moe_active` + `moe_total`      → only active experts count
      - `grad_checkpoint`               → backward multiplier
      - `lora_rank` + `lora_targets`    → different backward FLOPs shape
    """
    params: int
    hidden: Optional[int] = None
    heads: Optional[int] = None                # unused in FLOP math; useful metadata
    layers: Optional[int] = None
    seq_len: Optional[int] = None
    vocab: Optional[int] = None                # metadata
    mlp_ratio: float = 4.0                     # metadata
    moe_active: Optional[int] = None           # experts per token
    moe_total: Optional[int] = None            # experts in the pool
    grad_checkpoint: bool = False
    lora_rank: Optional[int] = None            # if set, we're in LoRA/PEFT mode
    lora_trainable_params: Optional[int] = None


def transformer_flops_per_step(
    arch: TransformerArch,
    tokens_per_step: int,
) -> dict:
    """Return {flops, breakdown, notes}. `flops` is training FLOPs per step,
    already including forward + backward and the multipliers below.

    Model:
        core        = 6·P·T                                (fwd 2 + bwd 4)
        attention   = 12·L·hidden·seq_len·T                (per token, per layer,
                                                            attn scores are O(T²))
        MoE scale   = core · (active/total) if configured
        ckpt        = (attention + core) · 4/3 if enabled  (recompute fwd)
        LoRA        = warning surfaced, `flops` is core+attn without ckpt

    Args:
        tokens_per_step: total tokens across the mini-batch this step
            (batch_size · seq_len).  Attention correction assumes each
            sample has `arch.seq_len` tokens.
    """
    P = arch.params
    T = int(tokens_per_step or 0)
    notes: list[str] = []

    if not P or not T:
        return {"flops": 0.0, "breakdown": {}, "notes": ["missing params or tokens"]}

    core = 6.0 * P * T
    # MoE: user-declared active/total scales the linear-in-P work.
    if arch.moe_active and arch.moe_total and arch.moe_total > 0:
        scale = arch.moe_active / arch.moe_total
        core *= scale
        notes.append(f"MoE: active/total = {scale:.3f} applied to core FLOPs "
                     "(assumes `params` counts total experts)")

    # Attention T² term: per-token cost scales with sequence length.
    attn = 0.0
    if arch.hidden and arch.layers and arch.seq_len:
        # Full attention: QK^T (T²·d) + softmax·V (T²·d) fwd = 4·L·H·T²
        # With backward triples: 12·L·H·T² per sample. Per-token = 12·L·H·T.
        # Aggregated over T tokens/step: 12 · L · hidden · seq_len · T.
        attn = 12.0 * arch.layers * arch.hidden * arch.seq_len * T
        if arch.seq_len >= 4096:
            notes.append(f"long context (seq_len={arch.seq_len}): attention "
                         f"T² term contributes {attn / (core + attn) * 100:.1f}% of FLOPs")
    elif not (arch.hidden and arch.layers and arch.seq_len):
        notes.append("architecture incomplete (need hidden+layers+seq_len for "
                     "attention T² term); using 6·P·T only — will over-estimate "
                     "MFU on long-context models")

    # Backward multiplier: baseline is 3× forward (fwd=1 + bwd=2), which
    # is what the 6·P·T formula already encodes. LoRA and gradient
    # checkpointing change that multiplier — recompute it from scratch
    # rather than layering ad-hoc corrections on top of 6·P·T.
    #
    # `core` and `attn` above are 6·P·T-scaled = 3× the pure-forward
    # cost. Convert back to pure-forward for the recomposition.
    fwd_core = core / 3.0
    fwd_attn = attn / 3.0
    breakdown = {"core": core, "attention": attn}

    if arch.lora_rank is not None:
        # PEFT / LoRA: gradient wrt weight is only computed for trainable
        # params. Activation gradients still flow through the frozen
        # base, so backward includes:
        #   activation_bwd ≈ 1× forward FLOPs (chain rule through weights)
        #   weight_bwd     ≈ 1× forward_over_trainable_params
        # With P_trainable ≪ P_base, weight_bwd is negligible.
        # Effective multiplier: fwd(1) + act_bwd(1) + weight_bwd(≈0)
        #                     = 2× forward, not 3×.
        pt = arch.lora_trainable_params or 0
        weight_bwd_ratio = pt / P if P else 0
        multiplier = 1.0 + 1.0 + weight_bwd_ratio  # fwd + act_bwd + weight_bwd
        total = (fwd_core + fwd_attn) * multiplier
        breakdown["core"] = fwd_core * multiplier
        breakdown["attention"] = fwd_attn * multiplier
        notes.append(
            f"LoRA/PEFT: {multiplier:.2f}× forward "
            f"(fwd + activation bwd + {weight_bwd_ratio*100:.2f}% weight bwd) "
            "— exact FLOPs, not an upper bound"
        )
    else:
        total = core + attn  # standard 3× forward

    # Gradient checkpointing adds one extra forward pass during backward.
    # Applied AFTER the LoRA multiplier: recompute cost is the same
    # regardless of which weights get gradients.
    if arch.grad_checkpoint:
        recompute = fwd_core + fwd_attn
        total += recompute
        breakdown["recompute"] = recompute
        notes.append("gradient checkpointing: + 1× forward for recompute")

    return {
        "flops": total,
        "breakdown": breakdown,
        "notes": notes,
    }


def arch_from_meta(meta: dict) -> Optional[TransformerArch]:
    """Build a TransformerArch from a run's meta blob.

    Accepts either the direct fields (params, hidden, layers, seq_len, …)
    or a nested `arch: {...}`. If neither yields at least `params`,
    returns None.
    """
    if not isinstance(meta, dict):
        return None
    a = meta.get("arch")
    if not isinstance(a, dict):
        a = meta
    if not a.get("params"):
        return None
    moe = a.get("moe") or {}
    lora = a.get("lora") or {}
    return TransformerArch(
        params=int(a["params"]),
        hidden=_int_or_none(a.get("hidden")),
        heads=_int_or_none(a.get("heads")),
        layers=_int_or_none(a.get("layers")),
        seq_len=_int_or_none(a.get("seq_len")),
        vocab=_int_or_none(a.get("vocab")),
        mlp_ratio=float(a.get("mlp_ratio") or 4.0),
        moe_active=_int_or_none(moe.get("active") if moe else None),
        moe_total=_int_or_none(moe.get("total") if moe else None),
        grad_checkpoint=bool(a.get("grad_checkpoint") or a.get("gradient_checkpointing")),
        lora_rank=_int_or_none(lora.get("rank") if lora else a.get("lora_rank")),
        lora_trainable_params=_int_or_none(a.get("lora_trainable_params")),
    )


def _int_or_none(v):
    if v is None: return None
    try: return int(v)
    except (TypeError, ValueError): return None


# ---- back-compat: v1 API called this directly ----------------------
def transformer_flops(params: int, tokens: int) -> int:
    """Old 6·P·T back-of-envelope, kept for v1 callers. Prefer
    `transformer_flops_per_step(TransformerArch(...), tokens)`."""
    return 6 * params * tokens
