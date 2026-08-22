import pytest

from gpuprof.flops import (
    TransformerArch, arch_from_meta, peak_tflops,
    transformer_flops, transformer_flops_per_step,
)


def test_peak_tflops_substring_match():
    assert peak_tflops("NVIDIA H100 80GB HBM3", "bf16") == 989.0
    assert peak_tflops("NVIDIA A100-SXM4-80GB", "bf16") == 312.0
    assert peak_tflops("Some Future GPU", "bf16") is None
    assert peak_tflops("A100", None) is None
    assert peak_tflops(None, "bf16") is None


def test_legacy_6PT():
    # Old API still returns the exact 6·P·T value.
    assert transformer_flops(100, 200) == 6 * 100 * 200


def test_flops_no_arch_returns_core_only():
    arch = TransformerArch(params=1_000_000_000)
    r = transformer_flops_per_step(arch, tokens_per_step=1_000_000)
    # No hidden/layers/seq_len → attention term is 0, so just 6·P·T
    assert r["flops"] == 6 * 1_000_000_000 * 1_000_000
    assert r["breakdown"]["attention"] == 0
    assert any("architecture incomplete" in n for n in r["notes"])


def test_flops_with_attention_term():
    # A 1.3B model at 4k context — attention should add measurably.
    arch = TransformerArch(
        params=1_300_000_000, hidden=2048, layers=24, seq_len=4096,
    )
    T = 32 * 4096  # 32 samples of 4k tokens
    r = transformer_flops_per_step(arch, tokens_per_step=T)
    core = 6 * 1_300_000_000 * T
    attn = 12 * 24 * 2048 * 4096 * T
    assert r["breakdown"]["core"] == core
    assert r["breakdown"]["attention"] == attn
    assert r["flops"] == core + attn
    # Long-context note should surface.
    assert any("long context" in n for n in r["notes"])


def test_flops_grad_checkpoint_multiplier():
    arch = TransformerArch(params=1_000_000_000, grad_checkpoint=True)
    r = transformer_flops_per_step(arch, tokens_per_step=1_000_000)
    # 6·P·T × 4/3 for the recompute
    assert r["flops"] == 6 * 1_000_000_000 * 1_000_000 * (4 / 3)
    assert any("checkpointing" in n for n in r["notes"])


def test_flops_lora_2x_forward_multiplier():
    """LoRA: fwd + activation bwd + tiny weight bwd ≈ 2× forward (not 3×
    like full training). With P_trainable=0 (edge case) it's exactly 2×."""
    P, T = 7_000_000_000, 1_000_000
    arch = TransformerArch(params=P, lora_rank=16, lora_trainable_params=0)
    r = transformer_flops_per_step(arch, tokens_per_step=T)
    # baseline core (which is 3× forward) is 6·P·T; forward alone is 2·P·T.
    # LoRA total: forward + activation_bwd = 2 · forward = 4·P·T.
    assert r["flops"] == 4 * P * T
    assert any("LoRA/PEFT" in n for n in r["notes"])


def test_flops_lora_with_trainable_params_adds_weight_bwd():
    P, Ptr, T = 7_000_000_000, 700_000, 1_000_000  # 0.01% trainable
    arch = TransformerArch(params=P, lora_rank=16, lora_trainable_params=Ptr)
    r = transformer_flops_per_step(arch, tokens_per_step=T)
    ratio = Ptr / P
    expected_mult = 1.0 + 1.0 + ratio  # 2 + tiny
    assert r["flops"] == pytest.approx(2 * P * T * expected_mult)


def test_flops_lora_plus_checkpointing():
    P, T = 7_000_000_000, 1_000_000
    arch = TransformerArch(params=P, lora_rank=16,
                           lora_trainable_params=0, grad_checkpoint=True)
    r = transformer_flops_per_step(arch, tokens_per_step=T)
    # LoRA base (2× forward = 4·P·T) + recompute (1× forward = 2·P·T) = 6·P·T.
    assert r["flops"] == 6 * P * T
    assert "recompute" in r["breakdown"]


def test_flops_moe_scales_core():
    arch = TransformerArch(
        params=8_000_000_000, moe_active=2, moe_total=8,
    )
    r = transformer_flops_per_step(arch, tokens_per_step=1_000_000)
    # active/total = 0.25 → 6·P·T × 0.25
    assert r["flops"] == 6 * 8_000_000_000 * 1_000_000 * 0.25
    assert any("MoE" in n for n in r["notes"])


def test_flops_zero_tokens_returns_zero():
    arch = TransformerArch(params=1_000_000_000)
    r = transformer_flops_per_step(arch, tokens_per_step=0)
    assert r["flops"] == 0.0
    r2 = transformer_flops_per_step(TransformerArch(params=0), 1)
    assert r2["flops"] == 0.0


def test_arch_from_meta_nested():
    meta = {"arch": {"params": 100, "hidden": 512, "layers": 4,
                     "seq_len": 128, "moe": {"active": 2, "total": 4},
                     "grad_checkpoint": True}}
    arch = arch_from_meta(meta)
    assert arch.params == 100 and arch.hidden == 512
    assert arch.moe_active == 2 and arch.moe_total == 4
    assert arch.grad_checkpoint is True


def test_arch_from_meta_flat():
    meta = {"params": 100, "hidden": 512, "layers": 4, "seq_len": 128}
    arch = arch_from_meta(meta)
    assert arch.params == 100 and arch.layers == 4


def test_arch_from_meta_missing_params():
    assert arch_from_meta({}) is None
    assert arch_from_meta({"hidden": 512}) is None
