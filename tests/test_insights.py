from gpuprof.flops import TransformerArch
from gpuprof.insights import (
    Ctx,
    rule_dataloader_stall,
    rule_low_mfu,
    rule_memory_pressure,
    rule_kernel_launch_overhead,
    rule_compilation_warmup,
    rule_first_step_outlier,
    rule_high_step_variance,
    rule_gradient_checkpointing_detected,
    rule_sdpa_suboptimal,
    rule_thermal_throttling,
    rule_pcie_saturation,
    rule_small_batch,
    rule_comm_dominant,
)


def _ctx(**overrides):
    base = dict(
        run_id=1, name="t", gpu_name="H100", meta={}, arch=None,
        n_steps=100, train_s=10.0, step_s=0.1,
        step_times=[0.1] * 100,
        phase_avg_s={"dataloader_wait": 0.0, "forward": 0.03,
                     "backward": 0.06, "optimizer": 0.005, "comm": 0.0},
        inter_step_gap_s=0.0, tokens_total=1000,
        measured_flops=None, peak_tflops=989.0, mfu=None,
        avg_sm_util=0.7, max_mem=None, mem_total=None,
        max_temp=None, sm_clock_deficit_pct=None,
        pcie_avg_rx_gbps=None, pcie_avg_tx_gbps=None,
        pcie_max_rx_gbps=None, traces=[],
    )
    base.update(overrides)
    return Ctx(**base)


# ---- dataloader stall ---------------------------------------------------

def test_dataloader_stall_via_inter_step():
    c = _ctx(step_s=0.1, inter_step_gap_s=0.05)  # 50% gap
    r = rule_dataloader_stall(c)
    assert r is not None and r["severity"] == "high"


def test_dataloader_stall_via_phase_when_gap_absent():
    c = _ctx(step_s=0.1, inter_step_gap_s=0.0,
             phase_avg_s={"dataloader_wait": 0.03, "forward": 0.03,
                          "backward": 0.03, "optimizer": 0.005, "comm": 0.0})
    r = rule_dataloader_stall(c)
    assert r is not None and "dataloader_wait phase" in r["title"]


def test_dataloader_stall_healthy_no_fire():
    c = _ctx(inter_step_gap_s=0.005, step_s=0.1)
    assert rule_dataloader_stall(c) is None


# ---- MFU ---------------------------------------------------------------

def test_low_mfu_fires_below_30pct():
    c = _ctx(mfu=0.10, peak_tflops=989.0, meta={"dtype": "bf16"})
    r = rule_low_mfu(c)
    assert r and "10.0%" in r["title"]


def test_low_mfu_healthy_no_fire():
    assert rule_low_mfu(_ctx(mfu=0.55)) is None
    assert rule_low_mfu(_ctx(mfu=None)) is None


# ---- memory ------------------------------------------------------------

def test_memory_pressure_fires_high():
    c = _ctx(max_mem=int(0.98 * 80e9), mem_total=int(80e9))
    r = rule_memory_pressure(c)
    assert r and r["severity"] == "high"


def test_memory_pressure_medium():
    c = _ctx(max_mem=int(0.92 * 80e9), mem_total=int(80e9))
    r = rule_memory_pressure(c)
    assert r and r["severity"] == "medium"


def test_memory_pressure_healthy():
    c = _ctx(max_mem=int(0.5 * 80e9), mem_total=int(80e9))
    assert rule_memory_pressure(c) is None


# ---- kernel launch overhead --------------------------------------------

def test_kernel_launch_overhead_fires():
    # step_s much larger than phase sum
    c = _ctx(step_s=0.1,
             phase_avg_s={"dataloader_wait": 0.0, "forward": 0.02,
                          "backward": 0.04, "optimizer": 0.005, "comm": 0.0})
    # sum=0.065; gap=0.035; gap/step=0.35 → fires (low)
    r = rule_kernel_launch_overhead(c)
    assert r and r["severity"] == "low"


def test_kernel_launch_overhead_healthy():
    # phase sum equals step
    c = _ctx(step_s=0.100,
             phase_avg_s={"dataloader_wait": 0.0, "forward": 0.03,
                          "backward": 0.06, "optimizer": 0.005, "comm": 0.005})
    assert rule_kernel_launch_overhead(c) is None


# ---- compilation warmup + step 0 outlier -------------------------------

def test_compilation_warmup_fires():
    step_times = [0.5, 0.15] + [0.1] * 20
    c = _ctx(step_times=step_times, step_s=0.12)
    r = rule_compilation_warmup(c)
    assert r and "warmup" in r["title"].lower()


def test_first_step_outlier_fires_large():
    step_times = [5.0] + [0.1] * 20
    c = _ctx(step_times=step_times, step_s=0.33)
    r = rule_first_step_outlier(c)
    assert r and "Step 0" in r["title"]


def test_first_step_outlier_no_fire_small_ratio():
    step_times = [0.15, 0.10, 0.10]
    c = _ctx(step_times=step_times)
    assert rule_first_step_outlier(c) is None


# ---- step variance -----------------------------------------------------

def test_step_variance_fires_when_p99_much_higher():
    # 90 fast steps, 10 slow ones
    step_times = [0.1] * 90 + [0.5] * 10
    c = _ctx(step_times=step_times)
    r = rule_high_step_variance(c)
    assert r and "tail" in r["title"].lower()


# ---- gradient checkpointing -------------------------------------------

def test_checkpoint_detected_from_ratio():
    # backward/forward = 3.5x → suspicious, none declared
    c = _ctx(phase_avg_s={"dataloader_wait": 0.0, "forward": 0.02,
                          "backward": 0.07, "optimizer": 0.005, "comm": 0.0},
             arch=None)
    r = rule_gradient_checkpointing_detected(c)
    assert r and "gradient checkpointing" in r["title"].lower()


def test_checkpoint_declared_but_ratio_low():
    arch = TransformerArch(params=100, grad_checkpoint=True)
    c = _ctx(phase_avg_s={"dataloader_wait": 0.0, "forward": 0.02,
                          "backward": 0.04, "optimizer": 0.005, "comm": 0.0},
             arch=arch)
    r = rule_gradient_checkpointing_detected(c)
    assert r and "declared" in r["title"]


# ---- SDPA / attention --------------------------------------------------

def test_sdpa_suboptimal_when_bmm_softmax_no_flash():
    traces = [{"step": 10, "kernels": [
        {"name": "aten::bmm"}, {"name": "aten::softmax"},
        {"name": "aten::gelu"},
    ]}]
    r = rule_sdpa_suboptimal(_ctx(traces=traces))
    assert r and "unfused" in r["title"].lower()


def test_sdpa_ok_when_flash_present():
    traces = [{"step": 10, "kernels": [{"name": "flash_attention_forward"}]}]
    assert rule_sdpa_suboptimal(_ctx(traces=traces)) is None


def test_sdpa_no_trace_no_fire():
    assert rule_sdpa_suboptimal(_ctx(traces=[])) is None


# ---- thermal / clocks --------------------------------------------------

def test_thermal_throttling_fires():
    c = _ctx(max_temp=87.0, sm_clock_deficit_pct=0.20)
    r = rule_thermal_throttling(c)
    assert r and "throttl" in r["title"].lower()


def test_thermal_healthy():
    c = _ctx(max_temp=70.0, sm_clock_deficit_pct=0.05)
    assert rule_thermal_throttling(c) is None


# ---- PCIe --------------------------------------------------------------

def test_pcie_saturation_fires():
    c = _ctx(pcie_max_rx_gbps=25.0, pcie_avg_rx_gbps=15.0)
    r = rule_pcie_saturation(c)
    assert r and "PCIe" in r["title"]


# ---- small batch / low intensity ---------------------------------------

def test_small_batch_low_mfu():
    c = _ctx(mfu=0.10, avg_sm_util=0.75)
    r = rule_small_batch(c)
    assert r and "arithmetic intensity" in r["title"].lower()


def test_small_batch_no_fire_high_mfu():
    assert rule_small_batch(_ctx(mfu=0.55, avg_sm_util=0.85)) is None


# ---- comm dominance ----------------------------------------------------

def test_comm_dominant_fires():
    c = _ctx(step_s=0.100,
             phase_avg_s={"dataloader_wait": 0.0, "forward": 0.02,
                          "backward": 0.04, "optimizer": 0.005, "comm": 0.03})
    r = rule_comm_dominant(c)
    assert r and "NCCL" in r["title"]


def test_comm_healthy():
    c = _ctx(step_s=0.100,
             phase_avg_s={"dataloader_wait": 0.0, "forward": 0.03,
                          "backward": 0.06, "optimizer": 0.005, "comm": 0.005})
    assert rule_comm_dominant(c) is None
