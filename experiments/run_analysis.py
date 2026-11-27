"""Quantitative FLOP / memory / communication analysis.

Run with:  python3 -m experiments.run_analysis
Writes results/analysis_summary.json and results/communication_scaling.png
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import GPTConfig
from src.analysis import (
    flops_forward_analytical,
    flops_forward_measured,
    flops_training_approx_6n,
    memory_budget,
    comm_bytes_data_parallel,
    comm_bytes_tensor_parallel,
    comm_bytes_pipeline_parallel_per_device,
    comm_bytes_pipeline_parallel_total,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUR_MODEL = GPTConfig(vocab_size=65, block_size=64, n_layer=4, n_head=4, d_model=128, d_ff=512)
# A GPT-2-small-scale config, for comparison only (never trained in this project):
GPT2_SMALL = GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, d_model=768, d_ff=3072)

B, T = 32, 64


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def flops_section():
    print("=== FLOPs (forward pass, this project's model, B=%d T=%d) ===" % (B, T))
    analytical = flops_forward_analytical(OUR_MODEL, B, T)
    measured = flops_forward_measured(OUR_MODEL, B, T)
    approx_6n_fwd = flops_training_approx_6n(OUR_MODEL, B, T) / 3  # 6N is fwd+bwd; fwd alone is ~2N
    print(f"  hand-derived (matmuls only):   {analytical:,}")
    print(f"  XLA cost_analysis (measured):  {measured:,.0f}")
    print(f"  ratio measured/analytical:     {measured / analytical:.4f}")
    print(f"  6N/3 large-model approximation: {approx_6n_fwd:,.0f} "
          f"(ratio to measured: {approx_6n_fwd / measured:.3f} -- "
          f"diverges because T={T} is not >> d_model={OUR_MODEL.d_model})")
    return {
        "analytical_flops": analytical,
        "measured_flops": measured,
        "ratio_measured_over_analytical": measured / analytical,
        "approx_6n_forward_flops": approx_6n_fwd,
    }


def memory_section():
    print("\n=== Memory budget (B=%d T=%d) ===" % (B, T))
    results = {}
    for name, config in [("this_project_model", OUR_MODEL), ("gpt2_small_scale", GPT2_SMALL)]:
        mb = memory_budget(config, B, T)
        print(f"-- {name} --")
        print(f"  params:              {human_bytes(mb.params_bytes)}")
        print(f"  grads:               {human_bytes(mb.grad_bytes)}")
        print(f"  optimizer state (Adam m,v): {human_bytes(mb.optimizer_state_bytes)}")
        print(f"  activations (no checkpointing):  {human_bytes(mb.activation_bytes)}")
        print(f"  activations (checkpointed):      {human_bytes(mb.activation_bytes_checkpointed)}")
        print(f"  TOTAL (no checkpointing):        {human_bytes(mb.total_bytes)}")
        print(f"  TOTAL (checkpointed):            {human_bytes(mb.total_bytes_checkpointed)}")
        results[name] = mb.__dict__
    return results


def communication_section():
    print("\n=== Communication volume vs device count (B=%d T=%d, M=8 microbatches) ===" % (B, T))
    device_counts = [2, 4, 8, 16, 32]
    M = 8
    data = {"data_parallel": [], "tensor_parallel": [], "pipeline_parallel_per_device": [],
            "pipeline_parallel_total": []}
    for p in device_counts:
        data["data_parallel"].append(comm_bytes_data_parallel(OUR_MODEL, p))
        data["tensor_parallel"].append(comm_bytes_tensor_parallel(OUR_MODEL, B, T, p))
        data["pipeline_parallel_per_device"].append(comm_bytes_pipeline_parallel_per_device(OUR_MODEL, B, T, p, M))
        data["pipeline_parallel_total"].append(comm_bytes_pipeline_parallel_total(OUR_MODEL, B, T, p, M))

    for p, dp, tp, ppd, ppt in zip(device_counts, data["data_parallel"], data["tensor_parallel"],
                                    data["pipeline_parallel_per_device"], data["pipeline_parallel_total"]):
        print(f"  P={p:3d}: data-parallel={human_bytes(dp):>10}  tensor-parallel={human_bytes(tp):>10}  "
              f"pipeline-parallel/device={human_bytes(ppd):>10}  pipeline-parallel/total={human_bytes(ppt):>10}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(device_counts, data["data_parallel"], "o-", label="data parallel (per device: ring all-reduce)")
    plt.plot(device_counts, data["tensor_parallel"], "s-", label="tensor parallel (per device: 2 all-reduces/block)")
    plt.plot(device_counts, data["pipeline_parallel_per_device"], "^-",
              label=f"pipeline parallel (per device, {M} microbatches)")
    plt.plot(device_counts, data["pipeline_parallel_total"], "^--", alpha=0.6,
              label=f"pipeline parallel (TOTAL across all links, {M} microbatches)")
    plt.xlabel("number of devices P")
    plt.ylabel("communication volume (bytes)")
    plt.yscale("log")
    plt.title("Communication volume vs. device count, by parallelism strategy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "communication_scaling.png"), dpi=130)
    plt.close()

    return {"device_counts": device_counts, **data}


def main():
    summary = {
        "flops": flops_section(),
        "memory": memory_section(),
        "communication": communication_section(),
    }
    with open(os.path.join(RESULTS_DIR, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote results/analysis_summary.json and results/communication_scaling.png")


if __name__ == "__main__":
    main()
