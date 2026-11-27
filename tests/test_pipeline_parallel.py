"""Correctness check for pipeline parallelism: running M microbatches through
a GPipe-style P-stage pipeline (P simulated devices) must give the same
per-microbatch output as running all n_layer blocks sequentially on a single
device, and the pipeline's own busy/idle step accounting must match the
textbook bubble-fraction formula (P-1)/(M+P-1).

Run with:
  XLA_FLAGS="--xla_force_host_platform_device_count=4" python3 -m tests.test_pipeline_parallel
"""

import os

assert "host_platform_device_count" in os.environ.get("XLA_FLAGS", ""), (
    "run this with XLA_FLAGS=--xla_force_host_platform_device_count=N set"
)

import jax
import jax.numpy as jnp

from src.model import GPTConfig, init_params, transformer_block
from src.parallel_pipeline import stack_blocks, split_into_stages, pipeline_forward_pmapped, bubble_fraction


def reference_forward(x, blocks: list[dict], n_head: int):
    for block in blocks:
        x = transformer_block(x, block, n_head)
    return x


def run_case(config: GPTConfig, blocks: list[dict], n_stages: int, M: int):
    B, T, D = 2, config.block_size, config.d_model
    key = jax.random.PRNGKey(42)
    microbatches = jax.random.normal(key, (M, B, T, D))

    ref_outputs = [reference_forward(microbatches[m], blocks, config.n_head) for m in range(M)]

    stacked = stack_blocks(blocks)
    sharded = split_into_stages(stacked, n_stages)
    pmapped = pipeline_forward_pmapped(config.n_head, n_stages)
    collected, active = pmapped(sharded, microbatches)
    # collected/active: (n_stages, n_steps, ...) -- only the LAST stage's entries are real outputs
    last_stage_collected = collected[n_stages - 1]

    n_steps = M + n_stages - 1
    for m in range(M):
        step_idx = m + n_stages - 1
        out = last_stage_collected[step_idx]
        diff = float(jnp.max(jnp.abs(out - ref_outputs[m])))
        assert diff < 1e-4, f"microbatch {m}: pipeline output does not match reference (diff={diff:.3e})"

    total_slots = n_stages * n_steps
    busy_slots = int(jnp.sum(active))
    empirical_bubble = 1.0 - busy_slots / total_slots
    expected_bubble = bubble_fraction(n_stages, M)
    print(f"P={n_stages} M={M}: all {M} microbatches match reference. "
          f"busy={busy_slots}/{total_slots} slots, bubble: empirical={empirical_bubble:.4f} "
          f"vs analytical (P-1)/(M+P-1)={expected_bubble:.4f}")
    assert abs(empirical_bubble - expected_bubble) < 1e-9


def main():
    n_devices = jax.device_count()
    print(f"simulated devices: {n_devices}")

    config = GPTConfig(vocab_size=32, block_size=8, n_layer=4, n_head=2, d_model=16, d_ff=32)
    key = jax.random.PRNGKey(0)
    params = init_params(key, config)
    blocks = params["blocks"]

    for n_stages in [2, 4]:
        if n_stages > n_devices:
            continue
        for M in [1, 3, 8]:
            run_case(config, blocks, n_stages, M)

    print("PASS: pipeline parallelism matches the single-device reference, and bubble accounting matches theory.")


if __name__ == "__main__":
    main()
