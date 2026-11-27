"""Quantitative reasoning about compute, memory, and communication cost.

Every number here is produced two ways where possible: a hand-derived
analytical formula from the architecture, and an empirical measurement
(XLA's own cost-analysis pass, or an actual traced communication volume),
so the two can be checked against each other rather than trusting either
one blindly.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.model import GPTConfig, count_params, gpt_forward, init_params


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def flops_forward_analytical(config: GPTConfig, B: int, T: int) -> int:
    """Sum of the FLOPs (multiply+add counted as 2) of every matmul in the
    forward pass. Excludes layer norm, softmax, GELU, and embedding lookups,
    which are O(B*T*D) or smaller and negligible next to the O(B*T*D^2) and
    O(B*T^2*D) matmul terms for any non-tiny D."""
    D, T_ff = config.d_model, config.d_ff
    per_block = (
        6 * B * T * D * D  # Q, K, V projections: 3 matmuls of (T,D)x(D,D)
        + 4 * B * T * T * D  # attention scores (Q@K^T) + weighted sum (weights@V)
        + 2 * B * T * D * D  # output projection
        + 4 * B * T * D * T_ff  # MLP: two matmuls (T,D)x(D,d_ff) and (T,d_ff)x(d_ff,D)
    )
    head = 2 * B * T * D * config.vocab_size
    return config.n_layer * per_block + head


def flops_forward_measured(config: GPTConfig, B: int, T: int) -> float:
    """XLA's own cost-analysis estimate for the compiled forward pass, as a
    cross-check on the analytical formula above."""
    key = jax.random.PRNGKey(0)
    params = init_params(key, config)
    x = jnp.zeros((B, T), dtype=jnp.int32)
    compiled = jax.jit(lambda p, x: gpt_forward(p, x, config)).lower(params, x).compile()
    return compiled.cost_analysis()["flops"]


def flops_training_approx_6n(config: GPTConfig, B: int, T: int) -> int:
    """The widely-used large-scale approximation: ~6 FLOPs per parameter per
    token for a full forward+backward step (2N forward, 4N backward),
    ignoring attention's O(T^2) term -- accurate when D >> T, which is the
    opposite regime from this project's small model / short context."""
    n_non_embedding = count_params(init_params(jax.random.PRNGKey(0), config)) - (
        config.vocab_size * config.d_model + config.block_size * config.d_model
    )
    return 6 * n_non_embedding * B * T


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass
class MemoryBudget:
    params_bytes: int
    grad_bytes: int
    optimizer_state_bytes: int
    activation_bytes: int
    activation_bytes_checkpointed: int

    @property
    def static_bytes(self) -> int:
        return self.params_bytes + self.grad_bytes + self.optimizer_state_bytes

    @property
    def total_bytes(self) -> int:
        return self.static_bytes + self.activation_bytes

    @property
    def total_bytes_checkpointed(self) -> int:
        return self.static_bytes + self.activation_bytes_checkpointed


def memory_budget(config: GPTConfig, B: int, T: int, dtype_bytes: int = 4) -> MemoryBudget:
    """AdamW optimizer state holds two extra fp32 moment buffers per
    parameter (the first and second moment estimates m and v), so training
    memory for parameters+grads+optimizer-state is 4x the parameter count in
    elements (1 for params, 1 for grads, 2 for Adam's m/v) -- 16 bytes/param
    at fp32, regardless of model size.

    Activation memory (no recomputation) is dominated, per block, by: the
    residual-stream copy (B*T*D), the attention score matrix (B*n_head*T*T),
    and the MLP hidden activation (B*T*d_ff) -- each needs to be kept around
    for the backward pass. With activation checkpointing, only the
    block-boundary residual stream is kept (B*T*D per block) and everything
    else is recomputed during backward, trading ~33% more forward-pass
    compute for a large memory reduction when d_ff and T are large relative
    to D.
    """
    n_params = count_params(init_params(jax.random.PRNGKey(0), config))
    params_bytes = n_params * dtype_bytes
    grad_bytes = n_params * dtype_bytes
    optimizer_state_bytes = 2 * n_params * dtype_bytes  # Adam: m, v

    D, T_ff, H = config.d_model, config.d_ff, config.n_head
    per_block_full = B * T * D + B * H * T * T + B * T * T_ff  # residual + attn scores + MLP hidden
    activation_bytes = config.n_layer * per_block_full * dtype_bytes

    per_block_checkpointed = B * T * D  # only the residual stream at each block boundary
    activation_bytes_checkpointed = config.n_layer * per_block_checkpointed * dtype_bytes

    return MemoryBudget(
        params_bytes=params_bytes,
        grad_bytes=grad_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        activation_bytes=activation_bytes,
        activation_bytes_checkpointed=activation_bytes_checkpointed,
    )


# ---------------------------------------------------------------------------
# Communication volume
# ---------------------------------------------------------------------------


def ring_allreduce_bytes_per_device(total_bytes: int, n_devices: int) -> float:
    """Bandwidth-optimal ring all-reduce: each device sends/receives
    2(P-1)/P * total_bytes -- approaches 2x the buffer size as P grows,
    independent of P, rather than scaling with P the way a naive
    all-to-all all-reduce would."""
    if n_devices <= 1:
        return 0.0
    return 2 * (n_devices - 1) / n_devices * total_bytes


def comm_bytes_data_parallel(config: GPTConfig, n_devices: int, dtype_bytes: int = 4) -> float:
    """One gradient all-reduce per training step."""
    n_params = count_params(init_params(jax.random.PRNGKey(0), config))
    return ring_allreduce_bytes_per_device(n_params * dtype_bytes, n_devices)


def comm_bytes_tensor_parallel(config: GPTConfig, B: int, T: int, n_devices: int, dtype_bytes: int = 4) -> float:
    """Two all-reduces per block per forward pass (attention output,
    MLP output), each of size B*T*d_model; roughly doubles for the backward
    pass, so we report forward and total(fwd+bwd) separately via the
    multiplier the caller applies."""
    activation_bytes = B * T * config.d_model * dtype_bytes
    per_allreduce = ring_allreduce_bytes_per_device(activation_bytes, n_devices)
    allreduces_per_block = 2  # attention output + MLP output
    return config.n_layer * allreduces_per_block * per_allreduce


def comm_bytes_pipeline_parallel_per_device(config: GPTConfig, B: int, T: int, n_devices: int,
                                             n_microbatches: int, dtype_bytes: int = 4) -> float:
    """Point-to-point sends, forward pass only. Unlike data/tensor
    parallel's all-reduce, each device here only ever talks to its two ring
    neighbors: an interior stage receives one activation and sends one
    activation per microbatch, i.e. 2*M*activation_bytes -- a volume that
    does NOT grow with the number of stages P. (The first and last stages
    do only one direction, so this is the worst-case/interior-stage cost.)
    This is the fairest number to compare against data/tensor parallel's
    genuinely per-device ring-allreduce volume."""
    if n_devices <= 1:
        return 0.0
    activation_bytes = B * T * config.d_model * dtype_bytes
    return 2 * n_microbatches * activation_bytes


def comm_bytes_pipeline_parallel_total(config: GPTConfig, B: int, T: int, n_devices: int, n_microbatches: int,
                                        dtype_bytes: int = 4) -> float:
    """Total bytes moved system-wide, summed over all (P-1) stage
    boundaries -- this DOES grow linearly with P, because more stages means
    more boundaries, even though no single device's own traffic grows."""
    activation_bytes = B * T * config.d_model * dtype_bytes
    n_boundaries = n_devices - 1
    return n_boundaries * n_microbatches * activation_bytes
