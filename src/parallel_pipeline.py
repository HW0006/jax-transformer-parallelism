"""GPipe-style pipeline parallelism.

The model's L transformer blocks are split into P contiguous stages, one per
device. A batch is split into M microbatches, which are fed into stage 0 one
per time-step; each stage hands its output to the next stage via a
point-to-point send (`jax.lax.ppermute`, restricted to the "shift to my
neighbor" permutation) so that at steady state, P microbatches are in flight
across the P stages simultaneously -- this is what makes pipelining better
than naive "one device idle while another works" model parallelism.

The whole fill-drain schedule is one `jax.lax.scan` over T = M + P - 1
micro-steps, wrapped in a single `jax.pmap` over the P stage-devices: every
device runs the *same* program (required by pmap/SPMD) but only does real
work when `0 <= step - device_id < M`, using `jax.lax.axis_index` to find its
own position and `jnp.where` to mask out the idle steps. Those idle steps are
the pipeline "bubble": with P stages and M microbatches, exactly (P-1) of the
P*(M+P-1) device-steps are unavoidably idle while the pipeline fills and
drains, for a bubble fraction of (P-1)/(M+P-1) -- independent of model size,
and shrinking as M grows relative to P.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.model import transformer_block

AXIS = "pipeline"


def stack_blocks(blocks: list[dict]):
    """List of per-layer param dicts -> one pytree with a leading `n_layer` axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *blocks)


def split_into_stages(stacked_blocks, n_stages: int):
    """(n_layer, ...) leaves -> (n_stages, layers_per_stage, ...) leaves."""
    n_layer = jax.tree_util.tree_leaves(stacked_blocks)[0].shape[0]
    assert n_layer % n_stages == 0, "n_layer must be divisible by n_stages"
    layers_per_stage = n_layer // n_stages
    return jax.tree_util.tree_map(
        lambda x: x.reshape((n_stages, layers_per_stage) + x.shape[1:]), stacked_blocks
    )


def run_stage(x: jnp.ndarray, stage_blocks, n_head: int) -> jnp.ndarray:
    """Sequentially apply this stage's blocks (leading axis = layers_per_stage) via scan."""

    def body(carry_x, block_params):
        return transformer_block(carry_x, block_params, n_head), None

    out, _ = jax.lax.scan(body, x, stage_blocks)
    return out


def pipeline_forward(stage_blocks_sharded, microbatches: jnp.ndarray, n_head: int, n_stages: int):
    """stage_blocks_sharded: pytree with leading axis n_stages (pmap-mapped).
    microbatches: (M, B, T, D) activations, REPLICATED (identical on every
    device) -- only device 0 actually consumes them, at the step matching
    its own microbatch index.

    Returns, per device: (collected, active) each of shape (T_steps, B, T, D) / (T_steps,).
    Only the last device's `collected` entries, at steps [P-1, P, ..., P+M-2],
    are the real pipeline outputs for microbatches [0, 1, ..., M-1].
    """
    M = microbatches.shape[0]
    pid = jax.lax.axis_index(AXIS)
    n_steps = microbatches.shape[0] + n_stages - 1
    zeros_like_mb = jnp.zeros_like(microbatches[0])

    def step(incoming, t):
        local_mb_idx = t - pid
        is_active = (local_mb_idx >= 0) & (local_mb_idx < M)
        safe_idx = jnp.clip(local_mb_idx, 0, M - 1)
        fresh_input = microbatches[safe_idx]
        stage_input = jnp.where(pid == 0, fresh_input, incoming)

        stage_output = run_stage(stage_input, stage_blocks_sharded, n_head)
        stage_output = jnp.where(is_active, stage_output, zeros_like_mb)

        perm = [(i, (i + 1) % n_stages) for i in range(n_stages)]
        next_incoming = jax.lax.ppermute(stage_output, AXIS, perm=perm)

        is_last_stage = pid == n_stages - 1
        collected = jnp.where(is_active & is_last_stage, stage_output, zeros_like_mb)
        return next_incoming, (collected, is_active)

    init_incoming = zeros_like_mb
    _, (collected, active) = jax.lax.scan(step, init_incoming, jnp.arange(n_steps))
    return collected, active


def pipeline_forward_pmapped(n_head: int, n_stages: int):
    def fn(stage_blocks_sharded, microbatches):
        return pipeline_forward(stage_blocks_sharded, microbatches, n_head, n_stages)

    return jax.pmap(fn, axis_name=AXIS, in_axes=(0, None))


def bubble_fraction(n_stages: int, n_microbatches: int) -> float:
    """Analytical GPipe bubble fraction: (P-1)/(M+P-1)."""
    return (n_stages - 1) / (n_microbatches + n_stages - 1)
