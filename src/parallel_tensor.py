"""Tensor parallelism (Megatron-LM style) for the MLP and attention sublayers.

The trick for a 2-matrix sublayer (Linear -> nonlinearity -> Linear) is to
split the *first* matrix along its output dimension ("column-parallel") and
the *second* matrix along its input dimension ("row-parallel"):

    h_p      = f(x @ W1_p)        # local on each shard p, no communication
    out_p    = h_p @ W2_p         # local partial sum, shape (..., d_model)
    out      = all_reduce(out_p)  # ONE collective combines the P partial sums

Because the elementwise nonlinearity between the two matmuls only needs the
columns of W1 that live on its own shard, no communication is needed until
the very end -- a single all-reduce per sublayer, regardless of how many
shards P there are. This is the same decomposition Megatron-LM uses for
transformer MLP and multi-head attention blocks, and it generalizes directly
to splitting attention heads across shards.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

AXIS = "tp"


def shard_column_parallel(w: jnp.ndarray, n_shards: int) -> jnp.ndarray:
    """(d_in, d_out) -> (n_shards, d_in, d_out // n_shards)."""
    d_in, d_out = w.shape
    assert d_out % n_shards == 0
    return w.reshape(d_in, n_shards, d_out // n_shards).transpose(1, 0, 2)


def shard_row_parallel(w: jnp.ndarray, n_shards: int) -> jnp.ndarray:
    """(d_in, d_out) -> (n_shards, d_in // n_shards, d_out)."""
    d_in, d_out = w.shape
    assert d_in % n_shards == 0
    return w.reshape(n_shards, d_in // n_shards, d_out)


def shard_vector(b: jnp.ndarray, n_shards: int) -> jnp.ndarray:
    """(d,) -> (n_shards, d // n_shards), for a column-parallel bias."""
    (d,) = b.shape
    assert d % n_shards == 0
    return b.reshape(n_shards, d // n_shards)


def tensor_parallel_mlp(x: jnp.ndarray, w1_shard: jnp.ndarray, b1_shard: jnp.ndarray,
                         w2_shard: jnp.ndarray, b2: jnp.ndarray) -> jnp.ndarray:
    """x: (..., d_model) identical on every shard.
    w1_shard: (d_model, d_ff/P) column-parallel. b1_shard: (d_ff/P,).
    w2_shard: (d_ff/P, d_model) row-parallel. b2: (d_model,) replicated (not sharded)."""
    h = jax.nn.gelu(x @ w1_shard + b1_shard)  # local
    partial_out = h @ w2_shard  # local partial sum over this shard's d_ff slice
    return jax.lax.psum(partial_out, axis_name=AXIS) + b2


def tensor_parallel_attention(x: jnp.ndarray, wq_shard, bq_shard, wk_shard, bk_shard,
                               wv_shard, bv_shard, wo_shard, bo: jnp.ndarray,
                               heads_per_shard: int) -> jnp.ndarray:
    """Each shard owns a disjoint subset of attention heads end to end: the
    Q/K/V projections are column-parallel (split along the head dimension)
    so each shard computes complete attention for its own heads with no
    communication, then the output projection is row-parallel and the
    partial outputs are all-reduced -- exactly the MLP pattern, applied to
    the per-head slice of the model instead of the hidden dimension.

    x: (B, T, d_model) identical on every shard.
    wq/wk/wv_shard: (d_model, heads_per_shard * d_head) column-parallel.
    wo_shard: (heads_per_shard * d_head, d_model) row-parallel. bo: (d_model,) replicated.
    """
    B, T, D = x.shape
    d_shard = wq_shard.shape[-1]
    d_head = d_shard // heads_per_shard

    q = x @ wq_shard + bq_shard
    k = x @ wk_shard + bk_shard
    v = x @ wv_shard + bv_shard

    def split_heads(t):
        return t.reshape(B, T, heads_per_shard, d_head).transpose(0, 2, 1, 3)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(d_head)
    causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    scores = jnp.where(causal_mask, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, d_shard)  # local, this shard's heads only

    partial_out = out @ wo_shard  # local partial sum
    return jax.lax.psum(partial_out, axis_name=AXIS) + bo
