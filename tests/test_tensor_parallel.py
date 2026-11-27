"""Correctness check for tensor parallelism: the column-parallel -> row-parallel
-> all-reduce decomposition of the MLP and attention sublayers must produce
exactly the same output as the single-device (unsharded) reference
computation in src/model.py, for any number of shards P that divides the
hidden/head dimension.

Run with:
  XLA_FLAGS="--xla_force_host_platform_device_count=4" python3 -m tests.test_tensor_parallel
"""

import os

assert "host_platform_device_count" in os.environ.get("XLA_FLAGS", ""), (
    "run this with XLA_FLAGS=--xla_force_host_platform_device_count=N set"
)

import jax
import jax.numpy as jnp

from src.model import GPTConfig, init_params, mlp as ref_mlp, causal_self_attention as ref_attention
from src.parallel_tensor import (
    AXIS,
    shard_column_parallel,
    shard_row_parallel,
    shard_vector,
    tensor_parallel_mlp,
    tensor_parallel_attention,
)


def test_mlp(config: GPTConfig, block_params: dict, x: jnp.ndarray, n_shards: int):
    ref_out = ref_mlp(x, block_params["mlp"])

    w1_s = shard_column_parallel(block_params["mlp"]["w1"], n_shards)
    b1_s = shard_vector(block_params["mlp"]["b1"], n_shards)
    w2_s = shard_row_parallel(block_params["mlp"]["w2"], n_shards)
    b2 = block_params["mlp"]["b2"]  # replicated, not sharded

    def shard_fn(w1_shard, b1_shard, w2_shard, x_local, b2_local):
        return tensor_parallel_mlp(x_local, w1_shard, b1_shard, w2_shard, b2_local)

    pmapped = jax.pmap(shard_fn, axis_name=AXIS, in_axes=(0, 0, 0, None, None))
    out = pmapped(w1_s, b1_s, w2_s, x, b2)  # (n_shards, B, T, d_model), identical across shards

    max_diff = float(jnp.max(jnp.abs(out[0] - ref_out)))
    spread = float(jnp.max(jnp.abs(out - out[0:1])))
    print(f"[mlp, P={n_shards}] max diff vs reference: {max_diff:.3e}, max spread across shards: {spread:.3e}")
    assert max_diff < 1e-4, "tensor-parallel MLP does not match single-device reference"
    assert spread < 1e-6, "shards disagree after all-reduce -- communication is broken"


def test_attention(config: GPTConfig, block_params: dict, x: jnp.ndarray, n_shards: int):
    ref_out = ref_attention(x, block_params["attn"], config.n_head)

    assert config.n_head % n_shards == 0
    heads_per_shard = config.n_head // n_shards
    attn = block_params["attn"]
    wq_s = shard_column_parallel(attn["wq"], n_shards)
    bq_s = shard_vector(attn["bq"], n_shards)
    wk_s = shard_column_parallel(attn["wk"], n_shards)
    bk_s = shard_vector(attn["bk"], n_shards)
    wv_s = shard_column_parallel(attn["wv"], n_shards)
    bv_s = shard_vector(attn["bv"], n_shards)
    wo_s = shard_row_parallel(attn["wo"], n_shards)
    bo = attn["bo"]  # replicated

    def shard_fn(wq, bq, wk, bk, wv, bv, wo, x_local, bo_local):
        return tensor_parallel_attention(x_local, wq, bq, wk, bk, wv, bv, wo, bo_local, heads_per_shard)

    pmapped = jax.pmap(shard_fn, axis_name=AXIS, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))
    out = pmapped(wq_s, bq_s, wk_s, bk_s, wv_s, bv_s, wo_s, x, bo)

    max_diff = float(jnp.max(jnp.abs(out[0] - ref_out)))
    spread = float(jnp.max(jnp.abs(out - out[0:1])))
    print(f"[attn, P={n_shards}] max diff vs reference: {max_diff:.3e}, max spread across shards: {spread:.3e}")
    assert max_diff < 1e-4, "tensor-parallel attention does not match single-device reference"
    assert spread < 1e-6, "shards disagree after all-reduce -- communication is broken"


def main():
    n_devices = jax.device_count()
    print(f"simulated devices: {n_devices}")

    config = GPTConfig(vocab_size=32, block_size=16, n_layer=1, n_head=4, d_model=32, d_ff=64)
    key = jax.random.PRNGKey(0)
    params = init_params(key, config)
    block_params = params["blocks"][0]

    x = jax.random.normal(jax.random.PRNGKey(1), (2, config.block_size, config.d_model))

    for n_shards in [1, 2, n_devices]:
        if n_shards > n_devices or config.n_head % n_shards != 0:
            continue
        test_mlp(config, block_params, x, n_shards)
        test_attention(config, block_params, x, n_shards)

    print("PASS: tensor parallelism matches the single-device reference at every shard count tested.")


if __name__ == "__main__":
    main()
