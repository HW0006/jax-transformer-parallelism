"""Correctness check for data parallelism: D steps of pmap'd data-parallel
training on D simulated devices (per-device batch B) must match
single-device training on the same concatenated batch of size D*B, since
averaging per-device gradient means is the same as the overall mean.

Run with:
  XLA_FLAGS="--xla_force_host_platform_device_count=4" python3 -m tests.test_data_parallel
"""

import os

assert "host_platform_device_count" in os.environ.get("XLA_FLAGS", ""), (
    "run this with XLA_FLAGS=--xla_force_host_platform_device_count=N set, "
    "e.g. XLA_FLAGS='--xla_force_host_platform_device_count=4' python3 -m tests.test_data_parallel"
)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.model import GPTConfig, init_params, cross_entropy_loss
from src.parallel_data import make_data_parallel_train_step, replicate, shard_batch, AXIS


def main():
    n_devices = jax.device_count()
    print(f"simulated devices: {n_devices}")
    assert n_devices > 1, "need more than 1 simulated device for this test to be meaningful"

    config = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, d_model=32, d_ff=64)
    per_device_batch = 6
    global_batch = per_device_batch * n_devices

    key = jax.random.PRNGKey(0)
    params0 = init_params(key, config)
    optimizer = optax.sgd(learning_rate=0.1)  # a deterministic optimizer makes exact comparison cleaner

    rng = np.random.default_rng(0)
    x_np = rng.integers(0, config.vocab_size, size=(global_batch, config.block_size)).astype(np.int32)
    y_np = rng.integers(0, config.vocab_size, size=(global_batch, config.block_size)).astype(np.int32)

    # --- single-device reference: one step on the full global batch ---
    def loss_fn(params, x, y):
        return cross_entropy_loss(params, x, y, config)

    ref_opt_state = optimizer.init(params0)
    ref_loss, ref_grads = jax.value_and_grad(loss_fn)(params0, jnp.array(x_np), jnp.array(y_np))
    ref_updates, _ = optimizer.update(ref_grads, ref_opt_state, params0)
    ref_params = optax.apply_updates(params0, ref_updates)

    # --- data-parallel: same batch, sharded across n_devices ---
    dp_step = make_data_parallel_train_step(config, optimizer)
    dp_params = replicate(params0, n_devices)
    dp_opt_state = jax.pmap(optimizer.init)(dp_params)
    x_sharded = shard_batch(jnp.array(x_np), n_devices)
    y_sharded = shard_batch(jnp.array(y_np), n_devices)

    dp_params, dp_opt_state, dp_loss = dp_step(dp_params, dp_opt_state, x_sharded, y_sharded)

    # every replica should hold identical params (same averaged update applied everywhere)
    replica_diffs = jax.tree_util.tree_map(
        lambda p: float(jnp.max(jnp.abs(p[1:] - p[0:1]))), dp_params
    )
    max_replica_diff = max(jax.tree_util.tree_leaves(replica_diffs))
    print(f"max difference between replicas after 1 step: {max_replica_diff:.3e}")
    assert max_replica_diff < 1e-6, "replicas diverged -- all-reduce is not synchronizing correctly"

    # compare replica 0's params to the single-device reference
    dp_params_r0 = jax.tree_util.tree_map(lambda p: p[0], dp_params)
    max_diff = max(
        jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda a, b: float(jnp.max(jnp.abs(a - b))), dp_params_r0, ref_params)
        )
    )
    loss_diff = abs(float(dp_loss[0]) - float(ref_loss))
    print(f"max param difference vs single-device reference: {max_diff:.3e}")
    print(f"loss difference vs single-device reference: {loss_diff:.3e}")

    assert max_diff < 1e-5, "data-parallel result does not match single-device reference"
    assert loss_diff < 1e-5, "data-parallel loss does not match single-device reference"
    print("PASS: data parallelism matches the single-device reference.")


if __name__ == "__main__":
    main()
