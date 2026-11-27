"""Data parallelism via jax.pmap.

The model is replicated identically on every device; a global batch is split
evenly across devices; each device computes the loss/gradient on its local
shard; gradients are averaged across devices with an all-reduce
(`jax.lax.pmean` over the `"devices"` axis) before every replica applies the
*same* averaged update. That all-reduce is the one piece of communication
data parallelism needs per step, and it's why data-parallel training with D
devices and per-device batch size B is mathematically identical to
single-device training on a batch of size D*B drawn from the same examples:
mean-of-means over equal-sized shards is the overall mean.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from src.model import GPTConfig, cross_entropy_loss

AXIS = "devices"


def make_data_parallel_train_step(config: GPTConfig, optimizer: optax.GradientTransformation):
    def loss_fn(params, x, y):
        return cross_entropy_loss(params, x, y, config)

    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        # all-reduce: average gradients (and the reported loss) across devices
        grads = jax.lax.pmean(grads, axis_name=AXIS)
        loss = jax.lax.pmean(loss, axis_name=AXIS)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return jax.pmap(step, axis_name=AXIS)


def replicate(pytree, n_devices: int):
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_devices,) + x.shape), pytree)


def shard_batch(x: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    """(D*B, ...) -> (D, B, ...)."""
    assert x.shape[0] % n_devices == 0
    per_device = x.shape[0] // n_devices
    return x.reshape((n_devices, per_device) + x.shape[1:])
