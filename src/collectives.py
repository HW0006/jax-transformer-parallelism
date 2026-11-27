"""A ring all-reduce implemented from scratch on top of `jax.lax.ppermute`
(point-to-point send/receive), instead of using JAX's built-in `lax.psum`.

The ring algorithm is the classic bandwidth-optimal all-reduce (as used by
NCCL/Horovod): split each device's buffer into P equal chunks, arrange the P
devices in a ring, and run two phases of P-1 point-to-point steps each:

  1. Scatter-reduce: at step k, every device sends one chunk to its ring
     neighbor and adds an incoming chunk into its own buffer. After P-1
     steps, device p holds the complete sum for exactly one chunk.
  2. All-gather: another P-1 steps circulate each device's completed chunk
     around the ring (overwrite instead of add) until every device holds
     every chunk's complete sum.

Each device sends 2(P-1) chunks of size (total/P) in total, i.e. a data
volume of 2(P-1)/P * total_size -- independent of P for large P, and much
better than a naive "everyone sends their full buffer to everyone" scheme,
which sends P times as much data per device.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def _send_recv(x, axis_name, n_devices):
    """Send to the next device in the ring, receive from the previous one."""
    perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    return jax.lax.ppermute(x, axis_name, perm=perm)


def ring_all_reduce_flat(x: jnp.ndarray, axis_name: str, n_devices: int) -> jnp.ndarray:
    """x: a 1-D array, identical shape on every device. Returns the elementwise
    sum across all devices, identical on every device (same contract as
    `jax.lax.psum(x, axis_name)`)."""
    pid = jax.lax.axis_index(axis_name)
    n = x.shape[0]
    pad = (-n) % n_devices
    x_padded = jnp.pad(x, (0, pad))
    chunk_size = x_padded.shape[0] // n_devices
    chunks = x_padded.reshape(n_devices, chunk_size)

    def scatter_reduce_step(chunks, k):
        send_idx = (pid - k) % n_devices
        recv_idx = (pid - k - 1) % n_devices
        send_chunk = jnp.take(chunks, send_idx, axis=0)
        received = _send_recv(send_chunk, axis_name, n_devices)
        updated = jnp.take(chunks, recv_idx, axis=0) + received
        chunks = chunks.at[recv_idx].set(updated)
        return chunks, None

    chunks, _ = jax.lax.scan(scatter_reduce_step, chunks, jnp.arange(n_devices - 1))

    def all_gather_step(chunks, k):
        send_idx = (pid + 1 - k) % n_devices
        recv_idx = (pid - k) % n_devices
        send_chunk = jnp.take(chunks, send_idx, axis=0)
        received = _send_recv(send_chunk, axis_name, n_devices)
        chunks = chunks.at[recv_idx].set(received)
        return chunks, None

    chunks, _ = jax.lax.scan(all_gather_step, chunks, jnp.arange(n_devices - 1))

    return chunks.reshape(-1)[:n]


def ring_all_reduce_pytree(tree, axis_name: str, n_devices: int):
    """Ring all-reduce applied leaf-by-leaf to an arbitrary pytree (e.g. a
    gradient pytree), flattening each leaf to 1-D for the ring algorithm and
    reshaping back."""

    def reduce_leaf(x):
        flat = ring_all_reduce_flat(x.reshape(-1), axis_name, n_devices)
        return flat.reshape(x.shape)

    return jax.tree_util.tree_map(reduce_leaf, tree)
