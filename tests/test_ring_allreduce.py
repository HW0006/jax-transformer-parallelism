"""Correctness check for the from-scratch ring all-reduce: it must produce
exactly the same result as jax.lax.psum on every simulated device, for
array sizes that are and are not evenly divisible by the device count.

Run with:
  XLA_FLAGS="--xla_force_host_platform_device_count=4" python3 -m tests.test_ring_allreduce
"""

import os

assert "host_platform_device_count" in os.environ.get("XLA_FLAGS", ""), (
    "run this with XLA_FLAGS=--xla_force_host_platform_device_count=N set"
)

import jax
import jax.numpy as jnp

from src.collectives import ring_all_reduce_flat

AXIS = "ring"


def check(n_devices: int, length: int, seed: int):
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (n_devices, length))

    def ring_fn(x_local):
        return ring_all_reduce_flat(x_local, AXIS, n_devices)

    def psum_fn(x_local):
        return jax.lax.psum(x_local, AXIS)

    ring_out = jax.pmap(ring_fn, axis_name=AXIS)(x)
    psum_out = jax.pmap(psum_fn, axis_name=AXIS)(x)

    diff = float(jnp.max(jnp.abs(ring_out - psum_out)))
    spread = float(jnp.max(jnp.abs(ring_out - ring_out[0:1])))
    status = "OK" if diff < 1e-4 and spread < 1e-5 else "FAIL"
    print(f"P={n_devices} length={length}: diff_vs_psum={diff:.3e} spread_across_devices={spread:.3e} [{status}]")
    assert diff < 1e-4, "ring all-reduce does not match jax.lax.psum"
    assert spread < 1e-5, "devices disagree after ring all-reduce"


def main():
    n_devices = jax.device_count()
    print(f"simulated devices: {n_devices}")
    for length in [n_devices * 10, n_devices * 10 + 1, n_devices * 10 - 1, 7, 1]:
        if length < 1:
            continue
        check(n_devices, length, seed=length)
    print("PASS: from-scratch ring all-reduce matches jax.lax.psum in every case tested.")


if __name__ == "__main__":
    main()
