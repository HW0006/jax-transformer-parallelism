"""Sanity tests for the from-scratch model: shapes, a near-uniform initial
loss (sanity check on initialization), and non-zero gradients everywhere.

Run with:  python3 -m tests.test_model
"""

import jax
import jax.numpy as jnp

from src.model import GPTConfig, init_params, gpt_forward, cross_entropy_loss, count_params


def main():
    config = GPTConfig(vocab_size=37, block_size=16, n_layer=3, n_head=4, d_model=32, d_ff=64)
    key = jax.random.PRNGKey(0)
    params = init_params(key, config)

    n_params = count_params(params)
    print(f"param count: {n_params:,}")
    assert n_params > 0

    B, T = 5, config.block_size
    x = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, config.vocab_size)
    y = jax.random.randint(jax.random.PRNGKey(2), (B, T), 0, config.vocab_size)

    logits = gpt_forward(params, x, config)
    assert logits.shape == (B, T, config.vocab_size), logits.shape
    print(f"logits shape OK: {logits.shape}")

    loss = cross_entropy_loss(params, x, y, config)
    expected = jnp.log(config.vocab_size)
    diff = abs(float(loss) - float(expected))
    print(f"initial loss={float(loss):.4f}, ln(vocab_size)={float(expected):.4f}, diff={diff:.4f}")
    assert diff < 0.5, "initial loss is far from the near-uniform expectation -- check initialization"

    grads = jax.grad(cross_entropy_loss)(params, x, y, config)
    leaves = jax.tree_util.tree_leaves(grads)
    n_zero_grad_leaves = sum(1 for g in leaves if float(jnp.sum(jnp.abs(g))) == 0.0)
    print(f"{n_zero_grad_leaves} / {len(leaves)} gradient leaves are identically zero")
    assert n_zero_grad_leaves == 0, "some parameters received no gradient at all"

    print("PASS: model shapes, initialization, and gradient flow all check out.")


if __name__ == "__main__":
    main()
