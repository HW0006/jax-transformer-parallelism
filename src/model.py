"""A decoder-only GPT-style transformer implemented from scratch in raw JAX.

No Flax, no Haiku: parameters are a plain pytree (nested dict of jnp arrays)
and the forward pass is a pure function `gpt_forward(params, idx, config)`.
Everything -- embeddings, layer norm, causal multi-head attention, the MLP,
residual connections, initialization scaling -- is written out explicitly so
the architecture's actual mechanics are visible and testable, rather than
hidden behind a library module.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int  # max context length
    n_layer: int
    n_head: int
    d_model: int
    d_ff: int  # MLP hidden size, conventionally 4 * d_model

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


# ---------------------------------------------------------------------------
# Parameter initialization
# ---------------------------------------------------------------------------


def _normal(key, shape, std):
    return std * jax.random.normal(key, shape)


def init_params(key: jax.Array, config: GPTConfig) -> dict:
    """GPT-2-style initialization: N(0, 0.02) for most weights, with the two
    per-block "residual projections" (attention output, MLP second linear)
    scaled by an extra 1/sqrt(2 * n_layer) so that the variance contributed
    to the residual stream by each of the 2*n_layer sublayers stays bounded
    as depth grows (Radford et al. 2019)."""
    keys = iter(jax.random.split(key, 4 + config.n_layer * 8))
    std = 0.02
    resid_std = std / jnp.sqrt(2.0 * config.n_layer)

    params = {
        "tok_emb": _normal(next(keys), (config.vocab_size, config.d_model), std),
        "pos_emb": _normal(next(keys), (config.block_size, config.d_model), std),
        "blocks": [],
        "ln_f": {"gamma": jnp.ones(config.d_model), "beta": jnp.zeros(config.d_model)},
        "head": _normal(next(keys), (config.d_model, config.vocab_size), std),
        "head_b": jnp.zeros(config.vocab_size),
    }

    for _ in range(config.n_layer):
        block = {
            "ln1": {"gamma": jnp.ones(config.d_model), "beta": jnp.zeros(config.d_model)},
            "attn": {
                "wq": _normal(next(keys), (config.d_model, config.d_model), std),
                "bq": jnp.zeros(config.d_model),
                "wk": _normal(next(keys), (config.d_model, config.d_model), std),
                "bk": jnp.zeros(config.d_model),
                "wv": _normal(next(keys), (config.d_model, config.d_model), std),
                "bv": jnp.zeros(config.d_model),
                "wo": _normal(next(keys), (config.d_model, config.d_model), resid_std),
                "bo": jnp.zeros(config.d_model),
            },
            "ln2": {"gamma": jnp.ones(config.d_model), "beta": jnp.zeros(config.d_model)},
            "mlp": {
                "w1": _normal(next(keys), (config.d_model, config.d_ff), std),
                "b1": jnp.zeros(config.d_ff),
                "w2": _normal(next(keys), (config.d_ff, config.d_model), resid_std),
                "b2": jnp.zeros(config.d_model),
            },
        }
        params["blocks"].append(block)

    return params


def count_params(params: dict) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(leaf.size for leaf in leaves))


# ---------------------------------------------------------------------------
# Forward pass building blocks
# ---------------------------------------------------------------------------


def layer_norm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + eps)
    return x_hat * gamma + beta


def causal_self_attention(x: jnp.ndarray, p: dict, n_head: int) -> jnp.ndarray:
    """x: (B, T, D). Explicit scaled dot-product attention with a causal mask,
    computed per head via a reshape rather than a library attention call."""
    B, T, D = x.shape
    d_head = D // n_head

    q = x @ p["wq"] + p["bq"]
    k = x @ p["wk"] + p["bk"]
    v = x @ p["wv"] + p["bv"]

    # (B, T, D) -> (B, n_head, T, d_head)
    def split_heads(t):
        return t.reshape(B, T, n_head, d_head).transpose(0, 2, 1, 3)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(d_head)  # (B, nh, T, T)
    causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    scores = jnp.where(causal_mask, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    out = weights @ v  # (B, nh, T, d_head)

    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ p["wo"] + p["bo"]


def mlp(x: jnp.ndarray, p: dict) -> jnp.ndarray:
    h = jax.nn.gelu(x @ p["w1"] + p["b1"])
    return h @ p["w2"] + p["b2"]


def transformer_block(x: jnp.ndarray, p: dict, n_head: int) -> jnp.ndarray:
    x = x + causal_self_attention(layer_norm(x, **p["ln1"]), p["attn"], n_head)
    x = x + mlp(layer_norm(x, **p["ln2"]), p["mlp"])
    return x


def gpt_forward(params: dict, idx: jnp.ndarray, config: GPTConfig) -> jnp.ndarray:
    """idx: (B, T) int32 token ids. Returns logits (B, T, vocab_size)."""
    B, T = idx.shape
    assert T <= config.block_size

    tok = params["tok_emb"][idx]  # (B, T, D)
    pos = params["pos_emb"][:T]  # (T, D)
    x = tok + pos

    for block in params["blocks"]:
        x = transformer_block(x, block, config.n_head)

    x = layer_norm(x, **params["ln_f"])
    logits = x @ params["head"] + params["head_b"]
    return logits


def cross_entropy_loss(params: dict, idx: jnp.ndarray, targets: jnp.ndarray, config: GPTConfig) -> jnp.ndarray:
    logits = gpt_forward(params, idx, config)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return jnp.mean(nll)
