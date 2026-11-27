"""Single-device training loop: optimizer construction, a jitted train step,
and a runner that logs loss curves. This is the baseline everything else
(data/tensor/pipeline parallelism) is checked against for numerical
correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.data import get_batch
from src.model import GPTConfig, cross_entropy_loss, init_params


@dataclass
class TrainConfig:
    steps: int
    batch_size: int
    peak_lr: float = 3e-4
    warmup_steps: int = 50
    schedule: str = "warmup_cosine"  # or "constant"
    grad_clip_norm: float | None = 1.0
    weight_decay: float = 0.01
    seed: int = 0
    eval_every: int = 50
    eval_batches: int = 20


def make_lr_schedule(cfg: TrainConfig) -> optax.Schedule:
    if cfg.schedule == "constant":
        return optax.constant_schedule(cfg.peak_lr)
    if cfg.schedule == "warmup_cosine":
        decay_steps = max(cfg.steps - cfg.warmup_steps, 1)
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.peak_lr,
            warmup_steps=cfg.warmup_steps,
            decay_steps=decay_steps,
            end_value=cfg.peak_lr * 0.1,
        )
    raise ValueError(f"unknown schedule {cfg.schedule!r}")


def make_optimizer(cfg: TrainConfig) -> optax.GradientTransformation:
    schedule = make_lr_schedule(cfg)
    chain = []
    if cfg.grad_clip_norm is not None:
        chain.append(optax.clip_by_global_norm(cfg.grad_clip_norm))
    chain.append(optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay))
    return optax.chain(*chain)


def make_train_step(config: GPTConfig, optimizer: optax.GradientTransformation):
    def loss_fn(params, x, y):
        return cross_entropy_loss(params, x, y, config)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grad_norm

    return train_step


def eval_loss(params, config: GPTConfig, data: np.ndarray, cfg: TrainConfig, rng: np.random.Generator) -> float:
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(rng, data, cfg.batch_size, config.block_size)
        x, y = jnp.array(x), jnp.array(y)
        losses.append(float(cross_entropy_loss(params, x, y, config)))
    return float(np.mean(losses))


@dataclass
class TrainLog:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def train(
    config: GPTConfig,
    train_cfg: TrainConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> tuple[dict, TrainLog]:
    key = jax.random.PRNGKey(train_cfg.seed)
    params = init_params(key, config)
    optimizer = make_optimizer(train_cfg)
    opt_state = optimizer.init(params)
    train_step = make_train_step(config, optimizer)
    schedule = make_lr_schedule(train_cfg)

    rng = np.random.default_rng(train_cfg.seed)
    eval_rng = np.random.default_rng(train_cfg.seed + 10_000)
    log = TrainLog()

    for step in range(train_cfg.steps):
        x, y = get_batch(rng, train_data, train_cfg.batch_size, config.block_size)
        x, y = jnp.array(x), jnp.array(y)
        params, opt_state, loss, grad_norm = train_step(params, opt_state, x, y)

        log.steps.append(step)
        log.train_loss.append(float(loss))
        log.grad_norm.append(float(grad_norm))
        log.lr.append(float(schedule(step)))

        if (step + 1) % train_cfg.eval_every == 0 or step == train_cfg.steps - 1:
            vloss = eval_loss(params, config, val_data, train_cfg, eval_rng)
            log.val_steps.append(step)
            log.val_loss.append(vloss)

    return params, log
