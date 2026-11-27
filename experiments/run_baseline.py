"""Training-dynamics experiments: a multi-seed baseline, an LR-schedule
ablation (warmup+cosine vs constant), and a gradient-clipping ablation.

Run with:  python3 -m experiments.run_baseline
Writes JSON logs and PNG plots to results/.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import GPTConfig
from src.train import TrainConfig, train
from src.data import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_CONFIG = GPTConfig(vocab_size=65, block_size=64, n_layer=4, n_head=4, d_model=128, d_ff=512)
SEEDS = [0, 1, 2]


def run_seeds(train_cfg_template: TrainConfig, train_data, val_data, config: GPTConfig):
    logs = []
    for seed in SEEDS:
        cfg = TrainConfig(**{**train_cfg_template.__dict__, "seed": seed})
        t0 = time.time()
        _, log = train(config, cfg, train_data, val_data)
        dt = time.time() - t0
        print(f"  seed={seed} done in {dt:.1f}s, final train loss={log.train_loss[-1]:.4f}")
        logs.append(log)
    return logs


def summarize_final(logs, window=50):
    finals = [float(np.mean(l.train_loss[-window:])) for l in logs]
    val_finals = [l.val_loss[-1] for l in logs]
    grad_norms = [float(np.mean(l.grad_norm)) for l in logs]
    grad_norm_max = [float(np.max(l.grad_norm)) for l in logs]
    return {
        "train_loss_mean": float(np.mean(finals)),
        "train_loss_std": float(np.std(finals)),
        "val_loss_mean": float(np.mean(val_finals)),
        "val_loss_std": float(np.std(val_finals)),
        "grad_norm_mean": float(np.mean(grad_norms)),
        "grad_norm_max_mean": float(np.mean(grad_norm_max)),
    }


def save_logs(name, logs):
    payload = [l.__dict__ for l in logs]
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump(payload, f)


def plot_loss_curves(logs_by_label: dict, title: str, out_name: str, key="train_loss"):
    plt.figure(figsize=(7, 4.5))
    colors = plt.cm.tab10.colors
    for i, (label, logs) in enumerate(logs_by_label.items()):
        for j, log in enumerate(logs):
            steps = log.steps if key == "train_loss" else log.val_steps
            ys = log.train_loss if key == "train_loss" else log.val_loss
            plt.plot(steps, ys, color=colors[i % 10], alpha=0.35 if j > 0 else 0.9,
                      label=label if j == 0 else None, linewidth=1.3)
    plt.xlabel("step")
    plt.ylabel("loss" if key == "train_loss" else "val loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, out_name), dpi=130)
    plt.close()


def plot_grad_norms(logs_by_label: dict, title: str, out_name: str):
    plt.figure(figsize=(7, 4.5))
    colors = plt.cm.tab10.colors
    for i, (label, logs) in enumerate(logs_by_label.items()):
        for j, log in enumerate(logs):
            plt.plot(log.steps, log.grad_norm, color=colors[i % 10], alpha=0.35 if j > 0 else 0.9,
                      label=label if j == 0 else None, linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("global grad norm")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, out_name), dpi=130)
    plt.close()


def main():
    tokenizer, train_data, val_data = load_dataset(
        os.path.join(os.path.dirname(__file__), "..", "data", "tiny_shakespeare.txt")
    )
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=64, n_layer=4, n_head=4, d_model=128, d_ff=512)
    summary = {}

    # --- 1. Main baseline: warmup+cosine, peak_lr=5e-4, clip=1.0, 1500 steps, 3 seeds ---
    print("Baseline (warmup_cosine, peak_lr=5e-4, clip=1.0)...")
    baseline_cfg = TrainConfig(steps=1500, batch_size=32, warmup_steps=50, peak_lr=5e-4,
                                schedule="warmup_cosine", grad_clip_norm=1.0, eval_every=300)
    baseline_logs = run_seeds(baseline_cfg, train_data, val_data, config)
    save_logs("baseline", baseline_logs)
    summary["baseline"] = summarize_final(baseline_logs)
    plot_loss_curves({"warmup_cosine (baseline)": baseline_logs}, "Baseline training loss (3 seeds)", "baseline_loss.png")

    # --- 2. LR schedule ablation: warmup_cosine vs constant, same budget ---
    print("Ablation: constant LR schedule...")
    constant_cfg = TrainConfig(**{**baseline_cfg.__dict__, "schedule": "constant"})
    constant_logs = run_seeds(constant_cfg, train_data, val_data, config)
    save_logs("schedule_ablation_constant", constant_logs)
    summary["schedule_constant"] = summarize_final(constant_logs)
    plot_loss_curves(
        {"warmup_cosine": baseline_logs, "constant": constant_logs},
        "LR schedule ablation: warmup+cosine vs constant", "schedule_ablation_loss.png",
    )

    # --- 3. Gradient-clipping ablation: higher peak LR where clipping matters ---
    print("Ablation: gradient clipping (peak_lr=3e-3)...")
    clip_cfg = TrainConfig(steps=300, batch_size=32, warmup_steps=10, peak_lr=3e-3,
                            schedule="warmup_cosine", grad_clip_norm=1.0, eval_every=100)
    clip_logs = run_seeds(clip_cfg, train_data, val_data, config)
    save_logs("clip_ablation_clipped", clip_logs)
    summary["clip_1.0"] = summarize_final(clip_logs)

    noclip_cfg = TrainConfig(**{**clip_cfg.__dict__, "grad_clip_norm": None})
    noclip_logs = run_seeds(noclip_cfg, train_data, val_data, config)
    save_logs("clip_ablation_unclipped", noclip_logs)
    summary["clip_none"] = summarize_final(noclip_logs)

    plot_loss_curves(
        {"clip=1.0": clip_logs, "clip=None": noclip_logs},
        "Gradient-clipping ablation (peak_lr=3e-3): loss", "clip_ablation_loss.png",
    )
    plot_grad_norms(
        {"clip=1.0": clip_logs, "clip=None": noclip_logs},
        "Gradient-clipping ablation (peak_lr=3e-3): global grad norm", "clip_ablation_gradnorm.png",
    )

    with open(os.path.join(RESULTS_DIR, "training_dynamics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary (mean +/- std over 3 seeds) ===")
    for name, s in summary.items():
        print(f"{name}: train_loss={s['train_loss_mean']:.4f}+/-{s['train_loss_std']:.4f}  "
              f"val_loss={s['val_loss_mean']:.4f}+/-{s['val_loss_std']:.4f}  "
              f"grad_norm_mean={s['grad_norm_mean']:.2f}  grad_norm_max_mean={s['grad_norm_max_mean']:.2f}")


if __name__ == "__main__":
    main()
