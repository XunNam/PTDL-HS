import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def find_history_file(out_dir: str, history_file: Optional[str] = None) -> str:
    if history_file:
        if not os.path.exists(history_file):
            raise FileNotFoundError(f"History file not found: {history_file}")
        return history_file

    candidates = glob.glob(os.path.join(out_dir, "training_history_*.json"))
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy file training_history_*.json trong: {out_dir}")
    return max(candidates, key=os.path.getctime)


def _extract_val_metrics(val_metrics: List[Dict]) -> Tuple[List[int], List[float], List[float], List[float]]:
    val_epochs = [int(m["epoch"]) for m in val_metrics if "epoch" in m]
    val_losses = [float(m.get("val_loss", 0.0)) for m in val_metrics if "epoch" in m]
    val_hits = [float(m.get("hit", 0.0)) for m in val_metrics if "epoch" in m]
    val_recalls = [float(m.get("recall", 0.0)) for m in val_metrics if "epoch" in m]
    return val_epochs, val_losses, val_hits, val_recalls


def _extract_train_eval_metrics(train_eval_metrics: List[Dict]) -> Tuple[List[int], List[float], List[float], List[float]]:
    epochs = [int(m["epoch"]) for m in train_eval_metrics if "epoch" in m]
    losses = [float(m.get("train_eval_loss", m.get("val_loss", 0.0))) for m in train_eval_metrics if "epoch" in m]
    hits = [float(m.get("train_hit", 0.0)) for m in train_eval_metrics if "epoch" in m]
    recalls = [float(m.get("train_recall", 0.0)) for m in train_eval_metrics if "epoch" in m]
    return epochs, losses, hits, recalls


def plot_latest_history(
    out_dir: str = "runs_decagon_full",
    history_file: Optional[str] = None,
    include_sampled_loss: bool = True,
    output_path: Optional[str] = None,
) -> str:
    history_path = find_history_file(out_dir=out_dir, history_file=history_file)
    print(f"📈 Đang vẽ biểu đồ từ file: {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = [int(x) for x in history.get("epochs", [])]
    sampled_losses = [float(x) for x in history.get("loss", [])]

    val_metrics = history.get("val_metrics", [])
    val_epochs, val_losses, val_hits, val_recalls = _extract_val_metrics(val_metrics)

    train_eval_metrics = history.get("train_eval_metrics", [])
    train_eval_epochs, train_eval_losses, train_hits, train_recalls = _extract_train_eval_metrics(train_eval_metrics)

    timestamp = os.path.basename(history_path).replace("training_history_", "").replace(".json", "")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Decagon Training Results - Run: {timestamp}", fontsize=14, fontweight="bold")

    # --- Left: losses ---
    if include_sampled_loss and epochs and sampled_losses:
        axes[0].plot(
            epochs,
            sampled_losses,
            label="Training (sampled triplet loss)",
            color="#4C78A8",
            linewidth=1.2,
            alpha=0.8,
        )
    if train_eval_epochs and train_eval_losses:
        axes[0].plot(
            train_eval_epochs,
            train_eval_losses,
            label="Train Eval Loss (all-relations BCE)",
            color="#59A14F",
            linewidth=2.0,
        )
    if val_epochs and val_losses and any(v > 0 for v in val_losses):
        axes[0].plot(
            val_epochs,
            val_losses,
            label="Validation Loss (all-relations BCE)",
            color="#F28E2B",
            linewidth=2.0,
        )
    axes[0].set_title("Loss Curves (Comparable vs Training Objective)")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # --- Right: retrieval metrics ---
    if val_epochs and val_hits:
        axes[1].plot(val_epochs, val_hits, label="Val Hit@K", color="#1f77b4", linewidth=1.8)
    if val_epochs and val_recalls:
        axes[1].plot(val_epochs, val_recalls, label="Val Recall@K", color="#ff7f0e", linewidth=1.8)
    if train_eval_epochs and train_hits and any(h > 0 for h in train_hits):
        axes[1].plot(train_eval_epochs, train_hits, label="Train Hit@K (eval-style)", color="#2ca02c", linestyle="--")
    if train_eval_epochs and train_recalls and any(r > 0 for r in train_recalls):
        axes[1].plot(
            train_eval_epochs,
            train_recalls,
            label="Train Recall@K (eval-style)",
            color="#d62728",
            linestyle="--",
        )
    axes[1].set_title("Retrieval Metrics (Hit@K / Recall@K)")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    plt.tight_layout()

    if output_path is None:
        target_dir = out_dir if history_file is None else os.path.dirname(history_path) or "."
        output_path = os.path.join(target_dir, f"plot_result_{timestamp}.png")

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Đã xuất biểu đồ thành công tại: {output_path}")
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs_decagon_full", help="Directory containing training_history_*.json")
    ap.add_argument("--history-file", default=None, help="Specific history JSON path (overrides --out-dir latest)")
    ap.add_argument("--hide-sampled-loss", action="store_true", help="Hide sampled triplet training loss curve")
    ap.add_argument("--output", default=None, help="Output PNG path")
    args = ap.parse_args()

    plot_latest_history(
        out_dir=args.out_dir,
        history_file=args.history_file,
        include_sampled_loss=not args.hide_sampled_loss,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
