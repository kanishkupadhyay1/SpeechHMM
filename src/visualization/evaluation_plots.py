"""
Evaluation Plotting Module.
Generates:
- results/plots/confusion_matrix.png
- results/plots/command_accuracy.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(metrics_json_path: str, output_path: str) -> None:
    """Plot high-resolution annotated confusion matrix heatmap."""
    with open(metrics_json_path, "r") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"])
    commands = data["commands"]

    plt.figure(figsize=(9, 8), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=commands,
        yticklabels=commands,
        cbar=True,
        linewidths=0.5,
        linecolor="#e2e8f0"
    )
    plt.title(f"Speech Command HMM Confusion Matrix\nAccuracy: {data['accuracy']*100:.2f}% | Macro F1: {data['macro_f1']*100:.2f}%",
              fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Command", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Command", fontsize=12, fontweight="bold", labelpad=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {output_path}")


def plot_command_accuracy(metrics_json_path: str, output_path: str) -> None:
    """Plot per-command precision, recall, and F1 bar chart."""
    with open(metrics_json_path, "r") as f:
        data = json.load(f)

    commands = data["commands"]
    per_class = data["per_class_metrics"]

    precision = [per_class[c]["precision"] * 100 for c in commands]
    recall = [per_class[c]["recall"] * 100 for c in commands]
    f1 = [per_class[c]["f1_score"] * 100 for c in commands]

    x = np.arange(len(commands))
    width = 0.25

    plt.figure(figsize=(12, 6), dpi=300)
    plt.bar(x - width, precision, width, label="Precision", color="#3b82f6", alpha=0.9)
    plt.bar(x, recall, width, label="Recall", color="#8b5cf6", alpha=0.9)
    plt.bar(x + width, f1, width, label="F1 Score", color="#10b981", alpha=0.9)

    plt.axhline(data["macro_f1"] * 100, color="#ef4444", linestyle="--", linewidth=1.5,
                label=f"Macro F1 ({data['macro_f1']*100:.1f}%)")

    plt.xlabel("Spoken Command", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Score (%)", fontsize=12, fontweight="bold", labelpad=10)
    plt.title("Per-Command Classification Performance (Held-out Test Split)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, commands, fontsize=11, fontweight="medium")
    plt.ylim(0, 105)
    plt.legend(frameon=True, facecolor="#f8fafc", loc="lower right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved per-command performance plot to {output_path}")


if __name__ == "__main__":
    metrics_path = os.path.join("results", "metrics", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        plot_confusion_matrix(metrics_path, os.path.join("results", "plots", "confusion_matrix.png"))
        plot_command_accuracy(metrics_path, os.path.join("results", "plots", "command_accuracy.png"))
    else:
        print("Metrics file not yet created.")
