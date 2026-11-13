"""
Likelihood Comparison Visualization Module.
Generates:
- results/plots/command_likelihood_comparison.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict

from src.features.mfcc_extractor import MFCCExtractor
from src.models.forward_algorithm import score_all_commands
from src.models.hmm_trainer import load_trained_models


def plot_likelihood_comparison(audio_path: str,
                               output_path: str,
                               models_dir: str = os.path.join("models", "hmm_models"),
                               extractor: MFCCExtractor = None) -> None:
    """Plot comparative forward log-likelihood scores across all candidate commands."""
    if extractor is None:
        extractor = MFCCExtractor()

    models = load_trained_models(models_dir)
    _, feat = extractor.process_file(audio_path)
    scores = score_all_commands(models, feat, use_custom=True)

    sorted_cmds = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    sorted_scores = [scores[c] for c in sorted_cmds]
    best_cmd = sorted_cmds[0]

    colors = ["#10b981" if c == best_cmd else "#3b82f6" for c in sorted_cmds]

    plt.figure(figsize=(10, 5), dpi=300)
    bars = plt.barh(sorted_cmds[::-1], sorted_scores[::-1], color=colors[::-1], alpha=0.85, edgecolor="#1e293b")

    plt.xlabel("Forward Log Likelihood ln P(X | lambda)", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Command Model", fontsize=11, fontweight="bold", labelpad=10)
    plt.title(f"Acoustic Likelihood Comparison (Winner: '{best_cmd}')\nAudio: {os.path.basename(audio_path)}",
              fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle=":", alpha=0.6)

    # Annotate score text on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}",
                 va="center", ha="left", fontsize=9, color="#1e293b", fontweight="medium")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved command likelihood comparison plot to {output_path}")
