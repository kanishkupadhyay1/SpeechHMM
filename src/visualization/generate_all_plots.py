"""
Comprehensive plot generation script.
Generates all 6 required plots in results/plots/:
1. confusion_matrix.png
2. command_accuracy.png
3. waveform_visualization.png
4. mfcc_visualization.png
5. hmm_state_visualization.png
6. command_likelihood_comparison.png
"""

import os
import glob
from src.visualization.evaluation_plots import plot_confusion_matrix, plot_command_accuracy
from src.visualization.mfcc_visualization import plot_waveform, plot_mfcc, plot_hmm_states
from src.visualization.likelihood_visualization import plot_likelihood_comparison


def generate_all(results_dir: str = "results",
                 dataset_dir: str = os.path.join("data", "raw", "speech_commands_v0.02"),
                 models_dir: str = os.path.join("models", "hmm_models")) -> None:
    plots_dir = os.path.join(results_dir, "plots")
    metrics_path = os.path.join(results_dir, "metrics", "evaluation_metrics.json")
    os.makedirs(plots_dir, exist_ok=True)

    print("Generating evaluation and acoustic plots...")

    # 1 & 2: Evaluation Plots
    if os.path.exists(metrics_path):
        plot_confusion_matrix(metrics_path, os.path.join(plots_dir, "confusion_matrix.png"))
        plot_command_accuracy(metrics_path, os.path.join(plots_dir, "command_accuracy.png"))
    else:
        print(f"Metrics not found at {metrics_path}")

    # Pick a sample WAV file for signal and HMM state visualizations
    sample_wavs = glob.glob(os.path.join(dataset_dir, "yes", "*.wav"))
    if not sample_wavs:
        sample_wavs = glob.glob(os.path.join(dataset_dir, "*", "*.wav"))

    if sample_wavs:
        sample_audio = sample_wavs[0]
        sample_cmd = os.path.basename(os.path.dirname(sample_audio))
        print(f"Using sample audio for acoustic visualizations: {sample_audio} ('{sample_cmd}')")

        # 3. Waveform
        plot_waveform(sample_audio, os.path.join(plots_dir, "waveform_visualization.png"))

        # 4. MFCC
        plot_mfcc(sample_audio, os.path.join(plots_dir, "mfcc_visualization.png"))

        # 5. HMM State Alignment
        plot_hmm_states(sample_audio, sample_cmd, os.path.join(plots_dir, "hmm_state_visualization.png"), models_dir)

        # 6. Command Likelihood Comparison
        plot_likelihood_comparison(sample_audio, os.path.join(plots_dir, "command_likelihood_comparison.png"), models_dir)

    print(f"All plots generated in {plots_dir}")


if __name__ == "__main__":
    generate_all()
