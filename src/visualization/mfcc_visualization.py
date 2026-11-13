"""
Waveform, MFCC, and HMM State Visualization Module.
Generates:
- results/plots/waveform_visualization.png
- results/plots/mfcc_visualization.png
- results/plots/hmm_state_visualization.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.features.mfcc_extractor import MFCCExtractor
from src.models.viterbi_decoder import decode_with_hmm_model
from src.models.hmm_trainer import load_trained_models


def plot_waveform(audio_path: str, output_path: str, extractor: MFCCExtractor = None) -> None:
    """Plot raw and normalized/pre-emphasized waveform."""
    if extractor is None:
        extractor = MFCCExtractor()

    raw_y, sr = extractor.load_audio(audio_path)
    clean_y = extractor.preprocess_signal(raw_y)

    time_axis = np.linspace(0, len(clean_y) / sr, len(clean_y))

    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(time_axis, clean_y, color="#2563eb", linewidth=1.0)
    plt.title(f"Acoustic Waveform: {os.path.basename(audio_path)}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Time (seconds)", fontsize=11, fontweight="medium")
    plt.ylabel("Normalized Amplitude", fontsize=11, fontweight="medium")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved waveform plot to {output_path}")


def plot_mfcc(audio_path: str, output_path: str, extractor: MFCCExtractor = None) -> None:
    """Plot 2D heatmap of extracted MFCC acoustic features."""
    if extractor is None:
        extractor = MFCCExtractor()

    clean_y, feat = extractor.process_file(audio_path)
    # feat shape: (T, D) -> transpose to (D, T) for display
    feat_t = feat.T

    plt.figure(figsize=(10, 5), dpi=300)
    img = plt.imshow(feat_t, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    cbar = plt.colorbar(img)
    cbar.set_label("Standardized Coefficient Value", fontsize=10)

    plt.title(f"MFCC Feature Spectrogram (D={feat.shape[1]} coefficients)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Time Frame (10 ms hop)", fontsize=11, fontweight="medium")
    plt.ylabel("MFCC + Delta Index", fontsize=11, fontweight="medium")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved MFCC spectrogram to {output_path}")


def plot_hmm_states(audio_path: str,
                    command: str,
                    output_path: str,
                    models_dir: str = os.path.join("models", "hmm_models"),
                    extractor: MFCCExtractor = None) -> None:
    """Plot Viterbi optimal hidden state sequence alignment."""
    if extractor is None:
        extractor = MFCCExtractor()

    models = load_trained_models(models_dir)
    if command not in models:
        print(f"Warning: Model for {command} not found.")
        return

    clean_y, feat = extractor.process_file(audio_path)
    model = models[command]
    best_score, states = decode_with_hmm_model(model, feat, use_custom=True)

    time_frames = np.arange(len(states))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=300)

    # Upper plot: Waveform envelope
    ax1.plot(clean_y, color="#3b82f6", alpha=0.8)
    ax1.set_title(f"Command '{command}' Waveform & Decoded HMM States", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Lower plot: Viterbi State Path (step plot)
    # Map frame indices to samples for visualization
    frame_samples = time_frames * extractor.hop_length
    ax2.step(frame_samples, states, where="mid", color="#8b5cf6", linewidth=2.5, label="Hidden State")
    ax2.set_ylabel("HMM State", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Sample Index", fontsize=10)
    ax2.set_yticks(np.arange(model.n_components))
    ax2.set_yticklabels([f"S{i}" for i in range(model.n_components)])
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved HMM state alignment plot to {output_path}")
