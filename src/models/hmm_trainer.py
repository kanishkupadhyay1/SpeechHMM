"""
HMM Training Module for Isolated Speech Command Recognition.
Trains command-specific Continuous Gaussian Hidden Markov Models (GaussianHMM)
using the Baum-Welch Expectation-Maximization (EM) algorithm.
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from hmmlearn.hmm import GaussianHMM

CORE_COMMANDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]


class CommandHMMTrainer:
    """
    Trains and serializes command-specific Gaussian Hidden Markov Models.
    """

    def __init__(self,
                 commands: Optional[List[str]] = None,
                 n_states: int = 5,
                 covariance_type: str = "diag",
                 n_iter: int = 25,
                 tol: float = 1e-2,
                 random_state: int = 42):
        self.commands = commands or CORE_COMMANDS
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.models: Dict[str, GaussianHMM] = {}

    def _init_left_to_right_transitions(self, n_states: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize Left-to-Right (Bakis) topology:
        Start only in initial state s_0, transition to current state or next state.
        """
        startprob = np.zeros(n_states)
        startprob[0] = 1.0

        transmat = np.zeros((n_states, n_states))
        for i in range(n_states):
            if i == n_states - 1:
                transmat[i, i] = 1.0
            else:
                transmat[i, i] = 0.6
                transmat[i, i + 1] = 0.4
        return startprob, transmat

    def build_model(self) -> GaussianHMM:
        """Create a new Gaussian HMM instance initialized with left-to-right topology."""
        startprob, transmat = self._init_left_to_right_transitions(self.n_states)
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            init_params="mc",
            params="stmc",
            random_state=self.random_state,
            verbose=False
        )
        model.startprob_ = startprob
        model.transmat_ = transmat
        return model

    def train_single_command(self,
                             command: str,
                             feature_sequences: List[np.ndarray],
                             lengths: List[int]) -> GaussianHMM:
        """
        Fit an HMM on all training sequences for a specific spoken command.
        Uses Baum-Welch (EM) to maximize observation sequence log-likelihood.
        """
        print(f"Training HMM for command '{command}' with {len(lengths)} sequences ({sum(lengths)} frames)...", flush=True)
        if len(feature_sequences) == 0:
            raise ValueError(f"No feature sequences provided for command {command}")

        # Concatenate all sequences into (total_frames, feature_dim)
        X = np.vstack(feature_sequences)

        model = self.build_model()
        model.fit(X, lengths)

        converged = getattr(model.monitor_, "converged", True)
        final_iter = getattr(model.monitor_, "iter", self.n_iter)
        print(f"  Command '{command}' HMM fitted in {final_iter} iterations (Converged: {converged})", flush=True)

        return model

    def train_all(self,
                  train_cache_path: str,
                  output_dir: str = os.path.join("models", "hmm_models")) -> Dict[str, GaussianHMM]:
        """
        Train HMMs for all commands from cached features and save models.
        """
        print(f"Loading training features from cache: {train_cache_path}...", flush=True)
        data = joblib.load(train_cache_path)
        all_features = data["features"]
        all_labels = data["labels"]
        all_lengths = data["lengths"]

        # Group by command
        command_data: Dict[str, Tuple[List[np.ndarray], List[int]]] = {cmd: ([], []) for cmd in self.commands}
        for feat, label, length in zip(all_features, all_labels, all_lengths):
            if label in command_data:
                command_data[label][0].append(feat)
                command_data[label][1].append(length)

        os.makedirs(output_dir, exist_ok=True)
        self.models = {}

        # Check existing models
        for cmd in self.commands:
            mpath = os.path.join(output_dir, f"{cmd}.pkl")
            if os.path.exists(mpath):
                try:
                    self.models[cmd] = joblib.load(mpath)
                    print(f"  Command '{cmd}' already trained -> loaded from {mpath}", flush=True)
                except Exception:
                    pass

        pending_cmds = [cmd for cmd in self.commands if cmd not in self.models]

        for cmd in pending_cmds:
            seqs, lens = command_data[cmd]
            if len(seqs) == 0:
                print(f"Warning: No training data found for command '{cmd}'", flush=True)
                continue
            model = self.train_single_command(cmd, seqs, lens)
            model_path = os.path.join(output_dir, f"{cmd}.pkl")
            joblib.dump(model, model_path)
            self.models[cmd] = model
            print(f"  Saved {cmd} HMM to {model_path}", flush=True)

        # Save label encoder and model metadata
        models_root = os.path.dirname(output_dir) if os.path.basename(output_dir) == "hmm_models" else output_dir
        label_encoder_path = os.path.join(models_root, "label_encoder.pkl")
        joblib.dump(self.commands, label_encoder_path)

        config_path = os.path.join(models_root, "model_config.json")
        config = {
            "commands": self.commands,
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "extractor_config": data.get("extractor_config", {})
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nAll models saved successfully to {output_dir}")
        print(f"Model config saved to {config_path}\n")
        return self.models


def load_trained_models(models_dir: str = os.path.join("models", "hmm_models")) -> Dict[str, GaussianHMM]:
    """Load all serialized command HMMs from disk."""
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    models = {}
    for fname in os.listdir(models_dir):
        if fname.endswith(".pkl"):
            cmd = fname[:-4]
            path = os.path.join(models_dir, fname)
            models[cmd] = joblib.load(path)
    return models


if __name__ == "__main__":
    trainer = CommandHMMTrainer()
    print("CommandHMMTrainer initialized for commands:", trainer.commands)
