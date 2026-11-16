"""
Prediction Service for Speech Command Recognition.
Coordinates audio preprocessing, MFCC extraction, forward likelihood scoring,
and Viterbi state decoding.
"""

import os
import time
import io
import json
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.special import softmax

from src.features.mfcc_extractor import MFCCExtractor
from src.models.forward_algorithm import score_all_commands
from src.models.viterbi_decoder import decode_with_hmm_model
from src.models.hmm_trainer import load_trained_models, CORE_COMMANDS


class PredictionService:
    """Singleton service for speech command inference."""

    def __init__(self,
                 models_dir: str = os.path.join("models", "hmm_models"),
                 config_path: str = os.path.join("models", "model_config.json")):
        self.models_dir = models_dir
        self.config_path = config_path
        self.extractor = MFCCExtractor()
        self.models: Dict[str, Any] = {}
        self.commands = CORE_COMMANDS
        self.config: Dict[str, Any] = {}
        self.is_loaded = False
        self.load_models()

    def load_models(self) -> None:
        """Load trained HMM models and configuration from disk."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            self.commands = self.config.get("commands", self.commands)

        if os.path.isdir(self.models_dir):
            try:
                self.models = load_trained_models(self.models_dir)
                if self.models:
                    self.commands = sorted(list(self.models.keys()))
                    self.is_loaded = True
                    print(f"PredictionService: Successfully loaded {len(self.models)} HMM models: {self.commands}")
            except Exception as e:
                print(f"PredictionService: Warning loading models: {e}")

    def recognize_audio(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Process uploaded audio and return command recognition results.
        """
        if not self.models:
            raise RuntimeError("No HMM models loaded. Please train models first.")

        start_time = time.time()

        # 1. Load and preprocess audio, extract MFCCs
        clean_y, feat = self.extractor.process_file(audio_bytes)

        # 2. Score all command HMMs via Forward Algorithm in log-space
        scores = score_all_commands(self.models, feat, use_custom=True)

        # 3. Clean and sanitize scores for strict JSON compliance (no -inf, inf, or nan)
        safe_scores: Dict[str, float] = {}
        for cmd, sc in scores.items():
            if np.isnan(sc) or sc == -np.inf or sc < -1e7:
                safe_scores[cmd] = -1e6
            elif sc == np.inf or sc > 1e7:
                safe_scores[cmd] = 1e6
            else:
                safe_scores[cmd] = float(sc)

        # 4. Best command selection (argmax)
        best_cmd = max(safe_scores, key=safe_scores.get)
        best_score = float(safe_scores[best_cmd])

        # 5. Viterbi decoding on winner model
        best_model = self.models[best_cmd]
        try:
            viterbi_score, state_path = decode_with_hmm_model(best_model, feat, use_custom=True)
            if isinstance(state_path, np.ndarray):
                state_list = state_path.tolist()
            else:
                state_list = list(state_path)
        except Exception:
            state_list = [0] * len(feat)

        # 6. Softmax normalization for relative confidence across supported vocabulary
        cmd_keys = list(safe_scores.keys())
        raw_vals = np.array([safe_scores[k] for k in cmd_keys], dtype=np.float64)
        if not np.all(np.isfinite(raw_vals)):
            raw_vals = np.nan_to_num(raw_vals, nan=-1e6, neginf=-1e6, posinf=1e6)

        # Safe stable softmax
        shifted = np.clip(raw_vals - np.max(raw_vals), -500.0, 0.0)
        norm_probs = softmax(shifted)
        if not np.all(np.isfinite(norm_probs)):
            norm_probs = np.full(len(cmd_keys), 1.0 / len(cmd_keys))

        prob_dict = {cmd_keys[i]: float(norm_probs[i]) for i in range(len(cmd_keys))}

        total_time_ms = (time.time() - start_time) * 1000.0

        return {
            "predicted_command": best_cmd,
            "confidence_normalized": float(prob_dict[best_cmd]),
            "best_log_likelihood": best_score,
            "processing_time_ms": float(total_time_ms),
            "scores": safe_scores,
            "normalized_probabilities": prob_dict,
            "viterbi_state_path": state_list,
            "status": "success"
        }

    def compute_mfcc_data(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Compute MFCC feature matrix and statistics for visualization."""
        clean_y, feat = self.extractor.process_file(audio_bytes)
        T, D = feat.shape
        feat_clean = np.nan_to_num(feat, nan=0.0, neginf=-10.0, posinf=10.0)
        return {
            "time_frames": int(T),
            "feature_dim": int(D),
            "hop_length": int(self.extractor.hop_length),
            "sample_rate": int(self.extractor.sample_rate),
            "mfcc_matrix": feat_clean.tolist(),
            "summary_statistics": {
                "mean": float(np.mean(feat_clean)),
                "std": float(np.std(feat_clean)),
                "min": float(np.min(feat_clean)),
                "max": float(np.max(feat_clean))
            }
        }
