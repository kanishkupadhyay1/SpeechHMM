"""
Model Evaluation Module for Speech Command Recognition.
Evaluates trained command HMMs on the official held-out test split.
Calculates:
- Overall Accuracy
- Macro Precision, Recall, F1 Score
- Per-class Precision, Recall, F1 Score
- Confusion Matrix
- Likelihood statistics
Saves results to:
- results/metrics/evaluation_metrics.json
- results/metrics/classification_report.csv
- results/predictions/test_predictions.csv
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.models.forward_algorithm import score_all_commands
from src.models.viterbi_decoder import decode_with_hmm_model
from src.models.hmm_trainer import load_trained_models, CORE_COMMANDS


def evaluate_test_set(test_cache_path: str,
                      models_dir: str = os.path.join("models", "hmm_models"),
                      output_dir: str = "results",
                      use_custom_forward: bool = False) -> Dict[str, Any]:
    """
    Run full evaluation on held-out test set using trained HMMs.
    """
    print(f"Loading trained models from {models_dir}...")
    models = load_trained_models(models_dir)
    commands = sorted(list(models.keys()))
    print(f"Loaded {len(models)} command models: {commands}")

    print(f"Loading test features from {test_cache_path}...")
    test_data = joblib.load(test_cache_path)
    features = test_data["features"]
    true_labels = test_data["labels"]
    file_paths = test_data["file_paths"]

    total_samples = len(features)
    print(f"Evaluating {total_samples} test recordings...")

    pred_records = []
    y_true = []
    y_pred = []

    start_eval_time = time.time()

    for idx, (feat, label, fpath) in enumerate(zip(features, true_labels, file_paths)):
        if label not in models:
            continue

        sample_start = time.time()
        # Compute forward log likelihood for all command models
        scores = score_all_commands(models, feat, use_custom=use_custom_forward)
        sample_time = time.time() - sample_start

        # Command with highest forward log likelihood wins
        best_cmd = max(scores, key=scores.get)
        best_score = scores[best_cmd]

        # Viterbi decoding on the predicted command model
        best_model = models[best_cmd]
        viterbi_score, state_path = decode_with_hmm_model(best_model, feat, use_custom=True)

        y_true.append(label)
        y_pred.append(best_cmd)

        record = {
            "file_path": fpath,
            "true_command": label,
            "predicted_command": best_cmd,
            "is_correct": int(label == best_cmd),
            "best_log_likelihood": float(best_score),
            "viterbi_log_score": float(viterbi_score),
            "state_sequence": ",".join(map(str, state_path.tolist())),
            "processing_time_sec": float(sample_time)
        }
        # Add per-command scores to record
        for c in commands:
            record[f"score_{c}"] = float(scores.get(c, -1e9))

        pred_records.append(record)

        if (idx + 1) % 500 == 0 or (idx + 1) == total_samples:
            interim_acc = accuracy_score(y_true, y_pred)
            print(f"  Processed {idx + 1}/{total_samples} test samples (Current Accuracy: {interim_acc * 100:.2f}%)")

    total_eval_time = time.time() - start_eval_time
    avg_inference_time = total_eval_time / (len(y_true) + 1e-6)

    # Compute overall and per-class metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=commands, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=commands, average="weighted", zero_division=0
    )

    per_class_prec, per_class_rec, per_class_f1, per_class_sup = precision_recall_fscore_support(
        y_true, y_pred, labels=commands, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=commands).tolist()

    # Per-class summary dictionary
    per_class_metrics = {}
    for i, c in enumerate(commands):
        per_class_metrics[c] = {
            "precision": float(per_class_prec[i]),
            "recall": float(per_class_rec[i]),
            "f1_score": float(per_class_f1[i]),
            "support": int(per_class_sup[i])
        }

    # Save test predictions CSV
    preds_dir = os.path.join(output_dir, "predictions")
    os.makedirs(preds_dir, exist_ok=True)
    df_preds = pd.DataFrame(pred_records)
    preds_csv_path = os.path.join(preds_dir, "test_predictions.csv")
    df_preds.to_csv(preds_csv_path, index=False)
    print(f"Predictions saved to {preds_csv_path}")

    # Save classification report CSV
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    report_dict = classification_report(y_true, y_pred, labels=commands, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(metrics_dir, "classification_report.csv")
    df_report.to_csv(report_csv_path)

    # Save evaluation_metrics.json
    final_metrics = {
        "dataset": "Google Speech Commands v0.02",
        "split": "test",
        "num_test_samples": len(y_true),
        "commands": commands,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm,
        "evaluation_time_total_sec": float(total_eval_time),
        "avg_inference_time_sec": float(avg_inference_time)
    }

    metrics_json_path = os.path.join(metrics_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Evaluation metrics saved to {metrics_json_path}")

    print("\n================== TEST SET EVALUATION ==================")
    print(f"Total Test Samples: {len(y_true)}")
    print(f"Accuracy:         {accuracy * 100:.2f}%")
    print(f"Macro Precision:  {macro_prec * 100:.2f}%")
    print(f"Macro Recall:     {macro_rec * 100:.2f}%")
    print(f"Macro F1 Score:   {macro_f1 * 100:.2f}%")
    print(f"Avg Inference:    {avg_inference_time * 1000:.2f} ms/sample")
    print("=========================================================\n")

    return final_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Speech Command HMMs on test set")
    parser.add_argument("--test-cache", default=os.path.join("data", "features", "test_features.joblib"),
                        help="Path to cached test features")
    parser.add_argument("--models-dir", default=os.path.join("models", "hmm_models"),
                        help="Path to trained models")
    parser.add_argument("--output-dir", default="results", help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    evaluate_test_set(args.test_cache, args.models_dir, args.output_dir)
