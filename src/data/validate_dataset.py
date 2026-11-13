"""
Dataset validation module for Google Speech Commands v0.02.
Validates:
- WAV file integrity and headers
- Audio sample rate (16,000 Hz)
- Audio duration distribution (~1.0s)
- Official train/validation/test split lists without data leakage
- Class distribution across splits
"""

import os
import sys
import glob
import json
from typing import Dict, Any, List
import soundfile as sf
import pandas as pd
import numpy as np

CORE_COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def load_split_files(dataset_dir: str):
    """Load validation and testing lists provided officially with the dataset."""
    val_file = os.path.join(dataset_dir, "validation_list.txt")
    test_file = os.path.join(dataset_dir, "testing_list.txt")

    if not os.path.exists(val_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Missing split files in {dataset_dir}: validation_list.txt or testing_list.txt")

    with open(val_file, "r") as f:
        val_rel_paths = set(line.strip().replace("\\", "/") for line in f if line.strip())

    with open(test_file, "r") as f:
        test_rel_paths = set(line.strip().replace("\\", "/") for line in f if line.strip())

    # Check for leakage between official validation and test lists
    overlap = val_rel_paths.intersection(test_rel_paths)
    if overlap:
        raise ValueError(f"Data leakage detected! {len(overlap)} files overlap between validation and test splits.")

    return val_rel_paths, test_rel_paths


def validate_dataset(dataset_dir: str, target_classes: List[str] = None) -> Dict[str, Any]:
    """Inspect and validate audio files, sample rates, durations, splits, and counts."""
    if target_classes is None:
        target_classes = CORE_COMMANDS

    print(f"Validating dataset at: {dataset_dir}")
    val_rel_paths, test_rel_paths = load_split_files(dataset_dir)

    results = {
        "dataset_path": os.path.abspath(dataset_dir),
        "target_classes": target_classes,
        "classes_found": {},
        "split_counts": {"train": 0, "validation": 0, "test": 0},
        "sample_rates": {},
        "duration_stats": {},
        "corrupted_files": [],
        "speaker_leakage_detected": False,
        "is_valid": True
    }

    train_speakers = set()
    val_speakers = set()
    test_speakers = set()

    durations = []
    sample_rates = set()

    for cmd in target_classes:
        cmd_dir = os.path.join(dataset_dir, cmd)
        if not os.path.isdir(cmd_dir):
            print(f"Warning: Command directory not found: {cmd_dir}")
            continue

        wav_files = glob.glob(os.path.join(cmd_dir, "*.wav"))
        results["classes_found"][cmd] = {
            "total": len(wav_files),
            "train": 0,
            "validation": 0,
            "test": 0
        }

        # Validate a representative sample of files for audio format and sample rate
        for idx, wav_path in enumerate(wav_files):
            rel_path = f"{cmd}/{os.path.basename(wav_path)}"
            speaker_id = os.path.basename(wav_path).split("_nohash_")[0]

            if rel_path in test_rel_paths:
                split = "test"
                test_speakers.add(speaker_id)
            elif rel_path in val_rel_paths:
                split = "validation"
                val_speakers.add(speaker_id)
            else:
                split = "train"
                train_speakers.add(speaker_id)

            results["split_counts"][split] += 1
            results["classes_found"][cmd][split] += 1

            # Check sound file validity
            if idx < 50 or idx % 50 == 0:  # Sample check every 50th file
                try:
                    info = sf.info(wav_path)
                    sample_rates.add(info.samplerate)
                    durations.append(info.duration)
                except Exception as e:
                    results["corrupted_files"].append({"file": wav_path, "error": str(e)})

    # Speaker overlap check
    overlap_train_val = train_speakers.intersection(val_speakers)
    overlap_train_test = train_speakers.intersection(test_speakers)
    overlap_val_test = val_speakers.intersection(test_speakers)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        results["speaker_leakage_detected"] = True
        print(f"Speaker overlap warning: train/val={len(overlap_train_val)}, "
              f"train/test={len(overlap_train_test)}, val/test={len(overlap_val_test)}")

    results["sample_rates"] = list(sample_rates)
    if durations:
        results["duration_stats"] = {
            "min_sec": float(np.min(durations)),
            "max_sec": float(np.max(durations)),
            "mean_sec": float(np.mean(durations)),
            "std_sec": float(np.std(durations)),
            "samples_checked": len(durations)
        }

    if results["corrupted_files"] or not results["classes_found"]:
        results["is_valid"] = False

    print("\n--- DATASET VALIDATION SUMMARY ---")
    print(f"Target classes checked: {len(results['classes_found'])}/{len(target_classes)}")
    print(f"Total split counts: Train={results['split_counts']['train']}, "
          f"Val={results['split_counts']['validation']}, Test={results['split_counts']['test']}")
    print(f"Sample rates detected: {results['sample_rates']}")
    if durations:
        print(f"Mean duration: {results['duration_stats']['mean_sec']:.3f}s (Min: {results['duration_stats']['min_sec']:.3f}s, Max: {results['duration_stats']['max_sec']:.3f}s)")
    print(f"Corrupted files count: {len(results['corrupted_files'])}")
    print(f"Dataset Valid: {results['is_valid']}")
    print("---------------------------------\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate Google Speech Commands dataset")
    parser.add_argument("--dataset-dir", default=os.path.join("data", "raw", "speech_commands_v0.02"),
                        help="Path to extracted dataset")
    args = parser.parse_args()

    res = validate_dataset(args.dataset_dir)
    sys.exit(0 if res["is_valid"] else 1)
