"""
Dataset preparation and manifest generation module.
Processes Google Speech Commands v0.02 audio according to official train, validation, and test splits.
Generates:
- data/processed/train_manifest.csv
- data/processed/validation_manifest.csv
- data/processed/test_manifest.csv
"""

import os
import glob
import pandas as pd
import soundfile as sf
import numpy as np
from typing import List, Tuple, Dict

CORE_COMMANDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]


def load_split_lists(dataset_dir: str) -> Tuple[set, set]:
    """Load official validation and test file relative paths."""
    val_file = os.path.join(dataset_dir, "validation_list.txt")
    test_file = os.path.join(dataset_dir, "testing_list.txt")

    if not os.path.exists(val_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Missing split file in {dataset_dir}")

    with open(val_file, "r") as f:
        val_rel_paths = set(line.strip().replace("\\", "/") for line in f if line.strip())

    with open(test_file, "r") as f:
        test_rel_paths = set(line.strip().replace("\\", "/") for line in f if line.strip())

    return val_rel_paths, test_rel_paths


def build_manifests(dataset_dir: str,
                    output_dir: str,
                    target_commands: List[str] = None,
                    max_samples_per_class: int = None) -> Dict[str, pd.DataFrame]:
    """
    Build train, validation, and test manifests for target commands.
    Ensures zero speaker leakage by strictly following official split lists.
    """
    if target_commands is None:
        target_commands = CORE_COMMANDS

    os.makedirs(output_dir, exist_ok=True)
    val_set, test_set = load_split_lists(dataset_dir)

    train_records = []
    val_records = []
    test_records = []

    print(f"Building manifests for commands: {target_commands}")

    for cmd in target_commands:
        cmd_dir = os.path.join(dataset_dir, cmd)
        if not os.path.exists(cmd_dir):
            print(f"Warning: Directory not found: {cmd_dir}")
            continue

        wav_files = sorted(glob.glob(os.path.join(cmd_dir, "*.wav")))
        cmd_train = 0
        cmd_val = 0
        cmd_test = 0

        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            rel_path = f"{cmd}/{fname}"
            speaker_id = fname.split("_nohash_")[0]

            record = {
                "file_path": os.path.abspath(wav_path),
                "relative_path": rel_path,
                "command": cmd,
                "speaker_id": speaker_id
            }

            if rel_path in test_set:
                test_records.append(record)
                cmd_test += 1
            elif rel_path in val_set:
                val_records.append(record)
                cmd_val += 1
            else:
                train_records.append(record)
                cmd_train += 1

        print(f"Command '{cmd}': Train={cmd_train}, Validation={cmd_val}, Test={cmd_test}")

    df_train = pd.DataFrame(train_records)
    df_val = pd.DataFrame(val_records)
    df_test = pd.DataFrame(test_records)

    # Optional sub-sampling for fast experimentation if specified
    if max_samples_per_class is not None and max_samples_per_class > 0:
        df_train = df_train.groupby("command", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=42)
        ).reset_index(drop=True)

    train_path = os.path.join(output_dir, "train_manifest.csv")
    val_path = os.path.join(output_dir, "validation_manifest.csv")
    test_path = os.path.join(output_dir, "test_manifest.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"\nManifests saved to {output_dir}:")
    print(f"  Train: {len(df_train)} samples -> {train_path}")
    print(f"  Validation: {len(df_val)} samples -> {val_path}")
    print(f"  Test: {len(df_test)} samples -> {test_path}\n")

    return {
        "train": df_train,
        "validation": df_val,
        "test": df_test
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare dataset manifests")
    parser.add_argument("--dataset-dir", default=os.path.join("data", "raw", "speech_commands_v0.02"),
                        help="Raw dataset directory")
    parser.add_argument("--output-dir", default=os.path.join("data", "processed"),
                        help="Processed manifest directory")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max training samples per class for training speedup")
    args = parser.parse_args()

    build_manifests(args.dataset_dir, args.output_dir, max_samples_per_class=args.max_samples)
