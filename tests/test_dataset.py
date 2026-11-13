"""
Tests for dataset integrity and validation.
"""

import os
import pytest
from src.data.validate_dataset import validate_dataset, load_split_files
from src.data.prepare_dataset import CORE_COMMANDS


def test_split_files_exist():
    raw_dir = os.path.join("data", "raw", "speech_commands_v0.02")
    assert os.path.isdir(raw_dir), f"Raw dataset directory does not exist: {raw_dir}"
    val_set, test_set = load_split_files(raw_dir)
    assert len(val_set) > 0, "Validation split list is empty"
    assert len(test_set) > 0, "Testing split list is empty"
    # Ensure zero overlap
    overlap = val_set.intersection(test_set)
    assert len(overlap) == 0, f"Speaker/data leakage detected between val and test: {len(overlap)} files"


def test_core_command_folders_exist():
    raw_dir = os.path.join("data", "raw", "speech_commands_v0.02")
    for cmd in CORE_COMMANDS:
        cmd_dir = os.path.join(raw_dir, cmd)
        assert os.path.isdir(cmd_dir), f"Missing command folder: {cmd_dir}"
        wavs = [f for f in os.listdir(cmd_dir) if f.endswith(".wav")]
        assert len(wavs) > 1000, f"Command {cmd} has insufficient audio files: {len(wavs)}"


def test_manifest_files():
    import pandas as pd
    manifest_dir = os.path.join("data", "processed")
    train_csv = os.path.join(manifest_dir, "train_manifest.csv")
    val_csv = os.path.join(manifest_dir, "validation_manifest.csv")
    test_csv = os.path.join(manifest_dir, "test_manifest.csv")

    assert os.path.exists(train_csv), "Missing train_manifest.csv"
    assert os.path.exists(val_csv), "Missing validation_manifest.csv"
    assert os.path.exists(test_csv), "Missing test_manifest.csv"

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    assert len(df_train) > 20000, f"Unexpected train count: {len(df_train)}"
    assert len(df_val) > 2000, f"Unexpected val count: {len(df_val)}"
    assert len(df_test) > 3000, f"Unexpected test count: {len(df_test)}"
