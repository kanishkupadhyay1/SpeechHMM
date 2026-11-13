"""
Tests for HMM model building, initialization, training, and serialization.
"""

import os
import numpy as np
import pytest
from src.models.hmm_trainer import CommandHMMTrainer


def test_hmm_initialization():
    trainer = CommandHMMTrainer(n_states=5, covariance_type="diag")
    assert trainer.n_states == 5
    assert trainer.covariance_type == "diag"
    assert len(trainer.commands) == 10

    model = trainer.build_model()
    assert model.n_components == 5
    assert model.startprob_[0] == 1.0
    assert model.transmat_.shape == (5, 5)


def test_hmm_single_command_training():
    trainer = CommandHMMTrainer(n_states=3, n_iter=10)
    # Generate mock training sequences with distinct patterns
    np.random.seed(42)
    seq1 = np.random.randn(40, 13)
    seq2 = np.random.randn(45, 13)
    feature_sequences = [seq1, seq2]
    lengths = [len(seq1), len(seq2)]

    model = trainer.train_single_command("test_cmd", feature_sequences, lengths)
    assert model.n_components == 3
    assert model.means_.shape == (3, 13)
    assert model.covars_.shape in ((3, 13), (3, 13, 13))
