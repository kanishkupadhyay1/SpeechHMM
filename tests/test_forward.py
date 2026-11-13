"""
Tests for Forward Algorithm implementation and numerical stability.
"""

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM
from src.models.forward_algorithm import forward_log_likelihood, score_hmm_model


def test_forward_algorithm_against_hmmlearn():
    """
    Validate that our independent log-space Forward algorithm formulation
    produces identical log likelihoods to hmmlearn's internal C-implementation.
    """
    np.random.seed(42)
    n_states = 4
    n_features = 13
    T = 50

    # Create random observation sequence
    X = np.random.randn(T, n_features)

    # Initialize GaussianHMM
    model = GaussianHMM(n_components=n_states, covariance_type="diag", random_state=42)
    model.startprob_ = np.array([0.7, 0.2, 0.1, 0.0])
    trans = np.array([
        [0.6, 0.4, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.7, 0.3],
        [0.0, 0.0, 0.0, 1.0]
    ])
    model.transmat_ = trans
    model.means_ = np.random.randn(n_states, n_features)
    model.covars_ = np.abs(np.random.randn(n_states, n_features)) + 0.1

    # hmmlearn score
    hmmlearn_score = model.score(X)

    # Custom Forward algorithm in log space
    custom_score, log_alpha = forward_log_likelihood(
        X=X,
        startprob=model.startprob_,
        transmat=model.transmat_,
        means=model.means_,
        covars=model.covars_,
        covariance_type="diag"
    )

    print(f"hmmlearn: {hmmlearn_score:.6f}, custom: {custom_score:.6f}")
    assert np.isclose(hmmlearn_score, custom_score, atol=1e-3), \
        f"Forward likelihood mismatch: hmmlearn={hmmlearn_score}, custom={custom_score}"


def test_forward_trellis_shape():
    np.random.seed(42)
    n_states = 5
    n_features = 39
    T = 101

    X = np.random.randn(T, n_features)
    startprob = np.ones(n_states) / n_states
    transmat = np.eye(n_states) * 0.5 + np.roll(np.eye(n_states), 1, axis=1) * 0.5
    means = np.random.randn(n_states, n_features)
    covars = np.ones((n_states, n_features))

    score, log_alpha = forward_log_likelihood(
        X, startprob, transmat, means, covars, covariance_type="diag"
    )

    assert log_alpha.shape == (T, n_states)
    assert not np.isnan(log_alpha).any()
    assert np.isfinite(score)
