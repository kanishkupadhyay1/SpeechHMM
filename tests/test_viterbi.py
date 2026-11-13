"""
Tests for Viterbi decoding algorithm and state sequence alignment.
"""

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM
from src.models.viterbi_decoder import viterbi_decode, decode_with_hmm_model


def test_viterbi_decoding_against_hmmlearn():
    """
    Validate that our custom log-space Viterbi implementation produces
    identical optimal hidden state sequences and log scores to hmmlearn.
    """
    np.random.seed(42)
    n_states = 4
    n_features = 13
    T = 60

    X = np.random.randn(T, n_features)

    model = GaussianHMM(n_components=n_states, covariance_type="diag", random_state=42)
    model.startprob_ = np.array([0.8, 0.2, 0.0, 0.0])
    trans = np.array([
        [0.7, 0.3, 0.0, 0.0],
        [0.0, 0.6, 0.4, 0.0],
        [0.0, 0.0, 0.8, 0.2],
        [0.0, 0.0, 0.0, 1.0]
    ])
    model.transmat_ = trans
    model.means_ = np.random.randn(n_states, n_features)
    model.covars_ = np.abs(np.random.randn(n_states, n_features)) + 0.1

    # hmmlearn Viterbi decode
    hmmlearn_log_prob, hmmlearn_states = model.decode(X, algorithm="viterbi")

    # Custom Viterbi decode
    custom_log_prob, custom_states = viterbi_decode(
        X=X,
        startprob=model.startprob_,
        transmat=model.transmat_,
        means=model.means_,
        covars=model.covars_,
        covariance_type="diag"
    )

    # Assert matching path score
    assert np.isclose(hmmlearn_log_prob, custom_log_prob, atol=1e-3), \
        f"Viterbi score mismatch: hmmlearn={hmmlearn_log_prob}, custom={custom_log_prob}"

    # Assert matching state sequence
    np.testing.assert_array_equal(
        hmmlearn_states,
        custom_states,
        err_msg="Decoded state path does not match hmmlearn state sequence"
    )


def test_viterbi_monotonicity_on_left_to_right():
    """Verify that on strict left-to-right topology, state sequence never backtracks."""
    np.random.seed(42)
    n_states = 5
    n_features = 39
    T = 80

    X = np.random.randn(T, n_features)
    startprob = np.zeros(n_states)
    startprob[0] = 1.0

    # Upper triangular transition matrix (no backward transitions)
    trans = np.zeros((n_states, n_states))
    for i in range(n_states):
        if i == n_states - 1:
            trans[i, i] = 1.0
        else:
            trans[i, i] = 0.5
            trans[i, i + 1] = 0.5

    means = np.random.randn(n_states, n_features)
    covars = np.ones((n_states, n_features))

    score, states = viterbi_decode(X, startprob, trans, means, covars)

    # In strict left-to-right, states[t+1] >= states[t]
    diffs = np.diff(states)
    assert np.all(diffs >= 0), f"Backward transition detected in left-to-right model: {states}"
