"""
Viterbi Algorithm Implementation for Hidden Markov Models in Log-Space.

Mathematical Formulation:
--------------------------
Given an observation sequence X = (x_1, x_2, ..., x_T) and HMM lambda = (A, B, pi),
the Viterbi algorithm finds the optimal single state sequence Q* = (q_1*, q_2*, ..., q_T*)
that maximizes P(Q, X | lambda).

Definitions in Log-Space:
  log_delta_t(j) = max_{q_1..q_{t-1}} ln P(q_1, ..., q_{t-1}, q_t = s_j, x_1, ..., x_t | lambda)
  psi_t(j) = argmax_{i=1..N} [ log_delta_{t-1}(i) + ln a_ij ]

1. Initialization (t = 1):
     log_delta_1(i) = ln pi_i + ln b_i(x_1)    for i = 1, ..., N
     psi_1(i) = 0

2. Recursion (t = 2, ..., T):
     log_delta_t(j) = max_{i=1..N} [ log_delta_{t-1}(i) + ln a_ij ] + ln b_j(x_t)
     psi_t(j) = argmax_{i=1..N} [ log_delta_{t-1}(i) + ln a_ij ]

3. Termination:
     best_log_score = max_{i=1..N} [ log_delta_T(i) ]
     q_T* = argmax_{i=1..N} [ log_delta_T(i) ]

4. Backtracking (t = T-1 down to 1):
     q_t* = psi_{t+1}(q_{t+1}*)
"""

import numpy as np
from typing import Tuple, Any, Dict
from src.models.forward_algorithm import compute_log_gaussian_emissions


def viterbi_decode(X: np.ndarray,
                   startprob: np.ndarray,
                   transmat: np.ndarray,
                   means: np.ndarray,
                   covars: np.ndarray,
                   covariance_type: str = "diag") -> Tuple[float, np.ndarray]:
    """
    Compute most likely hidden state sequence using Viterbi algorithm in log-space.

    Args:
        X: (T, D) Observation feature sequence.
        startprob: (N,) Initial state probabilities.
        transmat: (N, N) Transition probability matrix.
        means: (N, D) Gaussian emission means.
        covars: (N, D) or (N, D, D) Gaussian emission covariances.
        covariance_type: "diag" or "full".

    Returns:
        best_log_score: float, joint log probability of the best path
        best_state_path: (T,) int array of optimal hidden states
    """
    T, D = X.shape
    N = len(startprob)

    # Handle case where diag covars is passed as 3D (N, D, D) with diagonal elements
    if covariance_type == "diag" and covars.ndim == 3:
        covars = np.diagonal(covars, axis1=1, axis2=2)

    with np.errstate(divide="ignore"):
        log_start = np.where(startprob > 0, np.log(startprob), -np.inf)
        log_trans = np.where(transmat > 0, np.log(transmat), -np.inf)

    # Emission log probabilities: (T, N)
    log_B = compute_log_gaussian_emissions(X, means, covars, covariance_type)

    # Trellis matrices
    log_delta = np.full((T, N), -np.inf, dtype=np.float64)
    psi = np.zeros((T, N), dtype=np.int32)

    # 1. Initialization (t = 0)
    log_delta[0] = log_start + log_B[0]
    psi[0] = 0

    # 2. Recursion (t = 1 .. T-1)
    for t in range(1, T):
        # Shape: (N, 1) + (N, N) -> (N, N)
        # Row i is source state, Col j is destination state
        trans_scores = log_delta[t - 1, :, None] + log_trans

        # For each destination state j, find max over source states i
        log_delta[t] = np.max(trans_scores, axis=0) + log_B[t]
        psi[t] = np.argmax(trans_scores, axis=0)

    # 3. Termination
    best_log_score = float(np.max(log_delta[-1]))
    best_last_state = int(np.argmax(log_delta[-1]))

    # 4. Backtracking
    best_state_path = np.zeros(T, dtype=np.int32)
    best_state_path[-1] = best_last_state

    for t in range(T - 2, -1, -1):
        best_state_path[t] = psi[t + 1, best_state_path[t + 1]]

    return best_log_score, best_state_path


def decode_with_hmm_model(model: Any, X: np.ndarray, use_custom: bool = True) -> Tuple[float, np.ndarray]:
    """
    Decode state path for an observation sequence using hmmlearn GaussianHMM model.
    """
    if not use_custom:
        log_prob, state_sequence = model.decode(X, algorithm="viterbi")
        return float(log_prob), state_sequence

    cov_type = getattr(model, "covariance_type", "diag")
    covars = getattr(model, "covars_", getattr(model, "_covars_", None))
    return viterbi_decode(
        X=X,
        startprob=model.startprob_,
        transmat=model.transmat_,
        means=model.means_,
        covars=covars,
        covariance_type=cov_type
    )
