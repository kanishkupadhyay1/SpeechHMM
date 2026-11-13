"""
Forward Algorithm Implementation for Hidden Markov Models in Log-Space.

Mathematical Formulation:
--------------------------
Let lambda = (A, B, pi) denote a Hidden Markov Model with:
  - N hidden states: S = {s_1, s_2, ..., s_N}
  - Initial state distribution: pi_i = P(q_1 = s_i)
  - Transition probability matrix: A = [a_ij], where a_ij = P(q_t = s_j | q_{t-1} = s_i)
  - Gaussian emission distribution: b_j(x_t) = N(x_t; mu_j, Sigma_j)

Forward Variable:
  alpha_t(i) = P(x_1, x_2, ..., x_t, q_t = s_i | lambda)

Log-Space Formulation:
  To prevent floating-point underflow on long observation sequences, computation
  is performed strictly in natural log space:
    log_alpha_t(i) = ln alpha_t(i)

1. Initialization (t = 1):
     log_alpha_1(i) = ln pi_i + ln b_i(x_1)   for i = 1, ..., N

2. Induction (t = 2, ..., T):
     log_alpha_t(j) = logsumexp_{i=1..N} [ log_alpha_{t-1}(i) + ln a_ij ] + ln b_j(x_t)

3. Termination:
     ln P(X | lambda) = logsumexp_{i=1..N} [ log_alpha_T(i) ]
"""

import numpy as np
from scipy.special import logsumexp
from typing import Dict, Any, Tuple, Optional


def compute_log_gaussian_emissions(X: np.ndarray,
                                   means: np.ndarray,
                                   covars: np.ndarray,
                                   covariance_type: str = "diag") -> np.ndarray:
    """
    Compute log multivariate Gaussian density: log N(x_t; mu_j, Sigma_j) for all frames and states.

    Args:
        X: (T, D) Observation sequence of MFCC feature vectors.
        means: (N, D) Mean vectors for each hidden state.
        covars: (N, D) diagonal variances or (N, D, D) full covariance matrices.
        covariance_type: "diag" or "full".

    Returns:
        log_B: (T, N) log emission probabilities.
    """
    T, D = X.shape
    N = means.shape[0]
    log_B = np.zeros((T, N), dtype=np.float64)

    # Handle case where diag covars is passed as 3D (N, D, D) with diagonal elements
    if covariance_type == "diag" and covars.ndim == 3:
        covars = np.diagonal(covars, axis1=1, axis2=2)

    if covariance_type == "diag":
        # Diagonal covariance: Sigma_j is a vector of variances (N, D)
        covars_safe = np.maximum(covars, 1e-5)
        log_det = np.sum(np.log(covars_safe), axis=1)  # shape (N,)
        const_term = -0.5 * (D * np.log(2.0 * np.pi) + log_det)  # shape (N,)

        for j in range(N):
            diff = X - means[j]  # (T, D)
            quad = np.sum((diff ** 2) / covars_safe[j], axis=1)  # (T,)
            log_B[:, j] = const_term[j] - 0.5 * quad

    elif covariance_type == "full":
        for j in range(N):
            cov = covars[j] + np.eye(D) * 1e-5
            sign, log_det = np.linalg.slogdet(cov)
            inv_cov = np.linalg.inv(cov)
            const = -0.5 * (D * np.log(2.0 * np.pi) + log_det)
            diff = X - means[j]  # (T, D)
            quad = np.sum(diff @ inv_cov * diff, axis=1)  # (T,)
            log_B[:, j] = const - 0.5 * quad
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    return log_B


def forward_log_likelihood(X: np.ndarray,
                           startprob: np.ndarray,
                           transmat: np.ndarray,
                           means: np.ndarray,
                           covars: np.ndarray,
                           covariance_type: str = "diag") -> Tuple[float, np.ndarray]:
    """
    Compute the log likelihood ln P(X | lambda) using the Forward algorithm in log-space.

    Args:
        X: (T, D) Observation sequence.
        startprob: (N,) Initial state probabilities.
        transmat: (N, N) State transition matrix.
        means: (N, D) Gaussian emission means.
        covars: (N, D) or (N, D, D) Gaussian emission covariances.
        covariance_type: "diag" or "full".

    Returns:
        total_log_likelihood: float, ln P(X | lambda)
        log_alpha: (T, N) forward trellis matrix in log space
    """
    T, D = X.shape
    N = len(startprob)

    # Exact log transforms: 0 probability maps to -infinity
    with np.errstate(divide="ignore"):
        log_start = np.where(startprob > 0, np.log(startprob), -np.inf)
        log_trans = np.where(transmat > 0, np.log(transmat), -np.inf)

    # Compute emission log-probabilities: (T, N)
    log_B = compute_log_gaussian_emissions(X, means, covars, covariance_type)

    # Forward trellis in log space
    log_alpha = np.full((T, N), -np.inf, dtype=np.float64)

    # 1. Initialization (t = 0)
    log_alpha[0] = log_start + log_B[0]

    # 2. Induction (t = 1 .. T-1)
    for t in range(1, T):
        # Broadcast previous log_alpha: (N, 1) + (N, N) -> (N, N)
        # sum over i: logsumexp along axis 0
        log_alpha[t] = logsumexp(log_alpha[t - 1, :, None] + log_trans, axis=0) + log_B[t]

    # 3. Termination
    total_log_likelihood = float(logsumexp(log_alpha[-1]))
    return total_log_likelihood, log_alpha


def score_hmm_model(model: Any, X: np.ndarray, use_custom: bool = True) -> float:
    """
    Score an observation sequence against an hmmlearn GaussianHMM model.
    Supports both custom forward algorithm and hmmlearn score method.
    """
    if not use_custom:
        return float(model.score(X))

    cov_type = getattr(model, "covariance_type", "diag")
    covars = getattr(model, "covars_", getattr(model, "_covars_", None))
    log_lik, _ = forward_log_likelihood(
        X=X,
        startprob=model.startprob_,
        transmat=model.transmat_,
        means=model.means_,
        covars=covars,
        covariance_type=cov_type
    )
    return log_lik


def score_all_commands(models: Dict[str, Any], X: np.ndarray, use_custom: bool = True) -> Dict[str, float]:
    """
    Compute forward log likelihood for every trained command HMM.
    Returns dictionary mapping command name -> log likelihood.
    """
    scores = {}
    for cmd, model in models.items():
        try:
            scores[cmd] = score_hmm_model(model, X, use_custom=use_custom)
        except Exception as e:
            scores[cmd] = -1e9
    return scores
