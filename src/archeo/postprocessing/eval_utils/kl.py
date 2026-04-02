import numpy as np


def compute_kl_divergence_from_samples(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute KL(N0 || N1) where N0 and N1 are multivariate Gaussians.
    X and Y should be 2D arrays of shape (n, d) and (m, d) respectively.
    """

    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same dimension.")

    k = X.shape[1]
    mu0 = X.mean(axis=0)
    mu1 = Y.mean(axis=0)
    S0 = np.cov(X.T, bias=False)
    S1 = np.cov(Y.T, bias=False)

    invS1 = np.linalg.inv(S1)
    diff = (mu1 - mu0).reshape(k, 1)

    term_trace = np.trace(invS1 @ S0)
    term_quad = float((diff.T @ invS1 @ diff)[0, 0])
    term_logdet = np.log(np.linalg.det(S1) / np.linalg.det(S0))

    return 0.5 * (term_trace + term_quad - k + term_logdet)
