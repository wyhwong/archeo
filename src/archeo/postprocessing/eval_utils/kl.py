import numpy as np


def compute_kl_divergence_from_samples(X: np.ndarray, Y: np.ndarray) -> float:
    """Estimate Gaussian KL divergence between two sample clouds.

    The function fits multivariate Gaussian distributions to ``X`` and ``Y`` using
    sample means and sample covariance matrices, then computes
    ``KL(N_X || N_Y)``.

    Args:
        X (np.ndarray): Array of shape ``(n_samples_x, n_features)``.
        Y (np.ndarray): Array of shape ``(n_samples_y, n_features)``.

    Returns:
        float: Estimated KL divergence.

    Raises:
        ValueError: If feature dimensions of ``X`` and ``Y`` do not match.
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
