import numpy as np
from numpy.linalg import svd

def shrinkage_operator(X, tau):
    """Soft-thresholding for singular values"""
    U, s, Vt = svd(X, full_matrices=False)
    s_threshold = np.maximum(s - tau, 0)
    return U @ np.diag(s_threshold) @ Vt

def soft_threshold(X, tau):
    """Soft-thresholding for L1 norm"""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def lrmr_inpainting(X_obs, mask, lam=0.01, tol=1e-7, max_iter=500):
    """
    Low-Rank Matrix Recovery for HSI Inpainting
    X_obs: observed data (H*W, Bands) or (Bands, H*W)
    mask: same shape as X_obs, 1 for known pixels, 0 for missing
    lam: lambda in optimization
    """
    # Initialize
    Y = np.zeros_like(X_obs)
    L = np.zeros_like(X_obs)
    S = np.zeros_like(X_obs)
    mu = 1.25 / np.linalg.norm(X_obs)
    mu_max = 1e7
    rho = 1.5

    for _ in range(max_iter):
        # 1. Update L (low-rank part)
        L = shrinkage_operator(X_obs - S + (1/mu)*Y, 1/mu)

        # 2. Update S (sparse part)
        temp = X_obs - L + (1/mu)*Y
        S = soft_threshold(temp, lam/mu)

        # 3. Enforce observed pixels only
        LS = L + S
        LS = mask * X_obs + (1 - mask) * LS

        # 4. Update multiplier Y
        Z = X_obs - LS
        Y = Y + mu * Z

        # 5. Update mu
        mu = min(mu * rho, mu_max)

        # 6. Check convergence
        err = np.linalg.norm(Z, 'fro') / np.linalg.norm(X_obs, 'fro')
        if err < tol:
            break

    return L
