import numpy as np

def unfold_tensor(X, mode):
    """Unfold tensor X along given mode (0,1,2)."""
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1))

def fold_tensor(X_unf, mode, shape):
    """Fold matrix X_unf back into tensor of given shape along mode."""
    full_shape = list(shape)
    full_shape[mode] = -1
    X_fold = np.moveaxis(np.reshape(X_unf, [shape[mode]] + [s for i,s in enumerate(shape) if i!=mode]), 0, mode)
    return X_fold

def shrinkage_svd(X, tau):
    """Singular value thresholding"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0)
    return U @ np.diag(s_thresh) @ Vt

def halrtc_tensor_completion(tensor_obs, mask, alpha=None, lam=0.01, tol=1e-5, max_iter=500, verbose=False):
    """
    HaLRTC: High-accuracy Low-Rank Tensor Completion
    tensor_obs: observed HSI tensor H x W x B
    mask: binary tensor same shape, 1=observed, 0=missing
    alpha: weight for each mode, default equal
    lam: singular value threshold
    """
    H, W, B = tensor_obs.shape
    X = tensor_obs.copy()
    
    if alpha is None:
        alpha = [1/3, 1/3, 1/3]
    
    Y = [np.zeros_like(X) for _ in range(3)]
    mu = 1e-4
    rho = 1.5
    mu_max = 1e10
    
    norm_obs = np.linalg.norm(X*mask)
    if norm_obs == 0:
        norm_obs = 1.0
    
    for it in range(max_iter):
        X_prev = X.copy()
        for mode in range(3):
            X_unf = unfold_tensor(X + (1/mu)*Y[mode], mode)
            X_unf = shrinkage_svd(X_unf, alpha[mode]/mu)
            X[mode==0] = fold_tensor(X_unf, mode, X.shape) if mode==0 else X
            X = fold_tensor(X_unf, mode, X.shape)
        
        # enforce observed entries
        X = mask * tensor_obs + (1 - mask) * X
        
        # update multipliers
        for mode in range(3):
            Y[mode] = Y[mode] + mu*(X - fold_tensor(unfold_tensor(X, mode), mode, X.shape))
        
        mu = min(mu*rho, mu_max)
        err = np.linalg.norm(X - X_prev) / (norm_obs + 1e-8)
        if verbose and (it % 20 == 0 or err < tol):
            print(f"iter {it}, err={err:.3e}")
        if err < tol:
            if verbose: print(f"Converged at iter {it}, err={err:.3e}")
            break
    return X

# -------------------------
# Demo on synthetic HSI
# -------------------------
H, W, B = 64, 64, 30
rng = np.random.RandomState(0)
r = 5
spatial = rng.randn(H*W, r)
spectral = rng.randn(r, B)
hsi_clean = (spatial @ spectral).reshape(H, W, B)
hsi_clean = (hsi_clean - hsi_clean.min()) / (hsi_clean.max() - hsi_clean.min() + 1e-12)

mask = (rng.rand(H,W,B) >= 0.4).astype(float)
hsi_obs = hsi_clean * mask

recon = halrtc_tensor_completion(hsi_obs.copy(), mask.copy(), lam=0.01, tol=1e-5, max_iter=500, verbose=True)

# 简单指标
def psnr(gt, rec):
    mse = np.mean((gt - rec)**2)
    if mse == 0: return float('inf')
    return 20*np.log10(1.0/np.sqrt(mse))

psnr_before = psnr(hsi_clean, hsi_obs)
psnr_after = psnr(hsi_clean, recon)
print("PSNR before:", psnr_before, "PSNR after:", psnr_after)
