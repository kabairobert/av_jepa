import numpy as np
from eb_jepa.logging import get_logger

logger = get_logger(__name__)


def linear_probe_r2(z: np.ndarray, u: np.ndarray) -> dict:
    """Fit Ridge regression z -> u, return R2 per factor and mean."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    
    # Train/test split to prevent R2 inflation at high dimensions
    split = len(z) // 2
    z_train, u_train = z[:split], u[:split]
    z_test, u_test = z[split:], u[split:]

    reg = Ridge(alpha=1.0).fit(z_train, u_train)
    u_pred = reg.predict(z_test)
    
    if u_test.ndim == 1:
        return {'r2_u0': float(r2_score(u_test, u_pred)), 'r2_mean': float(r2_score(u_test, u_pred))}
    r2_per = [float(r2_score(u_test[:, i], u_pred[:, i])) for i in range(u_test.shape[1])]
    result = {f'r2_u{i}': v for i, v in enumerate(r2_per)}
    result['r2_mean'] = float(np.mean(r2_per))
    return result


def rankme_score(z: np.ndarray) -> float:
    """Compute RankMe score for representations: exp(-sum(p_i log p_i)) where p_i are normalized singular values."""
    # Center the data
    z = z - z.mean(axis=0)
    _, svals, _ = np.linalg.svd(z, full_matrices=False)
    
    # Avoid zero division and extremely small singular values
    svals = svals[svals > 1e-7]
    if len(svals) == 0:
        return 0.0
        
    p = svals / svals.sum()
    rankme = np.exp(-np.sum(p * np.log(p + 1e-7)))
    return float(rankme)


def vicreg_variance(z: np.ndarray, margin: float = 1.0) -> float:
    """Calculate variance hinge loss as in VICReg."""
    z = z - z.mean(axis=0)
    std = np.sqrt(z.var(axis=0) + 0.0001)
    std_loss = np.mean(np.maximum(0, margin - std))
    return float(std_loss)


def vicreg_covariance(z: np.ndarray) -> float:
    """Calculate off-diagonal covariance penalty as in VICReg."""
    batch_size = z.shape[0]
    num_features = z.shape[1]
    z = z - z.mean(axis=0)
    denom = batch_size - 1 if batch_size > 1 else 1
    cov = (z.T @ z) / denom
    
    if num_features > 1:
        # Zero out diagonal elements
        cov_off_diag = cov - np.diag(np.diag(cov))
        # Mean of squared off-diagonals
        return float(np.sum(cov_off_diag**2) / (num_features * (num_features - 1)))
    return 0.0


def pca_axis_alignment(z: np.ndarray, n_active: int = 2) -> float:
    """Measure how axis-aligned the top-n_active PCA components of z are.

    For each of the top-n_active eigenvectors, compute max |cosine| with any coord axis.
    Return the mean of these max cosines across the active components.

    Interpretation:
      1.0  = each active PC perfectly parallel to a coord axis (ideal)
      ~0.57 = random orientation in 3D (1/sqrt(3))
    """
    z_centered = z - z.mean(axis=0)
    _, _, Vt = np.linalg.svd(z_centered, full_matrices=False)
    n_dims = z.shape[1]
    identity = np.eye(n_dims)
    scores = []
    for i in range(min(n_active, Vt.shape[0])):
        cosines = np.abs(Vt[i] @ identity)   # |cos| with each coord axis
        scores.append(float(cosines.max()))
    return float(np.mean(scores))


def masked_retrieval_accuracy(z_a: np.ndarray, z_b: np.ndarray, mask: np.ndarray, ks=(1, 5)) -> dict:
    """For each masked z_A[i], find k-nearest masked z_B by L2 and cosine."""
    from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
    
    # Apply mask
    z_a_masked = z_a[:, mask]
    z_b_masked = z_b[:, mask]
    
    # If mask is empty, return 0 for everything
    if z_a_masked.shape[1] == 0:
        results = {}
        for name in ['l2', 'cos']:
            for k in ks:
                results[f'masked_retrieval_{name}@{k}'] = 0.0
        return results
        
    results = {}
    for dist_fn, name in [(euclidean_distances, 'l2'), (cosine_distances, 'cos')]:
        D = dist_fn(z_a_masked, z_b_masked)
        for k in ks:
            top_k = np.argsort(D, axis=1)[:, :k]
            hits = float(np.mean([i in top_k[i] for i in range(len(z_a_masked))]))
            results[f'masked_retrieval_{name}@{k}'] = hits
    return results


def retrieval_accuracy(z_a: np.ndarray, z_b: np.ndarray, ks=(1, 5)) -> dict:
    """For each z_A[i], find k-nearest z_B by L2 and cosine. Check if correct index in top-k."""
    from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
    results = {}
    for dist_fn, name in [(euclidean_distances, 'l2'), (cosine_distances, 'cos')]:
        D = dist_fn(z_a, z_b)
        for k in ks:
            top_k = np.argsort(D, axis=1)[:, :k]
            hits = float(np.mean([i in top_k[i] for i in range(len(z_a))]))
            results[f'retrieval_{name}@{k}'] = hits
    return results


def cca_score(z_a: np.ndarray, z_b: np.ndarray) -> dict:
    """CCA between z_A and z_B.

    Returns:
      cca_corr_dim{i}: canonical correlation for each component
      cca_effective_rank: number of components with corr > 0.5
      cca_diag_score: diagonality of cross-correlation matrix in CCA space
        = sum(|diag(C)|) / sum(|C|) where C[i,j] = corr(z_a_c[:,i], z_b_c[:,j])
        1.0 = dim-i of z_A maps exactly to dim-i of z_B
        0.0 = fully off-diagonal (rotated alignment)
    """
    from sklearn.cross_decomposition import CCA
    
    # Cap components to prevent statistical invalidity and hanging at high D
    n_components = min(z_a.shape[1], z_b.shape[1], max(3, z_a.shape[0] // 100))
    
    try:
        cca = CCA(n_components=n_components).fit(z_a, z_b)
        z_a_c, z_b_c = cca.transform(z_a, z_b)
        
        corrs = []
        for i in range(n_components):
            val = float(np.corrcoef(z_a_c[:, i], z_b_c[:, i])[0, 1])
            corrs.append(0.0 if np.isnan(val) else val)

        # Full cross-correlation matrix for diagonality score
        C = np.zeros((n_components, n_components))
        for i in range(n_components):
            for j in range(n_components):
                val = float(np.corrcoef(z_a_c[:, i], z_b_c[:, j])[0, 1])
                C[i, j] = 0.0 if np.isnan(val) else abs(val)
        total = C.sum()
        diag_score = float(np.diag(C).sum() / total) if total > 1e-8 else 0.0

    except Exception as exc:
        logger.warning("CCA failed: %s", exc)
        corrs = [0.0] * n_components
        diag_score = 0.0

    result = {f'cca_corr_dim{i}': c for i, c in enumerate(corrs)}
    result['cca_effective_rank'] = float(np.sum(np.array(corrs) > 0.5))
    result['cca_diag_score'] = diag_score
    return result
