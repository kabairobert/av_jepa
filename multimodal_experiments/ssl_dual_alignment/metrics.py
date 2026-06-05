import numpy as np
import torch
from typing import Optional
from eb_jepa.logging import get_logger
from multimodal_experiments.ssl_dual_alignment.dataset import PointType, DualDisentangleDataset
from multimodal_experiments.ssl_dual_alignment.vis import to_numpy

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


def compute_diagonality_ratio(z_a: np.ndarray, u: np.ndarray, num_zdims: int, k_shared: int) -> float:
    """Permutation-invariant diagonality ratio using Ridge R2 matrix."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    n_factors = u.shape[1]
    r2_matrix = np.zeros((num_zdims, n_factors))
    for i in range(num_zdims):
        zi = z_a[:, i:i+1]
        for j in range(n_factors):
            uj = u[:, j]
            reg = Ridge(alpha=1.0).fit(zi, uj)
            r2_matrix[i, j] = float(r2_score(uj, reg.predict(zi)))
    
    # Select top-k_shared z-dims by total R2 summed across all shared factors
    top_dim_idxs = np.argsort(r2_matrix.sum(axis=1))[::-1][:k_shared]
    sub = r2_matrix[top_dim_idxs, :]  # (k_shared, n_factors)
    
    # Greedy optimal assignment: maximise sum of selected (dim, factor) R2 pairs
    assigned_rows, assigned_cols = set(), set()
    diag_sum = 0.0
    for val, r, c in sorted(
        [(sub[r, c], r, c) for r in range(k_shared) for c in range(n_factors)],
        key=lambda x: -x[0]
    ):
        if r not in assigned_rows and c not in assigned_cols:
            diag_sum += val
            assigned_rows.add(r)
            assigned_cols.add(c)
            if len(assigned_rows) == k_shared:
                break
    total_sub = sub.sum()
    return float(diag_sum / total_sub if total_sub > 1e-12 else 0.5)


def compute_found_rank_metrics(
    z_a: np.ndarray,
    z_b: np.ndarray,
    num_zdims: int,
    cca_corr_spectrum: list[float],
    predictor_a2b: Optional[torch.nn.Module],
    z_a_clean: Optional[np.ndarray],
    z_b_clean: Optional[np.ndarray],
) -> dict:
    """Estimates the mutual-information subspace rank using Pearson, CCA, pred weights, and pred R2."""
    from sklearn.metrics import r2_score
    metrics = {}

    # 1. Pearson correlation rank
    pearson_spectrum = []
    for i in range(num_zdims):
        c = float(np.corrcoef(z_a[:, i], z_b[:, i])[0, 1])
        c = 0.0 if np.isnan(c) else c
        pearson_spectrum.append(c)
        metrics[f'geom/za_zb_pearson_dim{i}'] = c
    for thresh, tag in [(0.1, '1'), (0.3, '3'), (0.5, '5')]:
        metrics[f'geom/found_rank_pearson_{tag}'] = int(
            np.sum(np.abs(pearson_spectrum) > thresh)
        )

    # 2. CCA rank
    for thresh, tag in [(0.1, '1'), (0.3, '3'), (0.5, '5')]:
        metrics[f'geom/found_rank_cca_{tag}'] = int(
            np.sum(np.array(cca_corr_spectrum) > thresh)
        )

    # 3. Predictor weight rank & clean predictor R2
    if predictor_a2b is not None and hasattr(predictor_a2b, 'weight'):
        w_np = predictor_a2b.weight.detach().cpu().numpy()
        abs_w = np.abs(w_np)
        for i, wi in enumerate(abs_w):
            metrics[f'geom/pred_w_dim{i}'] = float(wi)
        for thresh, tag in [(0.3, '3'), (0.5, '5'), (0.7, '7')]:
            metrics[f'geom/found_rank_pred_w_{tag}'] = int(np.sum(abs_w > thresh))

        # 4. Predictor R2 on clean manifold points
        if z_a_clean is not None and z_b_clean is not None and z_a_clean.shape[0] >= 10:
            _pred_device = next(predictor_a2b.parameters()).device
            with torch.no_grad():
                _pred_zb = predictor_a2b(
                    torch.tensor(z_a_clean, dtype=torch.float32).to(_pred_device)
                ).cpu().numpy()
            pred_r2_spectrum = []
            for i in range(num_zdims):
                r2 = float(r2_score(z_b_clean[:, i], _pred_zb[:, i]))
                r2 = max(0.0, r2)
                pred_r2_spectrum.append(r2)
                metrics[f'geom/pred_r2_dim{i}'] = r2
            for thresh, tag in [(0.1, '1'), (0.3, '3'), (0.5, '5')]:
                metrics[f'geom/found_rank_pred_r2_{tag}'] = int(
                    np.sum(np.array(pred_r2_spectrum) > thresh)
                )

    return metrics


def compute_norm_diagnostics(
    z_a: np.ndarray,
    z_b: np.ndarray,
    dataset: DualDisentangleDataset,
    idxs: Optional[np.ndarray],
) -> dict:
    """Computes global norms and partition-wise norms (manifold, corrupt, external)."""
    norms_a = np.linalg.norm(z_a, axis=1)
    norms_b = np.linalg.norm(z_b, axis=1)
    
    metrics = {
        'geom/z_a_norm_mean': float(norms_a.mean()),
        'geom/z_b_norm_mean': float(norms_b.mean()),
    }

    pt_a = getattr(dataset, "point_type_a", None)
    pt_b = getattr(dataset, "point_type_b", None)
    if pt_a is not None and pt_b is not None:
        pt_a = to_numpy(pt_a)
        pt_b = to_numpy(pt_b)
        if idxs is not None:
            pt_a = pt_a[idxs]
            pt_b = pt_b[idxs]

        def _mean_or_nan(values, mask):
            return float(values[mask].mean()) if mask.any() else float("nan")

        metrics['geom/z_a_norm_manifold'] = _mean_or_nan(norms_a, pt_a == PointType.MANIFOLD)
        metrics['geom/z_a_norm_asym_corrupt'] = _mean_or_nan(norms_a, pt_a == PointType.ASYM_A_CORRUPT)
        metrics['geom/z_a_norm_external'] = _mean_or_nan(norms_a, pt_a == PointType.EXTERNAL)

        metrics['geom/z_b_norm_manifold'] = _mean_or_nan(norms_b, pt_b == PointType.MANIFOLD)
        metrics['geom/z_b_norm_asym_corrupt'] = _mean_or_nan(norms_b, pt_b == PointType.ASYM_B_CORRUPT)
        metrics['geom/z_b_norm_external'] = _mean_or_nan(norms_b, pt_b == PointType.EXTERNAL)

    return metrics
