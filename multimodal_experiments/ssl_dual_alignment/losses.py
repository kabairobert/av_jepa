import warnings
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared alias helper — single source of truth for congruence_mode strings.
# Callers: EBMJEPALoss.__init__, main.py, eval.py
# ---------------------------------------------------------------------------

_CONGRUENCE_ALIAS_MAP = {
    'none':           'none',
    'off':            'none',
    'cm_off':         'none',
    'pred_only':      'pred_only',
    'pred':           'pred_only',
    'cm_pred':        'pred_only',
    'full':           'pred_and_sparse',   # old 'full' closest to pred_and_sparse
    'pred_and_sparse': 'pred_and_sparse',
    'pred_sparse':    'pred_and_sparse',
    'cm_pred_sparse': 'pred_and_sparse',
}

_VALID_CONGRUENCE_MODES = ('none', 'pred_only', 'pred_and_sparse')


def canonicalize_congruence_mode(val: str) -> str:
    """Map any known congruence_mode alias to its canonical form.

    Raises ValueError for unknown values so callers catch config mistakes early.
    """
    canonical = _CONGRUENCE_ALIAS_MAP.get(str(val).strip().lower())
    if canonical is None:
        raise ValueError(
            f"Unknown congruence_mode '{val}'. "
            f"Valid values/aliases: {sorted(_CONGRUENCE_ALIAS_MAP.keys())}"
        )
    return canonical


class EBMJEPALoss(torch.nn.Module):
    """EBM JEPA Loss: Prediction error + Prior - Jacobian + Sparsity.

    Loss decomposition
    ------------------
    Four terms with fundamentally different roles:

      pred_loss   – cross-modal prediction quality; signal-congruence proxy
      jac_loss    – Jacobian regulariser; encourages volume-preserving geometry
      prior_loss  – latent prior (L1/L2); keeps encoders near a known prior
      sparse_loss – predictor weight sparsity; structural inductive bias

    Congruence gate
    ---------------
    A per-sample sigmoid weight  w_i = σ(−pred_per_i / τ)  is applied to
    the prediction loss (and optionally the sparsity penalty) so that
    noisy / incongruent samples contribute less to the gradient.

    Crucially, jac_loss and prior_loss are kept *uniformly weighted*:
      - jac_loss  reshapes the latent geometry globally; downweighting
        samples based on prediction error would corrupt the geometry
        exactly where the model is uncertain — the opposite of what we want.
      - prior_loss  is a per-sample magnitude penalty; it must stay
        uniform so that the prior is enforced regardless of prediction ease.

    Flow Variance Calibration
    -------------------------
    The combination of `prior_loss` and `jac_loss` acts as a Maximum Likelihood 
    objective matching the latent space to a prior distribution.
    For an L2 prior (`prior_loss = z^2`), the EBM loss matches a Gaussian:
      Loss = lambda_prior * z^2 - lambda_jac * log|det J|
    This analytical formulation mathematically forces the latent space to converge 
    to an Isotropic Gaussian N(0, σ^2) with variance:
      σ^2 = lambda_jac / (2 * lambda_prior)
    To achieve standard JEPA / SIGReg unit variance (σ^2 = 1.0) and preserve 
    unimodal information scales without collapsing them, we must set:
      lambda_jac = 1.0  and  lambda_prior = 0.5.

    Args:
        predictor_a2b, predictor_b2a : cross-modal predictor modules
        lambda_jac   : weight for Jacobian loss (kept uniform)
        lambda_prior : weight for prior loss (kept uniform)
        lambda_sparse: weight for predictor weight sparsity
        prior_type   : 'l1' (Laplace) or 'l2' (Gaussian)
        pred_loss    : 'l1' or 'l2' (smooth_l1) for cross-modal prediction
        congruence_mode : 'none' | 'pred_only' | 'pred_and_sparse'
            - 'none'           : all terms equally weighted (baseline)
            - 'pred_only'      : sigmoid gate on pred_loss only
            - 'pred_and_sparse': sigmoid gate on pred_loss AND sparse_loss
        congruence_tau : temperature τ for sigmoid gate (lower = sharper)

        noise_reweighting (deprecated): old alias; mapped automatically.
    """

    # Backward-compat: keep the class-level dict for any external code that
    # might reference EBMJEPALoss._CONGRUENCE_ALIAS_MAP directly.
    _CONGRUENCE_ALIAS_MAP = _CONGRUENCE_ALIAS_MAP

    def __init__(
        self,
        predictor_a2b,
        predictor_b2a,
        lambda_jac: float = 1.0,
        lambda_prior: float = 0.5,
        lambda_pred: float = 1.0,
        lambda_sparse: float = 0.1,
        prior_type: str = 'l1',
        pred_loss: str = 'l1',
        congruence_mode: str = 'none',
        congruence_tau: float = 0.5,
        # ---- deprecated alias ----
        noise_reweighting: str = None,
        reweighting_tau: float = None,
    ):
        super().__init__()
        # Backward-compat: map deprecated noise_reweighting -> congruence_mode
        if noise_reweighting is not None:
            mapped = canonicalize_congruence_mode(noise_reweighting)
            warnings.warn(
                f"noise_reweighting='{noise_reweighting}' is deprecated. "
                f"Use congruence_mode='{mapped}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            congruence_mode = mapped
        if reweighting_tau is not None:
            warnings.warn(
                "reweighting_tau is deprecated. Use congruence_tau instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            congruence_tau = reweighting_tau

        congruence_mode = canonicalize_congruence_mode(congruence_mode)

        self.predictor_a2b = predictor_a2b
        self.predictor_b2a = predictor_b2a
        self.lambda_jac = lambda_jac
        self.lambda_prior = lambda_prior
        self.lambda_pred = lambda_pred
        self.lambda_sparse = lambda_sparse
        self.prior_type = prior_type
        self.pred_loss = pred_loss
        self.congruence_mode = congruence_mode
        self.congruence_tau = congruence_tau

    # ------------------------------------------------------------------
    # Per-sample loss helpers
    # ------------------------------------------------------------------

    def _pred_loss_per_sample(self, pred, target):
        """Per-sample prediction loss, summed over dims, shape (N,)."""
        if self.pred_loss == 'l1':
            return F.l1_loss(pred, target, reduction='none').sum(dim=-1)
        else:  # l2 / MSE
            return F.mse_loss(pred, target, reduction='none').sum(dim=-1)

    def _prior_per_sample(self, z_a, z_b):
        """Per-sample prior loss, shape (N,)."""
        if self.prior_type == 'l1':
            return self.lambda_prior * (z_a.abs().sum(dim=-1) + z_b.abs().sum(dim=-1))
        else:  # l2 Gaussian
            return self.lambda_prior * (z_a.pow(2).sum(dim=-1) + z_b.pow(2).sum(dim=-1))

    # ------------------------------------------------------------------
    # Congruence gate
    # ------------------------------------------------------------------

    def _congruence_weights(self, pred_loss_per):
        """Sigmoid gate: w_i = σ(−pred_per_i / τ) ∈ (0, 1).

        High prediction error → low weight (noisy / incongruent sample).
        Returns:
            w_norm   : normalised weights (sum to 1) for sample-wise terms
            mean_raw : average sigmoid weight (0 to 1) for scalar terms
        """
        raw = torch.sigmoid(-pred_loss_per / self.congruence_tau)  # (N,)
        mean_raw = raw.mean()
        w_norm = raw / (raw.sum() + 1e-8)                          # normalised
        return w_norm, mean_raw

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, dual_model_outputs):
        # 1. Unpack latents & jacobians
        d = (dual_model_outputs.shape[1] - 2) // 2
        z_a = dual_model_outputs[:, :d]
        z_b = dual_model_outputs[:, d:2 * d]
        jac_a = dual_model_outputs[:, 2 * d]       # (N,)
        jac_b = dual_model_outputs[:, 2 * d + 1]   # (N,)

        # 2. Per-sample primary losses
        if self.lambda_pred > 0.0 and self.predictor_a2b is not None:
            loss_a2b = self._pred_loss_per_sample(self.predictor_a2b(z_a), z_b)   # (N,)
            loss_b2a = self._pred_loss_per_sample(self.predictor_b2a(z_b), z_a)   # (N,)
            pred_loss_per = loss_a2b + loss_b2a                                    # (N,)
        else:
            pred_loss_per = z_a.new_zeros(z_a.shape[0])

        prior_loss_per = self._prior_per_sample(z_a, z_b)                     # (N,)
        jac_per = jac_a + jac_b                                                # (N,)

        # 3. Sparsity penalty (scalar; predictor weights only, not biases)
        sparse_loss = pred_loss_per.new_zeros(())
        if self.lambda_sparse > 0 and self.lambda_pred > 0 and self.predictor_a2b is not None:
            for predictor in [self.predictor_a2b, self.predictor_b2a]:
                sparse_loss = sparse_loss + self.lambda_sparse * sum(
                    p.abs().sum()
                    for name, p in predictor.named_parameters()
                    if 'weight' in name
                )

        # 4. Aggregate with congruence gate
        jac_term   = -self.lambda_jac * jac_per.mean()          # uniform
        prior_term = prior_loss_per.mean()                       # uniform

        if self.lambda_pred == 0.0:
            return jac_term + prior_term

        if self.congruence_mode == 'none':
            pred_term = self.lambda_pred * pred_loss_per.mean()
            sparse_term = sparse_loss
        else:
            w_norm, mean_raw = self._congruence_weights(pred_loss_per)
            pred_term = self.lambda_pred * (w_norm * pred_loss_per).sum()
            if self.congruence_mode == 'pred_only':
                sparse_term = sparse_loss
            else:  # 'pred_and_sparse'
                sparse_term = sparse_loss * mean_raw

        return pred_term + jac_term + prior_term + sparse_term


def build_loss_from_config(cfg, predictor_a2b, predictor_b2a) -> torch.nn.Module:
    """Builds the loss module (either EBMJEPALoss or SupervisedFactorLoss) from configuration."""
    loss_cfg = cfg.loss
    loss_type = loss_cfg.get("type", "ebm")
    
    if loss_type == "ebm":
        return EBMJEPALoss(
            predictor_a2b,
            predictor_b2a,
            lambda_jac=loss_cfg.get("lambda_jac", 1.0),
            lambda_prior=loss_cfg.get("lambda_prior", 0.5),
            lambda_pred=loss_cfg.get("lambda_pred", 1.0),
            lambda_sparse=loss_cfg.get("lambda_sparse", 0.1),
            prior_type=loss_cfg.get("prior_type", 'l1'),
            pred_loss=loss_cfg.get("pred_loss", 'l1'),
            congruence_mode=loss_cfg.get("congruence_mode", "none"),
            congruence_tau=loss_cfg.get("congruence_tau", 0.5),
            noise_reweighting=loss_cfg.get("noise_reweighting"),
            reweighting_tau=loss_cfg.get("reweighting_tau"),
        )
    else:
        from multimodal_experiments.initial_trials.ssl_disentangling import SupervisedFactorLoss
        data_type = cfg.data.get('type', '2d')
        if data_type == '2d':
            dims = [1, 1]
        elif data_type == 'nd-kf-mlp':
            dims = [1] * cfg.data.get('k_shared', 2)
        else:
            dims = [1, 1, 1]
        return SupervisedFactorLoss(dimensions_per_factor=dims)

