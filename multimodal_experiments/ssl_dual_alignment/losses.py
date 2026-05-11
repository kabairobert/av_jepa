import torch
import torch.nn.functional as F


class EBMJEPALoss(torch.nn.Module):
    """EBM JEPA Loss: Prediction error + Prior - Jacobian + Sparsity.

    Args:
        prior_type: 'l1' (Laplace) or 'l2' (Gaussian)
        pred_loss:  'l1' or 'l2' (smooth_l1) for cross-modal prediction
        noise_reweighting: 'none' | 'pred_only' | 'full'
            - 'none':      all samples equally weighted (original behaviour)
            - 'pred_only': softmax re-weight applied to pred loss only;
                           Jacobian and prior losses unweighted
            - 'full':      softmax re-weight applied to all three terms;
                           motivates geometry reshaping away from noise
        reweighting_tau: temperature for softmax re-weighting
    """
    def __init__(
        self,
        predictor_a2b,
        predictor_b2a,
        lambda_jac: float = 1.0,
        lambda_prior: float = 0.5,
        lambda_sparse: float = 0.1,
        prior_type: str = 'l1',
        pred_loss: str = 'l1',
        noise_reweighting: str = 'none',
        reweighting_tau: float = 0.5,
    ):
        super().__init__()
        self.predictor_a2b = predictor_a2b
        self.predictor_b2a = predictor_b2a
        self.lambda_jac = lambda_jac
        self.lambda_prior = lambda_prior
        self.lambda_sparse = lambda_sparse
        self.prior_type = prior_type
        self.pred_loss = pred_loss
        self.noise_reweighting = noise_reweighting
        self.reweighting_tau = reweighting_tau

    def _pred_loss_per_sample(self, pred, target):
        """Per-sample prediction loss, summed over dims, shape (N,)."""
        if self.pred_loss == 'l1':
            return F.l1_loss(pred, target, reduction='none').sum(dim=-1)
        else:  # l2 / smooth_l1
            return F.smooth_l1_loss(pred, target, reduction='none').sum(dim=-1)

    def _prior_per_sample(self, z_a, z_b):
        """Per-sample prior loss, shape (N,)."""
        if self.prior_type == 'l1':
            return self.lambda_prior * (z_a.abs().sum(dim=-1) + z_b.abs().sum(dim=-1))
        else:  # l2 Gaussian
            return self.lambda_prior * (z_a.pow(2).sum(dim=-1) + z_b.pow(2).sum(dim=-1))

    def forward(self, dual_model_outputs):
        # 1. Unpack latents & jacobians
        d = (dual_model_outputs.shape[1] - 2) // 2
        z_a = dual_model_outputs[:, :d]
        z_b = dual_model_outputs[:, d:2 * d]
        jac_a = dual_model_outputs[:, 2 * d]
        jac_b = dual_model_outputs[:, 2 * d + 1]

        # 2. Per-sample losses
        loss_a2b = self._pred_loss_per_sample(self.predictor_a2b(z_a), z_b)   # (N,)
        loss_b2a = self._pred_loss_per_sample(self.predictor_b2a(z_b), z_a)   # (N,)
        pred_loss_per = loss_a2b + loss_b2a                                    # (N,)

        prior_loss_per = self._prior_per_sample(z_a, z_b)                     # (N,)
        jac_per = jac_a + jac_b                                                # (N,)

        # 3. Reweighting
        if self.noise_reweighting == 'none':
            total = (pred_loss_per - self.lambda_jac * jac_per + prior_loss_per).mean()

        elif self.noise_reweighting == 'pred_only':
            weights = torch.softmax(-pred_loss_per / self.reweighting_tau, dim=0)  # (N,)
            total = (
                (weights * pred_loss_per).sum()
                - self.lambda_jac * jac_per.mean()
                + prior_loss_per.mean()
            )

        else:  # full
            weights = torch.softmax(-pred_loss_per / self.reweighting_tau, dim=0)  # (N,)
            total = (
                (weights * pred_loss_per).sum()
                - self.lambda_jac * (weights * jac_per).sum()
                + (weights * prior_loss_per).sum()
            )

        # 4. Sparsity penalty — works for all predictor types (weights only, not biases)
        sparse_loss = torch.tensor(0.0, device=dual_model_outputs.device)
        if self.lambda_sparse > 0 and self.predictor_a2b is not None:
            for predictor in [self.predictor_a2b, self.predictor_b2a]:
                sparse_loss = sparse_loss + self.lambda_sparse * sum(
                    p.abs().sum()
                    for name, p in predictor.named_parameters()
                    if 'weight' in name
                )

        return total + sparse_loss
