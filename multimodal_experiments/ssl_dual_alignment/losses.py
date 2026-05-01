import torch
import torch.nn.functional as F

class EBMJEPALoss(torch.nn.Module):
    """EBM JEPA Loss: Prediction error + Prior - Jacobian + Sparsity."""
    def __init__(self, predictor_a2b, predictor_b2a, lambda_jac=1.0, lambda_prior=0.5, lambda_sparse=0.1, use_l1=False):
        super().__init__()
        self.predictor_a2b = predictor_a2b
        self.predictor_b2a = predictor_b2a
        self.lambda_jac = lambda_jac
        self.lambda_prior = lambda_prior
        self.lambda_sparse = lambda_sparse
        self.use_l1 = use_l1

    def forward(self, dual_model_outputs):
        # 1. Unpack latents & jacobians
        d = (dual_model_outputs.shape[1] - 2) // 2
        z_a, z_b = dual_model_outputs[:, :d], dual_model_outputs[:, d:2*d]
        jac_a, jac_b = dual_model_outputs[:, 2*d], dual_model_outputs[:, 2*d+1]

        # 2. Cross-predictive & prior losses
        if self.use_l1:
            loss_a2b = F.l1_loss(self.predictor_a2b(z_a), z_b, reduction='none').sum(dim=-1)
            loss_b2a = F.l1_loss(self.predictor_b2a(z_b), z_a, reduction='none').sum(dim=-1)
            prior_loss = self.lambda_prior * (torch.sum(torch.abs(z_a), dim=-1) + torch.sum(torch.abs(z_b), dim=-1))
        else:
            loss_a2b = F.smooth_l1_loss(self.predictor_a2b(z_a), z_b, reduction='none').sum(dim=-1)
            loss_b2a = F.smooth_l1_loss(self.predictor_b2a(z_b), z_a, reduction='none').sum(dim=-1)
            prior_loss = self.lambda_prior * (torch.sum(z_a**2, dim=-1) + torch.sum(z_b**2, dim=-1))
        
        # 3. Sparsity penalty for axis-alignment
        sparse_loss = 0
        if self.lambda_sparse > 0 and hasattr(self.predictor_a2b, 'weight'):
            sparse_loss = self.lambda_sparse * (torch.sum(torch.abs(self.predictor_a2b.weight)) + torch.sum(torch.abs(self.predictor_b2a.weight)))
        
        # 4. Final EBM loss
        return (loss_a2b + loss_b2a - self.lambda_jac * (jac_a + jac_b) + prior_loss).mean() + sparse_loss
