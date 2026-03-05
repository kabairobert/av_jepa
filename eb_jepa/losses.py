import torch
import torch.nn as nn
import torch.nn.functional as F


def sq_loss(x, y, reduction="mean"):
    """Simple square loss (MSE)."""
    return nn.functional.mse_loss(x, y, reduction=reduction)


def square_cost_seq(state, predi):
    """Square loss between two [B, C, T, H, W] sequences."""
    return sq_loss(state, predi)


class SquareLossSeq(nn.Module):
    """Square loss over a sequence [B, C, T, H, W] (feature dim at dim 1)."""

    def __init__(self, proj=None):
        super().__init__()
        self.proj = nn.Identity() if proj is None else proj

    def forward(self, state, predi):
        state = self.proj(state.transpose(0, 1).flatten(1).transpose(0, 1))
        predi = self.proj(predi.transpose(0, 1).flatten(1).transpose(0, 1))
        return square_cost_seq(state, predi)


class VCLoss(nn.Module):
    """Variance-Covariance loss attracting means to zero and covariance to identity."""

    def __init__(self, std_coeff, cov_coeff, proj=None):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.proj = nn.Identity() if proj is None else proj
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, x, actions=None):
        x = x.transpose(0, 1).flatten(1).transpose(0, 1)  # [B*T*H*W, C]
        fx = self.proj(x)  # [B*T*H*W, C']

        std_loss = self.std_loss_fn(fx)
        cov_loss = self.cov_loss_fn(fx)

        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        total_unweighted_loss = std_loss + cov_loss
        loss_dict = {
            "std_loss": std_loss.item(),
            "cov_loss": cov_loss.item(),
        }
        return loss, total_unweighted_loss, loss_dict


class HingeStdLoss(torch.nn.Module):
    def __init__(
        self,
        std_margin: float = 1.0,
    ):
        """
        Encourages each feature to maintain at least a minimum standard deviation.
        Features with std below the margin incur a penalty of (std_margin - std).
        Args:
            std_margin (float, default=1.0):
                Minimum desired standard deviation per feature.
        """
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            std_loss: Scalar tensor with the hinge loss on standard deviations
        """
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss


class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelation.

        Normalizes by D * (D - 1) where D is feature dimensionality.
        """
        super().__init__()

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        # Calculate off-diagonal loss
        cov_loss = self.off_diagonal(cov).pow(2).mean()

        return cov_loss


class TemporalSimilarityLoss(torch.nn.Module):
    def __init__(self):
        """
        Temporal Similarity Loss.
        Encourages consecutive frames to have similar representations by penalizing
        the squared difference between consecutive time steps.
        """
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, N, D] where T is time steps, N is batch size, D is feature dimension
        """
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device)
        sim_loss_t = (x[1:] - x[:-1]).pow(2).mean()
        return sim_loss_t


class InverseDynamicsLoss(torch.nn.Module):
    def __init__(self, idm: nn.Module):
        """
        Predicts actions from consecutive states and compares with ground truth actions.
        Args:
            idm (nn.Module): Inverse dynamics model that takes (state_t, state_t+1) and predicts action
        """
        super().__init__()
        self.idm = idm

    def forward(self, x: torch.Tensor, actions: torch.Tensor):
        """
        Args:
            x: [T, B, D] - States across time steps
            actions: [B, A, T] - Ground truth actions between consecutive states
        """
        if x.shape[0] <= 1 or actions is None:
            return torch.tensor(0.0, device=x.device)

        t, b, d = x.shape

        states_t = x[:-1].transpose(0, 1)  # [B, T-1, D]
        states_t_plus_1 = x[1:].transpose(0, 1)  # [B, T-1, D]

        states_t_flat = states_t.reshape(-1, d)  # [B*(T-1), D]
        states_t_plus_1_flat = states_t_plus_1.reshape(-1, d)  # [B*(T-1), D]

        pred_actions = self.idm(states_t_flat, states_t_plus_1_flat)  # [B*(T-1), A]
        target_actions = actions.transpose(1, 2)[:, :-1].reshape(
            -1, actions.size(1)
        )  # [B*(T-1), A]
        idm_loss = F.mse_loss(pred_actions, target_actions)

        return idm_loss


class VC_IDM_Sim_Regularizer(torch.nn.Module):
    def __init__(
        self,
        cov_coeff: float,
        std_coeff: float,
        sim_coeff_t: float,
        idm_coeff: float = 0.0,
        idm: nn.Module = None,
        std_margin: float = 1,
        first_t_only: bool = True,
        projector: nn.Module = None,
        spatial_as_samples: bool = False,
        sim_t_after_proj: bool = False,
        idm_after_proj: bool = False,
    ):
        """
        Composite Regularizer combining multiple losses

        This is a composite loss that combines:
        - Hinge Standard Deviation Loss
        - Covariance Decorrelation Loss
        - Temporal Similarity Loss
        - Inverse Dynamics Model Loss

        Args:
            cov_coeff (float): Weight for covariance loss
            std_coeff (float): Weight for std hinge loss
            sim_coeff_t (float): Weight for temporal similarity loss
            idm_coeff (float): Weight for inverse dynamics loss
            idm (nn.Module): Inverse dynamics model
            std_margin (float): Minimum desired std per feature
            first_t_only (bool): Use only first time slice for std/cov loss
            projector (nn.Module): Optional projection layer
            spatial_as_samples (bool): Treat spatial locations as samples
            sim_t_after_proj (bool): Apply temporal loss after projection
            idm_after_proj (bool): Apply IDM loss after projection
        """
        super().__init__()
        self.cov_coeff = cov_coeff
        self.std_coeff = std_coeff
        self.sim_coeff_t = sim_coeff_t
        self.idm_coeff = idm_coeff

        self.first_t_only = first_t_only
        self.projector = nn.Identity() if projector is None else projector
        self.spatial_as_samples = spatial_as_samples
        self.sim_t_after_proj = sim_t_after_proj
        self.idm_after_proj = idm_after_proj

        # Initialize individual loss components
        self.std_loss_fn = HingeStdLoss(std_margin=std_margin)
        self.cov_loss_fn = CovarianceLoss()
        self.sim_loss_fn = TemporalSimilarityLoss()
        self.idm_loss_fn = InverseDynamicsLoss(idm) if idm is not None else None

    def forward(self, x, actions=None):
        """
        Args:
            x: [B, C, T, H, W] - Input activations. Internally reshaped to either
                [1, B, D] when first_t_only=True or [T*B, D] otherwise, with D=C*H*W.
            actions: [B, A, T] - Optional actions for IDM loss
        """
        b, c, t, h, w = x.shape

        # divergent gradient paths for x_unprojected and x_projected
        x_unprojected = x.permute(2, 0, 1, 3, 4).reshape(t, b, -1)  # [T, B, C*H*W]

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, c)  # [B*T*H*W, C]
        x_proj = self.projector(x_flat)  # [B*T*H*W, C_out]
        c_out = x_proj.shape[-1]
        x_projected = x_proj.view(b, t, h, w, c_out)  # [B, T, H, W, C_out]
        x_projected_reshaped = x_projected.permute(2, 0, 1, 3, 4).reshape(
            t, b, -1
        )  # [T, B, C_out*H*W]

        # SIM_T LOSS
        if self.sim_t_after_proj:
            sim_loss_t = self.sim_loss_fn(x_projected_reshaped)
        else:
            sim_loss_t = self.sim_loss_fn(x_unprojected)

        # IDM LOSS
        idm_loss = torch.tensor(0.0, device=x.device)
        if self.idm_coeff > 0 and self.idm_loss_fn is not None and actions is not None:
            if self.idm_after_proj:
                idm_loss = self.idm_loss_fn(x_projected_reshaped, actions)
            else:
                idm_loss = self.idm_loss_fn(x_unprojected, actions)

        # STD and COV LOSS
        if self.spatial_as_samples:
            if self.first_t_only:
                # Use only first time: [B*H*W, C_out]
                x_for_vc = x_projected[:, 0].reshape(b * h * w, c_out)
                assert x_for_vc.shape == (b * h * w, c_out)
            else:
                # Use all times: [B*T*H*W, C_out]
                x_for_vc = x_projected.reshape(-1, c_out)
                assert x_for_vc.shape == (b * t * h * w, c_out)
        else:
            x_for_vc = x_projected.permute(0, 1, 4, 2, 3).reshape(
                b, t, -1
            )  # [B, T, C_out*H*W]
            if self.first_t_only:
                # Use only first time: [B, C_out*H*W]
                x_for_vc = x_for_vc[:, 0]
                assert x_for_vc.shape == (b, c_out * h * w)
            else:
                # Use all times: [B*T, C_out*H*W]
                x_for_vc = x_for_vc.reshape(-1, x_for_vc.size(-1))
                assert x_for_vc.shape == (b * t, c_out * h * w)
        # [B*T, C_out*H*W] if first_t_only=False and spatial_as_samples=False
        # or [B, C_out*H*W] if first_t_only=True and spatial_as_samples=False
        # or [B*H*W, C_out] if first_t_only=True spatial_as_samples=True
        # or [B*T*H*W, C_out] if first_t_only=False spatial_as_samples=True
        std_loss = self.std_loss_fn(x_for_vc)
        cov_loss = self.cov_loss_fn(x_for_vc)

        total_weighted_loss = (
            self.cov_coeff * cov_loss
            + self.std_coeff * std_loss
            + self.sim_coeff_t * sim_loss_t
            + self.idm_coeff * idm_loss
        )
        total_unweighted_loss = cov_loss + std_loss + sim_loss_t + idm_loss

        loss_dict = {
            "cov_loss": cov_loss.item(),
            "std_loss": std_loss.item(),
            "sim_loss_t": sim_loss_t.item(),
            "idm_loss": idm_loss if isinstance(idm_loss, float) else idm_loss.item(),
        }

        return total_weighted_loss, total_unweighted_loss, loss_dict


class VICRegLoss(nn.Module):
    """VICReg loss combining invariance, variance (std), and covariance terms."""

    def __init__(self, std_coeff=1.0, cov_coeff=1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, z1, z2):
        """Compute VICReg loss.

        Args:
            z1: [B, D] - First projection tensor
            z2: [B, D] - Second projection tensor

        Returns:
            dict with keys: loss, invariance_loss, var_loss, cov_loss
        """
        # Invariance loss (similarity)
        sim_loss = F.mse_loss(z1, z2)

        # Variance loss (applied to both views and summed)
        var_loss = self.std_loss_fn(z1) + self.std_loss_fn(z2)

        # Covariance loss (applied to both views and summed)
        cov_loss = self.cov_loss_fn(z1) + self.cov_loss_fn(z2)

        total_loss = sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss

        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }


######################################################
# BCS (Batched Characteristic Slicing) loss for SIGReg


def all_reduce(x, op):
    """All-reduce operation for distributed training."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x, op=op)
        return x
    else:
        return x


def epps_pulley(x, t_min=-3, t_max=3, n_points=10):                     # my-comments: t:(-3,3) is Std.Normal 99% interval, excluding noise, n_points: resolution of the grid for numerical approximation of continuous integral
    """Epps-Pulley test statistic for Gaussianity."""
    # integration points
    t = torch.linspace(t_min, t_max, n_points, device=x.device)
    # theoretical CF for N(0, 1)
    exp_f = torch.exp(-0.5 * t**2)
    # ECF
    x_t = x.unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    # weighted L2 distance
    err = exp_f * (ecf - exp_f).abs() ** 2
    T = torch.trapz(err, t, dim=1)
    return T

class BCS(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg."""

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd

    def forward(self, z1, z2):
        with torch.no_grad():
            dev = z1.device
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A /= A.norm(p=2, dim=0)
        view1 = z1 @ A
        view2 = z2 @ A

        self.step += 1
        bcs = (epps_pulley(view1).mean() + epps_pulley(view2).mean()) / 2
        invariance_loss = F.mse_loss(z1, z2).mean()
        total_loss = invariance_loss + self.lmbd * bcs
        return {"loss": total_loss, "bcs_loss": bcs, "invariance_loss": invariance_loss}


class VideoJEPA_BCS(nn.Module):
    """SIGReg regularizer for Video JEPA, following LeJEPA (Balestriero & LeCun, 2025).

    Applies the BCS Gaussianity test to the encoder output (or predictor input),
    using a learned projector before the random slicing.

    Matches VCLoss interface exactly:
        forward(x, actions=None) -> (weighted_loss, unweighted_loss, loss_dict)

    Args:
        num_slices: Number of random 1-D projections for the Epps-Pulley test.
        lmbd: Weight of the BCS loss term.
        proj: Optional learned projector (nn.Module). If None, uses nn.Identity.
    """

    def __init__(
        self,
        num_slices: int = 256,
        lmbd: float = 0.05,
        proj=None,
    ):
        """Args:
            num_slices: number of random 1-D projections for the Epps-Pulley test.
            lmbd:       weight of the BCS loss term.
            proj:       optional learned projector (nn.Module); nn.Identity if None.
            euler:      if True, use real-valued Euler ECF (epps_pulley_distributed_euler);
                        if False, use original complex ECF (epps_pulley).
                        The Euler ECF implementation uses distributed synchronization
                        (all-reduce) for unbiased gradients in multi-GPU training.
        """
        super().__init__()
        self.num_slices = num_slices
        self.lmbd = lmbd
        self.proj = nn.Identity() if proj is None else proj
        self.step = 0

    def forward(self, x, actions=None):
        """
        Args:
            x: [B, C, T, H, W] - raw encoder output (same as VCLoss input)
            actions: ignored, present for interface compatibility
        Returns:
            (weighted_loss, unweighted_bcs_loss, loss_dict)
        """
        # Flatten [B, C, T, H, W] -> [B*T*H*W, C], exactly as VCLoss does
        x_flat = x.transpose(0, 1).flatten(1).transpose(0, 1)  # [B*T*H*W, C]
        fx = self.proj(x_flat)  # [B*T*H*W, C']

        D = fx.shape[1]
        with torch.no_grad():
            g = torch.Generator(device=fx.device)
            g.manual_seed(self.step)
            A = torch.randn(D, self.num_slices, device=fx.device, generator=g)
            A /= A.norm(p=2, dim=0)  # unit columns: [C', num_slices]

        z_proj = fx @ A  # [B*T*H*W, num_slices]
        bcs_loss = epps_pulley(z_proj).mean()

        self.step += 1

        loss = self.lmbd * bcs_loss
        return loss, bcs_loss, {"bcs_loss": bcs_loss.item()}


class EppsPulleyScaleInvariant(nn.Module):
    """ Epps-Pulley test statistic using real-valued Euler expansion:
        - Following LeJEPA and (Balestriero & LeCun, 2025) and mirrors 
          lejepa.EppsPulley but without N * world_size scaling to make it scale invariant 
          to batch size and number of DDP workers unlike the original implementation. 
          This is important for stable training across different hardware setups and batch sizes.
        - Replaces complex exponential (1j*x) .exp() with cos/sin decomposition via
          Euler's identity: e^(itX) = cos(tX) + i.sin(tX) 
          to keep all tensors in float32 and avoid bfloat16→complex64 type promotion.
        - Uses a positive-half trapezoidal rule with precomputed `t', `phi' (=exp(-t^2/2))
          and trapezoid weights stored as buffers so the forward pass is lightweight.
    
    Buffers (float32, same names as lejepa.EppsPulley):
        t       : positive-half integration nodes [0, t_max], shape [n_points]
        phi     : exp(-t^2/2), shape [n_points]
        weights : trapezoid weights * phi, shape [n_points]
                  (phi is folded in so forward only needs err @ weights)

    Args:
        t_max (float): upper integration bound (default 3.0)
        n_points (int): number of positive-half quadrature nodes (default 10)
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 10):
        super().__init__()
        # Positive-half trapezoidal grid: t in [0, t_max], dt = t_max / (n_points - 1)
        # Interior weights doubled to account for the symmetric negative half.
        t = torch.linspace(0.0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # half-weight at t=0 and t=t_max
        phi = torch.exp(-0.5 * t ** 2)
        # Fold phi into weights so forward only computes err @ self.weights
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-slice Epps-Pulley statistics.

        Args:
            x: [N, M] real-valued projected samples (N samples, M slices)
        Returns:
            [M] per-slice test statistics
        """
        # Cast to float32 for stable trig (bfloat16 has only 7 mantissa bits)
        x_t = x.to(self.t.dtype).unsqueeze(2) * self.t  # [N, M, T]

        real_ecf = torch.cos(x_t).mean(0)  # [M, T]
        imag_ecf = torch.sin(x_t).mean(0)  # [M, T]

        # Synchronize linear ECF components across DDP ranks BEFORE squaring
        # (required for unbiased gradients; no-op outside distributed training)
        real_ecf = all_reduce(real_ecf, op="AVG")
        imag_ecf = all_reduce(imag_ecf, op="AVG")

        # Squared complex distance; phi already folded into self.weights
        err = (real_ecf - self.phi).square() + imag_ecf.square()  # [M, T]
        return err @ self.weights  # [M]


class VideoJEPA_BCS_Euler_buffer(nn.Module):
    """SIGReg regularizer using the buffered Epps-Pulley test (Balestriero & LeCun, 2025).

    Combines random 1-D projection (BCS slicing) with EppsPulleyScaleInvariant.
    Matches the VCLoss interface: forward(x, actions=None) -> (loss, bcs_loss, dict).

    Args:
        num_slices (int): number of random 1-D projections.
        lmbd (float): weight of the BCS loss term.
        proj: optional learned projector (nn.Module); nn.Identity if None.
        t_max (float): passed to EppsPulleyScaleInvariant.
        n_points (int): passed to EppsPulleyScaleInvariant.
    """

    def __init__(
        self,
        num_slices: int = 256,
        lmbd: float = 0.05,
        proj=None,
        t_max: float = 3.0,
        n_points: int = 10,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.lmbd = lmbd
        self.proj = nn.Identity() if proj is None else proj
        self.epps_pulley = EppsPulleyScaleInvariant(t_max=t_max, n_points=n_points)
        self.step = 0

    def forward(self, x, actions=None):
        # Flatten [B, C, T, H, W] -> [B*T*H*W, C]
        x_flat = x.transpose(0, 1).flatten(1).transpose(0, 1)
        fx = self.proj(x_flat)  # [N, C']

        D = fx.shape[1]
        with torch.no_grad():
            g = torch.Generator(device=fx.device)
            g.manual_seed(self.step)
            A = torch.randn(D, self.num_slices, device=fx.device, generator=g)
            A /= A.norm(p=2, dim=0)  # unit columns: [C', num_slices]

        z_proj = fx @ A  # [N, num_slices]
        bcs_loss = self.epps_pulley(z_proj).mean()

        self.step += 1
        loss = self.lmbd * bcs_loss
        return loss, bcs_loss, {"bcs_loss": bcs_loss.item()}


