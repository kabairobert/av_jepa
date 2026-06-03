import math
import random
from dataclasses import dataclass
import warnings
from typing import Callable, Generator, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn


class CheckerBoardMask:
    def __init__(self, axes: List[int], shape: List[int]):
        self._axes_ = list(axes)
        self._shape_ = list(shape)
        mask = np.zeros(shape, dtype=np.float64)
        dimension_count = int(np.prod(shape))
        current_indices = [0] * len(shape)
        mask[tuple(current_indices)] = np.sum(current_indices) % 2
        for _ in range(dimension_count):
            for s in range(len(shape) - 1, -1, -1):
                if current_indices[s] == shape[s] - 1:
                    current_indices[s] = 0
                else:
                    current_indices[s] += 1
                    break
            mask[tuple(current_indices)] = np.sum(current_indices) % 2
        self._mask_ = torch.tensor(mask.reshape(-1), dtype=torch.get_default_dtype())

    def call(self, inputs: torch.Tensor, is_positive: bool = True) -> torch.Tensor:
        mask = self._mask_ if is_positive else 1.0 - self._mask_
        x = inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
        masked = x * mask.to(device=x.device, dtype=x.dtype)
        return masked


class FlowLayer(nn.Module):
    def __init__(self, shape: List[int], axes: List[int]):
        super().__init__()
        self._shape_ = list(shape)
        self._axes_ = list(axes)

    def compute_jacobian_determinant(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


def _make_dense_stack(num_dims: int, hidden_units: int) -> nn.Sequential:
    stack = nn.Sequential(
        nn.Linear(num_dims, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, num_dims),
    ).to(dtype=torch.get_default_dtype())

    for module in stack.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return stack


def _make_affine_dense_stack(
    num_dims: int,
    hidden_units: int,
    num_hidden_layers: int = 2,
) -> nn.Sequential:
    """Subnet for ClampedAffineCoupling.

    Outputs 2*num_dims: first half = log-scale (s), second half = translation (t).
    Final layer is zero-initialized so the flow starts as pure identity:
    s=0, t=0 → exp(0)=1 → y2 = x2 (no distortion at step 0).
    """
    layers: List[nn.Module] = []
    in_dim = num_dims
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(in_dim, hidden_units), nn.GELU()]
        in_dim = hidden_units
    final = nn.Linear(in_dim, num_dims * 2)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers).to(dtype=torch.get_default_dtype())


class ActivationNormalization(FlowLayer):
    def __init__(self, shape: List[int], axes: List[int]):
        super().__init__(shape=shape, axes=axes)
        self._location_ = nn.Parameter(torch.zeros(*shape, dtype=torch.get_default_dtype()))
        self._scale_ = nn.Parameter(torch.ones(*shape, dtype=torch.get_default_dtype()))
        self._scale_constraint_eps = 1e-6

    def _scale_positive(self) -> torch.Tensor:
        return torch.clamp(self._scale_, min=self._scale_constraint_eps)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        location = self._location_.to(device=inputs.device, dtype=inputs.dtype)
        scale = self._scale_positive().to(device=inputs.device, dtype=inputs.dtype)
        y_hat = (inputs - location) / scale
        jacobian_determinant = self.compute_jacobian_determinant(inputs)
        return y_hat, jacobian_determinant

    def invert(self, outputs: torch.Tensor) -> torch.Tensor:
        location = self._location_.to(device=outputs.device, dtype=outputs.dtype)
        scale = self._scale_positive().to(device=outputs.device, dtype=outputs.dtype)
        return outputs * scale + location

    def compute_jacobian_determinant(self, x: torch.Tensor) -> torch.Tensor:
        scale = self._scale_positive().to(device=x.device, dtype=x.dtype)
        dimension_count = 1
        for axis in range(1, len(x.shape)):
            if axis not in self._axes_:
                dimension_count *= x.shape[axis]
        jacobian_determinant = -dimension_count * torch.sum(torch.log(scale))
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype) + jacobian_determinant


class Reflection(FlowLayer):
    def __init__(self, shape: List[int], axes: List[int], reflection_count: int):
        assert reflection_count >= 1
        super().__init__(shape=shape, axes=axes)
        self._reflection_count_ = reflection_count
        dim = int(np.prod(shape))
        normals = -1.0 + 2.0 * torch.rand(reflection_count, dim, dtype=torch.get_default_dtype())
        normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self._reflection_normals_ = nn.Parameter(normals)
        self._inverse_mode_ = False

    def _reflect_(self, x: torch.Tensor) -> torch.Tensor:
        x_new = x
        indices = list(range(self._reflection_count_))
        if self._inverse_mode_:
            indices.reverse()
        for r in indices:
            v_r = self._reflection_normals_[r]
            v_r = v_r / v_r.norm(p=2).clamp_min(1e-6)
            dot = torch.sum(x_new * v_r, dim=-1, keepdim=True)
            x_new = x_new - 2.0 * dot * v_r
        return x_new

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        old_shape = list(inputs.shape)
        x = inputs.reshape(inputs.shape[0], -1)
        y_hat = self._reflect_(x)
        y_hat = y_hat.reshape(old_shape)
        jacobian_determinant = self.compute_jacobian_determinant(inputs)
        return y_hat, jacobian_determinant

    def invert(self, outputs: torch.Tensor) -> torch.Tensor:
        previous = self._inverse_mode_
        self._inverse_mode_ = True
        reconstructed, _ = self.forward(outputs)
        self._inverse_mode_ = previous
        return reconstructed

    def compute_jacobian_determinant(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


class _PermutationBase(FlowLayer):
    def __init__(self, shape: List[int], axes: List[int], permutation: Sequence[int]):
        super().__init__(shape=shape, axes=axes)
        self.register_buffer('_permutation_tensor', torch.tensor(permutation, dtype=torch.long))
        self.register_buffer('_inverse_permutation_tensor', torch.tensor(np.argsort(permutation), dtype=torch.long))
        self._permutation_ = list(permutation)
        self._inverse_permutation_ = list(np.argsort(self._permutation_))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_hat = inputs[:, self._permutation_tensor]
        jacobian_determinant = self.compute_jacobian_determinant(inputs)
        return y_hat, jacobian_determinant

    def invert(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs[:, self._inverse_permutation_tensor]

    def compute_jacobian_determinant(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


class CheckerBoardPermutation(_PermutationBase):
    def __init__(self, shape: List[int], axes: List[int]):
        dimension_count = int(np.prod(shape))
        tensor = np.reshape(np.arange(dimension_count), shape)
        rope_values = [None] * dimension_count

        def is_end_of_axis(index: int, limit: int, direction: int) -> bool:
            if direction == 1:
                return index == limit - 1
            return index == 0

        current_indices = [0] * len(shape)
        directions = [1] * len(shape)
        rope_values[0] = tensor[tuple(current_indices)]
        for d in range(dimension_count - 1):
            for s in range(len(shape) - 1, -1, -1):
                if is_end_of_axis(current_indices[s], shape[s], directions[s]):
                    directions[s] = -directions[s]
                else:
                    current_indices[s] += directions[s]
                    break
            rope_values[d + 1] = tensor[tuple(current_indices)]

        for d in range(0, dimension_count - 1, 2):
            rope_values[d], rope_values[d + 1] = rope_values[d + 1], rope_values[d]

        current_indices = [0] * len(shape)
        directions = [1] * len(shape)
        tensor[tuple(current_indices)] = rope_values[0]
        for d in range(dimension_count - 1):
            for s in range(len(shape) - 1, -1, -1):
                if is_end_of_axis(current_indices[s], shape[s], directions[s]):
                    directions[s] = -directions[s]
                else:
                    current_indices[s] += directions[s]
                    break
            tensor[tuple(current_indices)] = rope_values[d + 1]

        permutation = list(np.reshape(tensor, [-1]))
        super().__init__(shape=shape, axes=axes, permutation=permutation)


class Coupling(FlowLayer):
    def __init__(self, shape: List[int], axes: List[int], compute_coupling_parameters: nn.Module, mask: CheckerBoardMask):
        super().__init__(shape=shape, axes=axes)
        self._compute_coupling_parameters_ = compute_coupling_parameters
        self._mask_ = mask

    def _couple_(self, inputs: torch.Tensor, coupling_parameters: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _decouple_(self, outputs: torch.Tensor, coupling_parameters: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_1 = self._mask_.call(inputs=inputs, is_positive=True)
        coupling_parameters = self._compute_coupling_parameters_(x_1)
        y_hat_1 = x_1
        y_hat_2 = self._mask_.call(inputs=self._couple_(inputs=inputs, coupling_parameters=coupling_parameters), is_positive=False)
        y_hat = y_hat_1 + y_hat_2
        jacobian_determinant = self.compute_jacobian_determinant(inputs)
        return y_hat, jacobian_determinant

    def invert(self, outputs: torch.Tensor) -> torch.Tensor:
        y_hat_1 = self._mask_.call(inputs=outputs, is_positive=True)
        coupling_parameters = self._compute_coupling_parameters_(y_hat_1)
        x_1 = y_hat_1
        x_2 = self._mask_.call(inputs=self._decouple_(outputs=outputs, coupling_parameters=coupling_parameters), is_positive=False)
        return x_1 + x_2

    def compute_jacobian_determinant(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


class AdditiveCoupling(Coupling):
    def _couple_(self, inputs: torch.Tensor, coupling_parameters: torch.Tensor) -> torch.Tensor:
        return inputs + coupling_parameters

    def _decouple_(self, outputs: torch.Tensor, coupling_parameters: torch.Tensor) -> torch.Tensor:
        return outputs - coupling_parameters


class ClampedAffineCoupling(Coupling):
    """Affine coupling layer with tanh-clamped log-scale (RealNVP/Glow style).

    Replaces AdditiveCoupling's  y2 = x2 + t  with  y2 = x2 * exp(s) + t,
    where s is bounded via  s_clamped = clamp * tanh(s / clamp).

    Unlike AdditiveCoupling, this produces a non-zero log-det, so the flow
    is no longer volume-preserving and can learn to suppress noise dimensions.

    forward() is overridden directly (not via _couple_/_decouple_) because
    the log-det computation needs access to s_clamped inside the same pass.
    """

    def __init__(
        self,
        shape: List[int],
        axes: List[int],
        compute_coupling_parameters: nn.Module,
        mask: CheckerBoardMask,
        clamp: float = 2.0,
    ):
        super().__init__(
            shape=shape, axes=axes,
            compute_coupling_parameters=compute_coupling_parameters,
            mask=mask,
        )
        self.clamp = clamp

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Unchanged half (positive mask)
        x_1 = self._mask_.call(inputs=inputs, is_positive=True)

        # 2. Compute s and t from unchanged half
        st = self._compute_coupling_parameters_(x_1)   # (N, 2*D)
        s, t = st.chunk(2, dim=-1)                     # each (N, D)

        # 3. Clamp log-scale: s_clamped ∈ (-clamp, +clamp)
        s_clamped = self.clamp * torch.tanh(s / self.clamp)

        # 4. Affine transform on negative-mask half: y2 = x2 * exp(s) + t
        y_hat_2 = self._mask_.call(
            inputs=(inputs * torch.exp(s_clamped) + t),
            is_positive=False,
        )
        y_hat = x_1 + y_hat_2

        # 5. Log-det = sum of active (negative-mask) log-scales over all dims
        s_active = self._mask_.call(inputs=s_clamped, is_positive=False)
        jacobian_determinant = s_active.sum(dim=tuple(range(1, s_active.dim())))

        return y_hat, jacobian_determinant

    def invert(self, outputs: torch.Tensor) -> torch.Tensor:
        y_1 = self._mask_.call(inputs=outputs, is_positive=True)
        st = self._compute_coupling_parameters_(y_1)
        s, t = st.chunk(2, dim=-1)
        s_clamped = self.clamp * torch.tanh(s / self.clamp)
        # Inverse affine law: x2 = (y2 - t) * exp(-s)
        x_2 = self._mask_.call(
            inputs=((outputs - t) * torch.exp(-s_clamped)),
            is_positive=False,
        )
        return y_1 + x_2


class FlowModel(nn.Module):
    def __init__(self, flow_layers: List[FlowLayer]):
        super().__init__()
        self.flow_layers = nn.ModuleList(flow_layers)

    def build(self, input_shape=None):
        return self

    def summary(self):
        print("FlowModel(")
        for i, layer in enumerate(self.flow_layers):
            print(f"  ({i}): {layer.__class__.__name__}")
        print(")")

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = inputs
        jacobians = []
        for layer in self.flow_layers:
            z, jacobian = layer(z)
            jacobians.append(jacobian)
        total_j = torch.stack(jacobians, dim=0).sum(dim=0) if jacobians else torch.zeros(inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
        return z, total_j

    def invert(self, outputs: torch.Tensor) -> torch.Tensor:
        x = outputs
        for layer in reversed(self.flow_layers):
            x = layer.invert(x)
        return x


class SupervisedFactorLoss(nn.Module):
    # =========================================================================
    # DEPRECATED LEGACY COMPONENT (Replaced by EBMJEPALoss in active framework)
    # =========================================================================
    def __init__(self, dimensions_per_factor: List[int]):
        super().__init__()
        warnings.warn(
            "SupervisedFactorLoss is deprecated and may be removed in future versions. "
            "Use EBMJEPALoss instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        factor_count = len(dimensions_per_factor)
        factor_masks = np.zeros((factor_count, int(np.sum(dimensions_per_factor))), dtype=np.float64)
        total = 0
        for idx, dimension_count in enumerate(dimensions_per_factor):
            factor_masks[idx, total : total + dimension_count] = 1.0
            total += dimension_count
        self.__factor_masks__ = torch.tensor(factor_masks, dtype=torch.get_default_dtype())
        self.__dimensions_per_factor__ = list(dimensions_per_factor)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        d = int(np.sum(self.__dimensions_per_factor__))
        z_tilde_a = y_pred[:, :d]
        z_tilde_b = y_pred[:, d : 2 * d]
        j_a = y_pred[:, 2 * d]
        j_b = y_pred[:, 2 * d + 1]

        y_true_N = torch.matmul(y_true, self.__factor_masks__.to(device=y_true.device, dtype=y_true.dtype))
        eps = 1e-6
        y_true_N = torch.clamp(y_true_N, -1 + eps, 1 - eps)

        term_1 = 0.5 * torch.sum(z_tilde_a.pow(2), dim=1)
        var = 1.0 - y_true_N.pow(2)
        diff = z_tilde_b - y_true_N * z_tilde_a
        term_2 = 0.5 * torch.sum(diff.pow(2) / var + torch.log(var), dim=1)
        loss = term_1 + term_2 - (j_a + j_b)
        return loss.mean()


def reset_random_number_generators(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def add_noise(x_1, x_2, noise_standard_deviation, rng=None):
    if rng is None:
        rng = np.random
    return (
        x_1 + rng.normal(scale=noise_standard_deviation[0], size=x_1.shape),
        x_2 + rng.normal(scale=noise_standard_deviation[1], size=x_2.shape),
    )


def create_data_set(S: np.ndarray, manifold_function: Callable, noise_standard_deviation: Tuple[float, float], rng=None) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random
    data_function = lambda x: add_noise(*manifold_function(x), noise_standard_deviation=noise_standard_deviation, rng=rng)
    Y = np.concatenate([rng.standard_normal(size=[len(S), 1]), (S[:, np.newaxis] - np.mean(S)) / np.std(S)], axis=1)
    Z_1, Z_2 = data_function(x=S)
    Z = np.concatenate([Z_1[:, np.newaxis], Z_2[:, np.newaxis]], axis=1)
    return Z.astype(np.float64), Y.astype(np.float64)


def self_supervised_dual_generator(z1_data, z2_data, batch_size, rng=None):
    # =========================================================================
    # DEPRECATED LEGACY GENERATOR (Replaced by PyTorch DataLoader)
    # =========================================================================
    warnings.warn(
        "self_supervised_dual_generator is deprecated and replaced by PyTorch's native DataLoader.",
        DeprecationWarning,
        stacklevel=2,
    )
    if rng is None:
        rng = np.random
    num_samples = len(z1_data)
    y_corr_target = np.tile([0.0, 0.9], (batch_size, 1))
    while True:
        idx = rng.choice(num_samples, batch_size, replace=False)
        Z1_batch = z1_data[idx]
        Z2_batch = z2_data[idx]
        yield (Z1_batch, Z2_batch), y_corr_target


def construct_layers(
    stage_count: int,
    num_dims: int,
    hidden_units: int = 128,
    coupling_type: str = 'additive',
    coupling_clamp: float = 2.0,
    affine_subnet_layers: int = 2,
) -> List[FlowLayer]:
    """Build a list of FlowLayer objects for a normalizing flow.

    Args:
        stage_count: Number of (Reflection + 2xCoupling + 2xPermutation + ActNorm) blocks.
        num_dims: Dimensionality of the data.
        hidden_units: Hidden layer width for coupling subnets.
        coupling_type: 'additive' (volume-preserving, default) or 'affine' (non-volume-preserving).
        coupling_clamp: Tanh clamp magnitude for affine coupling log-scale. Ignored for additive.
        affine_subnet_layers: Number of hidden layers in the affine subnet. Ignored for additive.
    """
    layers: List[FlowLayer] = [None] * (6 * stage_count + 1)
    layers[0] = ActivationNormalization(shape=[num_dims], axes=[1])
    for i in range(stage_count):
        layers[6 * i + 1] = Reflection(shape=[num_dims], axes=[1], reflection_count=1)
        for slot in (2, 4):
            mask = CheckerBoardMask(axes=[1], shape=[num_dims])
            if coupling_type == 'affine':
                subnet = _make_affine_dense_stack(
                    num_dims=num_dims,
                    hidden_units=hidden_units,
                    num_hidden_layers=affine_subnet_layers,
                )
                coupling: FlowLayer = ClampedAffineCoupling(
                    shape=[num_dims], axes=[1],
                    compute_coupling_parameters=subnet,
                    mask=mask,
                    clamp=coupling_clamp,
                )
            else:  # 'additive' (default, backward-compatible)
                subnet = _make_dense_stack(num_dims=num_dims, hidden_units=hidden_units)
                coupling = AdditiveCoupling(
                    shape=[num_dims], axes=[1],
                    compute_coupling_parameters=subnet,
                    mask=mask,
                )
            layers[6 * i + slot] = coupling
            layers[6 * i + slot + 1] = CheckerBoardPermutation(shape=[num_dims], axes=[1])
        layers[6 * i + 6] = ActivationNormalization(shape=[num_dims], axes=[1])
    return layers


def set_global_seed(seed: int):
    reset_random_number_generators(seed=seed)


def sample_curve_data(param_values: np.ndarray, curve_fn: Callable, noise_std: Tuple[float, float], rng=None) -> Tuple[np.ndarray, np.ndarray]:
    return create_data_set(S=param_values, manifold_function=curve_fn, noise_standard_deviation=noise_std, rng=rng)


def paired_batch_generator(data_a: np.ndarray, data_b: np.ndarray, batch_size: int):
    # =========================================================================
    # DEPRECATED LEGACY GENERATOR (Replaced by PyTorch DataLoader)
    # =========================================================================
    warnings.warn(
        "paired_batch_generator is deprecated and replaced by PyTorch's native DataLoader.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self_supervised_dual_generator(z1_data=data_a, z2_data=data_b, batch_size=batch_size)


def build_flow_layers(
    stage_count: int,
    num_dims: int,
    hidden_units: int = 128,
    coupling_type: str = 'additive',
    coupling_clamp: float = 2.0,
    affine_subnet_layers: int = 2,
) -> List[FlowLayer]:
    """Public entry point for building flow layers. See construct_layers for param docs."""
    return construct_layers(
        stage_count=stage_count,
        num_dims=num_dims,
        hidden_units=hidden_units,
        coupling_type=coupling_type,
        coupling_clamp=coupling_clamp,
        affine_subnet_layers=affine_subnet_layers,
    )
