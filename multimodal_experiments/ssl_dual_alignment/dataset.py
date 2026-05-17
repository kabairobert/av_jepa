import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from multimodal_experiments.initial_trials.ssl_disentangling import sample_curve_data

class DualDisentangleDataset(Dataset):
    """Paired modality dset. Shared latent source."""
    def __init__(
        self,
        data_type='2d',
        num_samples=4096,
        path_a=None,
        path_b=None,
        manifold_noise_a=None,
        manifold_noise_b=None,
        asymmetric_noise_magnitude=None,
        asymmetric_noise_rate_a=None,
        asymmetric_noise_rate_b=None,
        external_noise_ratio=None,
        noise_bbox_expansion: float = 0.0,
        # Fractional expansion of the noise bounding box beyond manifold extents.
        # 0.0 = tight (manifold range only, original behaviour).
        # 0.25 = +25% total per axis: each edge pushed out by 12.5% of the range.
        # Also scales asymmetric Gaussian spread by (1 + noise_bbox_expansion).
        # Set via config: data.noise_bbox_expansion
        seed=None,
    ):
        # Setup. File paths or synth.
        self.num_samples = num_samples
        self.data_type = data_type
        self.seed = seed

        # Alias resolution for data_type
        dt_upper = str(data_type).upper()
        if dt_upper == '3D1F':
            data_type = '3d-av-1f-common'
        elif dt_upper == '3D2F':
            data_type = '3d-2f-common'
        self.data_type = data_type

        # Set global seed for reproducibility of noise generation
        if seed is not None:
            np.random.seed(seed)

        if path_a is not None and path_b is not None:
            # External load.
            data_a = self._load_file(path_a)
            data_b = self._load_file(path_b)
            self.num_samples = data_a.shape[0]
            self.param_values = np.linspace(0, 1, self.num_samples)
            self.point_type_a = np.zeros(self.num_samples, dtype=np.int32)
            self.point_type_b = np.zeros(self.num_samples, dtype=np.int32)
        else:
            # Synth generation.
            param_values = np.linspace(0, 1, num_samples)
            self.param_values = param_values
            # Number of spiral turns for spiral-based shapes (use 1 for single-turn spirals)
            turns = 1

            # Noise params: manifold_noise_a and manifold_noise_b mandatory (can be 0 to disable)
            # Optional: asymmetric/external enabled only when provided (None = off)
            self.manifold_noise_a = 0.02 if manifold_noise_a is None else manifold_noise_a
            self.manifold_noise_b = 0.02 if manifold_noise_b is None else manifold_noise_b
            self.asymmetric_noise_magnitude = asymmetric_noise_magnitude
            self.asymmetric_noise_rate_a = asymmetric_noise_rate_a if asymmetric_noise_rate_a is not None else 0.0
            self.asymmetric_noise_rate_b = asymmetric_noise_rate_b if asymmetric_noise_rate_b is not None else 0.0
            self.external_noise_ratio = external_noise_ratio if external_noise_ratio is not None else 0.0
            # Bounding box expansion factor for all noise regions.
            self.noise_bbox_expansion = float(noise_bbox_expansion) if noise_bbox_expansion is not None else 0.0

            if data_type == '2d':
                # 2D shapes from 1D u.
                r"""
                2D data breakdown:
                Shared 1D source $u$ ($[0,1]$).

                Mod A: Spiral. $N \times 2$. Radius + angle change by $u$.
                Mod B: Cubic. $N \times 2$. linear $x$ + $x^3$ squiggle.

                Relation: Samples paired by same $u$. Diff geometry, same source. Goal -> map back to shared 1D latent.
                """
                def curve_a_fn(u: np.ndarray):
                    return ((0.8 * u + 0.2) * np.sin(turns * u * 2 * np.pi), (0.8 * u + 0.2) * np.cos(turns * u * 2 * np.pi))
                def curve_b_fn(u: np.ndarray):
                    x = u * 2.0 - 1.0
                    y = x**3 - 0.5 * x - 0.5
                    return (x, y)

                # Split total num_samples into manifold / asymmetric / external according to rates
                N = num_samples
                # Use floor to ensure 0.0 ratio results in strictly 0 samples
                n_external = int(np.floor(N * self.external_noise_ratio))
                n_asym_a = int(np.floor(N * self.asymmetric_noise_rate_a))
                n_asym_b = int(np.floor(N * self.asymmetric_noise_rate_b))
                n_manifold = N - (n_external + n_asym_a + n_asym_b)

                if n_manifold < 1:
                    raise ValueError("num_samples too small for configured noise rates")

                # Manifold pairs
                u_man = np.linspace(0, 1, n_manifold)
                data_a_man, _ = sample_curve_data(u_man, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a))
                data_b_man, _ = sample_curve_data(u_man, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b))

                parts_a = [data_a_man]
                parts_b = [data_b_man]
                parts_u = [u_man]
                # Point type codes:
                # 0=manifold  1=asym_a_good(A=manifold)  2=asym_b_corrupt(B=noisy)
                # 3=asym_b_good(B=manifold)  4=asym_a_corrupt(A=noisy)  5=external
                parts_pt_a = [np.zeros(n_manifold, dtype=np.int32)]
                parts_pt_b = [np.zeros(n_manifold, dtype=np.int32)]

                # Asymmetric: A-good, B-corrupted (asymmetric_noise_rate_a)
                if n_asym_a > 0:
                    u_asym_a = np.random.uniform(0, 1, n_asym_a)
                    a_good, _ = sample_curve_data(u_asym_a, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a))
                    b_good, _ = sample_curve_data(u_asym_a, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b))
                    if self.asymmetric_noise_magnitude is None:
                        mag = 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    else:
                        mag = self.asymmetric_noise_magnitude
                    b_corrupt = b_good + np.random.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_good.shape)
                    parts_a.append(a_good)
                    parts_b.append(b_corrupt)
                    parts_u.append(u_asym_a)
                    parts_pt_a.append(np.ones(n_asym_a, dtype=np.int32))           # A = manifold
                    parts_pt_b.append(np.full(n_asym_a, 2, dtype=np.int32))        # B = corrupted

                # Asymmetric: B-good, A-corrupted (asymmetric_noise_rate_b)
                if n_asym_b > 0:
                    u_asym_b = np.random.uniform(0, 1, n_asym_b)
                    a_good2, _ = sample_curve_data(u_asym_b, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a))
                    b_good2, _ = sample_curve_data(u_asym_b, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b))
                    if self.asymmetric_noise_magnitude is None:
                        mag2 = 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    else:
                        mag2 = self.asymmetric_noise_magnitude
                    a_corrupt = a_good2 + np.random.normal(scale=mag2 * (1.0 + self.noise_bbox_expansion), size=a_good2.shape)
                    parts_a.append(a_corrupt)
                    parts_b.append(b_good2)
                    parts_u.append(u_asym_b)
                    parts_pt_a.append(np.full(n_asym_b, 4, dtype=np.int32))        # A = corrupted
                    parts_pt_b.append(np.full(n_asym_b, 3, dtype=np.int32))        # B = manifold

                # External random points (both sides random, independent)
                if n_external > 0:
                    # Robust bounds inference: always sample full manifold range
                    u_bound = np.linspace(0, 1, 100)
                    temp_a, _ = sample_curve_data(u_bound, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a))
                    temp_b, _ = sample_curve_data(u_bound, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b))
                    min_a, max_a = temp_a.min(axis=0), temp_a.max(axis=0)
                    min_b, max_b = temp_b.min(axis=0), temp_b.max(axis=0)
                    # Expand bounding box: push each edge out by (expansion/2) * range
                    half = self.noise_bbox_expansion / 2.0
                    range_a = max_a - min_a
                    min_a, max_a = min_a - half * range_a, max_a + half * range_a
                    range_b = max_b - min_b
                    min_b, max_b = min_b - half * range_b, max_b + half * range_b
                    ext_a = min_a + (max_a - min_a) * np.random.rand(n_external, temp_a.shape[1])
                    ext_b = min_b + (max_b - min_b) * np.random.rand(n_external, temp_b.shape[1])
                    ext_u = np.random.uniform(0, 1, n_external)
                    parts_a.append(ext_a)
                    parts_b.append(ext_b)
                    parts_u.append(ext_u)
                    parts_pt_a.append(np.full(n_external, 5, dtype=np.int32))
                    parts_pt_b.append(np.full(n_external, 5, dtype=np.int32))

                data_a = np.vstack(parts_a)
                data_b = np.vstack(parts_b)
                param_values = np.concatenate(parts_u)
                self.point_type_a = np.concatenate(parts_pt_a)
                self.point_type_b = np.concatenate(parts_pt_b)

            elif data_type == '3d-av-1f-common':
                # 3D physical traits + rotation. 1D u.
                r"""
                3D data with 1 shared source:
                Both modalities are 3D vectors ($N \times 3$) derived from one shared 1D latent ($u \in [0, 1]$) as only source. Feature breakdown:

                Audio (A):
                * F1: Signal ($u$ rise).
                * F2: Signal ($\sin(u)$ curve). Not noise. Deterministic function of $u$.
                * F3: Independent Noise.

                Video (B):
                * F1: Signal ($u$ linear).
                * F2: Dependent Noise. $u$ sets noise width. Information in variance, not value.
                * F3: Independent Noise.

                # Summary:
                # A has more signal ($u$ + curve). B has signal + signal-dependent noise. Both rotated to hide source.
                # """
                N = num_samples
                n_external = int(np.floor(N * self.external_noise_ratio))
                n_asym_a = int(np.floor(N * self.asymmetric_noise_rate_a))
                n_asym_b = int(np.floor(N * self.asymmetric_noise_rate_b))
                n_manifold = N - (n_external + n_asym_a + n_asym_b)
                if n_manifold < 1:
                    raise ValueError("num_samples too small for configured noise rates")


                # Manifold samples
                u_man = np.linspace(0, 1, n_manifold)
                pitch = 1.0 / (1.2 - u_man)
                resonance = np.sin(u_man * np.pi)
                splashing_noise = np.random.normal(0, self.manifold_noise_a, n_manifold)
                data_a_unrot = np.stack([pitch, resonance, splashing_noise], axis=1)
                data_a_std = (data_a_unrot - data_a_unrot.mean(axis=0)) / data_a_unrot.std(axis=0)
                theta_y_a = np.pi / 4
                theta_z_a = np.pi / 3
                Ry_a = np.array([[np.cos(theta_y_a), 0, np.sin(theta_y_a)], [0, 1, 0], [-np.sin(theta_y_a), 0, np.cos(theta_y_a)]])
                Rz_a = np.array([[np.cos(theta_z_a), -np.sin(theta_z_a), 0], [np.sin(theta_z_a), np.cos(theta_z_a), 0], [0, 0, 1]])
                data_a_man = data_a_std @ (Ry_a @ Rz_a).T

                dim1_b = u_man
                dim2_b = np.random.normal(0, 1, n_manifold) * (0.5 + u_man)
                dim3_b = np.random.normal(0, self.manifold_noise_b, n_manifold)
                data_b_raw = np.column_stack((dim1_b, dim2_b, dim3_b))
                data_b_man = (data_b_raw - np.mean(data_b_raw, axis=0)) / np.std(data_b_raw, axis=0)

                parts_a = [data_a_man]
                parts_b = [data_b_man]
                parts_u = [u_man]
                parts_pt_a = [np.zeros(n_manifold, dtype=np.int32)]
                parts_pt_b = [np.zeros(n_manifold, dtype=np.int32)]

                # Asymmetric A-good, B-corrupt
                if n_asym_a > 0:
                    u_asym_a = np.random.uniform(0, 1, n_asym_a)
                    pitch_a = 1.0 / (1.2 - u_asym_a)
                    res_a = np.sin(u_asym_a * np.pi)
                    splash_a = np.random.normal(0, self.manifold_noise_a, n_asym_a)
                    a_good = np.stack([pitch_a, res_a, splash_a], axis=1)
                    a_good = (a_good - a_good.mean(axis=0)) / a_good.std(axis=0)

                    dim1_b = u_asym_a
                    dim2_b = np.random.normal(0, 1, n_asym_a) * (0.5 + u_asym_a)
                    dim3_b = np.random.normal(0, self.manifold_noise_b, n_asym_a)
                    b_good = (np.column_stack((dim1_b, dim2_b, dim3_b)) - np.mean(np.column_stack((dim1_b, dim2_b, dim3_b)), axis=0)) / np.std(np.column_stack((dim1_b, dim2_b, dim3_b)), axis=0)
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    b_corrupt = b_good + np.random.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_good.shape)
                    parts_a.append(a_good)
                    parts_b.append(b_corrupt)
                    parts_u.append(u_asym_a)
                    parts_pt_a.append(np.ones(n_asym_a, dtype=np.int32))
                    parts_pt_b.append(np.full(n_asym_a, 2, dtype=np.int32))

                # Asymmetric B-good, A-corrupt
                if n_asym_b > 0:
                    u_asym_b = np.random.uniform(0, 1, n_asym_b)
                    pitch_a = 1.0 / (1.2 - u_asym_b)
                    res_a = np.sin(u_asym_b * np.pi)
                    splash_a = np.random.normal(0, self.manifold_noise_a, n_asym_b)
                    a_good2 = np.stack([pitch_a, res_a, splash_a], axis=1)
                    a_good2 = (a_good2 - a_good2.mean(axis=0)) / a_good2.std(axis=0)

                    dim1_b = u_asym_b
                    dim2_b = np.random.normal(0, 1, n_asym_b) * (0.5 + u_asym_b)
                    dim3_b = np.random.normal(0, self.manifold_noise_b, n_asym_b)
                    b_good2 = (np.column_stack((dim1_b, dim2_b, dim3_b)) - np.mean(np.column_stack((dim1_b, dim2_b, dim3_b)), axis=0)) / np.std(np.column_stack((dim1_b, dim2_b, dim3_b)), axis=0)
                    mag2 = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    a_corrupt = a_good2 + np.random.normal(scale=mag2 * (1.0 + self.noise_bbox_expansion), size=a_good2.shape)
                    parts_a.append(a_corrupt)
                    parts_b.append(b_good2)
                    parts_u.append(u_asym_b)
                    parts_pt_a.append(np.full(n_asym_b, 4, dtype=np.int32))
                    parts_pt_b.append(np.full(n_asym_b, 3, dtype=np.int32))

                # External random points
                if n_external > 0:
                    # Robust bounds inference: always sample full manifold range
                    u_bound = np.linspace(0, 1, 100)
                    pitch_t = 1.0 / (1.2 - u_bound)
                    resonance_t = np.sin(u_bound * np.pi)
                    splash_t = np.random.normal(0, self.manifold_noise_a, len(u_bound))
                    temp_a_unrot = np.stack([pitch_t, resonance_t, splash_t], axis=1)
                    temp_a = (temp_a_unrot - temp_a_unrot.mean(axis=0)) / temp_a_unrot.std(axis=0)
                    dim1_tb = u_bound
                    dim2_tb = np.random.normal(0, 1, len(u_bound)) * (0.5 + u_bound)
                    dim3_tb = np.random.normal(0, self.manifold_noise_b, len(u_bound))
                    temp_b_raw = np.column_stack((dim1_tb, dim2_tb, dim3_tb))
                    temp_b = (temp_b_raw - np.mean(temp_b_raw, axis=0)) / np.std(temp_b_raw, axis=0)
                    min_a, max_a = temp_a.min(axis=0), temp_a.max(axis=0)
                    min_b, max_b = temp_b.min(axis=0), temp_b.max(axis=0)
                    half = self.noise_bbox_expansion / 2.0
                    range_a = max_a - min_a
                    min_a, max_a = min_a - half * range_a, max_a + half * range_a
                    range_b = max_b - min_b
                    min_b, max_b = min_b - half * range_b, max_b + half * range_b
                    ext_a = min_a + (max_a - min_a) * np.random.rand(n_external, temp_a.shape[1])
                    ext_b = min_b + (max_b - min_b) * np.random.rand(n_external, temp_b.shape[1])
                    ext_u = np.random.uniform(0, 1, n_external)
                    parts_a.append(ext_a)
                    parts_b.append(ext_b)
                    parts_u.append(ext_u)
                    parts_pt_a.append(np.full(n_external, 5, dtype=np.int32))
                    parts_pt_b.append(np.full(n_external, 5, dtype=np.int32))

                data_a = np.vstack(parts_a)
                data_b = np.vstack(parts_b)
                param_values = np.concatenate(parts_u)
                self.point_type_a = np.concatenate(parts_pt_a)
                self.point_type_b = np.concatenate(parts_pt_b)

            elif data_type == '3d-2f-common':
                # 2 common factors. u1 -> shape, u2 -> 3rd dim (stretch vs shear).
                r"""
                3D data with 2d shared source. Structure:
                * Base: 2D latent grid $\mathbf{u} = [u_1, u_2] \in [0,1] \times [0,1]$.
                * $u_1$: Same as $u$ from 2D (controls Spiral/Cubic).
                * $u_2$: New shared factor.

                Modality A (3D Spiral):
                1. Spiral logic on $u_1$ -> $(x, y)$.
                2. Add $u_2$ as 3rd dimension.
                3. Stretch $u_2$ width to 2 ($2 \times u_2$).
                4. Final: $(x, y, 2u_2) + \text{noise}$.

                Modality B (3D Cubic):
                1. Cubic logic on $u_1$ -> $(x, y)$.
                2. Add $u_2$ as shear -> diag increase in Dim 3.
                3. Multiplier 1 (width 1).
                4. Final: $(x, y, u_2) + \text{noise}$.
                """
                N = num_samples
                n_external = int(np.floor(N * self.external_noise_ratio))
                n_asym_a = int(np.floor(N * self.asymmetric_noise_rate_a))
                n_asym_b = int(np.floor(N * self.asymmetric_noise_rate_b))
                n_manifold = N - (n_external + n_asym_a + n_asym_b)
                if n_manifold < 1:
                    raise ValueError("num_samples too small for configured noise rates")

                u1_man = np.linspace(0, 1, n_manifold)
                u2_man = np.random.uniform(0, 1, n_manifold)

                def curve_a_fn_3d(u1_vals):
                    return ((0.8 * u1_vals + 0.2) * np.sin(turns * u1_vals * 2 * np.pi),
                            (0.8 * u1_vals + 0.2) * np.cos(turns * u1_vals * 2 * np.pi))

                xy_a_man, _ = sample_curve_data(u1_man, curve_a_fn_3d, (self.manifold_noise_a, self.manifold_noise_a))
                z_a_man = (u2_man * 2.0).reshape(-1, 1)
                data_a_man = np.hstack([xy_a_man, z_a_man])

                def curve_b_fn_3d(u1_vals):
                    x = u1_vals * 2.0 - 1.0
                    y = x**3 - 0.5 * x - 0.5
                    return (x, y)

                xy_b_man, _ = sample_curve_data(u1_man, curve_b_fn_3d, (self.manifold_noise_b, self.manifold_noise_b))
                z_b_man = u2_man.reshape(-1, 1)
                data_b_man = np.hstack([xy_b_man, z_b_man])

                parts_a = [data_a_man]
                parts_b = [data_b_man]
                parts_u = [np.column_stack([u1_man, u2_man])]
                parts_pt_a = [np.zeros(n_manifold, dtype=np.int32)]
                parts_pt_b = [np.zeros(n_manifold, dtype=np.int32)]

                # Asymmetric A-good, B-corrupt
                if n_asym_a > 0:
                    u1_as = np.random.uniform(0, 1, n_asym_a)
                    u2_as = np.random.uniform(0, 1, n_asym_a)
                    xy_a_as, _ = sample_curve_data(u1_as, curve_a_fn_3d, (self.manifold_noise_a, self.manifold_noise_a))
                    z_a_as = (u2_as * 2.0).reshape(-1, 1)
                    a_good = np.hstack([xy_a_as, z_a_as])

                    xy_b_as, _ = sample_curve_data(u1_as, curve_b_fn_3d, (self.manifold_noise_b, self.manifold_noise_b))
                    z_b_as = u2_as.reshape(-1, 1)
                    b_good = np.hstack([xy_b_as, z_b_as])
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    b_corrupt = b_good + np.random.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_good.shape)
                    parts_a.append(a_good)
                    parts_b.append(b_corrupt)
                    parts_u.append(np.column_stack([u1_as, u2_as]))
                    parts_pt_a.append(np.ones(n_asym_a, dtype=np.int32))
                    parts_pt_b.append(np.full(n_asym_a, 2, dtype=np.int32))

                # Asymmetric B-good, A-corrupt
                if n_asym_b > 0:
                    u1_as2 = np.random.uniform(0, 1, n_asym_b)
                    u2_as2 = np.random.uniform(0, 1, n_asym_b)
                    xy_a_as2, _ = sample_curve_data(u1_as2, curve_a_fn_3d, (self.manifold_noise_a, self.manifold_noise_a))
                    z_a_as2 = (u2_as2 * 2.0).reshape(-1, 1)
                    a_good2 = np.hstack([xy_a_as2, z_a_as2])

                    xy_b_as2, _ = sample_curve_data(u1_as2, curve_b_fn_3d, (self.manifold_noise_b, self.manifold_noise_b))
                    z_b_as2 = u2_as2.reshape(-1, 1)
                    b_good2 = np.hstack([xy_b_as2, z_b_as2])
                    mag2 = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    a_corrupt = a_good2 + np.random.normal(scale=mag2 * (1.0 + self.noise_bbox_expansion), size=a_good2.shape)
                    parts_a.append(a_corrupt)
                    parts_b.append(b_good2)
                    parts_u.append(np.column_stack([u1_as2, u2_as2]))
                    parts_pt_a.append(np.full(n_asym_b, 4, dtype=np.int32))
                    parts_pt_b.append(np.full(n_asym_b, 3, dtype=np.int32))

                # External random points
                if n_external > 0:
                    # Robust bounds inference: always sample full manifold range
                    u1_bound = np.linspace(0, 1, 100)
                    u2_bound = np.random.uniform(0, 1, 100)

                    temp_xy_a, _ = sample_curve_data(u1_bound, curve_a_fn_3d, (self.manifold_noise_a, self.manifold_noise_a))
                    temp_a = np.hstack([temp_xy_a, (2.0 * u2_bound).reshape(-1, 1)])

                    temp_xy_b, _ = sample_curve_data(u1_bound, curve_b_fn_3d, (self.manifold_noise_b, self.manifold_noise_b))
                    temp_b = np.hstack([temp_xy_b, u2_bound.reshape(-1, 1)])
                    min_a, max_a = temp_a.min(axis=0), temp_a.max(axis=0)
                    min_b, max_b = temp_b.min(axis=0), temp_b.max(axis=0)
                    half = self.noise_bbox_expansion / 2.0
                    range_a = max_a - min_a
                    min_a, max_a = min_a - half * range_a, max_a + half * range_a
                    range_b = max_b - min_b
                    min_b, max_b = min_b - half * range_b, max_b + half * range_b
                    ext_a = min_a + (max_a - min_a) * np.random.rand(n_external, temp_a.shape[1])
                    ext_b = min_b + (max_b - min_b) * np.random.rand(n_external, temp_b.shape[1])
                    ext_u = np.column_stack([np.random.uniform(0, 1, n_external), np.random.uniform(0, 1, n_external)])
                    parts_a.append(ext_a)
                    parts_b.append(ext_b)
                    parts_u.append(ext_u)
                    parts_pt_a.append(np.full(n_external, 5, dtype=np.int32))
                    parts_pt_b.append(np.full(n_external, 5, dtype=np.int32))

                data_a = np.vstack(parts_a)
                data_b = np.vstack(parts_b)
                param_values = np.vstack(parts_u)
                self.point_type_a = np.concatenate(parts_pt_a)
                self.point_type_b = np.concatenate(parts_pt_b)

            else:
                raise ValueError(f"Unknown data type {data_type}")

            # Apply random 3D rotations (for 3D data types) after all generation
            if data_type.startswith('3d') and data_a.shape[1] == 3:
                data_a = self._apply_random_rotation(data_a, seed_offset=0)
                data_b = self._apply_random_rotation(data_b, seed_offset=1)

            # Compute a universal cubic axis box based on (rotated) input spaces
            if data_type.startswith('3d') and data_a.shape[1] == 3:
                combined = np.vstack([data_a, data_b])
                mins = combined.min(axis=0)
                maxs = combined.max(axis=0)
                center = (mins + maxs) / 2.0
                ranges = maxs - mins
                half_size = float(np.max(ranges) / 2.0)
                min_box = center - half_size
                max_box = center + half_size
                self.axis_box = np.vstack([min_box, max_box])
            else:
                self.axis_box = None

        # Data cast.
        self.data_a = torch.tensor(data_a, dtype=torch.float32)
        self.data_b = torch.tensor(data_b, dtype=torch.float32)
        self.param_values = param_values
        self.corr_target = torch.tensor(np.tile([0.0, 0.9], (self.num_samples, 1)), dtype=torch.float32)
        self.num_samples = self.data_a.shape[0]

    def _apply_random_rotation(self, data: np.ndarray, seed_offset: int = 0) -> np.ndarray:
        """Apply uniform random 3D rotation to data. Seed ensures reproducibility per modality."""
        if self.seed is None:
            return data
        rotation_seed = self.seed + seed_offset
        rng = np.random.RandomState(rotation_seed)
        rot = Rotation.random(random_state=rng)
        rotation_matrix = rot.as_matrix()
        return data @ rotation_matrix.T

    def _load_file(self, path):
        # Load data. npy/pt support.
        if path.endswith('.npy'):
            return np.load(path)
        elif path.endswith('.pt'):
            return torch.load(path).cpu().numpy()
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        param_val = self.param_values[idx] if isinstance(self.param_values, np.ndarray) else self.param_values[idx].numpy()
        return {
            "data_a": self.data_a[idx],
            "data_b": self.data_b[idx],
            "corr_target": self.corr_target[idx],
            "param_values": torch.tensor(param_val, dtype=torch.float32),
            "point_type_a": int(self.point_type_a[idx]),
            "point_type_b": int(self.point_type_b[idx]),
        }
