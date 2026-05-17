import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from multimodal_experiments.initial_trials.ssl_disentangling import sample_curve_data

class DualDisentangleDataset(Dataset):
    """Paired modality dset. Shared latent source."""
    def __init__(
        self,
        data_type: str = '2d',
        num_samples: int = 4096,
        path_a=None,
        path_b=None,
        manifold_noise_a: float = None,
        manifold_noise_b: float = None,
        asymmetric_noise_magnitude: float = None,
        asymmetric_noise_rate_a: float = 0.0,
        asymmetric_noise_rate_b: float = 0.0,
        external_noise_ratio: float = 0.0,
        noise_bbox_expansion: float = 0.0,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seed = seed

        # Alias resolution for data_type
        dt_upper = str(data_type).upper()
        if dt_upper == '3D1F':
            data_type = '3d-av-1f-common'
        elif dt_upper == '3D2F':
            data_type = '3d-2f-common'
        self.data_type = data_type

        # Use local RandomState for isolated reproducibility
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        if path_a is not None and path_b is not None:
            # External load
            data_a = self._load_file(path_a)
            data_b = self._load_file(path_b)
            self.num_samples = data_a.shape[0]
            self.param_values = np.linspace(0, 1, self.num_samples)
            self.point_type_a = np.zeros(self.num_samples, dtype=np.int32)
            self.point_type_b = np.zeros(self.num_samples, dtype=np.int32)
        else:
            # Synth generation
            self.manifold_noise_a = 0.02 if manifold_noise_a is None else manifold_noise_a
            self.manifold_noise_b = 0.02 if manifold_noise_b is None else manifold_noise_b
            self.asymmetric_noise_magnitude = asymmetric_noise_magnitude
            self.asymmetric_noise_rate_a = float(asymmetric_noise_rate_a)
            self.asymmetric_noise_rate_b = float(asymmetric_noise_rate_b)
            self.external_noise_ratio = float(external_noise_ratio)
            self.noise_bbox_expansion = float(noise_bbox_expansion)
            turns = 1

            # Split total num_samples into manifold / asymmetric / external according to rates
            N = num_samples
            n_external = int(np.floor(N * self.external_noise_ratio))
            n_asym_a = int(np.floor(N * self.asymmetric_noise_rate_a))
            n_asym_b = int(np.floor(N * self.asymmetric_noise_rate_b))
            n_manifold = N - (n_external + n_asym_a + n_asym_b)
            
            if n_manifold < 1:
                raise ValueError("num_samples too small for configured noise rates")

            if data_type == '2d':
                # 2D shapes from 1D u.
                r"""
                2D data breakdown:
                Shared 1D source $u$ ($[0,1]$).

                Mod A: Spiral. $N \times 2$. Radius + angle change by $u$.
                Mod B: Cubic. $N \times 2$. linear $x$ + $x^3$ squiggle.

                Relation: Samples paired by same $u$. Diff geometry, same source. Goal -> map back to shared 1D latent.
                """
                def curve_a_fn(u):
                    return ((0.8 * u + 0.2) * np.sin(turns * u * 2 * np.pi), (0.8 * u + 0.2) * np.cos(turns * u * 2 * np.pi))
                def curve_b_fn(u):
                    x = u * 2.0 - 1.0
                    return (x, x**3 - 0.5 * x - 0.5)

                # Manifold pairs
                u_man = np.linspace(0, 1, n_manifold)
                data_a_man, _ = sample_curve_data(u_man, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
                data_b_man, _ = sample_curve_data(u_man, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)

                parts_a = [data_a_man]
                parts_b = [data_b_man]
                parts_u = [u_man]
                
                # Point type codes:
                # 0=manifold  1=asym_a_good(A=manifold)  2=asym_b_corrupt(B=noisy)
                # 3=asym_b_good(B=manifold)  4=asym_a_corrupt(A=noisy)  5=external
                parts_pt_a = [np.zeros(n_manifold, dtype=np.int32)]
                parts_pt_b = [np.zeros(n_manifold, dtype=np.int32)]

                # Asymmetric: A-good, B-corrupted
                if n_asym_a > 0:
                    u_asym_a = self.rng.uniform(0, 1, n_asym_a)
                    a_good, _ = sample_curve_data(u_asym_a, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
                    b_good, _ = sample_curve_data(u_asym_a, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    b_corrupt = b_good + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_good.shape)
                    
                    parts_a.append(a_good)
                    parts_b.append(b_corrupt)
                    parts_u.append(u_asym_a)
                    parts_pt_a.append(np.full(n_asym_a, 1, dtype=np.int32))
                    parts_pt_b.append(np.full(n_asym_a, 2, dtype=np.int32))

                # Asymmetric: B-good, A-corrupted
                if n_asym_b > 0:
                    u_asym_b = self.rng.uniform(0, 1, n_asym_b)
                    a_good2, _ = sample_curve_data(u_asym_b, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
                    b_good2, _ = sample_curve_data(u_asym_b, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)
                    mag2 = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    a_corrupt = a_good2 + self.rng.normal(scale=mag2 * (1.0 + self.noise_bbox_expansion), size=a_good2.shape)
                    
                    parts_a.append(a_corrupt)
                    parts_b.append(b_good2)
                    parts_u.append(u_asym_b)
                    parts_pt_a.append(np.full(n_asym_b, 4, dtype=np.int32))
                    parts_pt_b.append(np.full(n_asym_b, 3, dtype=np.int32))

                # External random points (both sides random, independent)
                if n_external > 0:
                    # Robust bounds inference: always sample full manifold range
                    u_bound = np.linspace(0, 1, 100)
                    temp_a, _ = sample_curve_data(u_bound, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
                    temp_b, _ = sample_curve_data(u_bound, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)
                    
                    min_a, max_a = temp_a.min(axis=0), temp_a.max(axis=0)
                    min_b, max_b = temp_b.min(axis=0), temp_b.max(axis=0)
                    
                    # Expand bounding box
                    half = self.noise_bbox_expansion / 2.0
                    range_a = max_a - min_a
                    min_a, max_a = min_a - half * range_a, max_a + half * range_a
                    range_b = max_b - min_b
                    min_b, max_b = min_b - half * range_b, max_b + half * range_b
                    
                    ext_a = min_a + (max_a - min_a) * self.rng.rand(n_external, 2)
                    ext_b = min_b + (max_b - min_b) * self.rng.rand(n_external, 2)
                    ext_u = self.rng.uniform(0, 1, n_external)
                    
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
                Both modalities are 3D vectors ($N \times 3$) derived from one shared 1D latent ($u \in [0, 1]$) as only source. 

                Audio (A):
                * F1: Signal ($u$ rise).
                * F2: Signal ($\sin(u)$ curve). Deterministic function of $u$.
                * F3: Independent Noise.

                Video (B):
                * F1: Signal ($u$ linear).
                * F2: Dependent Noise. $u$ sets noise width. Information in variance, not value.
                * F3: Independent Noise.
                """
                def generate_3d1f_raw_features(u_vals, rng, noise_a, noise_b):
                    pitch = 1.0 / (1.2 - u_vals)
                    resonance = np.sin(u_vals * np.pi)
                    splash = rng.normal(0, noise_a, len(u_vals))
                    raw_a = np.stack([pitch, resonance, splash], axis=1)

                    dim1_b = u_vals
                    dim2_b = rng.normal(0, 1, len(u_vals)) * (0.5 + u_vals)
                    dim3_b = rng.normal(0, noise_b, len(u_vals))
                    raw_b = np.column_stack([dim1_b, dim2_b, dim3_b])
                    return raw_a, raw_b

                # 1. Collect all 'u' values first
                u_man = np.linspace(0, 1, n_manifold)
                u_asym_a = self.rng.uniform(0, 1, n_asym_a) if n_asym_a > 0 else np.array([])
                u_asym_b = self.rng.uniform(0, 1, n_asym_b) if n_asym_b > 0 else np.array([])
                u_all = np.concatenate([u_man, u_asym_a, u_asym_b])

                # 2. Generate raw features for all signal points
                raw_a_all, raw_b_all = generate_3d1f_raw_features(u_all, self.rng, self.manifold_noise_a, self.manifold_noise_b)
                
                # 3. Global Normalization (ensure signal points share same space)
                mean_a, std_a = raw_a_all.mean(axis=0), raw_a_all.std(axis=0)
                mean_b, std_b = raw_b_all.mean(axis=0), raw_b_all.std(axis=0)
                norm_a_all = (raw_a_all - mean_a) / (std_a + 1e-8)
                norm_b_all = (raw_b_all - mean_b) / (std_b + 1e-8)
                
                # 4. Apply Rotation (Fixed canonical rotation for Modality A)
                theta_y_a, theta_z_a = np.pi / 4, np.pi / 3
                Ry_a = np.array([[np.cos(theta_y_a), 0, np.sin(theta_y_a)], [0, 1, 0], [-np.sin(theta_y_a), 0, np.cos(theta_y_a)]])
                Rz_a = np.array([[np.cos(theta_z_a), -np.sin(theta_z_a), 0], [np.sin(theta_z_a), np.cos(theta_z_a), 0], [0, 0, 1]])
                rot_a_all = norm_a_all @ (Ry_a @ Rz_a).T
                rot_b_all = norm_b_all # Modality B stays canonical
                
                # 5. Split back into segments and apply corruption
                data_a_man = rot_a_all[:n_manifold]
                data_b_man = rot_b_all[:n_manifold]

                parts_a = [data_a_man]
                parts_b = [data_b_man]
                parts_u = [u_man]
                parts_pt_a = [np.zeros(n_manifold, dtype=np.int32)]
                parts_pt_b = [np.zeros(n_manifold, dtype=np.int32)]

                # Asymmetric A-good, B-corrupt
                if n_asym_a > 0:
                    idx_start = n_manifold
                    idx_end = n_manifold + n_asym_a
                    a_good = rot_a_all[idx_start:idx_end]
                    b_good = rot_b_all[idx_start:idx_end]
                    
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    b_corrupt = b_good + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_good.shape)
                    
                    parts_a.append(a_good)
                    parts_b.append(b_corrupt)
                    parts_u.append(u_asym_a)
                    parts_pt_a.append(np.ones(n_asym_a, dtype=np.int32))
                    parts_pt_b.append(np.full(n_asym_a, 2, dtype=np.int32))

                # Asymmetric B-good, A-corrupt
                if n_asym_b > 0:
                    idx_start = n_manifold + n_asym_a
                    idx_end = n_manifold + n_asym_a + n_asym_b
                    a_good = rot_a_all[idx_start:idx_end]
                    b_good = rot_b_all[idx_start:idx_end]
                    
                    mag2 = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    a_corrupt = a_good + self.rng.normal(scale=mag2 * (1.0 + self.noise_bbox_expansion), size=a_good.shape)
                    
                    parts_a.append(a_corrupt)
                    parts_b.append(b_good)
                    parts_u.append(u_asym_b)
                    parts_pt_a.append(np.full(n_asym_b, 4, dtype=np.int32))
                    parts_pt_b.append(np.full(n_asym_b, 3, dtype=np.int32))

                # 6. External points (calculated using robust manifold bounds)
                if n_external > 0:
                    u_bound = np.linspace(0, 1, 100)
                    rb_a_raw, rb_b_raw = generate_3d1f_raw_features(u_bound, self.rng, self.manifold_noise_a, self.manifold_noise_b)
                    
                    rb_a_norm = (rb_a_raw - mean_a) / (std_a + 1e-8)
                    rb_b_norm = (rb_b_raw - mean_b) / (std_b + 1e-8)
                    rb_a_rot = rb_a_norm @ (Ry_a @ Rz_a).T
                    
                    min_a, max_a = rb_a_rot.min(axis=0), rb_a_rot.max(axis=0)
                    min_b, max_b = rb_b_norm.min(axis=0), rb_b_norm.max(axis=0)
                    
                    half = self.noise_bbox_expansion / 2.0
                    range_a, range_b = max_a - min_a, max_b - min_b
                    min_a, max_a = min_a - half * range_a, max_a + half * range_a
                    min_b, max_b = min_b - half * range_b, max_b + half * range_b
                    
                    ext_a = min_a + (max_a - min_a) * self.rng.rand(n_external, 3)
                    ext_b = min_b + (max_b - min_b) * self.rng.rand(n_external, 3)
                    ext_u = self.rng.uniform(0, 1, n_external)
                    
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
                def generate_3d2f_raw_features(u1_vals, u2_vals, rng, noise_a, noise_b):
                    def curve_a_fn_3d(u):
                        theta = 4 * np.pi * u
                        r = u
                        return (r * np.cos(theta), r * np.sin(theta))
                    def curve_b_fn_3d(u):
                        x = u * 2.0 - 1.0
                        return (x, x**3 - 0.5 * x - 0.5)
                    
                    # xy_a from spiral logic
                    temp_xy_a, _ = sample_curve_data(u1_vals, curve_a_fn_3d, (noise_a, noise_a), rng=rng)
                    raw_a = np.hstack([temp_xy_a, (2.0 * u2_vals).reshape(-1, 1)])
                    
                    # xy_b from cubic logic
                    temp_xy_b, _ = sample_curve_data(u1_vals, curve_b_fn_3d, (noise_b, noise_b), rng=rng)
                    raw_b = np.hstack([temp_xy_b, u2_vals.reshape(-1, 1)])
                    return raw_a, raw_b

                # 1. Collect all latent factor pairs first
                u1_man = np.linspace(0, 1, n_manifold)
                u2_man = self.rng.uniform(0, 1, n_manifold)
                
                u1_asym_a = self.rng.uniform(0, 1, n_asym_a) if n_asym_a > 0 else np.array([])
                u2_asym_a = self.rng.uniform(0, 1, n_asym_a) if n_asym_a > 0 else np.array([])
                
                u1_asym_b = self.rng.uniform(0, 1, n_asym_b) if n_asym_b > 0 else np.array([])
                u2_asym_b = self.rng.uniform(0, 1, n_asym_b) if n_asym_b > 0 else np.array([])
                
                u1_all = np.concatenate([u1_man, u1_asym_a, u1_asym_b])
                u2_all = np.concatenate([u2_man, u2_asym_a, u2_asym_b])

                # 2. Generate raw features for all signal points
                raw_a_all, raw_b_all = generate_3d2f_raw_features(u1_all, u2_all, self.rng, self.manifold_noise_a, self.manifold_noise_b)

                # 3. Global Normalization
                mean_a, std_a = raw_a_all.mean(axis=0), raw_a_all.std(axis=0)
                mean_b, std_b = raw_b_all.mean(axis=0), raw_b_all.std(axis=0)
                norm_a_all = (raw_a_all - mean_a) / (std_a + 1e-8)
                norm_b_all = (raw_b_all - mean_b) / (std_b + 1e-8)

                # 4. Split back into segments and apply corruption
                data_a_man = norm_a_all[:n_manifold]
                data_b_man = norm_b_all[:n_manifold]
                
                parts_a = [data_a_man]
                parts_b = [data_b_man]
                parts_u = [np.column_stack([u1_man, u2_man])]
                parts_pt_a = [np.zeros(n_manifold, dtype=np.int32)]
                parts_pt_b = [np.zeros(n_manifold, dtype=np.int32)]

                # Asymmetric A-good, B-corrupt
                if n_asym_a > 0:
                    idx_start = n_manifold
                    idx_end = n_manifold + n_asym_a
                    a_good = norm_a_all[idx_start:idx_end]
                    b_good = norm_b_all[idx_start:idx_end]
                    
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    b_corrupt = b_good + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_good.shape)
                    
                    parts_a.append(a_good)
                    parts_b.append(b_corrupt)
                    parts_u.append(np.column_stack([u1_asym_a, u2_asym_a]))
                    parts_pt_a.append(np.ones(n_asym_a, dtype=np.int32))
                    parts_pt_b.append(np.full(n_asym_a, 2, dtype=np.int32))

                # Asymmetric B-good, A-corrupt
                if n_asym_b > 0:
                    idx_start = n_manifold + n_asym_a
                    idx_end = n_manifold + n_asym_a + n_asym_b
                    a_good = norm_a_all[idx_start:idx_end]
                    b_good = norm_b_all[idx_start:idx_end]
                    
                    mag2 = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    a_corrupt = a_good + self.rng.normal(scale=mag2 * (1.0 + self.noise_bbox_expansion), size=a_good.shape)
                    
                    parts_a.append(a_corrupt)
                    parts_b.append(b_good)
                    parts_u.append(np.column_stack([u1_asym_b, u2_asym_b]))
                    parts_pt_a.append(np.full(n_asym_b, 4, dtype=np.int32))
                    parts_pt_b.append(np.full(n_asym_b, 3, dtype=np.int32))

                # 5. External points
                if n_external > 0:
                    # Robust bounds using dedicated grid
                    u1_bound = np.linspace(0, 1, 100)
                    u2_bound = np.linspace(0, 1, 100)
                    rb_a_raw, rb_b_raw = generate_3d2f_raw_features(u1_bound, u2_bound, self.rng, self.manifold_noise_a, self.manifold_noise_b)
                    
                    rb_a_norm = (rb_a_raw - mean_a) / (std_a + 1e-8)
                    rb_b_norm = (rb_b_raw - mean_b) / (std_b + 1e-8)
                    
                    min_a, max_a = rb_a_norm.min(axis=0), rb_a_norm.max(axis=0)
                    min_b, max_b = rb_b_norm.min(axis=0), rb_b_norm.max(axis=0)
                    
                    half = self.noise_bbox_expansion / 2.0
                    range_a, range_b = max_a - min_a, max_b - min_b
                    min_a, max_a = min_a - half * range_a, max_a + half * range_a
                    min_b, max_b = min_b - half * range_b, max_b + half * range_b
                    
                    ext_a = min_a + (max_a - min_a) * self.rng.rand(n_external, 3)
                    ext_b = min_b + (max_b - min_b) * self.rng.rand(n_external, 3)
                    ext_u = np.column_stack([self.rng.uniform(0, 1, n_external), self.rng.uniform(0, 1, n_external)])
                    
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
