import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from multimodal_experiments.initial_trials.ssl_disentangling import sample_curve_data

class DualDisentangleDataset(Dataset):
    """Paired modality dataset with a shared latent source.
    
    Supports 2D and 3D data types with various noise regimes (asymmetric, external).
    Ensures isolated reproducibility via local RandomState.
    """
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
        self.external_noise_ratio = float(external_noise_ratio)
        self.asymmetric_noise_rate_a = float(asymmetric_noise_rate_a)
        self.asymmetric_noise_rate_b = float(asymmetric_noise_rate_b)
        self.noise_bbox_expansion = float(noise_bbox_expansion)

        # Alias resolution for data_type
        dt_upper = str(data_type).upper()
        if dt_upper == '3D1F':
            data_type = '3d-av-1f-common'
        elif dt_upper == '3D2F':
            data_type = '3d-2f-common'
        self.data_type = data_type

        # Use local RandomState for isolated reproducibility (fixes worker-fork duplicates)
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        if path_a is not None and path_b is not None:
            # External load mode
            data_a = self._load_file(path_a)
            data_b = self._load_file(path_b)
            self.num_samples = data_a.shape[0]
            self.param_values = np.linspace(0, 1, self.num_samples)
            self.point_type_a = np.zeros(self.num_samples, dtype=np.int32)
            self.point_type_b = np.zeros(self.num_samples, dtype=np.int32)
        else:
            # Synthetic generation mode
            self.manifold_noise_a = 0.02 if manifold_noise_a is None else manifold_noise_a
            self.manifold_noise_b = 0.02 if manifold_noise_b is None else manifold_noise_b
            self.asymmetric_noise_magnitude = asymmetric_noise_magnitude
            
            # Number of spiral turns (1 = 360 degrees)
            turns = 1

            # 1. Calculate sample counts for each segment
            N = num_samples
            n_external = int(np.floor(N * self.external_noise_ratio)) if self.external_noise_ratio > 0 else 0
            n_asym_a = int(np.floor(N * self.asymmetric_noise_rate_a)) if self.asymmetric_noise_rate_a > 0 else 0
            n_asym_b = int(np.floor(N * self.asymmetric_noise_rate_b)) if self.asymmetric_noise_rate_b > 0 else 0
            n_manifold = N - (n_external + n_asym_a + n_asym_b)
            
            if n_manifold < 1:
                raise ValueError("num_samples too small for configured noise rates")

            # --------------------------------------------------------------
            # DATA TYPE: 2D
            # --------------------------------------------------------------
            if data_type == '2d':
                r"""
                2D data breakdown:
                Shared 1D source u ([0,1]).
                Mod A: Spiral. radius + angle determined by u.
                Mod B: Cubic squiggle.
                """
                def curve_a_fn(u_vals):
                    # Spiral: radius grows with u, angle turns with u
                    return ((0.8 * u_vals + 0.2) * np.sin(turns * u_vals * 2 * np.pi),
                            (0.8 * u_vals + 0.2) * np.cos(turns * u_vals * 2 * np.pi))
                
                def curve_b_fn(u_vals):
                    # Cubic: simple squiggle
                    x = u_vals * 2.0 - 1.0
                    return (x, x**3 - 0.5 * x - 0.5)

                # Generate Manifold
                u_man = np.linspace(0, 1, n_manifold)
                data_a_man, _ = sample_curve_data(u_man, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
                data_b_man, _ = sample_curve_data(u_man, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)

                parts_a, parts_b, parts_u = [data_a_man], [data_b_man], [u_man]
                # Point types: 0=manifold, 1=asym_a_good, 2=asym_b_corrupt, 3=asym_b_good, 4=asym_a_corrupt, 5=external
                parts_pt_a, parts_pt_b = [np.zeros(n_manifold, dtype=np.int32)], [np.zeros(n_manifold, dtype=np.int32)]

                # Add Asymmetric Noise (A-side clean, B-side noisy)
                if n_asym_a > 0:
                    u_as = self.rng.uniform(0, 1, n_asym_a)
                    a_g, _ = sample_curve_data(u_as, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
                    b_g, _ = sample_curve_data(u_as, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    b_c = b_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_g.shape)
                    parts_a.append(a_g); parts_b.append(b_c); parts_u.append(u_as)
                    parts_pt_a.append(np.full(n_asym_a, 1, dtype=np.int32)); parts_pt_b.append(np.full(n_asym_a, 2, dtype=np.int32))

                # Add Asymmetric Noise (B-side clean, A-side noisy)
                if n_asym_b > 0:
                    u_as = self.rng.uniform(0, 1, n_asym_b)
                    a_g, _ = sample_curve_data(u_as, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
                    b_g, _ = sample_curve_data(u_as, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    a_c = a_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=a_g.shape)
                    parts_a.append(a_c); parts_b.append(b_g); parts_u.append(u_as)
                    parts_pt_a.append(np.full(n_asym_b, 4, dtype=np.int32)); parts_pt_b.append(np.full(n_asym_b, 3, dtype=np.int32))

                # Add External Uniform Noise
                if n_external > 0:
                    # Robust bounds inference: sample full range
                    u_bound = np.linspace(0, 1, 100)
                    t_a, _ = sample_curve_data(u_bound, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
                    t_b, _ = sample_curve_data(u_bound, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)
                    min_a, max_a = t_a.min(axis=0), t_a.max(axis=0)
                    min_b, max_b = t_b.min(axis=0), t_b.max(axis=0)
                    half = self.noise_bbox_expansion / 2.0
                    min_a, max_a = min_a - half * (max_a - min_a), max_a + half * (max_a - min_a)
                    min_b, max_b = min_b - half * (max_b - min_b), max_b + half * (max_b - min_b)
                    parts_a.append(min_a + (max_a - min_a) * self.rng.rand(n_external, 2))
                    parts_b.append(min_b + (max_b - min_b) * self.rng.rand(n_external, 2))
                    parts_u.append(self.rng.uniform(0, 1, n_external))
                    parts_pt_a.append(np.full(n_external, 5, dtype=np.int32)); parts_pt_b.append(np.full(n_external, 5, dtype=np.int32))

                data_a, data_b = np.vstack(parts_a), np.vstack(parts_b)
                param_values = np.concatenate(parts_u)
                self.point_type_a, self.point_type_b = np.concatenate(parts_pt_a), np.concatenate(parts_pt_b)

            # --------------------------------------------------------------
            # DATA TYPE: 3D1F (Audio-Video like)
            # --------------------------------------------------------------
            elif data_type == '3d-av-1f-common':
                r"""
                3D data with 1 shared source u:
                A: Pitch/Resonance/Noise.
                B: Linear signal/Signal-dependent noise.
                Unified normalization applied across segments to prevent disjoint feature spaces.
                """
                def generate_3d1f_raw_features(u_vals, rng, na, nb):
                    # A side
                    pitch = 1.0 / (1.2 - u_vals)
                    resonance = np.sin(u_vals * np.pi)
                    splash = rng.normal(0, na, len(u_vals))
                    raw_a = np.stack([pitch, resonance, splash], axis=1)
                    # B side
                    d1 = u_vals
                    d2 = rng.normal(0, 1, len(u_vals)) * (0.5 + u_vals)
                    d3 = rng.normal(0, nb, len(u_vals))
                    raw_b = np.column_stack([d1, d2, d3])
                    return raw_a, raw_b

                # Collect all latents for global normalization
                u_man = np.linspace(0, 1, n_manifold)
                u_as_a = self.rng.uniform(0, 1, n_asym_a) if n_asym_a > 0 else np.array([])
                u_as_b = self.rng.uniform(0, 1, n_asym_b) if n_asym_b > 0 else np.array([])
                u_all = np.concatenate([u_man, u_as_a, u_as_b])

                # Generate and Normalize globally
                raw_a_all, raw_b_all = generate_3d1f_raw_features(u_all, self.rng, self.manifold_noise_a, self.manifold_noise_b)
                m_a, s_a = raw_a_all.mean(0), raw_a_all.std(0)
                m_b, s_b = raw_b_all.mean(0), raw_b_all.std(0)
                norm_a_all = (raw_a_all - m_a) / (s_a + 1e-8)
                norm_b_all = (raw_b_all - m_b) / (s_b + 1e-8)
                
                # Fixed rotation for modality A
                theta_y, theta_z = np.pi / 4, np.pi / 3
                Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
                Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
                rot_a_all = norm_a_all @ (Ry @ Rz).T
                
                parts_a, parts_b, parts_u = [rot_a_all[:n_manifold]], [norm_b_all[:n_manifold]], [u_man]
                parts_pt_a, parts_pt_b = [np.zeros(n_manifold, dtype=np.int32)], [np.zeros(n_manifold, dtype=np.int32)]

                if n_asym_a > 0:
                    idx_s, idx_e = n_manifold, n_manifold + n_asym_a
                    a_g, b_g = rot_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    b_c = b_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_g.shape)
                    parts_a.append(a_g); parts_b.append(b_c); parts_u.append(u_as_a)
                    parts_pt_a.append(np.ones(n_asym_a, dtype=np.int32)); parts_pt_b.append(np.full(n_asym_a, 2, dtype=np.int32))

                if n_asym_b > 0:
                    idx_s, idx_e = n_manifold+n_asym_a, n_manifold+n_asym_a+n_asym_b
                    a_g, b_g = rot_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    a_c = a_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=a_g.shape)
                    parts_a.append(a_c); parts_b.append(b_g); parts_u.append(u_as_b)
                    parts_pt_a.append(np.full(n_asym_b, 4, dtype=np.int32)); parts_pt_b.append(np.full(n_asym_b, 3, dtype=np.int32))

                if n_external > 0:
                    u_b = np.linspace(0, 1, 100)
                    r_a, r_b = generate_3d1f_raw_features(u_b, self.rng, self.manifold_noise_a, self.manifold_noise_b)
                    n_a, n_b = (r_a - m_a) / (s_a + 1e-8), (r_b - m_b) / (s_b + 1e-8)
                    ro_a = n_a @ (Ry @ Rz).T
                    min_a, max_a = ro_a.min(0), ro_a.max(0); min_b, max_b = n_b.min(0), n_b.max(0)
                    half = self.noise_bbox_expansion / 2.0
                    min_a, max_a = min_a - half*(max_a-min_a), max_a + half*(max_a-min_a)
                    min_b, max_b = min_b - half*(max_b-min_b), max_b + half*(max_b-min_b)
                    parts_a.append(min_a + (max_a-min_a)*self.rng.rand(n_external, 3))
                    parts_b.append(min_b + (max_b-min_b)*self.rng.rand(n_external, 3))
                    parts_u.append(self.rng.uniform(0, 1, n_external))
                    parts_pt_a.append(np.full(n_external, 5, dtype=np.int32)); parts_pt_b.append(np.full(n_external, 5, dtype=np.int32))

                data_a, data_b = np.vstack(parts_a), np.vstack(parts_b)
                param_values = np.concatenate(parts_u)
                self.point_type_a, self.point_type_b = np.concatenate(parts_pt_a), np.concatenate(parts_pt_b)

            # --------------------------------------------------------------
            # DATA TYPE: 3D2F (2 shared factors)
            # --------------------------------------------------------------
            elif data_type == '3d-2f-common':
                r"""
                3D data with 2 shared source factors u1, u2.
                A: 3D Spiral using u1 for rotation/radius and u2 for 3rd dim.
                B: 3D Cubic using u1 for squiggle and u2 for 3rd dim.
                Refined bounds inference to prevent coordinate collapse artifacts.
                """
                def generate_3d2f_raw_features(u1, u2, rng, na, nb):
                    def c_a(u_vals):
                        theta = turns * u_vals * 2 * np.pi
                        r = u_vals
                        return (r * np.cos(theta), r * np.sin(theta))
                    def c_b(u_vals):
                        x = u_vals * 2.0 - 1.0
                        return (x, x**3 - 0.5 * x - 0.5)
                    xy_a, _ = sample_curve_data(u1, c_a, (na, na), rng=rng)
                    xy_b, _ = sample_curve_data(u1, c_b, (nb, nb), rng=rng)
                    # A has stretched u2 (scale 2.0), B has standard u2 (scale 1.0)
                    return np.hstack([xy_a, (2.0 * u2).reshape(-1, 1)]), np.hstack([xy_b, u2.reshape(-1, 1)])

                # 1. Collect latent factors
                u1_m, u2_m = np.linspace(0, 1, n_manifold), self.rng.uniform(0, 1, n_manifold)
                u1_aa, u2_aa = self.rng.uniform(0, 1, n_asym_a), self.rng.uniform(0, 1, n_asym_a)
                u1_ab, u2_ab = self.rng.uniform(0, 1, n_asym_b), self.rng.uniform(0, 1, n_asym_b)
                u1_all = np.concatenate([u1_m, u1_aa, u1_ab]); u2_all = np.concatenate([u2_m, u2_aa, u2_ab])
                
                # 2. Generate and normalize globally
                raw_a_all, raw_b_all = generate_3d2f_raw_features(u1_all, u2_all, self.rng, self.manifold_noise_a, self.manifold_noise_b)
                m_a, s_a = raw_a_all.mean(0), raw_a_all.std(0)
                m_b, s_b = raw_b_all.mean(0), raw_b_all.std(0)
                norm_a_all = (raw_a_all - m_a) / (s_a + 1e-8)
                norm_b_all = (raw_b_all - m_b) / (s_b + 1e-8)
                
                parts_a, parts_b, parts_u = [norm_a_all[:n_manifold]], [norm_b_all[:n_manifold]], [np.column_stack([u1_m, u2_m])]
                parts_pt_a, parts_pt_b = [np.zeros(n_manifold, dtype=np.int32)], [np.zeros(n_manifold, dtype=np.int32)]

                if n_asym_a > 0:
                    idx_s, idx_e = n_manifold, n_manifold+n_asym_a
                    a_g, b_g = norm_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    b_c = b_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_g.shape)
                    parts_a.append(a_g); parts_b.append(b_c); parts_u.append(np.column_stack([u1_aa, u2_aa]))
                    parts_pt_a.append(np.ones(n_asym_a, dtype=np.int32)); parts_pt_b.append(np.full(n_asym_a, 2, dtype=np.int32))

                if n_asym_b > 0:
                    idx_s, idx_e = n_manifold+n_asym_a, n_manifold+n_asym_a+n_asym_b
                    a_g, b_g = norm_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    a_c = a_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=a_g.shape)
                    parts_a.append(a_c); parts_b.append(b_g); parts_u.append(np.column_stack([u1_ab, u2_ab]))
                    parts_pt_a.append(np.full(n_asym_b, 4, dtype=np.int32)); parts_pt_b.append(np.full(n_asym_b, 3, dtype=np.int32))

                if n_external > 0:
                    # Robust bounds inference: sample a large random set to cover the sheet area
                    u1_b, u2_b = self.rng.uniform(0, 1, 1000), self.rng.uniform(0, 1, 1000)
                    r_a, r_b = generate_3d2f_raw_features(u1_b, u2_b, self.rng, self.manifold_noise_a, self.manifold_noise_b)
                    n_a, n_b = (r_a - m_a) / (s_a + 1e-8), (r_b - m_b) / (s_b + 1e-8)
                    min_a, max_a = n_a.min(0), n_a.max(0); min_b, max_b = n_b.min(0), n_b.max(0)
                    half = self.noise_bbox_expansion / 2.0
                    min_a, max_a = min_a - half*(max_a-min_a), max_a + half*(max_a-min_a)
                    min_b, max_b = min_b - half*(max_b-min_b), max_b + half*(max_b-min_b)
                    parts_a.append(min_a + (max_a-min_a)*self.rng.rand(n_external, 3))
                    parts_b.append(min_b + (max_b-min_b)*self.rng.rand(n_external, 3))
                    parts_u.append(np.column_stack([self.rng.uniform(0, 1, n_external), self.rng.uniform(0, 1, n_external)]))
                    parts_pt_a.append(np.full(n_external, 5, dtype=np.int32)); parts_pt_b.append(np.full(n_external, 5, dtype=np.int32))

                data_a, data_b = np.vstack(parts_a), np.vstack(parts_b)
                param_values = np.vstack(parts_u)
                self.point_type_a, self.point_type_b = np.concatenate(parts_pt_a), np.concatenate(parts_pt_b)

            else:
                raise ValueError(f"Unknown data type {data_type}")

            # Apply random 3D rotations for all 3D data types
            if data_type.startswith('3d') and data_a.shape[1] == 3:
                data_a = self._apply_random_rotation(data_a, seed_offset=0)
                data_b = self._apply_random_rotation(data_b, seed_offset=1)
                # Compute universal cubic axis box for visualization
                comb = np.vstack([data_a, data_b])
                mi, ma = comb.min(0), comb.max(0); center = (mi + ma) / 2.0
                hs = float(np.max(ma - mi) / 2.0)
                self.axis_box = np.vstack([center - hs, center + hs])
            else:
                self.axis_box = None

        # Final conversion to tensors
        self.data_a = torch.tensor(data_a, dtype=torch.float32)
        self.data_b = torch.tensor(data_b, dtype=torch.float32)
        self.param_values = torch.tensor(param_values, dtype=torch.float32)
        self.point_type_a = torch.tensor(self.point_type_a, dtype=torch.long)
        self.point_type_b = torch.tensor(self.point_type_b, dtype=torch.long)
        self.corr_target = torch.tensor(np.tile([0.0, 0.9], (self.data_a.shape[0], 1)), dtype=torch.float32)
        self.num_samples = self.data_a.shape[0]

    def to(self, device):
        """Move the entire dataset to the specified device (GPU speedup)."""
        self.data_a = self.data_a.to(device)
        self.data_b = self.data_b.to(device)
        self.param_values = self.param_values.to(device)
        self.point_type_a = self.point_type_a.to(device)
        self.point_type_b = self.point_type_b.to(device)
        self.corr_target = self.corr_target.to(device)
        return self

    def _apply_random_rotation(self, data, seed_offset=0):
        if self.seed is None: return data
        rng = np.random.RandomState(self.seed + seed_offset)
        rot = Rotation.random(random_state=rng)
        return data @ rot.as_matrix().T

    def _load_file(self, path):
        if path.endswith('.npy'): return np.load(path)
        if path.endswith('.pt'): return torch.load(path).cpu().numpy()
        raise ValueError(f"Unsupported format: {path}")

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        return {
            "data_a": self.data_a[idx],
            "data_b": self.data_b[idx],
            "corr_target": self.corr_target[idx],
            "param_values": self.param_values[idx],
            "point_type_a": self.point_type_a[idx],
            "point_type_b": self.point_type_b[idx],
        }
