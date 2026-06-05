import torch
import numpy as np
import torch.nn as nn
import warnings
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from multimodal_experiments.initial_trials.ssl_disentangling import sample_curve_data

class PointType:
    MANIFOLD = 0
    ASYM_A_GOOD = 1
    ASYM_B_CORRUPT = 2
    ASYM_B_GOOD = 3
    ASYM_A_CORRUPT = 4
    EXTERNAL = 5


class FrozenRandomMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, depth: int = 2, generator: torch.Generator = None):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        layers = []
        if depth == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            for _ in range(depth - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
        # Deterministic weight init using local generator
        if generator is not None:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, generator=generator)
                    torch.nn.init.zeros_(m.bias)
            self.apply(init_weights)
            
        # Freeze
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        return self.net(x)


def _safe_float(val, default=0.0) -> float:
    return default if val is None else float(val)


class DualDisentangleDataset(Dataset):
    """Paired modality dataset with a shared latent source.
    
    Supports 2D, 3D, and N-Dimensional data types with various noise regimes.
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
        # Legacy parameter, preserved for backward compatibility
        asymmetric_noise_rate_a: float = 0.0,
        asymmetric_noise_rate_b: float = 0.0,
        asymmetric_noise_magnitude: float = None,
        # New split noise parameters
        asym_corrupt_rate_a: float = 0.0,
        asym_corrupt_rate_b: float = 0.0,
        asym_mismatch_rate_a: float = 0.0,
        asym_mismatch_rate_b: float = 0.0,
        external_noise_ratio: float = 0.0,
        noise_bbox_expansion: float = 0.0,
        seed: int = 42,
        # For ND-KF-MLP mode
        k_shared: int = 2,
        m_unique: int = 2,
        d_out: int = 16,
        shared_factor_dist: str = 'uniform',
        # New 3D3F2C shape parameters
        u3a_scale: float = 0.2,
        u3b_scale: float = 0.3,
        turns: float = 1.0,
        wave_amplitude: float = 1.0,
        # HD embedding: used by 3d-3f-2c-rot and 3d-3f-2c-mlp
        embed_dim: int = None,
        mlp_depth: int = 2,
    ):
        self.num_samples = num_samples
        self.seed = seed
        self.external_noise_ratio = _safe_float(external_noise_ratio, 0.0)
        self.noise_bbox_expansion = _safe_float(noise_bbox_expansion, 0.0)
        self.u3a_scale = _safe_float(u3a_scale, 0.2)
        self.u3b_scale = _safe_float(u3b_scale, 0.3)
        self.turns = _safe_float(turns, 1.0)
        self.wave_amplitude = _safe_float(wave_amplitude, 1.0)
        self.embed_dim = int(embed_dim) if embed_dim is not None else None
        self.mlp_depth = int(mlp_depth)
        self.shared_factor_dist = shared_factor_dist
        self.k_shared = int(k_shared)
        self.m_unique = int(m_unique)
        self.d_out = int(d_out)
        
        # Backward compatibility for old configs using "asymmetric_noise_rate" (defaulting to corrupt behavior)
        self.asym_corrupt_rate_a = max(_safe_float(asym_corrupt_rate_a, 0.0), _safe_float(asymmetric_noise_rate_a, 0.0))
        self.asym_corrupt_rate_b = max(_safe_float(asym_corrupt_rate_b, 0.0), _safe_float(asymmetric_noise_rate_b, 0.0))
        self.asym_mismatch_rate_a = _safe_float(asym_mismatch_rate_a, 0.0)
        self.asym_mismatch_rate_b = _safe_float(asym_mismatch_rate_b, 0.0)

        # Alias resolution for data_type
        data_type = str(data_type).strip().lower()
        if data_type in ('3d1f', '3d-av-1f-common'):
            data_type = '3d-av-1f-common'
        elif data_type in ('3d2f', '3d-2f-common'):
            data_type = '3d-2f-common'
        elif data_type in ('3d-3f-2c-rot', '3d3f2crot'):
            data_type = '3d-3f-2c-rot'
        elif data_type in ('3d-3f-2c-mlp', '3d3f2cmlp'):
            data_type = '3d-3f-2c-mlp'
        elif data_type in ('3d3f2c', '3d-3f-2c'):
            data_type = '3d-3f-2c'
        self.data_type = data_type

        # Use local RandomState for isolated reproducibility (fixes worker-fork duplicates)
        self.rng = np.random.RandomState(seed)
        # Separate torch generator for MLP weights to guarantee consistency
        self.torch_rng = torch.Generator().manual_seed(seed if seed is not None else 42)

        if path_a is not None and path_b is not None:
            # External load mode
            data_a = self._load_file(path_a)
            data_b = self._load_file(path_b)
            self.num_samples = data_a.shape[0]
            param_values = np.linspace(0, 1, self.num_samples)
            self.point_type_a = np.full(self.num_samples, PointType.MANIFOLD, dtype=np.int32)
            self.point_type_b = np.full(self.num_samples, PointType.MANIFOLD, dtype=np.int32)
        else:
            # Synthetic generation mode
            self.manifold_noise_a = 0.02 if manifold_noise_a is None else manifold_noise_a
            self.manifold_noise_b = 0.02 if manifold_noise_b is None else manifold_noise_b
            self.asymmetric_noise_magnitude = asymmetric_noise_magnitude

            # Calculate sample counts for each segment (completely disjoint partitions of N, no index overlap)
            N = num_samples
            n_ext = int(np.floor(N * self.external_noise_ratio))
            n_ac_a = int(np.floor(N * self.asym_corrupt_rate_a))
            n_ac_b = int(np.floor(N * self.asym_corrupt_rate_b))
            n_am_a = int(np.floor(N * self.asym_mismatch_rate_a))
            n_am_b = int(np.floor(N * self.asym_mismatch_rate_b))
            n_man = N - (n_ext + n_ac_a + n_ac_b + n_am_a + n_am_b)
            
            if n_man < 1:
                raise ValueError(f"num_samples too small for configured noise rates. Manifold={n_man}")

            if data_type == '2d':
                data_a, data_b, param_values, self.point_type_a, self.point_type_b = self._init_2d(
                    n_man, n_ext, n_ac_a, n_ac_b, n_am_a, n_am_b
                )
            elif data_type in ('3d-3f-2c', '3d-3f-2c-rot', '3d-3f-2c-mlp'):
                data_a, data_b, param_values, self.point_type_a, self.point_type_b = self._init_3d3f2c(
                    n_man, n_ext, n_ac_a, n_ac_b, n_am_a, n_am_b
                )
            elif data_type == 'nd-kf-mlp':
                data_a, data_b, param_values, self.point_type_a, self.point_type_b = self._init_nd_kf_mlp(
                    n_man, n_ext, n_ac_a, n_ac_b, n_am_a, n_am_b
                )
            elif data_type == '3d-av-1f-common':
                data_a, data_b, param_values, self.point_type_a, self.point_type_b = self._init_legacy_3d1f(
                    n_man, n_ext, n_ac_a, n_ac_b, n_am_a, n_am_b
                )
            elif data_type == '3d-2f-common':
                data_a, data_b, param_values, self.point_type_a, self.point_type_b = self._init_legacy_3d2f(
                    n_man, n_ext, n_ac_a, n_ac_b, n_am_a, n_am_b
                )
            else:
                raise ValueError(f"Unknown data type {data_type}")

            # ---------------------------------------------------------------
            # HD EMBEDDING: 3d-3f-2c-rot and 3d-3f-2c-mlp
            # After generating the 3D base (data_a, data_b with shape [N,3]),
            # pad with (embed_dim-3) Gaussian noise dims then apply transform.
            # Normalization is redone after the transform.
            # ---------------------------------------------------------------
            if data_type in ('3d-3f-2c-rot', '3d-3f-2c-mlp'):
                D = self.embed_dim
                if D is None:
                    raise ValueError(f"data_type='{data_type}' requires embed_dim to be set")
                if D < 3:
                    raise ValueError(f"embed_dim={D} must be >= 3 for {data_type}")

                # Pad both modalities with (D-3) N(0,1) noise dimensions
                def _pad(arr, rng, extra):
                    if extra <= 0:
                        return arr
                    return np.hstack([arr, rng.normal(0, 1, (arr.shape[0], extra))])

                extra = D - 3
                data_a = _pad(data_a, self.rng, extra)
                data_b = _pad(data_b, self.rng, extra)

                if data_type == '3d-3f-2c-rot':
                    # Random Haar-distributed orthogonal matrix in R^{D×D}, one per modality.
                    from scipy.stats import ortho_group
                    rng_rot_a = np.random.RandomState(self.seed if self.seed is not None else 0)
                    rng_rot_b = np.random.RandomState((self.seed if self.seed is not None else 0) + 1)
                    Q_a = ortho_group.rvs(D, random_state=rng_rot_a)
                    Q_b = ortho_group.rvs(D, random_state=rng_rot_b)
                    data_a = data_a @ Q_a.T
                    data_b = data_b @ Q_b.T

                elif data_type == '3d-3f-2c-mlp':
                    mlp_a = FrozenRandomMLP(D, D, depth=self.mlp_depth, generator=self.torch_rng).eval()
                    mlp_b = FrozenRandomMLP(D, D, depth=self.mlp_depth, generator=self.torch_rng).eval()

                    with torch.no_grad():
                        data_a = mlp_a(torch.tensor(data_a, dtype=torch.float32)).numpy()
                        data_b = mlp_b(torch.tensor(data_b, dtype=torch.float32)).numpy()

                # Re-normalize after embedding (zero mean, unit std per dim)
                m_a, s_a = data_a.mean(0), data_a.std(0)
                m_b, s_b = data_b.mean(0), data_b.std(0)
                data_a = (data_a - m_a) / (s_a + 1e-8)
                data_b = (data_b - m_b) / (s_b + 1e-8)
                self.axis_box = None

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
        self.data_a = torch.as_tensor(data_a, dtype=torch.float32)
        self.data_b = torch.as_tensor(data_b, dtype=torch.float32)
        self.param_values = torch.as_tensor(param_values, dtype=torch.float32)
        self.point_type_a = torch.as_tensor(self.point_type_a, dtype=torch.int8)
        self.point_type_b = torch.as_tensor(self.point_type_b, dtype=torch.int8)
        
        is_clean = (self.point_type_a == PointType.MANIFOLD) & (self.point_type_b == PointType.MANIFOLD)
        c_targ = torch.zeros((self.num_samples, 2), dtype=torch.float32)
        c_targ[is_clean, 1] = 0.9
        self.corr_target = c_targ
        
        self.num_samples = self.data_a.shape[0]

    def _init_2d(self, n_man, n_ext, n_ac_a, n_ac_b, n_am_a, n_am_b):
        total_n = n_man + n_ext + n_ac_a + n_ac_b + n_am_a + n_am_b
        data_a = np.empty((total_n, 2), dtype=np.float32)
        data_b = np.empty((total_n, 2), dtype=np.float32)
        param_values = np.empty(total_n, dtype=np.float32)
        point_type_a = np.empty(total_n, dtype=np.int32)
        point_type_b = np.empty(total_n, dtype=np.int32)
        
        idx = 0
        def add(a, b, u, ta, tb):
            nonlocal idx
            n = len(a)
            if n == 0: return
            data_a[idx:idx+n] = a
            data_b[idx:idx+n] = b
            param_values[idx:idx+n] = u
            point_type_a[idx:idx+n] = ta
            point_type_b[idx:idx+n] = tb
            idx += n

        u_man = np.linspace(0, 1, n_man) if n_man > 0 else np.array([])
        if n_man > 0:
            da_m, _ = sample_curve_data(u_man, self._curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
            db_m, _ = sample_curve_data(u_man, self._curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)
            add(da_m, db_m, u_man, np.zeros(n_man), np.zeros(n_man))

        n_asym_a_total = n_ac_a + n_am_a
        if n_asym_a_total > 0:
            u_as = self.rng.uniform(0, 1, n_asym_a_total)
            ag, _ = sample_curve_data(u_as, self._curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
            bg, _ = sample_curve_data(u_as, self._curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)
            mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
            bc = bg + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=bg.shape)
            add(ag, bc, u_as, np.full(n_asym_a_total, 1), np.full(n_asym_a_total, 2))

        n_asym_b_total = n_ac_b + n_am_b
        if n_asym_b_total > 0:
            u_as = self.rng.uniform(0, 1, n_asym_b_total)
            ag, _ = sample_curve_data(u_as, self._curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
            bg, _ = sample_curve_data(u_as, self._curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)
            mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
            ac = ag + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=ag.shape)
            add(ac, bg, u_as, np.full(n_asym_b_total, 4), np.full(n_asym_b_total, 3))

        if n_ext > 0:
            u_bound = np.linspace(0, 1, 100)
            t_a, _ = sample_curve_data(u_bound, self._curve_a_fn, (0,0), rng=self.rng)
            t_b, _ = sample_curve_data(u_bound, self._curve_b_fn, (0,0), rng=self.rng)
            mina, maxa = t_a.min(0), t_a.max(0)
            minb, maxb = t_b.min(0), t_b.max(0)
            h = self.noise_bbox_expansion / 2.0
            mina, maxa = mina - h*(maxa-mina), maxa + h*(maxa-mina)
            minb, maxb = minb - h*(maxb-minb), maxb + h*(maxb-minb)
            a = mina + (maxa - mina) * self.rng.rand(n_ext, 2)
            b = minb + (maxb - minb) * self.rng.rand(n_ext, 2)
            u = self.rng.uniform(0, 1, n_ext)
            add(a, b, u, np.full(n_ext, 5), np.full(n_ext, 5))

        return data_a, data_b, param_values, point_type_a, point_type_b

    def _init_3d3f2c(self, n_man, n_ext, n_ac_a, n_ac_b, n_am_a, n_am_b):
        total_n = n_man + n_ext + n_ac_a + n_ac_b + n_am_a + n_am_b
        data_a = np.empty((total_n, 3), dtype=np.float32)
        data_b = np.empty((total_n, 3), dtype=np.float32)
        param_values = np.empty((total_n, 2), dtype=np.float32)
        point_type_a = np.empty(total_n, dtype=np.int32)
        point_type_b = np.empty(total_n, dtype=np.int32)
        
        idx = 0
        def add(a, b, u, ta, tb):
            nonlocal idx
            n = len(a)
            if n == 0: return
            data_a[idx:idx+n] = a
            data_b[idx:idx+n] = b
            param_values[idx:idx+n] = u
            point_type_a[idx:idx+n] = ta
            point_type_b[idx:idx+n] = tb
            idx += n

        def gen_clean(n):
            u1 = self.rng.uniform(0, 1, n)
            u2 = self.rng.uniform(0, 1, n)
            u_noise_a = self.rng.normal(0, 1, n)
            u_noise_b = self.rng.normal(0, 1, n)
            a, b = self._gen_3d3f2c(u1, u2, u_noise_a, u_noise_b)
            return a, b

        if n_man > 0:
            u1 = np.linspace(0, 1, n_man)
            u2 = self.rng.uniform(0, 1, n_man)
            u_noise_a = self.rng.normal(0, 1, n_man)
            u_noise_b = self.rng.normal(0, 1, n_man)
            a, b = self._gen_3d3f2c(u1, u2, u_noise_a, u_noise_b)
            add(self._apply_manifold_noise(a, self.manifold_noise_a),
                self._apply_manifold_noise(b, self.manifold_noise_b),
                np.column_stack([u1, u2]), np.zeros(n_man), np.zeros(n_man))

        min_a, max_a, min_b, max_b = self._get_bbox()

        if n_ac_a > 0:
            a, b = self._generate_asym_corrupt(n_ac_a, True, gen_clean, min_a, max_a, 3)
            u1 = self.rng.uniform(0, 1, n_ac_a)
            u2 = self.rng.uniform(0, 1, n_ac_a)
            add(a, self._apply_manifold_noise(b, self.manifold_noise_b),
                np.column_stack([u1, u2]), np.full(n_ac_a, PointType.ASYM_B_CORRUPT), np.full(n_ac_a, PointType.ASYM_A_GOOD))

        if n_ac_b > 0:
            a, b = self._generate_asym_corrupt(n_ac_b, False, gen_clean, min_b, max_b, 3)
            u1 = self.rng.uniform(0, 1, n_ac_b)
            u2 = self.rng.uniform(0, 1, n_ac_b)
            add(self._apply_manifold_noise(a, self.manifold_noise_a), b,
                np.column_stack([u1, u2]), np.full(n_ac_b, PointType.ASYM_A_GOOD), np.full(n_ac_b, PointType.ASYM_B_CORRUPT))

        if n_am_a > 0:
            a, b = self._generate_asym_mismatch(n_am_a, True, gen_clean)
            u1 = self.rng.uniform(0, 1, n_am_a)
            u2 = self.rng.uniform(0, 1, n_am_a)
            add(self._apply_manifold_noise(a, self.manifold_noise_a),
                self._apply_manifold_noise(b, self.manifold_noise_b),
                np.column_stack([u1, u2]), np.full(n_am_a, PointType.ASYM_A_CORRUPT), np.full(n_am_a, PointType.ASYM_B_GOOD))

        if n_am_b > 0:
            a, b = self._generate_asym_mismatch(n_am_b, False, gen_clean)
            u1 = self.rng.uniform(0, 1, n_am_b)
            u2 = self.rng.uniform(0, 1, n_am_b)
            add(self._apply_manifold_noise(a, self.manifold_noise_a),
                self._apply_manifold_noise(b, self.manifold_noise_b),
                np.column_stack([u1, u2]), np.full(n_am_b, PointType.ASYM_B_GOOD), np.full(n_am_b, PointType.ASYM_A_CORRUPT))

        if n_ext > 0:
            a, b = self._generate_external(n_ext, min_a, max_a, min_b, max_b, 3, 3)
            u1 = self.rng.uniform(0, 1, n_ext)
            u2 = self.rng.uniform(0, 1, n_ext)
            add(a, b, np.column_stack([u1, u2]), np.full(n_ext, PointType.EXTERNAL), np.full(n_ext, PointType.EXTERNAL))

        # In-place normalization
        m_a, s_a = data_a.mean(0), data_a.std(0)
        m_b, s_b = data_b.mean(0), data_b.std(0)
        data_a -= m_a
        data_a /= (s_a + 1e-8)
        data_b -= m_b
        data_b /= (s_b + 1e-8)
        
        return data_a, data_b, param_values, point_type_a, point_type_b

    def _init_nd_kf_mlp(self, n_man, n_ext, n_ac_a, n_ac_b, n_am_a, n_am_b):
        in_dim = self.k_shared + self.m_unique
        mlp_a = FrozenRandomMLP(in_dim, self.d_out, depth=self.mlp_depth, generator=self.torch_rng).eval()
        mlp_b = FrozenRandomMLP(in_dim, self.d_out, depth=self.mlp_depth, generator=self.torch_rng).eval()

        total_n = n_man + n_ext + n_ac_a + n_ac_b + n_am_a + n_am_b
        data_a = np.empty((total_n, self.d_out), dtype=np.float32)
        data_b = np.empty((total_n, self.d_out), dtype=np.float32)
        param_values = np.empty((total_n, self.k_shared), dtype=np.float32)
        point_type_a = np.empty(total_n, dtype=np.int32)
        point_type_b = np.empty(total_n, dtype=np.int32)
        
        idx = 0
        def add(xa, xb, pu, ta, tb):
            nonlocal idx
            n = len(xa)
            if n == 0: return
            data_a[idx:idx+n] = xa
            data_b[idx:idx+n] = xb
            param_values[idx:idx+n] = pu
            point_type_a[idx:idx+n] = ta
            point_type_b[idx:idx+n] = tb
            idx += n

        # Estimate mean and standard deviation of MLP outputs to normalize them
        probe_us, probe_ua, probe_ub = self._sample_latents(10000, self.k_shared, self.m_unique)
        probe_xa, probe_xb = self._gen_mlp(probe_us, probe_ua, probe_ub, mlp_a, mlp_b)
        mean_a, std_a = probe_xa.mean(0), probe_xa.std(0)
        mean_b, std_b = probe_xb.mean(0), probe_xb.std(0)

        def norm_a(x): return (x - mean_a) / (std_a + 1e-8)
        def norm_b(x): return (x - mean_b) / (std_b + 1e-8)

        def gen_clean(n):
            us, ua, ub = self._sample_latents(n, self.k_shared, self.m_unique)
            xa, xb = self._gen_mlp(us, ua, ub, mlp_a, mlp_b)
            return norm_a(xa), norm_b(xb)

        if n_man > 0:
            chunk_size = 20000
            for i in range(0, n_man, chunk_size):
                end = min(i + chunk_size, n_man)
                n_chunk = end - i
                us, ua, ub = self._sample_latents(n_chunk, self.k_shared, self.m_unique)
                xa, xb = self._gen_mlp(us, ua, ub, mlp_a, mlp_b)
                add(norm_a(xa), norm_b(xb), us, np.zeros(n_chunk), np.zeros(n_chunk))

        _us, _ua, _ub = self._sample_latents(1000, self.k_shared, self.m_unique)
        _xa, _xb = self._gen_mlp(_us, _ua, _ub, mlp_a, mlp_b)
        _xa, _xb = norm_a(_xa), norm_b(_xb)
        min_a, max_a = _xa.min(0), _xa.max(0)
        min_b, max_b = _xb.min(0), _xb.max(0)
        h = self.noise_bbox_expansion / 2.0
        min_a, max_a = min_a - h*(max_a-min_a), max_a + h*(max_a-min_a)
        min_b, max_b = min_b - h*(max_b-min_b), max_b + h*(max_b-min_b)

        if n_ac_a > 0:
            xa, xb = self._generate_asym_corrupt(n_ac_a, True, gen_clean, min_a, max_a, self.d_out)
            us, _, _ = self._sample_latents(n_ac_a, self.k_shared, self.m_unique)
            add(xa, xb, us, np.full(n_ac_a, PointType.ASYM_B_CORRUPT), np.full(n_ac_a, PointType.ASYM_A_GOOD))

        if n_ac_b > 0:
            xa, xb = self._generate_asym_corrupt(n_ac_b, False, gen_clean, min_b, max_b, self.d_out)
            us, _, _ = self._sample_latents(n_ac_b, self.k_shared, self.m_unique)
            add(xa, xb, us, np.full(n_ac_b, PointType.ASYM_A_GOOD), np.full(n_ac_b, PointType.ASYM_B_CORRUPT))
            
        if n_am_a > 0:
            xa, xb = self._generate_asym_mismatch(n_am_a, True, gen_clean)
            us, _, _ = self._sample_latents(n_am_a, self.k_shared, self.m_unique)
            add(xa, xb, us, np.full(n_am_a, PointType.ASYM_A_CORRUPT), np.full(n_am_a, PointType.ASYM_B_GOOD))

        if n_am_b > 0:
            xa, xb = self._generate_asym_mismatch(n_am_b, False, gen_clean)
            us, _, _ = self._sample_latents(n_am_b, self.k_shared, self.m_unique)
            add(xa, xb, us, np.full(n_am_b, PointType.ASYM_B_GOOD), np.full(n_am_b, PointType.ASYM_A_CORRUPT))
            
        if n_ext > 0:
            xa, xb = self._generate_external(n_ext, min_a, max_a, min_b, max_b, self.d_out, self.d_out)
            add(xa, xb, self.rng.uniform(0, 1, (n_ext, self.k_shared)), np.full(n_ext, PointType.EXTERNAL), np.full(n_ext, PointType.EXTERNAL))

        # In-place normalization
        m_a, s_a = data_a.mean(0), data_a.std(0)
        m_b, s_b = data_b.mean(0), data_b.std(0)
        data_a -= m_a
        data_a /= (s_a + 1e-8)
        data_b -= m_b
        data_b /= (s_b + 1e-8)

        # Chunked noise addition via torch.randn to save memory
        def add_noise(arr, noise_level):
            if noise_level > 0:
                chunk_size = 20000
                for i in range(0, len(arr), chunk_size):
                    end = min(i + chunk_size, len(arr))
                    noise = torch.randn((end - i, arr.shape[1]), generator=self.torch_rng, dtype=torch.float32).numpy()
                    noise *= noise_level
                    arr[i:end] += noise

        add_noise(data_a, self.manifold_noise_a)
        add_noise(data_b, self.manifold_noise_b)
        
        return data_a, data_b, param_values, point_type_a, point_type_b


    def _init_legacy_3d1f(self, n_man, n_ext, n_ac_a, n_ac_b, n_am_a, n_am_b):
        import warnings
        warnings.warn(
            "data_type '3d-av-1f-common' is deprecated and will be removed in future versions.",
            DeprecationWarning,
            stacklevel=2,
        )
        total_n = n_man + n_ext + n_ac_a + n_ac_b + n_am_a + n_am_b
        data_a = np.empty((total_n, 3), dtype=np.float32)
        data_b = np.empty((total_n, 3), dtype=np.float32)
        param_values = np.empty(total_n, dtype=np.float32)
        point_type_a = np.empty(total_n, dtype=np.int32)
        point_type_b = np.empty(total_n, dtype=np.int32)

        idx = 0
        def add(a, b, u, ta, tb):
            nonlocal idx
            n = len(a)
            if n == 0: return
            data_a[idx:idx+n] = a
            data_b[idx:idx+n] = b
            param_values[idx:idx+n] = u
            point_type_a[idx:idx+n] = ta
            point_type_b[idx:idx+n] = tb
            idx += n

        u_man = np.linspace(0, 1, n_man)
        n_asym_a_total = n_ac_a + n_am_a
        n_asym_b_total = n_ac_b + n_am_b
        
        u_as_a = self.rng.uniform(0, 1, n_asym_a_total) if n_asym_a_total > 0 else np.array([])
        u_as_b = self.rng.uniform(0, 1, n_asym_b_total) if n_asym_b_total > 0 else np.array([])
        u_all = np.concatenate([u_man, u_as_a, u_as_b])

        raw_a_all, raw_b_all = self._generate_3d1f_raw_features(u_all, self.manifold_noise_a, self.manifold_noise_b)
        m_a, s_a = raw_a_all.mean(0), raw_a_all.std(0)
        m_b, s_b = raw_b_all.mean(0), raw_b_all.std(0)
        norm_a_all = (raw_a_all - m_a) / (s_a + 1e-8)
        norm_b_all = (raw_b_all - m_b) / (s_b + 1e-8)
        
        theta_y, theta_z = np.pi / 4, np.pi / 3
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
        rot_a_all = norm_a_all @ (Ry @ Rz).T
        
        if n_man > 0:
            add(rot_a_all[:n_man], norm_b_all[:n_man], u_man, np.zeros(n_man), np.zeros(n_man))

        if n_asym_a_total > 0:
            idx_s, idx_e = n_man, n_man + n_asym_a_total
            a_g, b_g = rot_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
            mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
            b_c = b_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_g.shape)
            add(a_g, b_c, u_as_a, np.ones(n_asym_a_total), np.full(n_asym_a_total, 2))

        if n_asym_b_total > 0:
            idx_s, idx_e = n_man+n_asym_a_total, n_man+n_asym_a_total+n_asym_b_total
            a_g, b_g = rot_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
            mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
            a_c = a_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=a_g.shape)
            add(a_c, b_g, u_as_b, np.full(n_asym_b_total, 4), np.full(n_asym_b_total, 3))

        if n_ext > 0:
            u_b = np.linspace(0, 1, 100)
            r_a, r_b = self._generate_3d1f_raw_features(u_b, self.manifold_noise_a, self.manifold_noise_b)
            n_a, n_b = (r_a - m_a) / (s_a + 1e-8), (r_b - m_b) / (s_b + 1e-8)
            ro_a = n_a @ (Ry @ Rz).T
            min_a, max_a = ro_a.min(0), ro_a.max(0); min_b, max_b = n_b.min(0), n_b.max(0)
            h = self.noise_bbox_expansion / 2.0
            min_a, max_a = min_a - h*(max_a-min_a), max_a + h*(max_a-min_a)
            min_b, max_b = min_b - h*(max_b-min_b), max_b + h*(max_b-min_b)
            a = min_a + (max_a-min_a)*self.rng.rand(n_ext, 3)
            b = min_b + (max_b-min_b)*self.rng.rand(n_ext, 3)
            u = self.rng.uniform(0, 1, n_ext)
            add(a, b, u, np.full(n_ext, 5), np.full(n_ext, 5))

        return data_a, data_b, param_values, point_type_a, point_type_b

    def _init_legacy_3d2f(self, n_man, n_ext, n_ac_a, n_ac_b, n_am_a, n_am_b):
        import warnings
        warnings.warn(
            "data_type '3d-2f-common' is deprecated and will be removed in future versions.",
            DeprecationWarning,
            stacklevel=2,
        )
        total_n = n_man + n_ext + n_ac_a + n_ac_b + n_am_a + n_am_b
        data_a = np.empty((total_n, 3), dtype=np.float32)
        data_b = np.empty((total_n, 3), dtype=np.float32)
        param_values = np.empty((total_n, 2), dtype=np.float32)
        point_type_a = np.empty(total_n, dtype=np.int32)
        point_type_b = np.empty(total_n, dtype=np.int32)

        idx = 0
        def add(a, b, u, ta, tb):
            nonlocal idx
            n = len(a)
            if n == 0: return
            data_a[idx:idx+n] = a
            data_b[idx:idx+n] = b
            param_values[idx:idx+n] = u
            point_type_a[idx:idx+n] = ta
            point_type_b[idx:idx+n] = tb
            idx += n

        n_asym_a_total = n_ac_a + n_am_a
        n_asym_b_total = n_ac_b + n_am_b

        u1_m, u2_m = np.linspace(0, 1, n_man), self.rng.uniform(0, 1, n_man)
        u1_aa, u2_aa = self.rng.uniform(0, 1, n_asym_a_total), self.rng.uniform(0, 1, n_asym_a_total)
        u1_ab, u2_ab = self.rng.uniform(0, 1, n_asym_b_total), self.rng.uniform(0, 1, n_asym_b_total)
        u1_all = np.concatenate([u1_m, u1_aa, u1_ab]) if total_n > n_ext else np.array([])
        u2_all = np.concatenate([u2_m, u2_aa, u2_ab]) if total_n > n_ext else np.array([])
        
        if len(u1_all) > 0:
            raw_a_all, raw_b_all = self._generate_3d2f_raw_features(u1_all, u2_all, self.manifold_noise_a, self.manifold_noise_b)
            m_a, s_a = raw_a_all.mean(0), raw_a_all.std(0)
            m_b, s_b = raw_b_all.mean(0), raw_b_all.std(0)
            norm_a_all = (raw_a_all - m_a) / (s_a + 1e-8)
            norm_b_all = (raw_b_all - m_b) / (s_b + 1e-8)
        
        if n_man > 0:
            add(norm_a_all[:n_man], norm_b_all[:n_man], np.column_stack([u1_m, u2_m]), np.zeros(n_man), np.zeros(n_man))

        if n_asym_a_total > 0:
            idx_s, idx_e = n_man, n_man+n_asym_a_total
            a_g, b_g = norm_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
            mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
            b_c = b_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_g.shape)
            add(a_g, b_c, np.column_stack([u1_aa, u2_aa]), np.ones(n_asym_a_total), np.full(n_asym_a_total, 2))

        if n_asym_b_total > 0:
            idx_s, idx_e = n_man+n_asym_a_total, n_man+n_asym_a_total+n_asym_b_total
            a_g, b_g = norm_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
            mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
            a_c = a_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=a_g.shape)
            add(a_c, b_g, np.column_stack([u1_ab, u2_ab]), np.full(n_asym_b_total, 4), np.full(n_asym_b_total, 3))

        if n_ext > 0:
            u1_b, u2_b = self.rng.uniform(0, 1, 1000), self.rng.uniform(0, 1, 1000)
            r_a, r_b = self._generate_3d2f_raw_features(u1_b, u2_b, self.manifold_noise_a, self.manifold_noise_b)
            if 'm_a' not in locals():
                m_a, s_a = r_a.mean(0), r_a.std(0)
                m_b, s_b = r_b.mean(0), r_b.std(0)
            n_a, n_b = (r_a - m_a) / (s_a + 1e-8), (r_b - m_b) / (s_b + 1e-8)
            min_a, max_a = n_a.min(0), n_a.max(0); min_b, max_b = n_b.min(0), n_b.max(0)
            h = self.noise_bbox_expansion / 2.0
            min_a, max_a = min_a - h*(max_a-min_a), max_a + h*(max_a-min_a)
            min_b, max_b = min_b - h*(max_b-min_b), max_b + h*(max_b-min_b)
            a = min_a + (max_a-min_a)*self.rng.rand(n_ext, 3)
            b = min_b + (max_b-min_b)*self.rng.rand(n_ext, 3)
            u = np.column_stack([self.rng.uniform(0, 1, n_ext), self.rng.uniform(0, 1, n_ext)])
            add(a, b, u, np.full(n_ext, 5), np.full(n_ext, 5))

        return data_a, data_b, param_values, point_type_a, point_type_b
    # Refactored private helpers for dataset generation
    # ------------------------------------------------------------------

    def _curve_a_fn(self, u_vals):
        return ((0.8 * u_vals + 0.2) * np.sin(self.turns * u_vals * 2 * np.pi),
                (0.8 * u_vals + 0.2) * np.cos(self.turns * u_vals * 2 * np.pi))

    def _curve_b_fn(self, u_vals):
        x = u_vals * 2.0 - 1.0
        return (x, x**3 - 0.5 * x - 0.5)

    def _generate_asym_corrupt(self, n, is_a_corrupt, gen_clean_fn, min_box, max_box, dim):
        """Generates asymmetric corrupted points. One modality is clean, other is random in bbox."""
        if n <= 0:
            return None, None
        a_clean, b_clean = gen_clean_fn(n)
        garbage = np.empty((n, dim), dtype=np.float32)
        chunk_size = 20000
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            n_chunk = end - i
            rand_vals = torch.rand((n_chunk, dim), generator=self.torch_rng, dtype=torch.float32).numpy()
            garbage[i:end] = min_box + (max_box - min_box) * rand_vals
        if is_a_corrupt:
            return garbage, b_clean
        else:
            return a_clean, garbage

    def _generate_asym_mismatch(self, n, is_a_fake, gen_clean_fn):
        """Generates asymmetric mismatch points."""
        if n <= 0:
            return None, None
        a1, b1 = gen_clean_fn(n)
        a2, b2 = gen_clean_fn(n)
        if is_a_fake:
            return a2, b1
        else:
            return a1, b2

    def _generate_external(self, n, min_box_a, max_box_a, min_box_b, max_box_b, dim_a, dim_b):
        """Generates external noise points (both modalities are random in their bboxes) in chunks to save memory."""
        if n <= 0:
            return None, None
        a = np.empty((n, dim_a), dtype=np.float32)
        b = np.empty((n, dim_b), dtype=np.float32)
        chunk_size = 20000
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            n_chunk = end - i
            a_rand = torch.rand((n_chunk, dim_a), generator=self.torch_rng, dtype=torch.float32).numpy()
            b_rand = torch.rand((n_chunk, dim_b), generator=self.torch_rng, dtype=torch.float32).numpy()
            a[i:end] = min_box_a + (max_box_a - min_box_a) * a_rand
            b[i:end] = min_box_b + (max_box_b - min_box_b) * b_rand
        return a, b

    def _gen_3d3f2c(self, u1, u2, u_noise_a, u_noise_b):
        # Base clean shapes without unstructured noise
        # Mod A: Volumetric Spiral
        r = u1 + self.u3a_scale * u_noise_a
        theta = 2 * np.pi * self.turns * u1
        x_a = r * np.cos(theta)
        y_a = r * np.sin(theta)
        z_a = u2
        a_clean = np.column_stack([x_a, y_a, z_a])
        
        # Mod B: Volumetric Wave
        x_b = u1
        y_b = u2
        v = 2 * u1 - 1
        z_b = self.wave_amplitude * (v**3 - 0.5 * v) + self.u3b_scale * u_noise_b
        b_clean = np.column_stack([x_b, y_b, z_b])
        return a_clean, b_clean

    def _apply_manifold_noise(self, arr, noise_level):
        return arr + self.rng.normal(0, noise_level, size=arr.shape)

    def _get_bbox(self, u_steps=1000):
        u1 = self.rng.uniform(0, 1, u_steps)
        u2 = self.rng.uniform(0, 1, u_steps)
        u_noise_a = self.rng.normal(0, 1, u_steps)
        u_noise_b = self.rng.normal(0, 1, u_steps)
        a, b = self._gen_3d3f2c(u1, u2, u_noise_a, u_noise_b)
        min_a, max_a = a.min(0), a.max(0)
        min_b, max_b = b.min(0), b.max(0)
        h = self.noise_bbox_expansion / 2.0
        min_a, max_a = min_a - h*(max_a-min_a), max_a + h*(max_a-min_a)
        min_b, max_b = min_b - h*(max_b-min_b), max_b + h*(max_b-min_b)
        return min_a, max_a, min_b, max_b

    def _sample_latents(self, n, k_shared, m_unique):
        if self.shared_factor_dist == 'normal':
            u_s = self.rng.normal(0, 1, (n, k_shared))
        else:
            u_s = self.rng.uniform(0, 1, (n, k_shared))
        u_ua = self.rng.normal(0, 1, (n, m_unique))
        u_ub = self.rng.normal(0, 1, (n, m_unique))
        return u_s, u_ua, u_ub

    def _gen_mlp(self, u_s, u_ua, u_ub, mlp_a, mlp_b, chunk_size=20000):
        za = torch.tensor(np.hstack([u_s, u_ua]), dtype=torch.float32)
        zb = torch.tensor(np.hstack([u_s, u_ub]), dtype=torch.float32)
        
        xa_list, xb_list = [], []
        with torch.no_grad():
            for i in range(0, len(za), chunk_size):
                xa_list.append(mlp_a(za[i:i+chunk_size]).cpu().numpy())
                xb_list.append(mlp_b(zb[i:i+chunk_size]).cpu().numpy())
                
        return np.vstack(xa_list), np.vstack(xb_list)

    def _generate_3d1f_raw_features(self, u_vals, na, nb):
        pitch = 1.0 / (1.2 - u_vals)
        resonance = np.sin(u_vals * np.pi)
        splash = self.rng.normal(0, na, len(u_vals))
        raw_a = np.stack([pitch, resonance, splash], axis=1)
        d1 = u_vals
        d2 = self.rng.normal(0, 1, len(u_vals)) * (0.5 + u_vals)
        d3 = self.rng.normal(0, nb, len(u_vals))
        raw_b = np.column_stack([d1, d2, d3])
        return raw_a, raw_b

    def _generate_3d2f_raw_features(self, u1, u2, na, nb):
        def c_a(u_vals):
            theta = self.turns * u_vals * 2 * np.pi
            r = u_vals
            return (r * np.cos(theta), r * np.sin(theta))
        def c_b(u_vals):
            x = u_vals * 2.0 - 1.0
            return (x, x**3 - 0.5 * x - 0.5)
        xy_a, _ = sample_curve_data(u1, c_a, (na, na), rng=self.rng)
        xy_b, _ = sample_curve_data(u1, c_b, (nb, nb), rng=self.rng)
        return np.hstack([xy_a, (2.0 * u2).reshape(-1, 1)]), np.hstack([xy_b, u2.reshape(-1, 1)])

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

    def set_training_mode(self, enabled: bool = True):
        """If True, __getitem__ returns only fields used by the training loop.

        Eval/vis code accesses dataset tensors directly (dataset.point_type_a, etc.)
        and is unaffected by this flag.
        """
        self._training_mode = enabled

    def __getitem__(self, idx):
        d = {
            "data_a": self.data_a[idx],
            "data_b": self.data_b[idx],
            "corr_target": self.corr_target[idx],
        }
        if not getattr(self, '_training_mode', False):
            d["param_values"] = self.param_values[idx]
            d["point_type_a"] = self.point_type_a[idx]
            d["point_type_b"] = self.point_type_b[idx]
        return d


def build_dataset_from_config(cfg, seed=None) -> DualDisentangleDataset:
    """Builds a DualDisentangleDataset instance from an OmegaConf configuration object."""
    data_cfg = cfg.data
    return DualDisentangleDataset(
        data_type=data_cfg.get('type', '2d'),
        num_samples=data_cfg.get('num_samples', 4096),
        path_a=data_cfg.get('path_a', None),
        path_b=data_cfg.get('path_b', None),
        manifold_noise_a=data_cfg.get('manifold_noise_a', None),
        manifold_noise_b=data_cfg.get('manifold_noise_b', None),
        asymmetric_noise_magnitude=data_cfg.get('asymmetric_noise_magnitude', None),
        asymmetric_noise_rate_a=data_cfg.get('asymmetric_noise_rate_a', 0.0),
        asymmetric_noise_rate_b=data_cfg.get('asymmetric_noise_rate_b', 0.0),
        asym_corrupt_rate_a=data_cfg.get('asym_corrupt_rate_a', 0.0),
        asym_corrupt_rate_b=data_cfg.get('asym_corrupt_rate_b', 0.0),
        asym_mismatch_rate_a=data_cfg.get('asym_mismatch_rate_a', 0.0),
        asym_mismatch_rate_b=data_cfg.get('asym_mismatch_rate_b', 0.0),
        external_noise_ratio=data_cfg.get('external_noise_ratio', 0.0),
        noise_bbox_expansion=data_cfg.get('noise_bbox_expansion', 0.0),
        seed=seed if seed is not None else cfg.meta.seed,
        k_shared=data_cfg.get('k_shared', 2),
        m_unique=data_cfg.get('m_unique', 2),
        d_out=data_cfg.get('d_out', 16),
        shared_factor_dist=data_cfg.get('shared_factor_dist', 'uniform'),
        u3a_scale=data_cfg.get('u3a_scale', 0.2),
        u3b_scale=data_cfg.get('u3b_scale', 0.3),
        turns=data_cfg.get('turns', 1.0),
        wave_amplitude=data_cfg.get('wave_amplitude', 1.0),
        embed_dim=data_cfg.get('embed_dim', None),
        mlp_depth=data_cfg.get('mlp_depth', 2),
    )
