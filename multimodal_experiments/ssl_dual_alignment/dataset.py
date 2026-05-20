import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from multimodal_experiments.initial_trials.ssl_disentangling import sample_curve_data

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
        d_out: int = 16
    ):
        self.num_samples = num_samples
        self.seed = seed
        self.external_noise_ratio = float(external_noise_ratio)
        self.noise_bbox_expansion = float(noise_bbox_expansion)
        
        # Backward compatibility for old configs using "asymmetric_noise_rate" (defaulting to corrupt behavior)
        self.asym_corrupt_rate_a = max(float(asym_corrupt_rate_a), float(asymmetric_noise_rate_a))
        self.asym_corrupt_rate_b = max(float(asym_corrupt_rate_b), float(asymmetric_noise_rate_b))
        self.asym_mismatch_rate_a = float(asym_mismatch_rate_a)
        self.asym_mismatch_rate_b = float(asym_mismatch_rate_b)

        # Alias resolution for data_type
        dt_upper = str(data_type).upper()
        if dt_upper == '3D1F':
            data_type = '3d-av-1f-common'
        elif dt_upper == '3D2F':
            data_type = '3d-2f-common'
        self.data_type = data_type

        # Use local RandomState for isolated reproducibility (fixes worker-fork duplicates)
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        # Separate torch generator for MLP weights to guarantee consistency
        self.torch_rng = torch.Generator().manual_seed(seed if seed is not None else 42)

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
            
            turns = 1

            # 1. Calculate sample counts for each segment
            N = num_samples
            n_ext = int(np.floor(N * self.external_noise_ratio))
            n_ac_a = int(np.floor(N * self.asym_corrupt_rate_a))
            n_ac_b = int(np.floor(N * self.asym_corrupt_rate_b))
            n_am_a = int(np.floor(N * self.asym_mismatch_rate_a))
            n_am_b = int(np.floor(N * self.asym_mismatch_rate_b))
            n_man = N - (n_ext + n_ac_a + n_ac_b + n_am_a + n_am_b)
            
            if n_man < 1:
                raise ValueError(f"num_samples too small for configured noise rates. Manifold={n_man}")

            # --------------------------------------------------------------
            # DATA TYPE: 2D
            # --------------------------------------------------------------
            if data_type == '2d':
                # Original 2D code preserved
                def curve_a_fn(u_vals):
                    return ((0.8 * u_vals + 0.2) * np.sin(turns * u_vals * 2 * np.pi),
                            (0.8 * u_vals + 0.2) * np.cos(turns * u_vals * 2 * np.pi))
                def curve_b_fn(u_vals):
                    x = u_vals * 2.0 - 1.0
                    return (x, x**3 - 0.5 * x - 0.5)

                u_man = np.linspace(0, 1, n_man)
                da_m, _ = sample_curve_data(u_man, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
                db_m, _ = sample_curve_data(u_man, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)

                pa, pb, pu = [da_m], [db_m], [u_man]
                pta, ptb = [np.zeros(n_man, dtype=np.int32)], [np.zeros(n_man, dtype=np.int32)]

                # For 2D, we map both corrupt and mismatch to the old asymmetric logic for legacy consistency
                n_asym_a_total = n_ac_a + n_am_a
                if n_asym_a_total > 0:
                    u_as = self.rng.uniform(0, 1, n_asym_a_total)
                    ag, _ = sample_curve_data(u_as, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
                    bg, _ = sample_curve_data(u_as, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    bc = bg + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=bg.shape)
                    pa.append(ag); pb.append(bc); pu.append(u_as)
                    pta.append(np.full(n_asym_a_total, 1, dtype=np.int32)); ptb.append(np.full(n_asym_a_total, 2, dtype=np.int32))

                n_asym_b_total = n_ac_b + n_am_b
                if n_asym_b_total > 0:
                    u_as = self.rng.uniform(0, 1, n_asym_b_total)
                    ag, _ = sample_curve_data(u_as, curve_a_fn, (self.manifold_noise_a, self.manifold_noise_a), rng=self.rng)
                    bg, _ = sample_curve_data(u_as, curve_b_fn, (self.manifold_noise_b, self.manifold_noise_b), rng=self.rng)
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    ac = ag + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=ag.shape)
                    pa.append(ac); pb.append(bg); pu.append(u_as)
                    pta.append(np.full(n_asym_b_total, 4, dtype=np.int32)); ptb.append(np.full(n_asym_b_total, 3, dtype=np.int32))

                if n_ext > 0:
                    u_bound = np.linspace(0, 1, 100)
                    t_a, _ = sample_curve_data(u_bound, curve_a_fn, (0,0), rng=self.rng)
                    t_b, _ = sample_curve_data(u_bound, curve_b_fn, (0,0), rng=self.rng)
                    mina, maxa = t_a.min(0), t_a.max(0)
                    minb, maxb = t_b.min(0), t_b.max(0)
                    h = self.noise_bbox_expansion / 2.0
                    mina, maxa = mina - h*(maxa-mina), maxa + h*(maxa-mina)
                    minb, maxb = minb - h*(maxb-minb), maxb + h*(maxb-minb)
                    pa.append(mina + (maxa - mina) * self.rng.rand(n_ext, 2))
                    pb.append(minb + (maxb - minb) * self.rng.rand(n_ext, 2))
                    pu.append(self.rng.uniform(0, 1, n_ext))
                    pta.append(np.full(n_ext, 5, dtype=np.int32)); ptb.append(np.full(n_ext, 5, dtype=np.int32))

                data_a, data_b = np.vstack(pa), np.vstack(pb)
                param_values = np.concatenate(pu)
                self.point_type_a, self.point_type_b = np.concatenate(pta), np.concatenate(ptb)

            # --------------------------------------------------------------
            # DATA TYPE: 3D-3F-2C (Mode 1: Volumetric Spiral/Wave Visualizer)
            # --------------------------------------------------------------
            elif data_type == '3d-3f-2c':
                r"""
                3D Observable, 3 Factors Total, 2 Factors Common.
                Uses explicit volumetric formulas for visual inspection of disentanglement.
                A: Volumetric Spiral (u3a creates thickness).
                B: Volumetric Wave (u3b creates vertical scatter).
                """
                def gen_3d3f2c(u1, u2, u3a, u3b):
                    # Base clean shapes without unstructured noise
                    # Mod A: Volumetric Spiral
                    r = u1 + 0.2 * u3a
                    theta = 2 * np.pi * turns * u1
                    x_a = r * np.cos(theta)
                    y_a = r * np.sin(theta)
                    z_a = u2
                    a_clean = np.column_stack([x_a, y_a, z_a])
                    
                    # Mod B: Volumetric Wave
                    x_b = u1
                    y_b = u2
                    v = 2 * u1 - 1
                    z_b = (v**3 - 0.5 * v) + 0.3 * u3b
                    b_clean = np.column_stack([x_b, y_b, z_b])
                    return a_clean, b_clean

                def apply_manifold_noise(arr, noise_level):
                    return arr + self.rng.normal(0, noise_level, size=arr.shape)

                def get_bbox(u_steps=1000):
                    u1 = self.rng.uniform(0, 1, u_steps)
                    u2 = self.rng.uniform(0, 1, u_steps)
                    u3a = self.rng.normal(0, 1, u_steps)
                    u3b = self.rng.normal(0, 1, u_steps)
                    a, b = gen_3d3f2c(u1, u2, u3a, u3b)
                    ma, Ma = a.min(0), a.max(0)
                    mb, Mb = b.min(0), b.max(0)
                    h = self.noise_bbox_expansion / 2.0
                    ma, Ma = ma - h*(Ma-ma), Ma + h*(Ma-ma)
                    mb, Mb = mb - h*(Mb-mb), Mb + h*(Mb-mb)
                    return ma, Ma, mb, Mb

                pa, pb, pu = [], [], []
                pta, ptb = [], []

                # 1. Manifold (Matching)
                if n_man > 0:
                    u1 = np.linspace(0, 1, n_man)
                    u2 = self.rng.uniform(0, 1, n_man)
                    u3a = self.rng.normal(0, 1, n_man)
                    u3b = self.rng.normal(0, 1, n_man)
                    a, b = gen_3d3f2c(u1, u2, u3a, u3b)
                    pa.append(apply_manifold_noise(a, self.manifold_noise_a))
                    pb.append(apply_manifold_noise(b, self.manifold_noise_b))
                    pu.append(np.column_stack([u1, u2]))
                    pta.append(np.zeros(n_man, dtype=np.int32))
                    ptb.append(np.zeros(n_man, dtype=np.int32))

                # Helper to sample random boxes
                ma, Ma, mb, Mb = get_bbox()

                # 2. Asymmetric Corrupted A (A is garbage, B is clean)
                if n_ac_a > 0:
                    u1 = self.rng.uniform(0, 1, n_ac_a)
                    u2 = self.rng.uniform(0, 1, n_ac_a)
                    u3b = self.rng.normal(0, 1, n_ac_a)
                    _, b = gen_3d3f2c(u1, u2, np.zeros(n_ac_a), u3b) # A is ignored anyway
                    a_garbage = ma + (Ma - ma) * self.rng.rand(n_ac_a, 3)
                    pa.append(a_garbage)
                    pb.append(apply_manifold_noise(b, self.manifold_noise_b))
                    pu.append(np.column_stack([u1, u2]))
                    pta.append(np.full(n_ac_a, 2, dtype=np.int32)) # Corrupted code
                    ptb.append(np.full(n_ac_a, 1, dtype=np.int32)) # Clean code

                # 3. Asymmetric Corrupted B (B is garbage, A is clean)
                if n_ac_b > 0:
                    u1 = self.rng.uniform(0, 1, n_ac_b)
                    u2 = self.rng.uniform(0, 1, n_ac_b)
                    u3a = self.rng.normal(0, 1, n_ac_b)
                    a, _ = gen_3d3f2c(u1, u2, u3a, np.zeros(n_ac_b))
                    b_garbage = mb + (Mb - mb) * self.rng.rand(n_ac_b, 3)
                    pa.append(apply_manifold_noise(a, self.manifold_noise_a))
                    pb.append(b_garbage)
                    pu.append(np.column_stack([u1, u2]))
                    pta.append(np.full(n_ac_b, 1, dtype=np.int32))
                    ptb.append(np.full(n_ac_b, 2, dtype=np.int32))

                # 4. Asymmetric Mismatched A (A is valid but wrong latent, B is clean)
                if n_am_a > 0:
                    u1_true = self.rng.uniform(0, 1, n_am_a)
                    u2_true = self.rng.uniform(0, 1, n_am_a)
                    u1_fake = self.rng.uniform(0, 1, n_am_a)
                    u2_fake = self.rng.uniform(0, 1, n_am_a)
                    u3a = self.rng.normal(0, 1, n_am_a)
                    u3b = self.rng.normal(0, 1, n_am_a)
                    
                    a_fake, _ = gen_3d3f2c(u1_fake, u2_fake, u3a, u3b)
                    _, b_true = gen_3d3f2c(u1_true, u2_true, u3a, u3b)
                    
                    pa.append(apply_manifold_noise(a_fake, self.manifold_noise_a))
                    pb.append(apply_manifold_noise(b_true, self.manifold_noise_b))
                    pu.append(np.column_stack([u1_true, u2_true]))
                    pta.append(np.full(n_am_a, 4, dtype=np.int32)) # Mismatched code
                    ptb.append(np.full(n_am_a, 3, dtype=np.int32)) # Clean code
                    
                # 5. Asymmetric Mismatched B (B is valid but wrong latent, A is clean)
                if n_am_b > 0:
                    u1_true = self.rng.uniform(0, 1, n_am_b)
                    u2_true = self.rng.uniform(0, 1, n_am_b)
                    u1_fake = self.rng.uniform(0, 1, n_am_b)
                    u2_fake = self.rng.uniform(0, 1, n_am_b)
                    u3a = self.rng.normal(0, 1, n_am_b)
                    u3b = self.rng.normal(0, 1, n_am_b)
                    
                    a_true, _ = gen_3d3f2c(u1_true, u2_true, u3a, u3b)
                    _, b_fake = gen_3d3f2c(u1_fake, u2_fake, u3a, u3b)
                    
                    pa.append(apply_manifold_noise(a_true, self.manifold_noise_a))
                    pb.append(apply_manifold_noise(b_fake, self.manifold_noise_b))
                    pu.append(np.column_stack([u1_true, u2_true]))
                    pta.append(np.full(n_am_b, 3, dtype=np.int32)) 
                    ptb.append(np.full(n_am_b, 4, dtype=np.int32))

                # 6. External (Both Garbage)
                if n_ext > 0:
                    pa.append(ma + (Ma - ma) * self.rng.rand(n_ext, 3))
                    pb.append(mb + (Mb - mb) * self.rng.rand(n_ext, 3))
                    pu.append(np.column_stack([self.rng.uniform(0, 1, n_ext), self.rng.uniform(0, 1, n_ext)]))
                    pta.append(np.full(n_ext, 5, dtype=np.int32))
                    ptb.append(np.full(n_ext, 5, dtype=np.int32))

                data_a, data_b = np.vstack(pa), np.vstack(pb)
                
                # Normalize 3D3F2C to keep coordinate bounds reasonable before rotation
                m_a, s_a = data_a.mean(0), data_a.std(0)
                m_b, s_b = data_b.mean(0), data_b.std(0)
                data_a = (data_a - m_a) / (s_a + 1e-8)
                data_b = (data_b - m_b) / (s_b + 1e-8)
                
                param_values = np.vstack(pu)
                self.point_type_a, self.point_type_b = np.concatenate(pta), np.concatenate(ptb)


            # --------------------------------------------------------------
            # DATA TYPE: ND-KF-MLP (Mode 2: Theoretical AV Benchmark)
            # --------------------------------------------------------------
            elif data_type == 'nd-kf-mlp':
                r"""
                N-Dimensional output, K-Factors Shared, M-Factors Unique.
                Observation is a frozen random MLP to simulate complex rendering.
                """
                class RandomFrozenMLP(nn.Module):
                    def __init__(self, in_dim, out_dim, hidden_dim=64):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(in_dim, hidden_dim),
                            nn.GELU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.GELU(),
                            nn.Linear(hidden_dim, out_dim)
                        )
                        # Freeze
                        for p in self.parameters():
                            p.requires_grad = False
                            
                    def forward(self, x):
                        return self.net(x)

                in_dim = k_shared + m_unique
                mlp_a = RandomFrozenMLP(in_dim, d_out).eval()
                mlp_b = RandomFrozenMLP(in_dim, d_out).eval()
                
                # Make sure the MLPs are deterministically initialized based on self.seed
                # For simplicity, we just seeded torch globally earlier or we rely on torch_rng.
                # Actually, best to explicitly initialize with generator to be safe:
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        torch.nn.init.xavier_uniform_(m.weight, generator=self.torch_rng)
                        torch.nn.init.zeros_(m.bias)
                mlp_a.apply(init_weights)
                mlp_b.apply(init_weights)

                pa, pb, pu = [], [], []
                pta, ptb = [], []

                def sample_latents(n):
                    u_s = self.rng.uniform(0, 1, (n, k_shared))
                    u_ua = self.rng.normal(0, 1, (n, m_unique))
                    u_ub = self.rng.normal(0, 1, (n, m_unique))
                    return u_s, u_ua, u_ub
                    
                def gen_mlp(u_s, u_ua, u_ub):
                    za = torch.tensor(np.hstack([u_s, u_ua]), dtype=torch.float32)
                    zb = torch.tensor(np.hstack([u_s, u_ub]), dtype=torch.float32)
                    with torch.no_grad():
                        xa = mlp_a(za).numpy()
                        xb = mlp_b(zb).numpy()
                    return xa, xb

                # 1. Manifold
                if n_man > 0:
                    us, ua, ub = sample_latents(n_man)
                    xa, xb = gen_mlp(us, ua, ub)
                    pa.append(xa); pb.append(xb); pu.append(us)
                    pta.append(np.zeros(n_man, dtype=np.int32)); ptb.append(np.zeros(n_man, dtype=np.int32))

                # Helper for bounds (estimating output range for garbage generation)
                _us, _ua, _ub = sample_latents(1000)
                _xa, _xb = gen_mlp(_us, _ua, _ub)
                ma, Ma = _xa.min(0), _xa.max(0)
                mb, Mb = _xb.min(0), _xb.max(0)
                h = self.noise_bbox_expansion / 2.0
                ma, Ma = ma - h*(Ma-ma), Ma + h*(Ma-ma)
                mb, Mb = mb - h*(Mb-mb), Mb + h*(Mb-mb)

                # 2. Asym Corrupt A
                if n_ac_a > 0:
                    us, ua, ub = sample_latents(n_ac_a)
                    _, xb = gen_mlp(us, ua, ub)
                    xa = ma + (Ma - ma) * self.rng.rand(n_ac_a, d_out)
                    pa.append(xa); pb.append(xb); pu.append(us)
                    pta.append(np.full(n_ac_a, 2, dtype=np.int32)); ptb.append(np.full(n_ac_a, 1, dtype=np.int32))

                # 3. Asym Corrupt B
                if n_ac_b > 0:
                    us, ua, ub = sample_latents(n_ac_b)
                    xa, _ = gen_mlp(us, ua, ub)
                    xb = mb + (Mb - mb) * self.rng.rand(n_ac_b, d_out)
                    pa.append(xa); pb.append(xb); pu.append(us)
                    pta.append(np.full(n_ac_b, 1, dtype=np.int32)); ptb.append(np.full(n_ac_b, 2, dtype=np.int32))
                    
                # 4. Asym Mismatch A
                if n_am_a > 0:
                    us_true, ua, ub = sample_latents(n_am_a)
                    us_fake, _, _ = sample_latents(n_am_a)
                    xa_fake, _ = gen_mlp(us_fake, ua, ub)
                    _, xb_true = gen_mlp(us_true, ua, ub)
                    pa.append(xa_fake); pb.append(xb_true); pu.append(us_true)
                    pta.append(np.full(n_am_a, 4, dtype=np.int32)); ptb.append(np.full(n_am_a, 3, dtype=np.int32))

                # 5. Asym Mismatch B
                if n_am_b > 0:
                    us_true, ua, ub = sample_latents(n_am_b)
                    us_fake, _, _ = sample_latents(n_am_b)
                    xa_true, _ = gen_mlp(us_true, ua, ub)
                    _, xb_fake = gen_mlp(us_fake, ua, ub)
                    pa.append(xa_true); pb.append(xb_fake); pu.append(us_true)
                    pta.append(np.full(n_am_b, 3, dtype=np.int32)); ptb.append(np.full(n_am_b, 4, dtype=np.int32))
                    
                # 6. External
                if n_ext > 0:
                    pa.append(ma + (Ma - ma) * self.rng.rand(n_ext, d_out))
                    pb.append(mb + (Mb - mb) * self.rng.rand(n_ext, d_out))
                    pu.append(self.rng.uniform(0, 1, (n_ext, k_shared)))
                    pta.append(np.full(n_ext, 5, dtype=np.int32)); ptb.append(np.full(n_ext, 5, dtype=np.int32))

                data_a, data_b = np.vstack(pa), np.vstack(pb)
                # Unstructured noise on final output (sensor noise)
                data_a += self.rng.normal(0, self.manifold_noise_a, size=data_a.shape)
                data_b += self.rng.normal(0, self.manifold_noise_b, size=data_b.shape)
                
                param_values = np.vstack(pu)
                self.point_type_a, self.point_type_b = np.concatenate(pta), np.concatenate(ptb)


            # Keep original legacy modes for backward compatibility if needed...
            elif data_type == '3d-av-1f-common':
                # Simplified preservation of old code to prevent breaking sweeps
                def generate_3d1f_raw_features(u_vals, rng, na, nb):
                    pitch = 1.0 / (1.2 - u_vals)
                    resonance = np.sin(u_vals * np.pi)
                    splash = rng.normal(0, na, len(u_vals))
                    raw_a = np.stack([pitch, resonance, splash], axis=1)
                    d1 = u_vals
                    d2 = rng.normal(0, 1, len(u_vals)) * (0.5 + u_vals)
                    d3 = rng.normal(0, nb, len(u_vals))
                    raw_b = np.column_stack([d1, d2, d3])
                    return raw_a, raw_b

                u_man = np.linspace(0, 1, n_man)
                n_asym_a_total = n_ac_a + n_am_a
                n_asym_b_total = n_ac_b + n_am_b
                
                u_as_a = self.rng.uniform(0, 1, n_asym_a_total) if n_asym_a_total > 0 else np.array([])
                u_as_b = self.rng.uniform(0, 1, n_asym_b_total) if n_asym_b_total > 0 else np.array([])
                u_all = np.concatenate([u_man, u_as_a, u_as_b])

                raw_a_all, raw_b_all = generate_3d1f_raw_features(u_all, self.rng, self.manifold_noise_a, self.manifold_noise_b)
                m_a, s_a = raw_a_all.mean(0), raw_a_all.std(0)
                m_b, s_b = raw_b_all.mean(0), raw_b_all.std(0)
                norm_a_all = (raw_a_all - m_a) / (s_a + 1e-8)
                norm_b_all = (raw_b_all - m_b) / (s_b + 1e-8)
                
                theta_y, theta_z = np.pi / 4, np.pi / 3
                Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
                Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
                rot_a_all = norm_a_all @ (Ry @ Rz).T
                
                pa, pb, pu = [rot_a_all[:n_man]], [norm_b_all[:n_man]], [u_man]
                pta, ptb = [np.zeros(n_man, dtype=np.int32)], [np.zeros(n_man, dtype=np.int32)]

                if n_asym_a_total > 0:
                    idx_s, idx_e = n_man, n_man + n_asym_a_total
                    a_g, b_g = rot_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    b_c = b_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_g.shape)
                    pa.append(a_g); pb.append(b_c); pu.append(u_as_a)
                    pta.append(np.ones(n_asym_a_total, dtype=np.int32)); ptb.append(np.full(n_asym_a_total, 2, dtype=np.int32))

                if n_asym_b_total > 0:
                    idx_s, idx_e = n_man+n_asym_a_total, n_man+n_asym_a_total+n_asym_b_total
                    a_g, b_g = rot_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    a_c = a_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=a_g.shape)
                    pa.append(a_c); pb.append(b_g); pu.append(u_as_b)
                    pta.append(np.full(n_asym_b_total, 4, dtype=np.int32)); ptb.append(np.full(n_asym_b_total, 3, dtype=np.int32))

                if n_ext > 0:
                    u_b = np.linspace(0, 1, 100)
                    r_a, r_b = generate_3d1f_raw_features(u_b, self.rng, self.manifold_noise_a, self.manifold_noise_b)
                    n_a, n_b = (r_a - m_a) / (s_a + 1e-8), (r_b - m_b) / (s_b + 1e-8)
                    ro_a = n_a @ (Ry @ Rz).T
                    ma, Ma = ro_a.min(0), ro_a.max(0); mb, Mb = n_b.min(0), n_b.max(0)
                    h = self.noise_bbox_expansion / 2.0
                    ma, Ma = ma - h*(Ma-ma), Ma + h*(Ma-ma)
                    mb, Mb = mb - h*(Mb-mb), Mb + h*(Mb-mb)
                    pa.append(ma + (Ma-ma)*self.rng.rand(n_ext, 3))
                    pb.append(mb + (Mb-mb)*self.rng.rand(n_ext, 3))
                    pu.append(self.rng.uniform(0, 1, n_ext))
                    pta.append(np.full(n_ext, 5, dtype=np.int32)); ptb.append(np.full(n_ext, 5, dtype=np.int32))

                data_a, data_b = np.vstack(pa), np.vstack(pb)
                param_values = np.concatenate(pu)
                self.point_type_a, self.point_type_b = np.concatenate(pta), np.concatenate(ptb)
                
            elif data_type == '3d-2f-common':
                # Legacy 3D2F
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
                    return np.hstack([xy_a, (2.0 * u2).reshape(-1, 1)]), np.hstack([xy_b, u2.reshape(-1, 1)])

                n_asym_a_total = n_ac_a + n_am_a
                n_asym_b_total = n_ac_b + n_am_b

                u1_m, u2_m = np.linspace(0, 1, n_man), self.rng.uniform(0, 1, n_man)
                u1_aa, u2_aa = self.rng.uniform(0, 1, n_asym_a_total), self.rng.uniform(0, 1, n_asym_a_total)
                u1_ab, u2_ab = self.rng.uniform(0, 1, n_asym_b_total), self.rng.uniform(0, 1, n_asym_b_total)
                u1_all = np.concatenate([u1_m, u1_aa, u1_ab]); u2_all = np.concatenate([u2_m, u2_aa, u2_ab])
                
                raw_a_all, raw_b_all = generate_3d2f_raw_features(u1_all, u2_all, self.rng, self.manifold_noise_a, self.manifold_noise_b)
                m_a, s_a = raw_a_all.mean(0), raw_a_all.std(0)
                m_b, s_b = raw_b_all.mean(0), raw_b_all.std(0)
                norm_a_all = (raw_a_all - m_a) / (s_a + 1e-8)
                norm_b_all = (raw_b_all - m_b) / (s_b + 1e-8)
                
                pa, pb, pu = [norm_a_all[:n_man]], [norm_b_all[:n_man]], [np.column_stack([u1_m, u2_m])]
                pta, ptb = [np.zeros(n_man, dtype=np.int32)], [np.zeros(n_man, dtype=np.int32)]

                if n_asym_a_total > 0:
                    idx_s, idx_e = n_man, n_man+n_asym_a_total
                    a_g, b_g = norm_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    b_c = b_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=b_g.shape)
                    pa.append(a_g); pb.append(b_c); pu.append(np.column_stack([u1_aa, u2_aa]))
                    pta.append(np.ones(n_asym_a_total, dtype=np.int32)); ptb.append(np.full(n_asym_a_total, 2, dtype=np.int32))

                if n_asym_b_total > 0:
                    idx_s, idx_e = n_man+n_asym_a_total, n_man+n_asym_a_total+n_asym_b_total
                    a_g, b_g = norm_a_all[idx_s:idx_e], norm_b_all[idx_s:idx_e]
                    mag = self.asymmetric_noise_magnitude if self.asymmetric_noise_magnitude is not None else 5.0 * max(self.manifold_noise_a, self.manifold_noise_b)
                    a_c = a_g + self.rng.normal(scale=mag * (1.0 + self.noise_bbox_expansion), size=a_g.shape)
                    pa.append(a_c); pb.append(b_g); pu.append(np.column_stack([u1_ab, u2_ab]))
                    pta.append(np.full(n_asym_b_total, 4, dtype=np.int32)); ptb.append(np.full(n_asym_b_total, 3, dtype=np.int32))

                if n_ext > 0:
                    u1_b, u2_b = self.rng.uniform(0, 1, 1000), self.rng.uniform(0, 1, 1000)
                    r_a, r_b = generate_3d2f_raw_features(u1_b, u2_b, self.rng, self.manifold_noise_a, self.manifold_noise_b)
                    n_a, n_b = (r_a - m_a) / (s_a + 1e-8), (r_b - m_b) / (s_b + 1e-8)
                    ma, Ma = n_a.min(0), n_a.max(0); mb, Mb = n_b.min(0), n_b.max(0)
                    h = self.noise_bbox_expansion / 2.0
                    ma, Ma = ma - h*(Ma-ma), Ma + h*(Ma-ma)
                    mb, Mb = mb - h*(Mb-mb), Mb + h*(Mb-mb)
                    pa.append(ma + (Ma-ma)*self.rng.rand(n_ext, 3))
                    pb.append(mb + (Mb-mb)*self.rng.rand(n_ext, 3))
                    pu.append(np.column_stack([self.rng.uniform(0, 1, n_ext), self.rng.uniform(0, 1, n_ext)]))
                    pta.append(np.full(n_ext, 5, dtype=np.int32)); ptb.append(np.full(n_ext, 5, dtype=np.int32))

                data_a, data_b = np.vstack(pa), np.vstack(pb)
                param_values = np.vstack(pu)
                self.point_type_a, self.point_type_b = np.concatenate(pta), np.concatenate(ptb)

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
        
        # Corr target: 0 for anything that is corrupted, mismatched, or external. 0.9 for true manifold.
        # We can map point_type == 0 to 0.9, others to 0.0
        # However, legacy code just mapped everything but external to 0.9.
        # To be rigorous with the new definitions:
        is_clean = (self.point_type_a == 0) & (self.point_type_b == 0)
        c_targ = np.where(is_clean.numpy()[:, None], [0.0, 0.9], [0.0, 0.0])
        self.corr_target = torch.tensor(c_targ, dtype=torch.float32)
        
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
