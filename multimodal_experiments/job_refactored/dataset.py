import torch
import numpy as np
from torch.utils.data import Dataset
from multimodal_experiments.initial_trials.ssl_disentangling import sample_curve_data

class DualDisentangleDataset(Dataset):
    """Paired modality dset. Shared latent source."""
    def __init__(self, data_type='2d', num_samples=4096, path_a=None, path_b=None):
        # Setup. File paths or synth.
        self.num_samples = num_samples
        self.data_type = data_type
        
        if path_a is not None and path_b is not None:
            # External load.
            data_a = self._load_file(path_a)
            data_b = self._load_file(path_b)
            self.num_samples = data_a.shape[0]
            self.param_values = np.linspace(0, 1, self.num_samples)
        else:
            # Synth generation.
            param_values = np.linspace(0, 1, num_samples)
            self.param_values = param_values
            
            if data_type == '2d':
                # 2D shapes from 1D u.
                """
                2D data breakdown:
                Shared 1D source $u$ ($[0,1]$).

                Mod A: Spiral. $N \times 2$. Radius + angle change by $u$.
                Mod B: Cubic. $N \times 2$. linear $x$ + $x^3$ squiggle.

                Relation: Samples paired by same $u$. Diff geometry, same source. Goal -> map back to shared 1D latent.
                """
                def curve_a_fn(u: np.ndarray):
                    return ((0.8 * u + 0.2) * np.sin(2 * u * 2 * np.pi), (0.8 * u + 0.2) * np.cos(2 * u * 2 * np.pi))
                def curve_b_fn(u: np.ndarray):
                    x = u * 2.0 - 1.0
                    y = x**3 - 0.5 * x - 0.5
                    return (x, y)
                    
                data_a, _ = sample_curve_data(param_values, curve_a_fn, [0.02, 0.02])
                data_b, _ = sample_curve_data(param_values, curve_b_fn, [0.02, 0.02])
                
            elif data_type == '3d-av-1f-common':
                # 3D physical traits + rotation. 1D u.
                """
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

                Summary: 
                A has more signal ($u$ + curve). B has signal + signal-dependent noise. Both rotated to hide source.
                """
                
                # Modality A: Audio.
                pitch = 1.0 / (1.2 - param_values)
                resonance = np.sin(param_values * np.pi)
                splashing_noise = np.random.normal(0, 0.1, num_samples)
                data_a_unrot = np.stack([pitch, resonance, splashing_noise], axis=1)
                data_a_std = (data_a_unrot - data_a_unrot.mean(axis=0)) / data_a_unrot.std(axis=0)
                theta_y_a = np.pi / 4
                theta_z_a = np.pi / 3
                Ry_a = np.array([[np.cos(theta_y_a), 0, np.sin(theta_y_a)], [0, 1, 0], [-np.sin(theta_y_a), 0, np.cos(theta_y_a)]])
                Rz_a = np.array([[np.cos(theta_z_a), -np.sin(theta_z_a), 0], [np.sin(theta_z_a), np.cos(theta_z_a), 0], [0, 0, 1]])
                data_a = data_a_std @ (Ry_a @ Rz_a).T

                # Modality B: Video.
                dim1_b = param_values
                dim2_b = np.random.normal(0, 1, num_samples) * (0.5 + param_values)
                dim3_b = np.random.normal(0, 0.15, num_samples)
                data_b_raw = np.column_stack((dim1_b, dim2_b, dim3_b))
                data_b_std = (data_b_raw - np.mean(data_b_raw, axis=0)) / np.std(data_b_raw, axis=0)
                theta_x_b = -np.pi / 3
                theta_y_b = np.pi / 6
                Rx_b = np.array([[1, 0, 0], [0, np.cos(theta_x_b), -np.sin(theta_x_b)], [0, np.sin(theta_x_b), np.cos(theta_x_b)]])
                Ry_b = np.array([[np.cos(theta_y_b), 0, np.sin(theta_y_b)], [0, 1, 0], [-np.sin(theta_y_b), 0, np.cos(theta_y_b)]])
                data_b = data_b_std @ (Rx_b @ Ry_b).T

            elif data_type == '3d-2f-common':
                # 2 common factors. u1 -> shape, u2 -> 3rd dim (stretch vs shear).
                """
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
                u1 = param_values
                u2 = np.random.uniform(0, 1, num_samples)

                # Mod A: 3D Spiral. Stretch width 2.
                def curve_a_fn_3d(u1_vals):
                    return ((0.8 * u1_vals + 0.2) * np.sin(2 * u1_vals * 2 * np.pi), 
                            (0.8 * u1_vals + 0.2) * np.cos(2 * u1_vals * 2 * np.pi))

                xy_a, _ = sample_curve_data(u1, curve_a_fn_3d, [0.02, 0.02])
                z_a = (u2 * 2.0).reshape(-1, 1)
                data_a = np.hstack([xy_a, z_a])

                # Mod B: 3D Cubic. Shear width 1.
                def curve_b_fn_3d(u1_vals):
                    x = u1_vals * 2.0 - 1.0
                    y = x**3 - 0.5 * x - 0.5
                    return (x, y)

                xy_b, _ = sample_curve_data(u1, curve_b_fn_3d, [0.02, 0.02])
                z_b = u2.reshape(-1, 1)
                data_b = np.hstack([xy_b, z_b])

            else:
                raise ValueError(f"Unknown data type {data_type}")
            
        # Data cast. Double precision.
        self.data_a = torch.tensor(data_a, dtype=torch.float64)
        self.data_b = torch.tensor(data_b, dtype=torch.float64)
        self.corr_target = torch.tensor(np.tile([0.0, 0.9], (self.num_samples, 1)), dtype=torch.float64)
        
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
        return {
            "data_a": self.data_a[idx],
            "data_b": self.data_b[idx],
            "corr_target": self.corr_target[idx],
            "param_values": self.param_values[idx]
        }
