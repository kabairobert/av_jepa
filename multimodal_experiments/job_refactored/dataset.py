import torch
import numpy as np
from torch.utils.data import Dataset
from multimodal_experiments.initial_trials.ssl_disentangling import sample_curve_data

class DualDisentangleDataset(Dataset):
    def __init__(self, data_type='2d', num_samples=4096):
        self.num_samples = num_samples
        self.data_type = data_type
        
        param_values = np.linspace(0, 1, num_samples)
        self.param_values = param_values
        
        if data_type == '2d':
            def curve_a_fn(u: np.ndarray):
                return ((0.8 * u + 0.2) * np.sin(2 * u * 2 * np.pi), (0.8 * u + 0.2) * np.cos(2 * u * 2 * np.pi))
            def curve_b_fn(u: np.ndarray):
                x = u * 2.0 - 1.0
                y = x**3 - 0.5 * x - 0.5
                return (x, y)
                
            data_a, _ = sample_curve_data(param_values, curve_a_fn, [0.02, 0.02])
            data_b, _ = sample_curve_data(param_values, curve_b_fn, [0.02, 0.02])
            
        elif data_type == '3d':
            # Modality A
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

            # Modality B
            dim1_b = param_values
            dim2_b = np.random.normal(0, 1, num_samples) * (0.5 + param_values) # from notebook output
            dim3_b = np.random.normal(0, 0.15, num_samples)
            data_b_raw = np.column_stack((dim1_b, dim2_b, dim3_b))
            data_b_std = (data_b_raw - np.mean(data_b_raw, axis=0)) / np.std(data_b_raw, axis=0)
            theta_x_b = -np.pi / 3
            theta_y_b = np.pi / 6
            Rx_b = np.array([[1, 0, 0], [0, np.cos(theta_x_b), -np.sin(theta_x_b)], [0, np.sin(theta_x_b), np.cos(theta_x_b)]])
            Ry_b = np.array([[np.cos(theta_y_b), 0, np.sin(theta_y_b)], [0, 1, 0], [-np.sin(theta_y_b), 0, np.cos(theta_y_b)]])
            data_b = data_b_std @ (Rx_b @ Ry_b).T
        else:
            raise ValueError(f"Unknown data type {data_type}")
            
        self.data_a = torch.tensor(data_a, dtype=torch.float64)
        self.data_b = torch.tensor(data_b, dtype=torch.float64)
        self.corr_target = torch.tensor(np.tile([0.0, 0.9], (num_samples, 1)), dtype=torch.float64) # Dummy target for original loss
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            "data_a": self.data_a[idx],
            "data_b": self.data_b[idx],
            "corr_target": self.corr_target[idx],
            "param_values": self.param_values[idx]
        }
