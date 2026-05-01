import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb

def plot_original_spaces(data_a, data_b, param_values):
    """Plots raw Modality A and Modality B datasets."""
    is_3d = data_a.shape[1] >= 3
    fig = plt.figure(figsize=(12, 6))
    
    if is_3d:
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(data_a[:, 0], data_a[:, 1], data_a[:, 2],
                    c=param_values, cmap='turbo', s=5, alpha=0.5)
        ax1.set_title('Modality A')
        ax1.set_xlabel('Dim 1'); ax1.set_ylabel('Dim 2'); ax1.set_zlabel('Dim 3')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(data_b[:, 0], data_b[:, 1], data_b[:, 2],
                    c=param_values, cmap='turbo', s=5, alpha=0.5)
        ax2.set_title('Modality B')
        ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2'); ax2.set_zlabel('Dim 3')
    else:
        ax1 = fig.add_subplot(121)
        ax1.scatter(data_a[:, 0], data_a[:, 1], c=param_values, cmap='turbo', alpha=0.5)
        ax1.set_title('Modality A')
        ax1.set_xlabel('Dim 1'); ax1.set_ylabel('Dim 2')
        ax1.axis('equal')

        ax2 = fig.add_subplot(122)
        ax2.scatter(data_b[:, 0], data_b[:, 1], c=param_values, cmap='turbo', alpha=0.5)
        ax2.set_title('Modality B')
        ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2')
        ax2.axis('equal')

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)
    return fig

def plot_dual_geometry_reshaping_view(dual_model, data_a, data_b, param_values, device):
    """Plots 4-way view: Input A -> Output A -> Output B -> Input B."""
    dual_model.eval()
    with torch.no_grad():
        output_a, _ = dual_model.model_a(torch.tensor(data_a, device=device, dtype=torch.float64))
        output_b, _ = dual_model.model_b(torch.tensor(data_b, device=device, dtype=torch.float64))

    output_a = output_a.detach().cpu().numpy()
    output_b = output_b.detach().cpu().numpy()
    
    # Normalize color code
    color_code = (param_values - np.min(param_values)) / (np.max(param_values) - np.min(param_values) + 1e-12)

    is_3d = data_a.shape[1] >= 3
    fig = plt.figure(figsize=(18, 4))
    fig.suptitle('Self-Supervised Dual Geometry Reshaping')

    if is_3d:
        axs = [fig.add_subplot(1, 4, i+1, projection='3d') for i in range(4)]
        axs[0].scatter(data_a[:, 0], data_a[:, 1], data_a[:, 2], c=color_code, cmap='turbo', s=10, alpha=0.85)
        axs[1].scatter(output_a[:, 0], output_a[:, 1], output_a[:, 2], c=color_code, cmap='turbo', s=10, alpha=0.85)
        axs[2].scatter(output_b[:, 0], output_b[:, 1], output_b[:, 2], c=color_code, cmap='turbo', s=10, alpha=0.85)
        axs[3].scatter(data_b[:, 0], data_b[:, 1], data_b[:, 2], c=color_code, cmap='turbo', s=10, alpha=0.85)
        
        for i in range(4):
            axs[i].set_xlabel('Dim 1')
            axs[i].set_ylabel('Dim 2')
            axs[i].set_zlabel('Dim 3')
    else:
        axs = [fig.add_subplot(1, 4, i+1) for i in range(4)]
        axs[0].scatter(data_a[:, 0], data_a[:, 1], c=color_code, cmap='turbo', s=10, alpha=0.85)
        axs[1].scatter(output_a[:, 0], output_a[:, 1], c=color_code, cmap='turbo', s=10, alpha=0.85)
        axs[2].scatter(output_b[:, 0], output_b[:, 1], c=color_code, cmap='turbo', s=10, alpha=0.85)
        axs[3].scatter(data_b[:, 0], data_b[:, 1], c=color_code, cmap='turbo', s=10, alpha=0.85)

        for i in range(4):
            axs[i].set_xlabel('Dim 1')
            axs[i].set_ylabel('Dim 2')
            axs[i].axis('equal')

    axs[0].set_title('Input Space A')
    axs[1].set_title('Output Space A')
    axs[2].set_title('Output Space B')
    axs[3].set_title('Input Space B')

    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.85, wspace=0.3)
    return fig

def log_plots_to_wandb(dual_model, dataset, device, step, wandb_run):
    """Generates and logs visualizations to W&B."""
    data_a = dataset.data_a.numpy()
    data_b = dataset.data_b.numpy()
    param_values = dataset.param_values
    
    fig_spaces = plot_original_spaces(data_a, data_b, param_values)
    fig_reshaping = plot_dual_geometry_reshaping_view(dual_model, data_a, data_b, param_values, device)
    
    if wandb_run:
        wandb.log({
            "original_spaces": wandb.Image(fig_spaces),
            "geometry_reshaping": wandb.Image(fig_reshaping)
        }, step=step)
    
    plt.close(fig_spaces)
    plt.close(fig_reshaping)
