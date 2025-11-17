import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Dict
from .config import OptimizationConfig

def plot_data(train: torch.Tensor, result: torch.Tensor, acqv: float | torch.Tensor, opt_name: str, config: OptimizationConfig):
    '''plotting results (only 2 parameters for simplicity)'''
    fig, ax = plt.subplots()

    if result is None:
        return None
    
    train_par_1 = train[:,0]
    train_par_2 = train[:,1]
    ax.scatter(train_par_1, train_par_2, color='blue', label='Training', alpha=0.7)
    
    res_par_1 = result[:,0]
    res_par_2 = result[:,1]
    ax.scatter(res_par_1, res_par_2, color='red', label='Results', alpha=0.7)

    ax.set_xlabel(config.optimization_parameters[0])
    ax.set_ylabel(config.optimization_parameters[1])
    a = acqv.tolist()
    if isinstance(a, list):
        title = f'Results with \'{opt_name}\'\nacq_values: ' + ', '.join([f'{v:.4f}' for v in a])
    else:
        title = f'Results with \'{opt_name}\'\nacq_value: {a:.4f}'
    ax.set_title(title)
    ax.legend()
    
    return fig

def plot_all(train_data: torch.Tensor, results: Dict, config: OptimizationConfig):
    '''plotting results (only 2 parameters for simplicity)'''
    fig, ax = plt.subplots(figsize=(8, 6))

    train_x = train_data[:, 0]
    train_y = train_data[:, 1]
    ax.scatter(train_x, train_y, c="blue", alpha=0.6, label="Training")

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (color, (name, entry)) in zip(colors, results.items()):
        (candidates, acq_val), _ = entry

        x = candidates[:, 0]
        y = candidates[:, 1]
        
        # Scatter
        ax.scatter(x, y, c=[color], label=name, alpha=0.8)

    ax.set_xlabel(config.optimization_parameters[0])
    ax.set_ylabel(config.optimization_parameters[1])
    ax.set_title("Training + Optimization results")

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig