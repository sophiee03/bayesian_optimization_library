import matplotlib.pyplot as plt
import torch
from typing import List
from .config import OptimizationConfig

def plot_data(train: torch.Tensor, result: torch.Tensor, acqv: float | torch.Tensor, opt_name: str, config: OptimizationConfig):
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