import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Dict, List
from ..config import OptimizationConfig
from tabulate import tabulate

def visualize_data(d: torch.Tensor, headers: List):
    """function to print data
    
    Args:
        d (Tensor): data to visualize
        headers (List[str]): parameter/metrics names
    """
    if isinstance(d, torch.Tensor):
        data = d.cpu().numpy().tolist()
    else:
        data = d
    print(tabulate(data, headers=headers, floatfmt=([".6f"]*len(headers)), tablefmt="simple_grid"))

def handle_acqv(a):
    """function to handle acquisition values format in plots"""
    if isinstance(a, list):
        result = []
        for item in a:
            if isinstance(item, (torch.Tensor, np.ndarray)):
                result.extend(handle_acqv(item))
            else:
                result.append(float(item))
        return result
    # Tensor 
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().tolist() if a.numel() > 1 else [a.detach().cpu().item()]
    # Array 
    if isinstance(a, np.ndarray):
        return a.tolist() if a.size > 1 else [float(a)]
    # Scalar
    return [float(a)]

def plot_train(train_data: torch.Tensor, config: OptimizationConfig):
    """plotting results measured from training dataset"""
    fig, ax = plt.subplots()

    train_par_1 = train_data[:,0]
    train_par_2 = train_data[:,1]
    ax.scatter(train_par_1, train_par_2, color="blue", label="Training", alpha=0.7)
    
    ax.set_xlabel(config.objective_metrics[0])
    ax.set_ylabel(config.objective_metrics[1])
    ax.set_title(f"Output metrics from training")
    
    return fig

def plot_data(train: torch.Tensor, result: torch.Tensor, acqv: float | torch.Tensor, opt_name: str, config: OptimizationConfig):
    """plotting results (only the first 2 parameters for simplicity), useful to compare these parameters with training choices"""
    fig, ax = plt.subplots()

    if result is None:
        return None

    train_par_1 = train[:,0]
    train_par_2 = train[:,1]
    ax.scatter(train_par_1, train_par_2, color="blue", label="Training", alpha=0.7)
    
    res_par_1 = result[:,0]
    res_par_2 = result[:,1]
    ax.scatter(res_par_1, res_par_2, color="red", label="Results", alpha=0.7)

    ax.set_xlabel(config.optimization_parameters[0])
    ax.set_ylabel(config.optimization_parameters[1])
    a = acqv.tolist()
    if isinstance(a, list):
        title = f"Results with \"{opt_name}\"\nacq_values: " + ", ".join([f"{v:.4f}" for v in a])
    else:
        title = f"Results with \"{opt_name}\"\nacq_value: {a:.4f}"
    ax.set_title(title)
    ax.legend()
    
    return fig

def plot_all(train_data: torch.Tensor, results: Dict, config: OptimizationConfig):
    """plotting candidates and training set for all the optimizers used"""
    fig, ax = plt.subplots()

    ax.scatter(train_data[:,0], train_data[:,1], c="blue", alpha=0.6, label="Training")
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    lines = []
    for (color, (name, entry)) in zip(colors, results.items()):
        (candidates, acq_val), _ = entry

        acq_list = handle_acqv(acq_val)
        acq_str = ", ".join([f"{val:.4f}" for val in acq_list])
        lines.append(f"{name} - acq_val: [{acq_str}]")
        
        ax.scatter(candidates[:,0], candidates[:,1], c=[color], label=name, alpha=0.8)

    ax.set_xlabel(config.optimization_parameters[0])
    ax.set_ylabel(config.optimization_parameters[1])
    ax.set_title(f"Candidates: \n"+"\n".join(lines))

    ax.legend()
    plt.tight_layout()
    return fig