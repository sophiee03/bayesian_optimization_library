import torch
from ..config import OptimizationConfig
from botorch.utils.transforms import normalize, unnormalize

def minimization_transformation(data, config: OptimizationConfig):
    """Function that change sign of the metrics to minimize, to transform this problem in a maximization problem
    
    Args:
        data (List[List]): data given as input for training
        config (OptimizationConfig): configuration of the BayesianOptimization (to check which metrics are to minimize)

    Returns:
        List[List]: data with the metrics to minimize multiplied by -1
    """
    for col in range(len(config.objective_metrics)):
        if config.goal[col].lower()=="min":
            for row in range(len(data)):
                data[row][col] *= -1
    return data

def normalize_val(x: torch.Tensor, bounds: torch.Tensor):
    """Function to normalize data with the BoTorch method normalize()
    
    Args:
        x (Tensor): parameters to normalize
        bounds (Tensor): bounds for each parameter

    Returns:
        Tensor: parameters normalized in [0,1]
    """
    return normalize(x, bounds)

def denormalize_val(candidates: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Function to denormalize data to the original bounds with the BoTorch method unnormalize()
    
    Args:
        candidates (Tensor): candidates generated
        bounds (Tensor): original bounds of each parameter

    Returns: 
        List: candidates denormalized
    """
    return unnormalize(candidates, bounds).tolist()