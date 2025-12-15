import torch
from ..config import OptimizationConfig
from botorch.utils.transforms import normalize, unnormalize

def minimization_transformation(data, config: OptimizationConfig):
    """function to change sign of the metrics to minimize, to transform this problem in a maximization problem
    
    Args:
        data (List[List]): data given as input for training
        config (OptimizationConfig): configuration of the BayesianOptimization (to check which metrics are to minimize)

    Returns:
        List[List]: data with the metrics to minimize multiplied by -1
    """
    for col in range(len(config.objective_metrics)):
        if config.goal[col]=='MIN':
            for row in range(len(data)):
                data[row][col] *= -1
    return data

def normalize_val(x: torch.Tensor, bounds: torch.Tensor):
    """function to normalize data
    
    Args:
        x (Tensor): parameters to normalize
        bounds (Tensor): bounds of parameters

    Returns:
        X_norm (Tensor): parameters normalized
    """
    X_norm = normalize(x, bounds)

    return X_norm

def denormalize_val(candidates: torch.Tensor, bounds: torch.Tensor, config: OptimizationConfig) -> torch.Tensor:
    """function to denormalize data to the original bounds and handle modelsize generated
    
    Args:
        candidates (Tensor): candidates generated
        bounds (Tensor): bounds of the parameters
        config (OptimizationConfig): configuration of the BayesianOptimization

    Returns: 
        cand_denormalized (List): candidates denormalized
    """
    cand_denormalized = unnormalize(candidates, bounds).tolist()
    #TO REMOVE
    for n,r in enumerate(cand_denormalized):
        for par_n, par in enumerate(config.optimization_parameters):
            if par == 'MODEL_SIZE':
                cand_denormalized[n][par_n] = str(handle_modelsize(r[par_n]))
    
    return cand_denormalized

#TO REMOVE
def handle_modelsize(n: float):
    """function to transform the modelsize value generated into the relative string"""
    if n <= 3700506.0:      #small
        return 'small'
    elif n <= 9853386.0:    #medium
        return 'medium'
    else:                   #large
        return 'large'