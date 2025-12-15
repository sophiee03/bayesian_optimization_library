import torch
from ..config import OptimizationConfig
from botorch.utils.transforms import normalize, standardize, unnormalize

def minimization_transformation(data):
    """function to make negative values of the metrics to minimize, to transform this problem in a maximization problem
    
    Args:
        data (Dict): data given as input for training

    Returns:
        Dict: data with the metrics to minimize multiplied by -1
    """
    for col in range(len(data['metrics'][0])):
        if data['metrics'][1][col]=='MIN':
            for row in range(len(data['metrics'][2])):
                data['metrics'][2][row][col] *= -1
    return data

def normalize_val(x: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor):
    """function to normalize and standardize data
    
    Args:
        x (Tensor): parameters to normalize
        y (Tensor): metrics to standardize
        bounds (Tensor): bounds of parameters

    Returns:
        X_norm (Tensor): parameters normalized
        Y_stand (Tensor): metrics standardized
    """
    # Here we trust that etl module returned no NaN values
    X_norm = normalize(x, bounds)
    
    if y.dim() >= 2 and y.shape[1] > 1:
        Y_stand=torch.zeros_like(y)
        for i in range(y.shape[1]):
            y_col = y[:,i:i+1]
            mean = y_col.mean(dim=0)
            std = y_col.std(dim=0)
            std[std==0] = 1.0
            Y_stand[:,i] = ((y_col - mean) / std).squeeze(-1)
    else:
        Y_stand = (standardize(y))
        if Y_stand.dim == 1:
            Y_stand = Y_stand.unsqueeze(-1)

    return X_norm, Y_stand

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
    for n,r in enumerate(cand_denormalized):
        for par_n, par in enumerate(config.optimization_parameters):
            if par == 'MODEL_SIZE':
                cand_denormalized[n][par_n] = str(handle_modelsize(r[par_n]))
    
    return cand_denormalized

def handle_modelsize(n: float):
    """ TO REMOVE
    function to transform the modelsize value generated into the relative string"""
    if n <= 3700506.0:      #small
        return 'small'
    elif n <= 9853386.0:    #medium
        return 'medium'
    else:                   #large
        return 'large'