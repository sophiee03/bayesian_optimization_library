import torch
import logging
from .config import Timer, OptimizationConfig
from botorch.utils.transforms import normalize, standardize, unnormalize

def normalize_val(config: OptimizationConfig, x: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor):
    '''function to normalize and standardize data''' 
    logger = logging.getLogger('BO')
    timer = Timer(logger)

    # Here we trust that etl module returned no NaN values
    with timer.measure('normalization/standardization'):
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

    if config.verbose:
        logger.info(f"   -> Data normalized and standardized        [{timer.get_opt_time('normalization/standardization'):.4f}s]")
    
    return X_norm, Y_stand

def denormalize_val(candidates: torch.Tensor, bounds: torch.Tensor, config: OptimizationConfig) -> torch.Tensor:
    '''function to denormalize data to the original bounds and handle modelsize generated'''
    cand_denormalized = unnormalize(candidates, bounds).tolist()
    for n,r in enumerate(cand_denormalized):
        for par_n, par in enumerate(config.optimization_parameters):
            if par == 'MODEL_SIZE':
                cand_denormalized[n][par_n] = str(handle_modelsize(r[par_n]))
    
    return cand_denormalized

def handle_modelsize(n: float):
    '''function to transform the modelsize value generated into the relative string'''
    if n <= 3700506.0:      #small
        return 'small'
    elif n <= 9853386.0:    #medium
        return 'medium'
    else:                   #large
        return 'large'