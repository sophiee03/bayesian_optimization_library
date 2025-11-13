import torch
import logging
from botorch.utils.transforms import normalize, standardize, unnormalize

def normalize_val(x: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor):
    '''function to normalize data'''
    logger = logging.getLogger('BO')
    X_norm = normalize(x, bounds)
    
    if y.dim() >= 2 and y.shape[1] > 1:
        Y_stand=torch.zeros_like(y)
        for i in range(y.shape[1]):
            y_col = y[:,i:i+1]
            if torch.isnan(y_col).any():
                raise ValueError(f"Output column {i} contains NaN in extracted data!")
            mean = y_col.mean(dim=0)
            std = y_col.std(dim=0)
            std[std==0] = 1.0
            Y_stand[:,i] = ((y_col - mean) / std).squeeze(-1)
    else:
        Y_stand = (standardize(y))
        if Y_stand.dim == 1:
            Y_stand = Y_stand.unsqueeze(-1)
    logger.info("   -> Data normalized and standardized")
    return X_norm, Y_stand

def denormalize_val(t: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    '''function to denormalize data to the original bounds'''
    if not isinstance(bounds, torch.Tensor):
        raise ValueError(f"Bounds Problem")
    
    return unnormalize(t, bounds)