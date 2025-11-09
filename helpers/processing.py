'''     Data processing before/after train_model/bo_loop execution      '''
import torch
from botorch.utils.transforms import normalize, standardize, unnormalize
from .find_bounds import bounds     #for now we have a file that define manually bounds/bounds_norm

def normalize_val(x: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor):
    '''function to normalize data bounds in [0,1]'''
    print(f"    -> Normalizing inputs")
    X_norm = normalize(x, bounds)
    if y.dim() == 2 and y.shape[1] > 1:
        Y_stand = [standardize(y[:, i:i+1]) for i in range(y.shape[1])]
    else:
        Y_stand = standardize(y)
    return X_norm, Y_stand

def denormalize_val(c: torch.Tensor) -> torch.Tensor:
    '''function to denormalize bounds in the original ones'''
    print(f"    -> Denormalizing outputs")
    return unnormalize(c, bounds)