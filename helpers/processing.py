import torch
from botorch.utils.transforms import normalize, standardize, unnormalize

def normalize_val(x: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor):
    '''function to normalize data'''
    print(f"    -> Normalizing and standardizing")
    #normalize data to have the same scale for each parameter
    X_norm = normalize(x, bounds)
    #standardize metrics 
    if y.dim() >= 2 and y.shape[1] > 1:
        Y_stand=[]
        for i in range(y.shape[1]):
            y_col = y[:,i:i+1]
            #check if there are NaN values
            if torch.isnan(y_col).any():
                raise ValueError(f"Output column {i} contains NaN in original data!")
            mean = y_col.mean(dim=0)
            std = y_col.std(dim=0)
            std[std==0] = 1.0
            Y_stand.append((y_col - mean) / std)
    else:
        Y_stand = standardize(y)
    return X_norm, Y_stand

def denormalize_val(t: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    '''function to denormalize values to the original bounds'''
    print(f"    -> Denormalizing outputs")
    return unnormalize(t, bounds)