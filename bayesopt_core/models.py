import torch
from .config import OptimizationConfig, Objective
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll

def train_model(config: OptimizationConfig, X_normalized: torch.Tensor, Y_data: torch.Tensor):
    """function to create and train the model

    Args:
        config: configuration of the BayesianOptimization (needed for model choice)
        X_normalized (Tensor): training parameters normalized
        Y_data (Tensor): training metrics

    Returns:
        (SingleTaskGP/ModelListGP):instance of the model created and trained
    """ 
    if config.objective == Objective.SINGLE:
        model = SingleTaskGP(X_normalized, Y_data, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    elif config.objective == Objective.MULTI:
        gps = [SingleTaskGP(X_normalized, Y_data[:,y].unsqueeze(-1), outcome_transform=Standardize(m=1)) for y in range(Y_data.shape[1])]
        model = ModelListGP(*gps)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    
    fit_gpytorch_mll(mll)

    return model
