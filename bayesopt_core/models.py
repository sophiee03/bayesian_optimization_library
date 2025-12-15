import torch
from .config import OptimizationConfig, Objective
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll

def train_model(config: OptimizationConfig, X_normalized: torch.Tensor, Y_standardized: torch.Tensor):
    """function to create and train the model

    Args:
        config: configuration of the BayesianOptimization (needed for model choice)
        X_normalized (Tensor): training parameters normalized
        Y_standardized (Tensor): training metrics standardized

    Returns:
        (SingleTaskGP/ModelListGP):instance of the model created and trained
    """ 
    if config.objective == Objective.SINGLE:
        model = SingleTaskGP(X_normalized, Y_standardized)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    elif config.objective == Objective.MULTI:
        gps = [SingleTaskGP(X_normalized, Y_standardized[:,y].unsqueeze(-1)) for y in range(Y_standardized.shape[1])]
        model = ModelListGP(*gps)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    
    fit_gpytorch_mll(mll)

    return model
