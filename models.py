import torch
from helpers.config import Objective
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll

def train_model(objective: Objective, X_normalized: torch.Tensor, Y_standardized: torch.Tensor):
    '''train the model with the required number of tasks'''
    print(f"    -> Training the model")    
    if objective == Objective.SINGLE:
        model = SingleTaskGP(X_normalized, Y_standardized)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    elif objective == Objective.MULTI:
        gps = [SingleTaskGP(X_normalized, y) for y in Y_standardized]
        model = ModelListGP(*gps)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

    return model
