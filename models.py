'''     model train routine     '''

import torch
from typing import Dict, List
from helpers.utils import set_objective
from helpers.processing import normalize_val
from helpers.config import Objective
from helpers.find_bounds import bounds  #for now we have a file that define manually bounds/bounds_norm
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll

def train_model(obj: Dict, data: List[Dict]):
    '''train the model with the required nnumber of tasks, given the input data from the etl module'''
    print(f"    -> Training the model")
    maximize, minimize, objective = set_objective(obj)
    X_names, Y_names, X_val, Y_val = []
    for l in data:
        for n,v in l.items():
            if n in maximize:
                Y_names.append(n)
                Y_val.append(v)     #for maximization
            elif n in minimize:
                Y_names.append(n)
                Y_val.append(-v)    #for minimization
            else:
                X_names.append(n)
                X_val.append(v)

        X_train = torch.tensor(X_val, dtype=torch.float64)
        Y_train = torch.tensor(Y_val, dtype=torch.float64)

        X_normalized, Y_standardized = normalize_val(X_train, Y_train, bounds)
    
    if objective == Objective.SINGLE:
        model = SingleTaskGP(X_normalized, Y_standardized)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    elif objective == Objective.MULTI:
        gps = [SingleTaskGP(X_normalized, y) for y in Y_standardized]
        model = ModelListGP(*gps)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

    return model, X_normalized, X_names, torch.cat(Y_standardized, dim=1), objective
