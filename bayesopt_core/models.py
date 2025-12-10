import torch, logging
from .helpers.config import OptimizationConfig, Timer, Objective
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll

def train_model(config: OptimizationConfig, X_normalized: torch.Tensor, Y_standardized: torch.Tensor):
    '''train the model with the normalized training set'''
    logger = logging.getLogger('BO')
    timer = Timer(logger)

    with timer.measure('model_training'):
        if config.objective == Objective.SINGLE:
            model = SingleTaskGP(X_normalized, Y_standardized)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
        elif config.objective == Objective.MULTI:
            gps = [SingleTaskGP(X_normalized, Y_standardized[:,y].unsqueeze(-1)) for y in range(Y_standardized.shape[1])]
            model = ModelListGP(*gps)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
        
        fit_gpytorch_mll(mll)

    if config.verbose:
        logger.info(f"   -> Model created and trained               [{timer.get_opt_time('model_training'):.4f}s]")
    return model
