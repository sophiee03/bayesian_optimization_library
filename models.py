import torch, logging
from helpers.config import OptimizationConfig, Timer, Objective
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.multitask import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll

def train_model(config: OptimizationConfig, X_normalized: torch.Tensor, Y_standardized: torch.Tensor):
    '''train the model with the required number of tasks'''
    logger = logging.getLogger('BO')
    timer = Timer(logger)

    with timer.measure('model_training'):
        if config.objective == Objective.SINGLE:
            model = SingleTaskGP(X_normalized, Y_standardized)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
        elif config.objective == Objective.MULTI:
            if config.multi_model == 'kroneckermultitaskgp':
                model = KroneckerMultiTaskGP(X_normalized, Y_standardized)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
            else:
                gps = [SingleTaskGP(X_normalized, Y_standardized[:,y].unsqueeze(-1)) for y in range(Y_standardized.shape[1])]
                model = ModelListGP(*gps)
                mll = SumMarginalLogLikelihood(model.likelihood, model)
        
        fit_gpytorch_mll(mll)

    logger.info(f"   -> Model created and trained               [{timer.get_opt_time('model_training'):.4f}s]")
    return model
