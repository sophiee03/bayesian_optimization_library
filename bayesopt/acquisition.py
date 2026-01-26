import torch
from .config import OptimizationConfig, ACQF
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound, qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.objective import ScalarizedPosteriorTransform

def generate_acqf(config: OptimizationConfig, model, X: torch.Tensor, Y_std: torch.Tensor):
    """Function to generate the acquisition function
    If the objective is SINGLE the choice is automatically set to LogExpectedImprovement. 
    If the objective is MULTI it allows to choose among the following BoTorch acquisition functions:
        - qLogEI = MC-based batch log expected improvement
        - qLogNEI = MC-based batch noisy expected improvement
        - qLogEHVI = parallel log expected hypervolume improvement supporting 2+ outcomes
        - qUCB = MC-based batch upper confidence bound

    Args:
        config (OptimizationConfig): configuration of the BayesianOptimization (needed for function choice and settings)
        model (SingleTaskGP/ModelListGP): model trained
        X (Tensor): training parameters normalized
        Y (Tensor): training metrics standardized

    Returns:
        AcquisitionFunction: instance of the acquisition function generated
    """
    if len(config.objective_metrics) == 1:
        acq = LogExpectedImprovement(
            model=model, 
            best_f=Y_std.max()
        )
    else:
        if config.acqf == ACQF[3]:
            ref_point = Y_std.min(dim=0)[0] - 0.1 * (Y_std.max(dim=0)[0] - Y_std.min(dim=0)[0])
            partitioning = FastNondominatedPartitioning(ref_point, Y_std)
            ref_point = ref_point.tolist()
            acq = qLogExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                partitioning=partitioning
            )
        elif config.acqf == ACQF[0]:
            weights = torch.ones(len(config.objective_metrics), dtype=torch.float64) / len(config.objective_metrics)
            posterior_transform = ScalarizedPosteriorTransform(weights=weights)
            acq = qLogExpectedImprovement(
                model=model, 
                best_f=Y_std.max(),
                posterior_transform=posterior_transform
            )
        elif config.acqf == ACQF[1]:
            weights = torch.ones(len(config.objective_metrics), dtype=torch.float64) / len(config.objective_metrics)
            posterior_transform = ScalarizedPosteriorTransform(weights=weights)
            acq = qLogNoisyExpectedImprovement(
                model,
                X_baseline=X,
                posterior_transform=posterior_transform
            )
        elif config.acqf == ACQF[2]:
            weights = torch.ones(len(config.objective_metrics), dtype=torch.float64) / len(config.objective_metrics)
            posterior_transform = ScalarizedPosteriorTransform(weights=weights)
            acq = qUpperConfidenceBound(
                model=model,
                beta=config.beta,
                posterior_transform=posterior_transform
            )
    return acq