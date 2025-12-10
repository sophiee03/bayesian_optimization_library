import torch
from .config import OptimizationConfig, ACQF, Objective
from botorch.acquisition import qLogExpectedImprovement, qLogNoisyExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.objective import ScalarizedPosteriorTransform

def generate_acqf(config: OptimizationConfig, model, X: torch.Tensor, Y: torch.Tensor):
    '''function to generate the acquisition function for single or multi objective
        - qlogei = MC-based batch log expected improvement
        - qlognei = MC-based batch noisy expected improvement
        - qlogehvi = parallel log expected hypervolume improvement supporting 2+ outcomes
        - qucb = MC-based batch upper confidence bound
    '''
    if config.objective == Objective.SINGLE:
        acq = qLogExpectedImprovement(
            model=model, 
            best_f=Y.max()
        )
    else:
        if config.acqf == ACQF[3]: 
            ref_point = Y.min(dim=0)[0] - 0.1 * (Y.max(dim=0)[0] - Y.min(dim=0)[0])
            partitioning = FastNondominatedPartitioning(ref_point, Y)
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
                best_f=Y.max(),
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