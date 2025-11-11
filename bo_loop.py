import torch
from helpers.config import Objective, OptimizationConfig
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.optim import optimize_acqf
from helpers.config import BoundsGenerator

def bo_loop(conf: OptimizationConfig, model, dim: int, Y: torch.Tensor, bm: BoundsGenerator):
    '''bayesian optimization routine that provides the n candidates required with its acquisition_value'''
    if conf.objective == Objective.SINGLE:
        acq = qLogExpectedImprovement(
            model = model,
            best_f = Y.max()
        )
    else:
        ref_point = Y.min(dim=0)[0] - 0.1 * (Y.max(dim=0)[0] - Y.min(dim=0)[0])
        partitioning = FastNondominatedPartitioning(ref_point, Y)
        ref_point = ref_point.tolist()
        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning
        )

    candidates, acq_value = optimize_acqf(
        acq,
        bounds=bm.generate_norm_bounds(dim),
        q = conf.n_candidates,
        num_restarts=conf.n_restarts,
        raw_samples=conf.raw_samples
    )
    return candidates, acq_value