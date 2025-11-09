'''     loop to perform bayesian optimization with Botorch      '''

import torch
from helpers.config import Objective, OptimizationConfig
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.optim import optimize_acqf
from helpers.find_bounds import bounds, norm_bounds   #for now we have a file that define manually bounds/bounds_norm
from helpers.processing import denormalize_val

def bo_loop(conf: OptimizationConfig, model, x_names, Y: torch.Tensor):
    '''bayesian optimization routine that provides the number of candidates required with its acquisition_value'''
    print(f"    -> Starting BAYESIAN OPTIMIZATION with {Y.dim()} objectives")
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
        norm_bounds,
        q = conf.n_candidates,
        num_restarts=conf.n_restarts,
        raw_samples=conf.raw_samples
    )

    candidates_denorm = denormalize_val(candidates)
    c = candidates_denorm.tolist()
    print(f'='*60)
    print(f"    Candidates found:   [acq_val = {acq_value:.4f}]")
    print(f"                    {x_names}")
    for i,c in enumerate(c):
        print(f"        Candidate {i+1}:        {c[0]:.1f},         {c[1]:.1f},         {c[2]:.4f}")

    return candidates_denorm, acq_value