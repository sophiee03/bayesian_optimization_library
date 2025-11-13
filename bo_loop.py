import torch, logging
from helpers.config import Objective, OptimizationConfig, Timer
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.optim import optimize_acqf, optimize_acqf_cyclic
from botorch.optim.initializers import gen_batch_initial_conditions
from helpers.config import BoundsGenerator

logger = logging.getLogger('BO')
timer = Timer(logger)

def generate_acqf(conf: OptimizationConfig, model, Y: torch.Tensor):
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
    return acq

@timer.time_function('optimize_acqf')
def basic_optim(acqf, bm: BoundsGenerator, dim: int, conf: OptimizationConfig):
    candidates, acq_value = optimize_acqf(
        acqf,
        bounds=bm.generate_norm_bounds(dim),
        q = conf.n_candidates,
        num_restarts=conf.n_restarts,
        raw_samples=conf.raw_samples
    )
    return candidates, acq_value

@timer.time_function('optimize_acqf_cyclic')
def cyclic_optim(acqf, bm: BoundsGenerator, dim: int, conf: OptimizationConfig):
    candidates, acq_value = optimize_acqf_cyclic(
        acqf,
        bounds=bm.generate_norm_bounds(dim),
        q = conf.n_candidates,
        num_restarts=conf.n_restarts,
        raw_samples=conf.raw_samples,
        options={"batch_limit": 5, "maxiter": 200}
    )
    return candidates, acq_value

@timer.time_function('batch_initial_condition')
def batch_init_cond_optim(acqf, bm: BoundsGenerator, dim: int, conf: OptimizationConfig):
    batch_initial_conditions = gen_batch_initial_conditions(
        acqf,
        bounds=bm.generate_norm_bounds(dim),
        q = conf.n_candidates,
        num_restarts= conf.n_restarts,
        raw_samples= conf.raw_samples,
        options={
            "seed": 42,
            "init_batch_limit": 32,
            "batch_limit": 5,
        }
    )
    candidates, acq_value = optimize_acqf(
        acqf,
        bounds=bm.generate_norm_bounds(dim),
        q = conf.n_candidates,
        num_restarts=conf.n_restarts,
        raw_samples=conf.raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={"batch_limit": 5, "maxiter": 200}
    )
    return candidates, acq_value

def bo_loop(config: OptimizationConfig, model, dim: int, Y, bm: BoundsGenerator):
    with timer.measure('acquisition_function'):
        acqf = generate_acqf(config, model, Y)
    
    logger.info(f"   -> Created acquisition function [{timer.get_opt_time('acquisition_function'):.4f}s]")

    result_basic = basic_optim(acqf, bm, dim, config)
    time_basic = timer.get_opt_time('optimize_acqf')
    
    result_cyclic = cyclic_optim(acqf, bm, dim, config)
    time_cyclic = timer.get_opt_time('optimize_acqf_cyclic')
    
    result_batch = batch_init_cond_optim(acqf, bm, dim, config)
    time_batch = timer.get_opt_time('batch_initial_condition')

    results = {
        'optimize_acqf': (result_basic, time_basic),
        'optimize_acqf_cyclic': (result_cyclic, time_cyclic),
        'batch_initial_condition': (result_batch, time_batch)
    }

    logger.info("   -> Optimization completed")

    return results
    