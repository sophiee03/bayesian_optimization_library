import torch, logging
from helpers.config import Objective, OptimizationConfig, Timer, OPTIMIZERS, BoundsGenerator
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.optim import optimize_acqf, optimize_acqf_cyclic
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.acquisition.objective import ScalarizedPosteriorTransform

logger = logging.getLogger('BO')
timer = Timer(logger)

def generate_acqf(config: OptimizationConfig, model, X: torch.Tensor, Y: torch.Tensor):
    '''function to generate the acquisition function for single or multi objective'''
    if config.objective == Objective.SINGLE:
        acq = qLogExpectedImprovement(
            model=model, 
            best_f=Y.max()
        )
    else:
    # FOR QLogEHVI
    #    ref_point = Y.min(dim=0)[0] - 0.1 * (Y.max(dim=0)[0] - Y.min(dim=0)[0])
    #    partitioning = FastNondominatedPartitioning(ref_point, Y)
    #    ref_point = ref_point.tolist()
    #    acq = qLogExpectedHypervolumeImprovement(
    #        model=model,
    #        ref_point=ref_point,
    #        partitioning=partitioning
    #    )
    # FOR QLogEI
        weights = torch.ones(len(config.objective_metrics), dtype=torch.float64) / len(config.objective_metrics)
        posterior_transform = ScalarizedPosteriorTransform(weights=weights)
        acq = qLogExpectedImprovement(
            model=model, 
            best_f=Y.max(),
            posterior_transform=posterior_transform
        )
    # FOR UCB
    #    weights = torch.ones(len(config.objective_metrics), dtype=torch.float64) / len(config.objective_metrics)
    #    posterior_transform = ScalarizedPosteriorTransform(weights=weights)
    #    acq = qUpperConfidenceBound(
    #        model=model,
    #        beta=2,
    #        posterior_transform=posterior_transform
    #    )
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

def bo_loop(config: OptimizationConfig, model, X, Y, bm: BoundsGenerator):
    '''
        generate the acquisition function and find candidates with their acq_values
        returns:
            Dict{
                'optimize_acqf': (candidate, acq_value), time,
                'optimize_acqf_cyclic': (candidate, acq_value), time,
                'batch_init_cond': (candidate, acq_value), time
            }
    '''
    with timer.measure('acquisition_function'):
        acqf = generate_acqf(config, model,X, Y)
    
    if config.verbose:
        logger.info(f"   -> Created acquisition function            [{timer.get_opt_time('acquisition_function'):.4f}s]")

    if config.optimizers == OPTIMIZERS[3]:
        result_basic = basic_optim(acqf, bm, X.shape[1], config)
        time_basic = timer.get_opt_time('optimize_acqf')
        
        result_cyclic = cyclic_optim(acqf, bm, X.shape[1], config)
        time_cyclic = timer.get_opt_time('optimize_acqf_cyclic')
        
        result_batch = batch_init_cond_optim(acqf, bm, X.shape[1], config)
        time_batch = timer.get_opt_time('batch_initial_condition')

    elif config.optimizers == OPTIMIZERS[0]:
        result_basic = basic_optim(acqf, bm, X.shape[1], config)
        time_basic = timer.get_opt_time('optimize_acqf')
        result_cyclic, time_cyclic, result_batch, time_batch = (None, None, None, None)

    elif config.optimizers == OPTIMIZERS[1]:
        result_batch = batch_init_cond_optim(acqf, bm, X.shape[1], config)
        time_batch = timer.get_opt_time('batch_initial_condition')
        result_cyclic, time_cyclic, result_basic, time_basic = (None, None, None, None)

    elif config.optimizers == OPTIMIZERS[2]:
        result_cyclic = cyclic_optim(acqf, bm, X.shape[1], config)
        time_cyclic = timer.get_opt_time('optimize_acqf_cyclic')
        result_basic, time_basic, result_batch, time_batch = (None, None, None, None)

    results = {
        'optimize_acqf': (result_basic, time_basic),
        'optimize_acqf_cyclic': (result_cyclic, time_cyclic),
        'batch_init_cond': (result_batch, time_batch)
    }

    if config.verbose:
        logger.info("   -> Optimization completed")

    return results