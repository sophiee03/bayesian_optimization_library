import logging
from .config import OptimizationConfig, Timer, OPTIMIZERS, BoundsGenerator
from botorch.optim import optimize_acqf, optimize_acqf_cyclic
from botorch.optim.initializers import gen_batch_initial_conditions
from .acquisition import generate_acqf

logger = logging.getLogger('BO')
timer = Timer(logger)

@timer.time_function('optimize_acqf')
def basic_optim(acqf, bm: BoundsGenerator, dim: int, conf: OptimizationConfig):
    '''optimizing parameters with optimize_acqf (BoTorch basic)'''
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
    '''optimizing parameters with optimize_acqf_cyclic '''
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
    '''optimizing parameters with optimize_acqf only after generating a batch of initial conditions for random-restart optimization'''
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

def get_candidates(config: OptimizationConfig, model, X, Y, bm: BoundsGenerator):
    ''' generate the acquisition function and find candidates with their acq_values
        returns the candidates with their acq_val and time needed for optimization
    '''
    with timer.measure('acquisition_function'):
        acqf = generate_acqf(config, model,X, Y)
    
    if config.verbose:
        logger.info(f"   -> Created acquisition function            [{timer.get_opt_time('acquisition_function'):.4f}s]")

    if config.optimizers == OPTIMIZERS[0]:
        result = basic_optim(acqf, bm, X.shape[1], config)
        time = timer.get_opt_time('optimize_acqf')

    elif config.optimizers == OPTIMIZERS[1]:
        result = batch_init_cond_optim(acqf, bm, X.shape[1], config)
        time = timer.get_opt_time('batch_initial_condition')

    elif config.optimizers == OPTIMIZERS[2]:
        result = cyclic_optim(acqf, bm, X.shape[1], config)
        time = timer.get_opt_time('optimize_acqf_cyclic')

    if config.verbose:
        logger.info("   -> Optimization completed")

    return result, time