from .config import OptimizationConfig, OPTIMIZERS, BoundsGenerator
from botorch.optim import optimize_acqf, optimize_acqf_cyclic
from botorch.optim.initializers import gen_batch_initial_conditions
from .acquisition import generate_acqf

def basic_optim(acqf, bm: BoundsGenerator, config: OptimizationConfig):
    """function to optimize the acquisition function generated with optimize_acqf
    
    Args: 
        acqf: acquisition function generated
        bm: instance of BoundsGenerator to generate normalized bounds for the parameters
        config: configuration of the BayesianOptimization (needed for optimizers settings) 
    
    Returns: 
        candidates (Tensor): candidates generated
        acq_values (float | List(float)): acquisition values
    """   
    candidates, acq_value = optimize_acqf(
        acqf,
        bounds=bm.generate_norm_bounds(len(config.optimization_parameters)),
        q = config.n_candidates,
        num_restarts=config.n_restarts,
        raw_samples=config.raw_samples
    )
    return candidates, acq_value

def cyclic_optim(acqf, bm: BoundsGenerator, config: OptimizationConfig):
    """function to optimize the acquisition function generated with optimize_acqf_cyclic

    Args: 
        acqf: acquisition function generated
        bm: instance of BoundsGenerator to generate normalized bounds for the parameters
        config: configuration of the BayesianOptimization (needed for optimizers settings) 
    
    Returns: 
        candidates (Tensor): candidates generated
        acq_values (float | List(float)): acquisition values
    """   
    candidates, acq_value = optimize_acqf_cyclic(
        acqf,
        bounds=bm.generate_norm_bounds(len(config.optimization_parameters)),
        q = config.n_candidates,
        num_restarts=config.n_restarts,
        raw_samples=config.raw_samples,
        options={"batch_limit": 5, "maxiter": 200}
    )
    return candidates, acq_value

def batch_init_cond_optim(acqf, bm: BoundsGenerator, config: OptimizationConfig):
    """function to optimize the acquisition function generated with an initial condition
        
    Args: 
        acqf (AcquisitionFunction): acquisition function generated
        bm (BoundsGenerator): instance of BoundsGenerator to generate normalized bounds for the parameters
        config (OptimizationConfigs): configuration of the BayesianOptimization (needed for optimizers settings) 
    
    Returns: 
        candidates (Tensor): candidates generated
        acq_values (float | List(float)): acquisition values
    """    
    batch_initial_conditions = gen_batch_initial_conditions(
        acqf,
        bounds=bm.generate_norm_bounds(len(config.optimization_parameters)),
        q = config.n_candidates,
        num_restarts= config.n_restarts,
        raw_samples= config.raw_samples,
        options={
            "seed": 42,
            "init_batch_limit": 32,
            "batch_limit": 5,
        }
    )
    candidates, acq_value = optimize_acqf(
        acqf,
        bounds=bm.generate_norm_bounds(len(config.optimization_parameters)),
        q = config.n_candidates,
        num_restarts=config.n_restarts,
        raw_samples=config.raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={"batch_limit": 5, "maxiter": 200}
    )
    return candidates, acq_value

def get_candidates(config: OptimizationConfig, model, X, Y, bm: BoundsGenerator):
    """function that generate the acquisition function and find candidates
    
    Args: 
        config (OptimizationConfig): configuration of the BayesianOptimization (needed for optimizers settings)
        model (SingleTaskGP/ModelListGP): model trained with training dataset
        X (Tensor): training parameters normalized
        Y (Tensor): training metrics standardized
        bm (BoundsGenerator): instance to generate normalized bounds for the parameters
    
    Returns: 
        (candidates (List[List]), acq_values (List[float])): candidates generated and its acquisition values
    """
    acqf = generate_acqf(config, model,X, Y)
    
    if config.optimizers == OPTIMIZERS[0]:
        result = basic_optim(acqf, bm, config)

    elif config.optimizers == OPTIMIZERS[1]:
        result = batch_init_cond_optim(acqf, bm, config)

    elif config.optimizers == OPTIMIZERS[2]:
        result = cyclic_optim(acqf, bm, config)

    return result