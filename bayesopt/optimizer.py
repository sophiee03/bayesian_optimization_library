from .config import OptimizationConfig, OPTIMIZERS, BoundsGenerator
from botorch.optim import optimize_acqf, optimize_acqf_cyclic
from botorch.optim.initializers import gen_batch_initial_conditions
from .acquisition import generate_acqf
import torch

def basic_optim(acqf, bm: BoundsGenerator, config: OptimizationConfig):
    """Function to perform optimization with optimize_acqf
    
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
    """Function to perform optimization with optimize_acqf_cyclic

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
    """Function to perform optimization with an initial condition and optimize_acqf
        
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
    """Function that generate the acquisition function and call the optimizer chosen to obtain candidates
    
    Args: 
        config (OptimizationConfig): configuration of the BayesianOptimization (needed for optimizers settings)
        model (SingleTaskGP/ModelListGP): model trained with training dataset
        X (Tensor): training parameters normalized
        Y (Tensor): training metrics
        bm (BoundsGenerator): instance to generate normalized bounds for the parameters
    
    Returns: 
        Tuple: candidates generated and its acquisition values
    """

    if len(config.objective_metrics) == 1:
        Y_std, _ = model.outcome_transform(Y)
    else:
        std_list = []
        for i, m in enumerate(model.models):
            yi = Y[:,i].unsqueeze(-1)
            yistd, _ = m.outcome_transform(yi)
            std_list.append(yistd)

        Y_std = torch.cat(std_list, dim=-1)

    acqf = generate_acqf(config, model, X, Y_std)
    
    if config.optimizers == OPTIMIZERS[0]:
        result = basic_optim(acqf, bm, config)

    elif config.optimizers == OPTIMIZERS[1]:
        result = batch_init_cond_optim(acqf, bm, config)

    elif config.optimizers == OPTIMIZERS[2]:
        result = cyclic_optim(acqf, bm, config)

    return result