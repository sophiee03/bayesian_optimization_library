from dataclasses import dataclass, fields
from enum import Enum
from typing import List, Dict, Callable
from functools import wraps
import torch, logging, time
from contextlib import contextmanager

PRECISIONS = {
    'param_lr': '.6f',
    'param_epochs': '.1f',
    'param_batch_size': '.1f',
    'param_seed': '.1f',
    'ACC_val': '.6f',
    'cpu_usage': '.4f',
    'disk_usage': '.4f',
    'memory_usage': '.4f',
    'MSE_train': '.6f',
    'MSE_val': '.6f',
    #_________________________
    'EPOCHS': '.1f',
    'LR': '.6f',
    'BATCH_SIZE': '.1f',
    'DROPOUT_RATE': '.2f',
    'MODEL_SIZE': '',
    'emissions': '.6f',
    'accuracy': '.6f',
    'power_consumption': '.6f'
}

METRICS = {
    'ACC_val': 'MAX',
    'cpu_usage': 'MIN',
    'MSE_train': 'MIN', 
    'MSE_val': 'MIN',
    'disk_usage': 'MIN', 
    'memory_usage': 'MIN',
    #_________________________
    'emissions': 'MIN',
    'accuracy': 'MAX',
    'power_consumption': 'MIN'
}

VALID_PARAMETERS = ['param_lr', 'param_epochs', 'param_batch_size', 'param_seed', 'EPOCHS', 'BATCH_SIZE', 'MODEL_SIZE', 'DROPOUT_RATE', 'LR']

OPTIMIZERS = ['optimize_acqf', 'batch_init_cond', 'optimize_acqf_cyclic', 'optimize_acqf optimize_acqf_cyclic batch_init_cond']

class Objective(Enum):
    SINGLE = "SINGLE"
    MULTI = "MULTI"

@dataclass
class  OptimizationConfig:
    '''
    Configuration choices:
        - obj_metrics -> list of metrics to maximize (str)
        - optimization_parameters -> list of parameters to optimize
        - objective -> number of parameters we want to maximize/minimize
        - n_candidates -> number of candidates the bo_loop will return
        - n_restarts -> number of restarts for the optimization routine
        - raw_samples -> number of random samples when initializing the optimization
        - optimizers -> choice of the optimizer(s) to use
        - multi_model -> choice between modellistgp and kroneckermultitaskgp
        - verbose
    '''
    objective_metrics: List[str]
    optimization_parameters: List[str]
    objective: Objective = Objective.SINGLE
    n_candidates: int = 1
    n_restarts: int = 10
    raw_samples: int = 500
    optimizers: str = 'optimize_acqf optimize_acqf_cyclic batch_init_cond'
    multi_model: str = 'modellistgp'
    verbose: bool = True

    def __post_init__(self):
        '''validate configuration after initialization'''
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                setattr(self, f.name, f.default)

        if self.objective not in Objective:
            raise ValueError(f"objective must be single or multi")
        
        if self.n_candidates < 1:
            raise ValueError(f"n_candidates must be at least 1")
        
        if self.n_restarts < 1:
            raise ValueError(f"iterations must be at least 1")
        
        if self.raw_samples < 1:
            raise ValueError(f"raw_samples must at least be 1 \n    (it is suggested to have at least 2*d samples where d is the number of parameters)")
        
        if self.optimizers is None or self.optimizers not in OPTIMIZERS:
            raise ValueError(f"optimizer not recognized: {self.optimizers}")

    def _objective(self) -> Objective:
        '''returns how many tasks we are executing'''
        return self.objective
    
    def _details(self):
        '''print a summary of the configuration choices adopted'''
        print(f"{'-'*60}")
        print(f"CONFIGURATION DETAILS:")
        print(f"    executing a {self.objective} model with:")
        print(f"    - Parameters to optimize: {self.optimization_parameters}")
        print(f"    - Metrics to maximize/minimize: {self.objective_metrics}")
        print(f"    {self.n_candidates} candidates are required")
        print(f"    It will be used a {self.optimizers} optimizer(s) with the following parameters:")
        print(f"    -> n_restarts: {self.n_restarts}    raw_samples: {self.raw_samples}")
        print(f"{'-'*60}")
    
@dataclass
class BoundsGenerator():
    '''configuration for parameter bounds needed in normalization'''
    margin: float = 0.02
    force_positive: bool = True

    def generate_bounds(self, t: torch.Tensor):
        '''generate the bounds based on the min and max value of the parameter data'''
        lower = t.min(dim=0)[0]
        upper = t.max(dim=0)[0]

        range = (upper - lower)*self.margin

        lower = lower-range
        upper = upper+range

        # excludes negative bounds
        if self.force_positive:
            lower = torch.clamp(lower, min=0.0)

        return torch.stack([lower, upper])
    
    def generate_norm_bounds(self, n: int):
        '''generate bounds normalized [0.0][1.0]'''
        return torch.tensor([   [0.0]*n,
                                [1.0]*n], dtype=torch.float64)
    
class Timer:
    '''register and confront timings'''
    def __init__(self, logger: logging.Logger = None):
        self.timings: Dict[str, float] = {}
        self.logger = logger or logging.getLogger(__name__)

    @contextmanager
    def measure(self, name: str):
        '''context manager to measure a specific block of code'''
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._record_timings(name, elapsed)

    def _record_timings(self, name: str, elapsed: float):
        '''register timings measured'''
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name] = elapsed

    def time_function(self, name: str = None):
        '''register the time that a function takes to execute'''
        def decorator(func: Callable) -> Callable:
            optimizer_name = name or func.__name__
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.perf_counter() - start
                    self._record_timings(optimizer_name, elapsed)
            return wrapper
        return decorator

    def get_opt_time(self, name: str) -> float:
        '''obtain only the time measure required'''
        if name not in self.timings.keys():
            raise ValueError(f"Time not found for {name} optimizer")
        return self.timings[name]

    def print_summary(self):
        '''print a summary of each execution time recorded'''
        sorted_opt = sorted(
            self.timings.keys(),
            key=lambda x: sum(self.timings[x]),
            reverse=True
        )
        self.logger.info("=" * 60)
        for name in sorted_opt:
            self.logger.info(f"total elapsed time for {name} optimization:       {self.timings[name].sum():.4f}")
        self.logger.info("=" * 60)

    def reset(self):
        '''clear all times registered'''
        self.timings.clear()

