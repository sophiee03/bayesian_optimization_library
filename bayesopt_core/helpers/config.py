from dataclasses import dataclass, fields
from enum import Enum
from typing import List, Dict, Callable
from functools import wraps
import torch, logging, time
from contextlib import contextmanager

class Type(Enum):
    METRIC = 'metric',
    PARAMETER = 'parameter'

ATTRIBUTES = {
    'EPOCHS': (Type.PARAMETER, '.0f'),
    'LR': (Type.PARAMETER, '.6f'),
    'BATCH_SIZE': (Type.PARAMETER, '.0f'),
    'DROPOUT': (Type.PARAMETER, '.2f'),
    'MODEL_SIZE': (Type.PARAMETER, ''),
    'emissions': (Type.METRIC, '.6f', 'MIN'),
    'accuracy': (Type.METRIC, '.6f', 'MAX'),
    'cpu_energy': (Type.METRIC, '.4f', 'MIN'),
    'cpu_power': (Type.METRIC, '.4f', 'MIN'),
    'emissions_rate': (Type.METRIC, '.4f', 'MIN'),
    'energy_consumed': (Type.METRIC, '.4f', 'MIN'),
    'gpu_energy': (Type.METRIC, '.4f', 'MIN'),
    'gpu_power': (Type.METRIC, '.4f', 'MIN'),
    'ram_energy': (Type.METRIC, '.4f', 'MIN'),
    'ram_power': (Type.METRIC, '.4f', 'MIN')
}

OPTIMIZERS = ['optimize_acqf', 'batch_init_cond', 'optimize_acqf_cyclic']
ACQF = ['qlogei', 'qlognei', 'ucb', 'qlogehvi']

class Objective(Enum):
    SINGLE = "SINGLE"
    MULTI = "MULTI"

@dataclass
class  OptimizationConfig:
    '''Configuration choices:
        - objective_metrics       = list of metrics to maximize
        - optimization_parameters = list of parameters to optimize
        - objective               = number of metrics we want to maximize/minimize
        - n_candidates            = number of candidates the bo_loop will return
        - n_restarts              = number of restarts for the optimization routine
        - raw_samples             = number of random samples when initializing the optimization
        - optimizers              = choice of the optimizer to use
        - acqf                    = choice of the acquisition function to use
        - verbose                 = flag to follow BO workflow
        - default                 = flag to adopt default condifguration
    '''
    objective_metrics: List[str]
    optimization_parameters: List[str]
    objective: Objective
    n_candidates: int = 1
    n_restarts: int = 10
    raw_samples: int = 500
    optimizers: str = OPTIMIZERS[0]
    acqf: str = ACQF[2]
    beta: float = 1.0
    verbose: bool = False
    default: bool = False

    def __post_init__(self):
        '''validate configuration after initialization'''
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                setattr(self, f.name, f.default)
        
        if self.n_candidates < 1:
            raise ValueError(f"n_candidates must be at least 1")
        
        if self.n_restarts < 1:
            raise ValueError(f"iterations must be at least 1")
        
        if self.raw_samples < 1:
            raise ValueError(f"raw_samples must at least be 1 \n    (it is suggested to have at least 2*d samples where d is the number of parameters)")

    def _details(self):
        '''print a summary of the configuration choices'''
        print(f"{'-'*60}")
        print(f"CONFIGURATION DETAILS:")
        print(f"    executing a {self.objective} model with:")
        print(f"    - Parameters to optimize: {self.optimization_parameters}")
        print(f"    - Metrics to maximize/minimize: {self.objective_metrics}")
        print(f"    {self.n_candidates} candidates are required")
        print(f"    It will be used a {self.optimizers} optimizer(s) with the following parameters:")
        print(f"    -> n_restarts: {self.n_restarts}    raw_samples: {self.raw_samples}")
        print(f"{'-'*60}")
    
    def return_dict(self):
        '''generate a dictionary of the configuration attributes'''
        d = {
            'objective_metrics': self.objective_metrics,
            'optimization_parameters': self.optimization_parameters,
            'objective': 'SINGLE' if self.objective == Objective.SINGLE else 'MULTI',
            'n_candidates': self.n_candidates,
            'n_restarts': self.n_restarts,
            'raw_samples': self.raw_samples,
            'optimizers': self.optimizers,
            'acqf': self.acqf,
            'verbose': self.verbose,
            'default': self.default
        }
        return d

@dataclass
class BoundsGenerator():
    '''class to create parameter bounds needed in normalization'''
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
    '''class to register and observe timings'''
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
        '''obtain only the measured time required'''
        if name not in self.timings.keys():
            raise ValueError(f"Time not found for {name} optimizer")
        return self.timings[name]

    def reset(self):
        '''clear all times registered'''
        self.timings.clear()