from dataclasses import dataclass
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
    'MSE_val': '.6f'
}

METRICS = {
    'ACC_val': 'MAX',
    'cpu_usage': 'MIN',
    'MSE_train': 'MIN', 
    'MSE_val': 'MIN',
    'disk_usage': 'MIN', 
    'memory_usage': 'MIN'
}

VALID_PARAMETERS = {'param_lr', 'param_epochs', 'param_batch_size', 'param_seed'}

class Objective(Enum):
    SINGLE = "SINGLE"
    MULTI = "MULTI"

@dataclass
class  OptimizationConfig:
    '''
    Configuration choices:
        - obj_metrics -> list of metrics to maximize (str)
        - optimization_parameters -> list of parameters to optimize
        - objective -> number of parameters we want to maximize/minimize             (default = 1)
        - n_candidates -> number of candidates the bo_loop will return               (default = 1)
        - n_restarts -> number of restarts for the optimization routine              (default = 10)
        - raw_samples -> number of random samples when initializing the optimization (default = 1000)
        - verbose
    '''
    objective_metrics: List[str]
    optimization_parameters: List[str]
    objective: Objective = Objective.SINGLE
    n_candidates: int = 1
    n_restarts: int = 10
    raw_samples: int = 1000
    verbose: bool = True

    def __post_init__(self):
        '''validate configuration after initialization'''
        if self.objective not in Objective:
            raise ValueError(f"objective must be single or multi")
        
        if self.n_candidates < 1:
            raise ValueError(f"n_candidates must be at least 1")
        
        if self.n_restarts < 1:
            raise ValueError(f"iterations must be at least 1")
        
        if self.raw_samples < 1:
            raise ValueError(f"raw_samples must at least be 1 \n    (it is suggested to have at least 2*d samples where d is the number of parameters)")
        
    def _objective(self) -> Objective:
        '''returns how many tasks we are executing'''
        return self.objective
    
@dataclass
class BoundsGenerator():
    '''configuration for parameter bounds needed in normalization'''
    margin: float = 0.02
    force_positive: bool = True

    def generate_bounds(self, t: torch.Tensor):
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
        self.timings.clear()

