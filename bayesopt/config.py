from dataclasses import dataclass, fields
from typing import List, Dict, Callable
from functools import wraps
import torch, logging, time
from contextlib import contextmanager

OPTIMIZERS = ["optimize_acqf", "batch_init_cond", "optimize_acqf_cyclic"]
ACQF = ["qlogei", "qlognei", "ucb", "qlogehvi"]

@dataclass
class  OptimizationConfig:
    """class to define the configuration choices for the optimization
    
    Attributes:
        objective_metrics (List): metrics to maximize
        optimization_parameters (List): parameters to optimize
        goal (List[str]): objective (MAX/MIN) of each metric
        ground_truth_dim (int): input dim
        n_candidates (int = 1): number of candidates the routine will produce
        n_restarts (int = 10): number of restarts for the optimization routine
        raw_samples (int = 200): number of random samples when initializing the optimization
        optimizers (str = 'optimize_acqf'): choice of the optimizer to use
        acqf (str = 'ucb'): choice of the acquisition function to use
        beta (float = 1.0): attribute to balance exploration-exploitation
        verbose (bool = False): flag to follow BO workflow
    """
    objective_metrics: List[str]
    optimization_parameters: List[str]
    goal: List[str]
    ground_truth_dim: int
    n_candidates: int = 1
    n_restarts: int = 10
    raw_samples: int = 200
    optimizers: str = OPTIMIZERS[0]
    acqf: str = ACQF[2]
    beta: float = 1.0
    verbose: bool = False

    def __post_init__(self):
        """validate configuration after initialization"""
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                setattr(self, f.name, f.default)
        
        if self.n_candidates < 1:
            raise ValueError(f"n_candidates must be at least 1")
        
        if self.n_restarts < 1:
            raise ValueError(f"n_restarts must at least be 1")
        
        if self.raw_samples < 1:
            raise ValueError(f"raw_samples must at least be 1 (it is suggested to have at least 2*d samples where d is the number of parameters)")

        if self.optimizers not in OPTIMIZERS:
            raise ValueError(f"optimizer not recognized")
        
        if self.acqf not in ACQF:
            raise ValueError(f"acquisition function not recognized")
        
        if len(self.goal) != len(self.objective_metrics):
            raise ValueError(f"each metric must have a goal (max or min)")
        
    def details(self):
        """print a summary of the configuration choices"""
        print(f"{'-'*60}")
        print(f"CONFIGURATION DETAILS:")
        print(f"    executing a {'SINGLE' if len(self.objective_metrics)==1 else 'MULTI'} model with:")
        print(f"    - Parameters to optimize: {self.optimization_parameters}")
        print(f"    - Metrics to maximize/minimize: {self.objective_metrics}")
        print(f"    {self.n_candidates} candidates are required to be generated, with a ground truth of {self.ground_truth_dim} runs")
        print(f"    It will be used a {self.acqf} acquisition function \nwith {self.optimizers} as optimizer setup with the following settings:")
        print(f"    -> n_restarts: {self.n_restarts}    raw_samples: {self.raw_samples}")
        print(f"{'-'*60}")
    
    def return_dict(self):
        """generate a dictionary of the configuration attributes"""
        d = {
            'objective_metrics': self.objective_metrics,
            'optimization_parameters': self.optimization_parameters,
            'goal': self.goal,
            'ground_truth_dim': self.ground_truth_dim,
            'n_candidates': self.n_candidates,
            'n_restarts': self.n_restarts,
            'raw_samples': self.raw_samples,
            'optimizers': self.optimizers,
            'acqf': self.acqf,
            'beta': self.beta,
            'verbose': self.verbose
        }
        return d

@dataclass
class BoundsGenerator():
    """class to create parameter bounds needed in normalization
    Attributes:
        margin (float): used to add a range to the max and min value found of the parameters
        force_positive (bool): flag to force the bounds to be strictly positive
    """
    margin: float = 0.02
    force_positive: bool = True

    def generate_bounds(self, t: torch.Tensor):
        """generate the bounds based on the min and max value of the parameter

        Args:
            t (Tensor): parameters data
        """
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
        """generate bounds normalized [[0.0][1.0]]*n

        Args:
            n (int): number of parameters used
        """
        return torch.tensor([   [0.0]*n,
                                [1.0]*n], dtype=torch.float64)
    
class Timer:
    """class to register and observe timings"""
    def __init__(self, logger: logging.Logger = None):
        self.timings: Dict[str, float] = {}
        self.logger = logger or logging.getLogger(__name__)

    @contextmanager
    def measure(self, name: str):
        """context manager to measure a specific block of code
        
        Args:
            name (str): saving name
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._record_timings(name, elapsed)

    def _record_timings(self, name: str, elapsed: float):
        """register timings measured
        
        Args:
            name (str): saving name
            elapsed (float): time recorded
        """
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name] = elapsed

    def time_function(self, name: str = None):
        """register the time that a function takes to execute
        
        Args:
            name (str): saving name
        """
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
        """obtain only the measured time required
        
        Args:
            name (str): saved name
        """
        if name not in self.timings.keys():
            raise ValueError(f"Time not found for {name} optimizer")
        return self.timings[name]

    def reset(self):
        """clear all times registered"""
        self.timings.clear()