from dataclasses import dataclass
from enum import Enum
from typing import List
import torch

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
        - verbose S
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
    margin: float = 0.1

    def generate_bounds(self, t: torch.Tensor):
        lower = t.min(dim=0)[0]
        upper = t.max(dim=0)[0]

        range = (upper - lower)*self.margin
        lower = lower-range
        upper = upper+range
        return torch.stack([lower, upper])
    
    def generate_norm_bounds(self, n: int):
        return torch.tensor([   [0.0]*n,
                                [1.0]*n], dtype=torch.float64)
    
