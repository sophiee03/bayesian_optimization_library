from dataclasses import dataclass
from enum import Enum
import torch

class Maximization:
    MAXIMIZE = ['ACC_val'],
    MINIMIZE = ['cpu_usage', 'MSE_train', 'MSE_val', 'disk_usage', 'memory_usage']

class Objective(Enum):
    SINGLE = "SINGLE"
    MULTI = "MULTI"

@dataclass
class  OptimizationConfig:
    '''Configuration choices:
        - objective -> number of parameters we want to maximize/minimize             (default = 1)
        - n_candidates -> number of candidates the bo_loop will return               (default = 1)
        - n_restarts -> number of restarts for the optimization routine              (default = 10)
        - raw_samples -> number of random samples when initializing the optimization (default = 1000)
    '''
    objective: Objective = Objective.SINGLE
    n_candidates: int = 1
    n_restarts: int = 10
    raw_samples: int = 1000

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
        #find max and min for each parameter
        lower = t.min(dim=0)[0]
        upper = t.max(dim=0)[0]
        #computes a margin to add to the bounds
        range = (upper - lower)*self.margin
        lower = torch.tensor(lower-range, dtype=torch.float64)
        upper = torch.tensor(upper+range, dtype=torch.float64)
        return torch.stack([lower, upper])
    
    def generate_norm_bounds(self, n: int):
        return torch.tensor([[0.0]*n,
                        [1.0]*n], dtype=torch.float64)
    
