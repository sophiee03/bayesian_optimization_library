from dataclasses import dataclass
from typing import List, Dict
import torch
from .config import OptimizationConfig, Timer, BoundsGenerator
from .helpers.logger import setup_console_logger
from .data.data_loader import load_data
from .helpers.processing import normalize_val, denormalize_val
from .models import train_model
from .optimizer import get_candidates

@dataclass
class OptimizationResults:
    '''dataclass for Bayesian Optimization results'''
    candidates: List[List]
    acq_values: List[float]
    time: float

class BayesianOptimizer:
    '''class to perform Bayesian Optimization routine
    data loading -> normalization -> train model -> generate acqf -> optimize -> denormalize -> return candidates
    '''
    def __init__(self, config: OptimizationConfig):
        '''initialize optimization'''
        self.config = config
        self.logger = setup_console_logger()
        self.timer = Timer(self.logger)
        self.bounds_manager = BoundsGenerator()
        self.X_data, self.Y_data = None, None
        self.X_norm, self.Y_stand = None, None
        self.model = None
        self.original_bounds = None

    def get_and_prepare_data(self, folder, data_needed):
        '''load data and normalize/standardize them with bounds generated'''
        self.X_data, self.Y_data = load_data(self.config, folder, data_needed)
        self.original_bounds = self.bounds_manager.generate_bounds(self.X_data).to(dtype=torch.float64)
        self.X_normalized, self.Y_standardized = normalize_val(self.config, self.X_data, self.Y_data, self.original_bounds)

    def model_training(self):
        '''train the model with the normalized data'''
        if self.X_normalized is None or self.Y_standardized is None:
            raise RuntimeError("You must load and prepare data before training the model!")
        self.model = train_model(self.config, self.X_normalized, self.Y_standardized)

    def optimize(self):
        '''optimization performed by acquisition function and denormalization of results'''
        if self.model == None:
            raise RuntimeError("You must train the model before performing optimization!")
        
        (candidates, val), opt_time = get_candidates(self.config, self.model, self.X_normalized, self.Y_standardized, self.bounds_manager)
        
        candidates_denormalized = denormalize_val(candidates, self.original_bounds, self.config)
        acq_values = val if isinstance(val, list) else val.tolist()
        
        results = OptimizationResults(candidates_denormalized, acq_values, opt_time)
        return results
    
    def run(self, folder: str, data_needed: Dict = 
            {'input': ['DROPOUT', 'BATCH_SIZE', 'EPOCHS', 'LR', 'MODEL_SIZE'], 
            'output': ['accuracy', 'emissions']}) -> OptimizationResults:
        '''run all the Bayesian Optimization pipeline and return final results'''
        self.get_and_prepare_data(folder, data_needed)
        self.model_training()
        results = self.optimize()
        return results

