from dataclasses import dataclass
from typing import List, Dict
import torch
from .config import OptimizationConfig, Timer, BoundsGenerator
from .helpers.logger import setup_console_logger
from .helpers.processing import normalize_val, denormalize_val, minimization_transformation
from .models import train_model
from .optimizer import get_candidates
from .helpers.visualization import visualize_data

@dataclass
class OptimizationResults:
    """dataclass for Bayesian Optimization results
    
    Attributes:
        candidates (List[List]): matrix of candidates
        acq_value (List[float]): acquisition values recorded
        time (float): elapsed time for teh entire routine
    """
    candidates: List[List]
    acq_values: List[float]
    time: float

class BayesianOptimizer:
    """class to perform Bayesian Optimization routine
    data processing -> train model -> generate acqf -> optimize -> denormalize -> return candidates

    Attributes:
        config (OptimizationConfig): configuration settings
        logger (Logger): to log info
        timer (Timer): to measure elapsed time
        bounds_manager (BoundsGenerator): to generate bounds
        X_data (Tensor): parameters used as training data 
        Y_data (Tensor): metrics used as training data (already transformed for minimization)
        X_norm (Tensor): parameters normalized 
        Y_stand (Tensor): metrics standardized
        model (SingleTaskGP/ModelListGP): model instance
        original_bounds (Tensor): initial bounds of parameters
    """
    def __init__(self, config: OptimizationConfig):
        """initialize optimization
        
        Args:
            config (OptimizationConfig): configuration of the BayesianOptimization
        """
        self.config = config
        self.logger = setup_console_logger()
        self.timer = Timer(self.logger)
        self.bounds_manager = BoundsGenerator()
        self.X_data, self.Y_data = None, None
        self.X_norm, self.Y_stand = None, None
        self.model = None
        self.original_bounds = None

    def get_and_prepare_data(self, data: Dict):
        """process and normalize/standardize data with bounds generated
        
        Args: 
            data (Dict): dictionary containing parameters (names and values) and metrics (name, goal and values)
        """
        transformed_data = minimization_transformation(data)

        if self.config.verbose:
            self.logger.info('   -> Data transformed')

        self.X_data = torch.tensor(transformed_data['parameters'][1], dtype=torch.float64)
        self.Y_data = torch.tensor(transformed_data['metrics'][2], dtype=torch.float64)

        self.original_bounds = self.bounds_manager.generate_bounds(self.X_data).to(dtype=torch.float64)

        if self.config.verbose:
            self.logger.info('   -> Bounds generated')

        self.X_normalized, self.Y_standardized = normalize_val(self.X_data, self.Y_data, self.original_bounds)

        if self.config.verbose:
            self.logger.info('   -> Data normalized and standardized')

    def model_training(self):
        """train the model with the normalized data"""
        if self.X_normalized is None or self.Y_standardized is None:
            raise RuntimeError("You must load and prepare data before training the model!")
        self.model = train_model(self.config, self.X_normalized, self.Y_standardized)

        if self.config.verbose:
            self.logger.info('   -> Model trained')

    def optimize(self):
        """optimization performed by acquisition function and denormalization of results"""
        if self.model == None:
            raise RuntimeError("You must train the model before performing optimization!")
        
        candidates, val = get_candidates(self.config, self.model, self.X_normalized, self.Y_standardized, self.bounds_manager)

        if self.config.verbose:
            self.logger.info('   -> Candidates obtained')

        candidates_denormalized = denormalize_val(candidates, self.original_bounds, self.config)
        if self.config.verbose:
            self.logger.info('   -> Candidates denormalized')

        acq_values = val if isinstance(val, list) else val.tolist()
        
        return candidates_denormalized, acq_values
    
    def run(self, data: Dict) -> OptimizationResults:
        """Run all the Bayesian Optimization pipeline and return final results
        
        Args:
            data (Dict): data provided by the user as training dataset

        Returns:
            OptimizationResults: instance with candidates, acq_values and time required for optimization
        """
        with self.timer.measure('tot_optimization'):
            if self.config.verbose:
                self.logger.info('   -> Starting Bayesian Optimization')
            self.get_and_prepare_data(data)
            self.model_training()
            candidates, acq_value = self.optimize()

        if self.config.verbose:
            self.logger.info(f'   -> Bayesian Optimization finished, took {round(self.timer.get_opt_time('tot_optimization'), 3)}s')
            visualize_data(candidates, self.config.optimization_parameters)

        return OptimizationResults(candidates, acq_value, self.timer.get_opt_time('tot_optimization'))


