from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from tabulate import tabulate
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
        candidates (List[List]): matrix of candidates denormalized
        acq_value (List[float]): acquisition values recorded
        time (float): elapsed time for the entire routine
        posterior (botorch.posteriors.Posterior): BoTorch posterior object with predictions (mean and variance)
    """
    candidates: List[List]
    acq_values: List[float]
    time: float
    posterior: Any

class BayesianOptimizer:
    """class to perform Bayesian Optimization routine
    data processing -> train model -> generate acqf -> optimize -> denormalize -> make predictions on results -> return OptimizationResults

    Attributes:
        config (OptimizationConfig): configuration settings
        logger (Logger): to log info
        timer (Timer): to measure elapsed time
        bounds_manager (BoundsGenerator): to generate bounds
        X_data (Tensor): parameters used as training data 
        Y_data (Tensor): metrics used as training data (already transformed for minimization)
        X_norm (Tensor): parameters normalized 
        model (SingleTaskGP/ModelListGP): model instance
        original_bounds (Tensor): initial bounds of parameters
        posterior (botorch.posteriors.Posterior): BoTorch posterior object with predictions (mean and variance)
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
        self.X_data = None
        self.Y_data = None
        self.X_norm = None
        self.model = None
        self.original_bounds = None
        self.posterior = None

    def change_config(self, new_conf: OptimizationConfig):
        """change configuration settings
        
        Args:
            new_conf (OptimizationConfig): new configuration to adopt
        """
        self.config = new_conf
        print(f"Successfully changed BayesianOptimizer configuration: \n {self.config.details()}")

    def prepare_data(self, data: Dict):
        """process and normalize data with bounds generated
        
        Args: 
            data (Dict): dictionary containing parameters and metrics
        """
        data['metrics'] = minimization_transformation(data['metrics'], self.config)

        if self.config.verbose:
            self.logger.info('   -> Data transformed')

        self.X_data = torch.tensor(data['parameters'], dtype=torch.float64)
        self.Y_data = torch.tensor(data['metrics'], dtype=torch.float64)

        self.original_bounds = self.bounds_manager.generate_bounds(self.X_data).to(dtype=torch.float64)

        if self.config.verbose:
            self.logger.info('   -> Bounds generated')

        self.X_norm = normalize_val(self.X_data, self.original_bounds)

        if self.config.verbose:
            self.logger.info('   -> Data normalized')

    def model_training(self):
        """train the model with the normalized data"""
        if self.X_norm is None:
            raise RuntimeError("You must load and prepare data before training the model!")
        
        self.model = train_model(self.config, self.X_norm, self.Y_data)

        if self.config.verbose:
            self.logger.info('   -> Model trained')

    def optimize(self):
        """optimization performed by acquisition function and denormalization of results
        
        Returns:
            candidates_normalized (Tensor): candidates normalized in [0,1]
            candidates_denormalized (Tensor): candidates in the original bounds
            acq_values (List[float]): acquisition values recorded
        """
        if self.model == None:
            raise RuntimeError("You must train the model before performing optimization!")
        
        candidates_norm, val = get_candidates(self.config, self.model, self.X_norm, self.Y_data, self.bounds_manager)

        if self.config.verbose:
            self.logger.info('   -> Candidates obtained')

        candidates_denorm = denormalize_val(candidates_norm, self.original_bounds)
        if self.config.verbose:
            self.logger.info('   -> Candidates denormalized')

        acq_values = val if isinstance(val, list) else val.tolist()
        
        return candidates_norm, candidates_denorm, acq_values
    
    def estimate(self, candidates: torch.Tensor):
        """generate the posterior of the candidates generated

        Args:
            candidates (Tensor): candidates normalized generated
        """
        if not isinstance(candidates, torch.Tensor):
            candidates = torch.tensor(candidates, dtype=torch.float64)
        with torch.no_grad():
            self.posterior = self.model.posterior(candidates)
    
    def print_estimations(self, mean, std):
        """visualize the estimations for the candidates generated
        
        Args:
            mean (Tensor)
            std (Tensor)
        """
        mean_copy = mean.clone()
        # the metrics to minimize will be negative so we have to change sign 
        for j, goal in enumerate(self.config.goal):
            if goal == 'MIN':
                mean_copy[:, j] *= -1

        for i in range(self.config.n_candidates):
            print(f'{'='*60}')
            print(f'CANDIDATE {i+1}')
            print(f'{'='*60}')
            table = []
            for j, metric in enumerate(self.config.objective_metrics):
                table.append([metric, mean_copy[i][j], std[i][j]])
            print(tabulate(table, headers=['METRIC', 'MEAN', 'STD'], tablefmt='simple_grid', floatfmt='.6f'), '\n')
    
    def run(self, data: Dict) -> OptimizationResults:
        """Run all the Bayesian Optimization pipeline and return final results
        
        Args:
            data (Dict): data provided by the user as training dataset

        Returns:
            OptimizationResults: instance with candidates, acq_values, time and posterior generated by the optimization
        """
        with self.timer.measure('tot_optimization'):
            if self.config.verbose:
                self.logger.info('   -> Starting Bayesian Optimization')
            self.prepare_data(data)
            self.model_training()
            norm_candidates, denorm_candidates, acq_value = self.optimize()

        if self.config.verbose:
            self.logger.info(f'   -> Bayesian Optimization finished, took {round(self.timer.get_opt_time('tot_optimization'), 3)}s')
            visualize_data(denorm_candidates, self.config.optimization_parameters)
            self.logger.info(f'   -> Estimating candidates')
        
        self.estimate(norm_candidates)

        if self.config.verbose:
            self.print_estimations(
                self.posterior.mean, 
                self.posterior.variance.sqrt()
            )

        return OptimizationResults(denorm_candidates, acq_value, self.timer.get_opt_time('tot_optimization'), self.posterior)


