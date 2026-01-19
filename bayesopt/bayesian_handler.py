from dataclasses import dataclass
from typing import List, Dict, Any
import torch, os, json
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
    def __init__(self, config: OptimizationConfig, bounds: List = None, save_dir: str=None):
        """initialize optimization
        
        Args:
            config (OptimizationConfig): configuration of the BayesianOptimization
            bounds (List[List]): optional bounds passed as input to define the parameters domain
            save_dir (str): directory where the library will save the logs
        """
        self.config = config
        self.logger = setup_console_logger()
        self.timer = Timer(self.logger)
        self.bounds_manager = BoundsGenerator()
        self.X_data = None
        self.Y_data = None
        self.X_norm = None
        self.model = None
        self.original_bounds = bounds
        self.posterior = None
        self.save_dir = save_dir

    def change_config(self, new_conf: OptimizationConfig):
        """change configuration settings
        
        Args:
            new_conf (OptimizationConfig): new configuration to adopt
        """
        self.config = new_conf

    def change_bounds(self, new_bounds: List[List]):
        """change bounds
        
        Args: 
            new_bounds (List[List]): new bounds to update the original ones
        """
        self.original_bounds = torch.tensor(new_bounds, dtype=torch.float64)

    def prepare_data(self, data: Dict):
        """process and normalize data with bounds generated
        
        Args: 
            data (Dict): dictionary containing parameters and metrics
        """
        data['metrics'] = minimization_transformation(data['metrics'], self.config)

        if self.config.verbose:
            self.logger.info("   -> Data transformed")

        self.X_data = torch.tensor(data['parameters'], dtype=torch.float64)
        self.Y_data = torch.tensor(data['metrics'], dtype=torch.float64)

        if self.original_bounds is None:
            self.original_bounds = self.bounds_manager.generate_bounds(self.X_data).to(dtype=torch.float64)

        if self.config.verbose:
            self.logger.info("   -> Bounds generated")

        self.X_norm = normalize_val(self.X_data, self.original_bounds)

        if self.config.verbose:
            self.logger.info("   -> Data normalized")

    def model_training(self):
        """train the model with the normalized data"""
        if self.X_norm is None:
            raise RuntimeError("You must load and prepare data before training the model!")
        
        self.model = train_model(self.config, self.X_norm, self.Y_data)

        if self.config.verbose:
            self.logger.info("   -> Model trained")

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
            self.logger.info("   -> Candidates obtained")

        candidates_denorm = denormalize_val(candidates_norm, self.original_bounds)
        if self.config.verbose:
            self.logger.info("   -> Candidates denormalized")

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
            if goal == "MIN":
                mean_copy[:, j] *= -1

        for i in range(self.config.n_candidates):
            print(f"{'='*60}")
            print(f"CANDIDATE {i+1}")
            print(f"{'='*60}")
            table = []
            for j, metric in enumerate(self.config.objective_metrics):
                table.append([metric, mean_copy[i][j], std[i][j]])
            print(tabulate(table, headers=['METRIC', 'MEAN', 'STD'], tablefmt='simple_grid', floatfmt='.6f'), '\n')

    def generate_json(self, res: OptimizationResults):
        json_path = self.save_dir
        os.makedirs(json_path, exist_ok=True)
        json_file = f'{json_path}/{self.config.ground_truth_dim}_{self.config.n_candidates}_{self.config.beta}_logs.json'

        with open(json_file, 'w') as f:
            json.dump({
                'config': self.config.return_dict(),
                'X_ground_truth': self.X_data.tolist(),
                'Y_ground_truth': minimization_transformation(self.Y_data, self.config).tolist(),
                'X_norm': self.X_norm.tolist(),
                'bounds': self.original_bounds.tolist(),
                'results': {
                    'candidates': res.candidates,
                    'acq_values': res.acq_values,
                    'posterior': {
                        'mean': minimization_transformation(res.posterior.mean, self.config).tolist(),
                        'variance': res.posterior.variance.sqrt().tolist()
                    },
                    'elapsed_time': float(res.time),
                },
            }, f, indent=2)
        if self.config.verbose == True:
            print(f"JSON generated: {json_file}")

    def update_training_set(self, new_data: Dict[str, List]):
        """method to update the ground truth data with the new ones passed
        
        Args:
            new_data (Dict): dictionary containing parameters and metrics values 
        """
        new_x = torch.tensor(new_data['parameters'], dtype=torch.float64)
        new_y = torch.tensor(minimization_transformation(new_data['metrics'], self.config), dtype=torch.float64)
        self.X_data = torch.cat([self.X_data, new_x])
        self.Y_data = torch.cat([self.Y_data, new_y])
        self.config.ground_truth_dim = len(self.X_data)
        self.X_norm = normalize_val(self.X_data, self.original_bounds)
    
    def run(self, data: Dict = None) -> OptimizationResults:
        """Run all the Bayesian Optimization pipeline and return final results
        
        Args:
            data (Dict): optional data provided by the user as training dataset 
            (if data are not already saved in the BayesianOptimizer instance)

        Returns:
            OptimizationResults: instance with candidates, acq_values, time and posterior generated by the optimization
        """
        with self.timer.measure("tot_optimization"):
            if self.config.verbose:
                self.logger.info("   -> Starting Bayesian Optimization")
            if data is not None:
                self.prepare_data(data)
            elif self.X_data is None or self.Y_data is None or self.X_norm is None:
                raise ValueError("You have to provide data if they are not already saved in the BayesianOptimizer instance")
            self.model_training()
            norm_candidates, denorm_candidates, acq_value = self.optimize()

        if self.config.verbose:
            self.logger.info(f"   -> Bayesian Optimization finished, took {round(self.timer.get_opt_time('tot_optimization'), 3)}s")
            visualize_data(denorm_candidates, self.config.optimization_parameters)
            self.logger.info(f"   -> Estimating candidates")
        
        self.estimate(norm_candidates)

        if self.config.verbose:
            self.print_estimations(
                self.posterior.mean, 
                self.posterior.variance.sqrt()
            )
        
        result = OptimizationResults(denorm_candidates, acq_value, self.timer.get_opt_time("tot_optimization"), self.posterior)
        
        if self.save_dir != None:
            self.generate_json(result)

        return result


