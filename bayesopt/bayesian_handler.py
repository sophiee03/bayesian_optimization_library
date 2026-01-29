from dataclasses import dataclass
from typing import List, Dict, Any
import torch, os, json
from datetime import datetime
from tabulate import tabulate
import yprov4ml
from .config import OptimizationConfig, Timer, BoundsGenerator
from .helpers.processing import normalize_val, denormalize_val, minimization_transformation
from .models import train_model
from .optimizer import get_candidates
from .helpers.visualization import visualize_data

@dataclass
class OptimizationResults:
    """Dataclass for Bayesian Optimization results
    
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
    """Class to perform Bayesian Optimization routine

    Attributes:
        config (OptimizationConfig): configuration settings
        timer (Timer): to measure elapsed time
        bounds_manager (BoundsGenerator): to generate bounds
        X_data (Tensor): parameters used as training data 
        Y_data (Tensor): metrics used as training data (already transformed for minimization)
        X_norm (Tensor): parameters normalized 
        model (SingleTaskGP/ModelListGP): model instance
        original_bounds (Tensor): initial bounds of parameters
        posterior (botorch.posteriors.Posterior): BoTorch posterior object with predictions (mean and variance)
    """
    def __init__(self, config: OptimizationConfig, bounds: List = None, save_dir: str = "."):
        """Initialize optimization
        
        Args:
            config (OptimizationConfig): configuration of the BayesianOptimization
            bounds (List[List]): optional bounds passed as input to define the parameters domain
            save_dir (str): optional folder in which the logs will be saved
        """
        self.config = config
        self.timer = Timer()
        self.bounds_manager = BoundsGenerator()
        self.X_data = None
        self.Y_data = None
        self.X_norm = None
        self.model = None
        self.mll =None
        self.original_bounds = bounds
        self.posterior = None
        self.save_dir = save_dir

    def change_config(self, n_cand = None, n_restarts = None, raw_samples = None, optimizers = None, acqf = None, beta = None, verbose = None):
        """Method to change configuration settings (only for certain attributes is possible otherwise the optimization is not consistent)
        
        Args:
            n_candidates (int)
            n_restarts (int)
            raw_samples (int)
            optimizers (str)
            acqf (str)
            beta (float)
            verbose (bool)
        """

        new_conf = OptimizationConfig(
            objective_metrics=self.config.objective_metrics,
            optimization_parameters=self.config.optimization_parameters,
            goal=self.config.goal,
            ground_truth_dim=self.config.ground_truth_dim,
            n_candidates=n_cand,
            n_restarts=n_restarts,
            raw_samples=raw_samples,
            optimizers=optimizers,
            acqf=acqf,
            beta=beta,
            verbose=verbose
        )
        self.config = new_conf

    def change_bounds(self, new_bounds: List[List]):
        """Method to change bounds
        
        Args: 
            new_bounds (List[List]): new bounds
        """
        self.original_bounds = torch.tensor(new_bounds, dtype=torch.float64)

    def prepare_data(self, data: Dict):
        """Method to process metrics based on their direction and normalize parameters with their bounds
        
        Args: 
            data (Dict): dictionary containing parameters and metrics
        """
        data['metrics'] = minimization_transformation(data['metrics'], self.config)

        if self.config.verbose:
            print("   -> Data transformed")

        self.X_data = torch.tensor(data['parameters'], dtype=torch.float64)
        self.Y_data = torch.tensor(data['metrics'], dtype=torch.float64)

        if self.original_bounds is None:
            self.original_bounds = self.bounds_manager.generate_bounds(self.X_data).to(dtype=torch.float64)

        if self.config.verbose:
            print("   -> Bounds generated")

        self.X_norm = normalize_val(self.X_data, self.original_bounds)

        if self.config.verbose:
            print("   -> Data normalized")

    def model_training(self):
        """Method to train the model with the normalized data"""
        if self.X_norm is None:
            raise RuntimeError("You must load and prepare data before training the model!")
        
        self.model, self.mll = train_model(self.config, self.X_norm, self.Y_data)

        if self.config.verbose:
            print("   -> Model trained")

    def optimize(self):
        """Method to perform optimization by acquisition function and denormalization of results
        
        Returns:
            candidates_normalized (Tensor): candidates normalized in [0,1]
            candidates_denormalized (Tensor): candidates in the original bounds
            acq_values (List[float]): acquisition values recorded
        """
        if self.model == None:
            raise RuntimeError("You must train the model before performing optimization!")
        
        candidates_norm, val = get_candidates(self.config, self.model, self.X_norm, self.Y_data, self.bounds_manager)

        if self.config.verbose:
            print("   -> Candidates obtained")

        candidates_denorm = denormalize_val(candidates_norm, self.original_bounds)
        if self.config.verbose:
            print("   -> Candidates denormalized")

        acq_values = val if isinstance(val, list) else val.tolist()
        
        return candidates_norm, candidates_denorm, acq_values
    
    def estimate(self, candidates: torch.Tensor):
        """Method to generate the posterior of the candidates generated

        Args:
            candidates (Tensor): candidates normalized generated
        """
        if not isinstance(candidates, torch.Tensor):
            candidates = torch.tensor(candidates, dtype=torch.float64)
        with torch.no_grad():
            self.posterior = self.model.posterior(candidates)
    
    def print_estimations(self, mean, std):
        """Method to visualize the estimations for the candidates generated
        
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

    def logs_dict(self):
        """Method to get a dictionary with the information to log"""
        return {
            'config': self.config.return_dict(),
            'X_ground_truth': self.X_data.tolist(),
            'Y_ground_truth': minimization_transformation(self.Y_data, self.config).tolist(),
            'X_norm': self.X_norm.tolist(),
            'bounds': self.original_bounds.tolist(),
        }

    def log_experiment(self, res: OptimizationResults):
        """Method to log runtime informations into a JSON

        Args: 
            res (OptimizationResults): instance of optimization results obtained
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_file = f'data_{timestamp}_logs.json'
        json_path = f"{self.save_dir}/{json_file}"

        with open(json_path, 'w') as f:
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
            print(f"JSON generated: {json_path}")
        
        yprov4ml.log_artifact(f"{json_file}", json_path, is_input=False, log_copy_in_prov_directory=True)

    def update_training_set(self, new_data: Dict[str, List]):
        """Method to append the new data passed to the training set
        
        Args:
            new_data (Dict): dictionary containing parameters and metrics values 
        """
        new_x = torch.tensor(new_data['parameters'], dtype=torch.float64)
        new_y = torch.tensor(minimization_transformation(new_data['metrics'], self.config), dtype=torch.float64)
        self.X_data = torch.cat([self.X_data, new_x])
        self.Y_data = torch.cat([self.Y_data, new_y])
        self.config.ground_truth_dim = len(self.X_data)
        self.X_norm = normalize_val(self.X_data, self.original_bounds)
    
    def run(self, data: Dict = None, exp_name: str = None) -> OptimizationResults:
        """Method to run all the Bayesian Optimization pipeline and return final results
        
        Args:
            data (Dict): optional data provided by the user as training dataset 
            (if data are not already saved in the BayesianOptimizer instance)
            exp_name (str): optional name to call the experimenti (otherwise will be created with date and time)

        Returns:
            OptimizationResults: instance with candidates, acq_values, time and posterior generated by the optimization
        """
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if exp_name == None:
            exp_name = f"data_{timestamp}_logs.json"

        yprov4ml.start_run(
            prov_user_namespace = "www.example.org",
            experiment_name = exp_name,  
            provenance_save_dir = f"{self.save_dir}/prov",
            save_after_n_logs = 100,
            collect_all_processes = True, 
            disable_codecarbon = True, 
        )

        with self.timer.measure("tot_optimization"):
            if self.config.verbose:
                print("   -> Starting Bayesian Optimization")
            if data is not None:
                self.prepare_data(data)
            elif self.X_data is None or self.Y_data is None or self.X_norm is None:
                raise ValueError("You have to provide data if they are not already saved in the BayesianOptimizer instance")
            self.model_training()
            norm_candidates, denorm_candidates, acq_value = self.optimize()

        if self.config.verbose:
            print(f"   -> Bayesian Optimization finished, took {round(self.timer.get_opt_time('tot_optimization'), 3)}s")
            visualize_data(denorm_candidates, self.config.optimization_parameters)
            print(f"   -> Estimating candidates")
        
        self.estimate(norm_candidates)

        if self.config.verbose:
            self.print_estimations(
                self.posterior.mean, 
                self.posterior.variance.sqrt()
            )
        
        result = OptimizationResults(denorm_candidates, acq_value, self.timer.get_opt_time("tot_optimization"), self.posterior)
        
        self.log_experiment(result)

        yprov4ml.end_run(
            create_graph=False, 
            create_svg=False, 
            crate_ro_crate=False
        )

        return result


