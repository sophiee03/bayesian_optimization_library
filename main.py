from bo_loop import bo_loop
import traceback, warnings, torch, argparse
import matplotlib.pyplot as plt
from helpers.logger import setup_console_logger
from models import train_model
from helpers.config import OptimizationConfig,BoundsGenerator, METRICS, VALID_PARAMETERS, OPTIMIZERS
from data_loader import load_data
from helpers.plot import plot_data
from helpers.utils import visualize_data
from helpers.processing import normalize_val, denormalize_val

def parse_arg():
    parser = argparse.ArgumentParser(description = "Bayesian Optimization with BoTorch")
    parser.add_argument('--folder', type=str, default='./prov', 
                        help="Folder to extract training data")
    parser.add_argument('--output', nargs='+', type=str, choices=list(METRICS.keys()),
                        help="Metrics to maximize/minimize")
    parser.add_argument('--input', nargs='+', type=str, choices=list(VALID_PARAMETERS),
                        help="Parameters to optimize")
    parser.add_argument('--multi_model', type=str, choices=['modellistgp', 'kroneckermultitaskgp'],
                        help="type of model")
    parser.add_argument('--n_candidates', type=int,
                        help="number of candidates to generate")
    parser.add_argument('--n_restarts', type=int, 
                        help="Number of restarts of the optimizer")
    parser.add_argument('--raw_samples', type=int, 
                        help="number of raw samples to generate before choosing the candidates")
    parser.add_argument('--optimizer', choices=list(OPTIMIZERS), 
                        help="optimizer to choose the candidates")
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()

def main(args):
    logger = setup_console_logger()

    metrics = args.output
    parameters = args.input

    # ETL Module
    data_needed = {
        'output': metrics,
        'input': parameters
    }
    X_data, Y_data, objective = load_data(args.folder, data_needed)

    # problem initialization
    config = OptimizationConfig( 
        objective_metrics=data_needed['output'], 
        optimization_parameters=data_needed['input'], 
        objective=objective,
        n_candidates=args.n_candidates,
        n_restarts=args.n_restarts,
        raw_samples=args.raw_samples,
        verbose=args.verbose,
        optimizers=args.optimizer,
        multi_model = args.multi_model
    )

    if config.verbose:
        print(f"Working data with {objective.value} objective:\nINPUT:")
        visualize_data(X_data, config.optimization_parameters)
        print("OUTPUT:")
        visualize_data(Y_data, config.objective_metrics)

    # normalize parameters and standardize metrics with their bounds
    bounds_manager = BoundsGenerator()
    original_bounds = bounds_manager.generate_bounds(X_data).to(dtype=torch.float64)
    if config.verbose:
        print(f"\nBounds Generated:")
        visualize_data(original_bounds, config.optimization_parameters)

    X_normalized, Y_standardized = normalize_val(X_data, Y_data, original_bounds)
    if config.verbose:
        print(f"\nNormalized data: \n INPUT:")
        visualize_data(X_normalized, config.optimization_parameters)
        print("OUTPUT:")
        visualize_data(Y_standardized, config.objective_metrics)

    # training model
    model = train_model(config, X_normalized, Y_standardized)

    # generate and optimize candidates
    results = bo_loop(
        config, model, X_normalized.shape[1], Y_standardized, bounds_manager
    )
    
    fig = []
    for k, v in results.items():
        if k not in config.optimizers.split(' '):
            continue

        (candidates, acq_val), elapsed_time = v

        candidates_denorm = denormalize_val(candidates, original_bounds)

        fig.append(plot_data(X_data, candidates_denorm, acq_val, k, config))
        
        print(f'='*60)
        print(f"Candidates suggested with method \'{k}\'\n     -- acq_value=[{acq_val.tolist()}] \n     -- Elapsed time: {elapsed_time:.4f}s")
        visualize_data(candidates_denorm, config.optimization_parameters)
    
    for f in fig:
        f.show()

    plt.show()

if __name__ == '__main__':
    try:
        args = parse_arg()

        warnings.filterwarnings('ignore')
        main(args)
    except Exception as e:
        print(f"An Error Occured:")
        print(f"   → {type(e).__name__}: {e}")
        print("\nError details:")
        traceback.print_exc()