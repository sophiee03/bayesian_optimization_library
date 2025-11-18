from bo_loop import bo_loop
import traceback, warnings, torch, argparse
import matplotlib.pyplot as plt
from helpers.logger import setup_console_logger
from models import train_model
from helpers.config import OptimizationConfig, Objective, BoundsGenerator, ATTRIBUTES, OPTIMIZERS
from data_loader import load_data
from helpers.plot import plot_data, plot_all
from helpers.utils import visualize_data
from helpers.processing import normalize_val, denormalize_val

def parse_arg():
    parser = argparse.ArgumentParser(description = "Bayesian Optimization with BoTorch")
    parser.add_argument('--folder', type=str, default='../prov2', 
                        help="Folder to extract training data")
    parser.add_argument('--output', nargs='+', type=str, choices=list(ATTRIBUTES.keys()),
                        help="Metrics to maximize/minimize")
    parser.add_argument('--input', nargs='+', type=str, choices=list(ATTRIBUTES.keys()),
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
    parser.add_argument('--default', action='store_true')

    return parser.parse_args()

def main(args):
    logger = setup_console_logger()

    if not args.default:
        metrics = args.output
        parameters = args.input

        data_needed = {
            'output': metrics,
            'input': parameters
        }

        config = OptimizationConfig( 
            objective_metrics=data_needed['output'], 
            optimization_parameters=data_needed['input'], 
            objective=Objective.SINGLE if len(data_needed['output']) == 1 else Objective.MULTI,
            n_candidates=args.n_candidates,
            n_restarts=args.n_restarts,
            raw_samples=args.raw_samples,
            optimizers=args.optimizer,
            multi_model = args.multi_model,
            verbose=args.verbose,
        )
    else:
        data_needed = {
            'output': ['accuracy', 'emissions'],
            'input': ['LR', 'BATCH_SIZE', 'DROPOUT_RATE', 'EPOCHS']
        }
        config = OptimizationConfig(
            objective_metrics = data_needed['output'],
            optimization_parameters = data_needed['input'],
            objective = Objective.MULTI,
            n_candidates=3,
            n_restarts=10,
            raw_samples=200,
            optimizers='optimize_acqf',
            multi_model='modellistgp'
        )

    X_data, Y_data = load_data(config, args.folder, data_needed)

    bounds_manager = BoundsGenerator()
    original_bounds = bounds_manager.generate_bounds(X_data).to(dtype=torch.float64)

    X_normalized, Y_standardized = normalize_val(config, X_data, Y_data, original_bounds)

    model = train_model(config, X_normalized, Y_standardized)

    results = bo_loop(config, model, X_normalized, Y_standardized, bounds_manager)
    
    final_results = {}
    for k, tuple in results.items():
        if k not in config.optimizers.split(' '):
            continue
        
        (candidates, acq_val), time = tuple
        candidates_denorm = denormalize_val(candidates, original_bounds)
        acq = acq_val.tolist()
        if not isinstance(acq_val, list):
            acq = [acq_val]
        final_results[f'{k}'] = (candidates_denorm, acq), time

    if config.verbose:
        for k, v in results.items():
            (candidates, acq_val), elapsed_time = v
            print(f'='*60)
            print(f"Candidates suggested with method \'{k}\'\n     -- acq_value=[{acq_val.tolist()}] \n     -- Elapsed time: {elapsed_time:.4f}s")
            visualize_data(candidates_denorm, config.optimization_parameters)
    
    plot_all(X_data, final_results, config)
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