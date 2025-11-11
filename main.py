from bo_loop import bo_loop
import traceback
from models import train_model
from helpers.config import OptimizationConfig,BoundsGenerator
from data_loader import load_data
from helpers.utils import visualize_data, validate_input_data, validate_output_data
from helpers.processing import normalize_val, denormalize_val

def main():
    # taking as input metrics to maximize/minimize and parameters to optimize
    inserted = input(f'SELECT THE METRIC(S) YOU WANT TO MAXIMIZE/MINIMIZE\n') 
    metrics = list(inserted.split(' '))
    validate_input_data(metrics)

    inserted = input(f'SELECT THE PARAMETER(S) YOU WANT TO OPTIMIZE\n')
    parameters = list(inserted.split(' '))
    validate_output_data(parameters)

    # ETL Module
    data_needed = {
        'output': metrics,
        'input': parameters
    }
    X_data, Y_data, objective = load_data('./prov', data_needed)

    # problem initialization
    config = OptimizationConfig( 
        data_needed['output'], 
        data_needed['input'], 
        objective,
        n_candidates=3, 
        verbose=False
    )
    if config.verbose:
        print(f"Working data with {objective.value} objective:\nINPUT:")
        visualize_data(X_data, config.optimization_parameters)
        print("OUTPUT:")
        visualize_data(Y_data, config.objective_metrics)

    # normalize parameters and standardize metrics with their bounds
    bounds_manager = BoundsGenerator()
    original_bounds = bounds_manager.generate_bounds(X_data)
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
    model = train_model(objective, X_normalized, Y_standardized)

    # generate and optimize candidates
    candidates, acq_val = bo_loop(
        config, model, X_normalized.shape[1], Y_standardized, bounds_manager
    )

    candidates_denorm = denormalize_val(candidates, original_bounds)
    print(f'='*60)
    print(f"Candidates suggested with acq_value=[{acq_val}]")
    visualize_data(candidates_denorm, config.optimization_parameters)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An Error Occured:")
        print(f"   → {type(e).__name__}: {e}")
        print("\nError details:")
        traceback.print_exc()