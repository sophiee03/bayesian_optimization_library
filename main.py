from bo_loop import bo_loop
from models import train_model
from helpers.config import OptimizationConfig
from data_loader import load_data
from helpers.config import BoundsGenerator
from helpers.processing import normalize_val, denormalize_val

#taking parameters to optimize and metrics to maximize as input
metrics = input('Which metric(s) do you want to maximize/minimize? (use a \' \' to separate each metric)\n')
if metrics is None:
    raise ValueError(f"You have to provide at least one metric to maximize/minimize")
parameters = input ('Which parameters do you want to optimize? (use a \' \' to separate each metric)\n')
if parameters is None:
    raise ValueError(f"You have to provide at least one parameter to optimize")
data_needed = {
    'output': list(metrics.split(' ')),
    'input': list(parameters.split(' '))
}

print(f"... Starting Execution ...")

#extract data
input, output, objective = load_data('./prov', data_needed)

#define problem and bounds
config = OptimizationConfig(objective, 3)
bounds_manager = BoundsGenerator()
original_bounds = bounds_manager.generate_bounds(input)

#normalize and standardize values
X_normalized, Y_standardized = normalize_val(input, output, original_bounds)

#training model
model = train_model(objective, X_normalized, Y_standardized)

#generate and optimize candidates 
candidates, acq_val = bo_loop(
    config, model, X_normalized.shape[1], Y_standardized, bounds_manager
)

#denormalize candidates
candidates_denorm = denormalize_val(candidates, original_bounds)
c = candidates_denorm.tolist()

#print results
print(f'='*60)
print(f"    Candidates found:   [acq_val = {acq_val:.4f}]")
print(f"                    {data_needed['input']}")
for i,c in enumerate(c):
    print(f"        Candidate {i+1}:        {c[0]:.1f},         {c[1]:.1f},         {c[2]:.4f}")
print(f"... Done ...")
