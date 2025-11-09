from bo_loop import bo_loop
from models import train_model
from helpers.config import OptimizationConfig
from data_loader import load_data

print(f"... Starting Execution ...")
data, obj = load_data('prov')
config = OptimizationConfig(obj, 3)
model, X, param_names, Y, objective = train_model(obj)
candidates, acq_val = bo_loop(
    config, model, param_names, Y
)
print(f"... Done ...")