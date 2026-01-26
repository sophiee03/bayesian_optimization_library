# BAYESIAN OPTIMIZATION LIBRARY
Library to perform Bayesian Optimization using BoTorch. 
It's purpose is to resolve HyperParameterTuning (HPT) problems since it finds the best parameters to maximize the objective metric(s) (or minimize them if you set it to). 
It allows to optimize with MULTI or SINGLE objective (either in both directions MAX/MIN)
The configuration is fully customizable, you can choose different: parameters, metrics, number of candidates to generate, number of restarts, number of raw samples to generate, the optimizer and acquisition function to use, the beta value for exploration-exploitation balance.

# Install and use the library
1. Install the library 
```
pip install bayesopt
```
2. Use the BayesianOptimizer class to perform bayesian optimization on your training datasets [view examples](https://github.com/sophiee03/bayesian_optimization_library/tree/main/examples)

# What you need to know
- The library has a BayesianOptimizer method to perform the entire optimization, it is called `.run()` and takes as arguments:
    - a dictionary of this form:
        ```
        data = {
            'parameters': [[par1, par2, par3][par1, par2, par3]],
            'metrics': [[metric1, metric2][metric1, metric2]]
        }
        ```
    - optionally a list with the bounds of the parameters (otherwise they will be generated automatically)
An example of this simple execution is provided in the [basic candidate generation example](https://github.com/sophiee03/bayesian_optimization_library/blob/main/examples/basic_candidate_generation.ipynb)
- If you prefer to see the entire pipeline you can opt to use each method of the BayesianOptimizer class as shown in the [detailed candidate generation example](https://github.com/sophiee03/bayesian_optimization_library/blob/main/examples/detailed_cand_generation.ipynb)
- It is also possible to change configuration without the creation of another instance as shown in [this example](https://github.com/sophiee03/bayesian_optimization_library/blob/main/examples/change_config_in_execution.ipynb)
- If you want to iteratively execute the optimization, the method `.update_training_dataset()` allow to add the candidates executed without re-loading the entire training dataset
    NB: when you have already loaded the dataset it is saved in the BayesianOptimizer instance so it is not necessary to re-pass the data dictionary when calling `.run()`
- If you want to modify the bounds you can use the method `.change_bounds()`

# Project Structure
```
bayesian_optimization_library/
├── bayesopt/
│   ├── __init__.py
│   ├── acquisition.py
│   ├── bayesian_handler.py
│   ├── config.py
│   ├── models.py
│   ├── optimizer.py
│   └── helpers/
│       ├── __init__.py
│       ├── processing.py
│       └── visualization.py
├── examples/
│   ├── basic_candidate_generation.ipynb
│   ├── basic_candidate_execution.ipynb
│   ├── detailed_cand_generation.ipynb
│   └── change_config_in_execution.ipynb
├── pyproject.toml
├── README.md
└── .gitignore
```
