# BAYESIAN OPTIMIZATION WITH BOTORCH
Library to perform Bayesian Optimization based on BoTorch library. It is useful for HyperParameterTuning (HPT) problems since it finds parameters that maximizes the objective metrics (or minimizes them if you set it to). 

# Install and use the library
1. Install the library 
```
pip install BayesianOptimization
```
2. Use the BayesianOptimizer class to perform bayesian optimization on your datasets

NB: examples of use are provided in the folder examples

# Project Structure
```
bayesian_optimization_library/
├── bayesopt_core/
│   ├── __init__.py
│   ├── acquisition.py
│   ├── bayesian_handler.py
│   ├── config.py
│   ├── models.py
│   ├── optimizer.py
│   └── helpers/
│       ├── __init__.py
│       ├── logger.py
│       ├── processing.py
│       ├── results_to_csv.py
│       └── visualization.py
├── examples/
│   ├── basic_candidates_generation.ipynb
│   ├── basic_candidate_execution.ipynb
│   ├── change_config_in_execution.ipynb
│   ├── detailed_cand_generation.ipynb
│   └── saving_candidates_to_csv.ipynb
├── pyproject.toml
├── README.md
└── .gitignore

```

# Benchmark Tests
