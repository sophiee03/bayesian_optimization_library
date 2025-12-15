# BAYESIAN OPTIMIZATION WITH BOTORCH
Library to perform Bayesian optimization based on BoTorch library. It is useful for HyperParameterTuning (HPT) problems since it finds parameters that maximizes the objective metrics. 

# Install and use the library
1. Install the library 
```
pip install BayesianOptimization
```
3. Use the BayesianOptimizer class to perform bayesian optimization on your datasets

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
│       └── visualization.py
├── examples/
│   ├── basic_candidates_generation.ipynb
│   └── candidate_exec_with_yprov4ml.ipynb
├── pyproject.toml
├── README.md
└── .gitignore

```

# Benchmark Tests
