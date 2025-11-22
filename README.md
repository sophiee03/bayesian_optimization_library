# BAYESIAN OPTIMIZATION WITH BOTORCH

DEFAULT EXECUTION: python main.py --default
      - it will execute with a multi-objective (accuracy maximization and emissions minimization)
      - it will train the model with a modellistGP
      - it will provide 3 candidates generated with qUCB and optimize_acqf

CUSTOMIZED EXECUTION: it is possible to customize these attributes:
      --folder (where the training data to retrive are stored)
      --output (metrics to maximize/minimize)
      --input (parameters to optimize)
      --multi_model (model to use for training)
      --n_candidates (number of candidates to generate)
      --n_restarts (number of restarts to find the best candidate)
      --raw_samples (number of samples to generate among which the optimizer will choose)
      --optimizer (optimizer to use)
      --verbose (to see workflow and timings)

NB: the etl module is a submodule so it is necessary to install it 
