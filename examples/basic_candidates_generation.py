from bayesopt_core import *

if __name__ == '__main__':
    bayesopt = BayesianOptimizer(OptimizationConfig(
        ['accuracy', 'emissions'],                                  #change here the metrics to maximize/minimize
        ['DROPUOT', 'BATCH_SIZE', 'EPOCHS', 'LR', 'MODEL_SIZE'],    #change here the parameters to optimize
        objective=Objective.MULTI,
        n_candidates=1,                                             #change here the number of candidates you want to obtain
        n_restarts=10,
        raw_samples=200,
        optimizers='optimize_acqf',                                 #change here the optimizer to use
        acqf='ucb',                                                 #change here the acquisition function to use
        beta=1.5                                                    #if using UCB you can balance beta otherwise it will be default
    ))

    res = bayesopt.run(folder='../test/prov')                       #change here the folder of the dataset

    print(res)