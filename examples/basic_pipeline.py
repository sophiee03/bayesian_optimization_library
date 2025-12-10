from bayesopt_core import *

if __name__ == '__main__':
    bayesopt = BayesianOptimizer(OptimizationConfig(
        ['accuracy', 'emissions'],
        ['DROPUOT', 'BATCH_SIZE', 'EPOCHS', 'LR', 'MODEL_SIZE'],
        objective=Objective.MULTI,
        n_candidates=3,
        n_restarts=10,
        raw_samples=200,
        optimizers='optimize_acqf',
        acqf='ucb',
        beta=1.5
    ))

    res = bayesopt.run(folder='./test/prov')

    print(res)