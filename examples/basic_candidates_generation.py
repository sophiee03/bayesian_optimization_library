from bayesopt_core import *
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from etl.extractors.provenance_extractor import ProvenanceExtractor

if __name__ == '__main__':
    data_needed = {
        'input': ['DROPOUT', 'BATCH_SIZE', 'EPOCHS', 'LR', 'MODEL_SIZE'],
        'output': ['accuracy', 'emissions']
    }
    extractor = ProvenanceExtractor('../test/prov', data_needed)
    inp, out = extractor.extract_all()
    # cols are parameters/metrics, rows are runs

    bayesopt = BayesianOptimizer(OptimizationConfig(
        data_needed['output'],
        data_needed['input'],
        objective=Objective.MULTI,
        n_candidates=1,
        n_restarts=10,
        raw_samples=200,
        optimizers='optimize_acqf',
        acqf='ucb',
        beta=1.5,
        verbose=True
    ))

    data = {
        'parameters': (data_needed['input'], inp),
        'metrics': (data_needed['output'], ['MAX', 'MIN'], out)
    }

    res = bayesopt.run(data) 
    print(res)