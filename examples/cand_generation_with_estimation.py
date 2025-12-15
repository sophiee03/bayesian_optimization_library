from bayesopt_core import *
from exec_with_yprov4ml import yprov_execution
import sys, os, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from etl.extractors.provenance_extractor import ProvenanceExtractor
def yprov_execution(candidates):    
    for n, candidate in enumerate(candidates):
        print(f'Executing candidate {n+1}...')
        cmd = [
            sys.executable, '../test/cifar10.py',
            '--dropout', str(candidate[0]),
            '--batch-size', str(int(candidate[1])),
            '--epochs', str(int(candidate[2])),
            '--lr', str(candidate[3]),
            '--model-size', str('small')    # manually set
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=False,
                cwd='../test/'
            )
            print(f'Execution {n+1} completed!')
            print(f'{'='*60}')
        except Exception as e:
            print(f'An error occured: {e}')


if __name__ == '__main__':
    data_needed = {
        'input': ['DROPOUT', 'BATCH_SIZE', 'EPOCHS', 'LR'],
        'output': ['accuracy', 'emissions']
    }
    extractor = ProvenanceExtractor('../test/prov', data_needed)
    inp, out = extractor.extract_all()
    # cols are parameters/metrics, rows are runs

    bayesopt = BayesianOptimizer(OptimizationConfig(
        data_needed['output'],
        data_needed['input'],
        ['MAX', 'MIN'],
        objective=Objective.MULTI,
        n_candidates=3,
        n_restarts=10,
        raw_samples=200,
        optimizers='optimize_acqf',
        acqf='ucb',
        beta=1.5,
        verbose=True
    ))

    data = {
        'parameters': (data_needed['input'], inp),
        'metrics': (data_needed['output'], out)
    }

    res = bayesopt.run(data)
    print(f'{'='*60}\nACTUAL EXECUTION:\n{'='*60}')
    yprov_execution(res.candidates)




