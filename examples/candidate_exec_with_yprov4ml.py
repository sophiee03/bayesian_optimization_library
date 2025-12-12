from bayesopt_core import *
import sys, subprocess

# FOR NOW THIS EXECUTION IS NOT ITERABLE (you can execute it once with the number of candidates required)
# if you try to re-execute other candidates the old ones are not included in the training set 
# so the new ones probably results equal to the first iterations ones
# if you want to execute it in an iterable manner change the provenance_save_dir (in cifar10.py) to the training set one
# note that if you do that, the candidates are not 'distinguishable' from the training data

if __name__ == '__main__':
    #for now the configuration is editable here (later we can find another way for the user to setup optimization settings)
    bayesopt = BayesianOptimizer(OptimizationConfig(
        #NB: if you change here the parameters/metrics the execution will not work
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

    #generate n candidates
    results = bayesopt.run('../test/prov')

    #execute candidates generated
    print(f'OBTAINED {len(results.candidates)} CANDIDATE(s):')
    for n in range(len(results.candidates)):
        print(f'{results.candidates[n]}')
    print(f'ACQUISITION VALUE(S): {results.acq_values}')
    print(f'{'='*60}')

    for n, candidate in enumerate(results.candidates):
        cmd = [
            sys.executable, '../test/cifar10.py',
            '--dropout', str(candidate[0]),
            '--batch-size', str(int(candidate[1])),
            '--epochs', str(int(candidate[2])),
            '--lr', str(candidate[3]),
            '--model-size', str(candidate[4])
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=False,
                cwd='../test/'
            )
            print(f'Execution {n+1} completed')
            print(f'{'='*60}')
        except Exception as e:
            print(f'An error occured: {e}')