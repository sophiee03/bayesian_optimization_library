from bayesopt_core import *
import sys, subprocess

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