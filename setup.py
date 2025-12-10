from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = 'BayesianOptimization',
    version = '0.1.0',
    packages = find_packages(where='./bayesopt_core'),
    package_dir = {'':'bayesopt_core'},
    install_requires = required,
)