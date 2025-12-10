from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = 'bayesopt_core',
    version='0.1.0',
    url = 'https://github.com/sophiee03/bayesian_optimization_library',
    packages = find_packages(exclude=('testing')),
    install_requires = required,
)