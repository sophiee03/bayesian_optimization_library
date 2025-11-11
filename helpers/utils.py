from typing import List
import torch
from tabulate import tabulate
from .config import METRICS, VALID_PARAMETERS, PRECISIONS

def visualize_data(d: torch.Tensor, headers: List=None):
    '''function to print data'''
    data = d.cpu().numpy().tolist()
    prec=set_precisions(headers)
    print(tabulate(data, headers=headers, floatfmt=prec, tablefmt="fancy_grid"))

def validate_input_data(d: List):
    '''check if the metrics to maximize/minimize are valid'''
    if len(d)==0:
        raise ValueError(f"You have to provide at least one metric to maximize/minimize\nChoose among: {METRICS.keys()}")
    for m in d:
        if m not in METRICS.keys():
            raise ValueError(f"Invalid metric: {m}")
        
def validate_output_data(d: List):
    '''check if the parameters to optimize are valid'''
    if len(d)==0:
        raise ValueError(f"You have to provide at least one parameter to optimize\nChoose among: {VALID_PARAMETERS}")
    for p in d:
        if p not in VALID_PARAMETERS:
            raise ValueError(f"Invalid parameter: {p}")
        
def set_precisions(x_names: List[str]):
    '''defines the precision for the output format'''
    prec = []
    for x in x_names:
        prec.append(PRECISIONS[f'{x}'])
    return prec

def print_params():
    print(f" {p} " for p in VALID_PARAMETERS)

def print_metrics():
    print([f" {k} " for k,m in METRICS.items()])