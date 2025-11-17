from typing import List
import torch
from tabulate import tabulate
from .config import METRICS, VALID_PARAMETERS, PRECISIONS

def visualize_data(d: torch.Tensor, headers: List=None):
    '''function to print data'''
    data = d.cpu().numpy().tolist()
    prec=set_precisions(headers)
    print(tabulate(data, headers=headers, floatfmt=prec, tablefmt="fancy_grid"))
        
def set_precisions(x_names: List[str]):
    '''defines the precision for the output format'''
    prec = []
    for x in x_names:
        prec.append(PRECISIONS[f'{x}'])
    return prec

def print_params():
    print(f" {p} " for p in VALID_PARAMETERS)

def print_metrics():
    print([f" {k} " for k in METRICS.keys()])