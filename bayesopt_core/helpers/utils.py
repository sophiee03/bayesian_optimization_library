from typing import List
import torch
from tabulate import tabulate
from ..config import ATTRIBUTES

def visualize_data(d: torch.Tensor, headers: List=None):
    '''function to print data'''
    if isinstance(d, torch.Tensor):
        data = d.cpu().numpy().tolist()
    else:
        data = d
    prec=set_precisions(headers)
    print(tabulate(data, headers=headers, floatfmt=prec, tablefmt="grid"))
        
def set_precisions(x_names: List[str]):
    '''defines the precision for the output format'''
    prec = []
    for x in x_names:
        prec.append(ATTRIBUTES[f'{x}'][1])

    return prec