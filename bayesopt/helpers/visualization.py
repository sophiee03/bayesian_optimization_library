import torch
from tabulate import tabulate
from typing import List

def visualize_data(d: torch.Tensor, headers: List):
    """Function to print data
    
    Args:
        d (Tensor): data to visualize
        headers (List[str]): headers names
    """
    if isinstance(d, torch.Tensor):
        data = d.cpu().numpy().tolist()
    else:
        data = d
    print(tabulate(data, headers=headers, floatfmt=([".6f"]*len(headers)), tablefmt="simple_grid"))
