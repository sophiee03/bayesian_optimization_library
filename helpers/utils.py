''' utility functions '''

from typing import Dict
from .config import Objective

def set_objective(y: Dict) -> Objective:
    '''
    setting objectives and Objective instance
    '''
    print(f"    -> Setting objective(s)")
    maximize = y['max']
    minimize = y['min']
    print(f"        The goal is to MAXIMIZE: {maximize} and MINIMIZE: {minimize}")
    if len(maximize) + len(minimize) <= 0:
        raise ValueError("You must provide at least one objective")
    elif len(maximize) + len(minimize) == 1:
        return maximize, minimize, Objective.SINGLE
    else:
        return maximize, minimize, Objective.MULTI
