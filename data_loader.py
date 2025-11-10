from etl import ProvenanceExtractor
import torch
from typing import Dict
from helpers.config import Maximization, Objective

def load_data(data_folder: str, data_needed: Dict):
    '''use etl module to provide training data and parameters to optimize'''
    print("    -> Loading Data")
    #extract data from etl module
    extractor = ProvenanceExtractor(data_folder, data_needed)
    inp, out = extractor.extract_all()

    #negate the metrics to minimize to perform maximization
    for n,key in enumerate(data_needed['output']):
        if key in Maximization.MINIMIZE:
            for row in out:
                row[n] = -row[n]

    X_ = torch.tensor(inp, dtype=torch.float64)
    Y_ = torch.tensor(out, dtype=torch.float64)
    objective = Objective.SINGLE if Y_.shape[1] == 1 else Objective.MULTI
    return X_, Y_, objective
