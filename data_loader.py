from etl import ProvenanceExtractor
import torch
import logging
from typing import Dict
from helpers.config import METRICS, Objective, Timer

def load_data(data_folder: str, data_needed: Dict):
    '''use etl module to provide training data'''
    logger = logging.getLogger('BO')
    timer = Timer(logger)

    with timer.measure('data_loading'):
        extractor = ProvenanceExtractor(data_folder, data_needed)
        inp, out = extractor.extract_all()

    logger.info(f"   -> Retrieved experiment data               [{timer.get_opt_time('data_loading'):.4f}s]")
    for n,key in enumerate(data_needed['output']):
        if METRICS[key]=='MIN':
            for row in range(len(out)):
                out[row][n] = -out[row][n]

    X_ = torch.tensor(inp, dtype=torch.float64)
    Y_ = torch.tensor(out, dtype=torch.float64)
    objective = Objective.SINGLE if Y_.shape[1] == 1 else Objective.MULTI
    return X_, Y_, objective
