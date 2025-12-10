from .etl import ProvenanceExtractor
import torch
import logging
from typing import Dict
from ..config import ATTRIBUTES, Timer, OptimizationConfig

def load_data(config: OptimizationConfig, data_folder: str, data_needed: Dict):
    '''use etl module to provide training data'''
    logger = logging.getLogger('BO')
    timer = Timer(logger)

    with timer.measure('data_loading'):
        extractor = ProvenanceExtractor(data_folder, data_needed)
        inp, out = extractor.extract_all()
        if inp is None or out is None:
            raise RuntimeError("An error occured in extracting data, check the folder and its content!")

    if config.verbose:
        logger.info(f"   -> Retrieved experiment data               [{timer.get_opt_time('data_loading'):.4f}s]")

    # make negative the values of the metrics to minimize
    for n,key in enumerate(data_needed['output']):
        if ATTRIBUTES[key][2]=='MIN':
            for row in range(len(out)):
                out[row][n] = -out[row][n]

    X_ = torch.tensor(inp, dtype=torch.float64)
    Y_ = torch.tensor(out, dtype=torch.float64)

    return X_, Y_
