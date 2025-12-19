import csv, torch, os
from datetime import datetime
from ..config import OptimizationConfig
from ..bayesian_handler import OptimizationResults
from typing import List, Dict, Optional

class CSVResults:
    """class to write to a csv file the results obtained from Bayesian Optimization
    
    Attributes:
        csv_path (str): path for the file to be accessed/created
        parameter_names (List(str)): names of the parameters optimized (for headers)
        metrics_names (List(str)): names of the metrics (for headers)
        metrics_estim_names (List(str)): names of the estimated metrics (for headers)
    """
    def __init__(self, headers: Dict[str, List[str]], path: str):
        self.csv_path = path
        self.parameter_names = headers['parameters']
        self.metrics_names = headers['metrics']
        self.metrics_estim_names = [f'estimated {m}' for m in headers['metrics']]
        self.ensure_exists()

    def build_headers(self):
        headers = ['timestamp', 'candidate_id', 'acq_value', 'optimizer', 'acqf', 'n_restarts', 'raw_samples']

        headers.extend(self.parameter_names)

        for metric in self.metrics_estim_names:
            headers.append(f'{metric} (mean)')
            headers.append(f'{metric} (std)')

        headers.append('execution status')
        headers.extend(self.metrics_names)
        return headers

    def ensure_exists(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.build_headers())
    
    def log_candidates(self, results: OptimizationResults, config: OptimizationConfig, exec_results: Optional[List[List]]=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # shape [n_candidates x n_metrics]
        mean = results.posterior.mean.detach().numpy()
        std = results.posterior.variance.sqrt().detach().numpy()

        rows_to_write = []

        if isinstance(results.acq_values, list):
            acq_val = results.acq_values
        else:
            acq_val = [results.acq_values]*len(results.candidates)

        for i, candidate in enumerate(results.candidates):
            if isinstance(candidate, torch.Tensor):
                candidate = candidate.tolist()

            row = [timestamp, i+1, round(acq_val[i], 6), config.optimizers, config.acqf, config.n_restarts, config.raw_samples]
            for param_val in candidate:
                if isinstance(param_val, float):
                    row.append(round(param_val, 6))
                else:
                    row.append(param_val)

            for m in range(len(self.metrics_estim_names)):
                if m < mean.shape[1]:
                    row.append(round(mean[i][m].item(), 6))
                    row.append(round(std[i][m].item(), 6))
                else:
                    row.extend([None, None])
            
            if exec_results and i < len(exec_results):
                row.append('executed')
                for m in range(len(self.metrics_names)):
                    value = exec_results[i][m]
                    if isinstance(value, float):
                        row.append(round(value, 6))
                    else:
                        row.append(value)
            else:
                row.append('not_executed')
                row.extend([None]*len(self.metrics_names))
            
            rows_to_write.append(row)

        self.append_rows(rows_to_write)

    def append_rows(self, rows: List[List]):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
