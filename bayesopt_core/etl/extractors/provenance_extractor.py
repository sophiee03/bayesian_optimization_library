from pathlib import Path
import os
import numpy as np
from .extractor_factory import ExtractorFactory
import json

"""
optimization_params format
{
    'output': ['ACC_val', 'cpu_usage'],
    'input': ['param_lr', 'param_batch_size', 'param_epochs']   
}
"""

class ProvenanceExtractor:
    def __init__(self, provenance_folder, optimization_params):
        self.provenance_folder = Path(provenance_folder)
        # Maybe here I could validate optimization_params
        # - check that it's a dictionary with the expected keys
        # - check that the values correspond to some metrics files in the provenance folder
        # ! The validation of the second point could be done in the _extract_experiment_data method
        # where we have access to the metrics files
        # Maybe it's better to raise an error asap if the optimization_params are not valid
        self.optimization_params = optimization_params

    
    def _list_experiments(self):
        return [p for p in self.provenance_folder.iterdir() if p.is_dir()]
    
    def _get_metrics_dir(self, experiment_path):
        for entry in os.listdir(experiment_path):
            full_path = os.path.join(experiment_path, entry)

            if os.path.isdir(full_path) and entry.startswith("metrics"):
                metrics_dir = Path(full_path)
                return list(metrics_dir.glob("*.*"))
        
        raise FileNotFoundError(f"Metrics directory not found in {experiment_path}")

        # metrics_dir = experiment_path / "metrics"

        # if not metrics_dir.exists():
        #     raise FileNotFoundError(f"Metrics directory not found in {experiment_path}")
        # return list(metrics_dir.glob("*.*"))
    
    def _extract_experiment_data(self, experiment_path):
        # Here we should combine data from the JSON file ("MODEL_SIZE", "DROPOUT", "BATCH_SIZE", "EPOCHS", "LR")
        # and the parameters that are present under the metrics*/ folder
        records = {}

        # 1 - Find and parse the JSON
        for file in os.listdir(experiment_path):
            # Hopefully there will always be 1 json file, otherwise im going to blow up smth
            if file.endswith(".json"):
                filename = os.path.join(experiment_path, file)
                json_file = json.load(open(filename))
                # Again, if these names change I blow up smth
                activity = json_file["activity"]
                training_params = activity["context:Context.TRAINING"]
                training_params.pop("yProv4ML:level", None) # I remove this so that the training_params dict is ready
                training_params["DROPOUT"] = [np.float64(training_params["DROPOUT"])]
                training_params["BATCH_SIZE"] = [np.float64(training_params["BATCH_SIZE"])]
                training_params["EPOCHS"] = [np.float64(training_params["EPOCHS"])]
                training_params["LR"] = [np.float64(training_params["LR"])]
                accuracy = [np.float64(activity["context:Context.TESTING"]["accuracy"])]
                records = training_params
                records["accuracy"] = accuracy


        metrics_files = self._get_metrics_dir(experiment_path)

        for metric_file in metrics_files:
            extractor = ExtractorFactory.get_extractor(metric_file)
            data = extractor.extract(metric_file)
            metric_name = data.attrs["_name"]
            metric_value = data["values"].values
            records[f"{metric_name}"] = metric_value
        
        return records

    def extract_all(self):
        input = []
        output = []
        for experiment_dir in self._list_experiments():
            experiment_data = self._extract_experiment_data(experiment_dir)

            # If, for some reason there are no data, avoid to put empty lists in the final result
            if len(experiment_data) == 0:
                continue
            
            tmp_input = []
            tmp_output = []
            for key in self.optimization_params['input']:
                if key in experiment_data:
                    value = experiment_data[key]
                    if len(value) > 1:
                        tmp_input.append(value.sum().astype(np.float64))
                    else:
                        tmp_input.append(experiment_data[key][0].astype(np.float64))

            input.append(tmp_input)

            for key in self.optimization_params['output']:
                if key in experiment_data:
                    value = experiment_data[key]
                    if len(value) > 1:
                        tmp_output.append(value.sum().astype(np.float64))
                    else:
                        tmp_output.append(experiment_data[key][0].astype(np.float64))
            output.append(tmp_output)   
        return input, output
