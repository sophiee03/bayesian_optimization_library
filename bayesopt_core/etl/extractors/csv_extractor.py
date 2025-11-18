import pandas as pd
from .base import BaseExtractor

class CSVExtractor(BaseExtractor):
    def extract(self, file_path):
        return pd.read_csv(file_path)