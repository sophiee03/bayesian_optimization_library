import xarray as xr
from .base import BaseExtractor

class NetCDFExtractor(BaseExtractor):
    def extract(self, file_path):
        return xr.open_dataset(file_path)