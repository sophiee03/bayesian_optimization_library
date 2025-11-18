import xarray as xr
from .base import BaseExtractor

class ZarrExtractor(BaseExtractor):
    def extract(self, file_path):
        return xr.open_zarr(file_path)