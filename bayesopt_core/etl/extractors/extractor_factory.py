import os
from .csv_extractor import CSVExtractor
from .netcdf_extractor import NetCDFExtractor
from .zarr_extractor import ZarrExtractor
from .base import BaseExtractor

class ExtractorFactory:
    """Factory class to create extractor instances based on file type."""

    _extractor_mapping = {
        ".csv": CSVExtractor(),
        ".nc": NetCDFExtractor(),
        ".zarr": ZarrExtractor()
    }

    @staticmethod
    def get_extractor(file_path: str) -> BaseExtractor:
        """Get the appropriate extractor based on the file extension.

        Args:
            file_path (str): The path to the data file.

        Returns:
            BaseExtractor: An instance of the appropriate extractor.

        Raises:
            ValueError: If no extractor is found for the given file type.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ExtractorFactory._extractor_mapping:
            raise ValueError(f"No extractor found for file type: {ext}")
        return ExtractorFactory._extractor_mapping[ext]
