import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd

# Idea :
# Give me a file, and I'll know how to load it into a pandas DataFrame.
# This code only knows how to handle ZIP files, but it is designed so you can easily add support for CSV, Excel, JSON, etc. later.


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass


class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a zip file.")

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")

        if len(csv_files) > 1:
            raise ValueError(
                "Multiple CSV files found. Please specify which one to use."
            )

        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        return df


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        if file_extension == ".zip":
            return ZipDataIngestor()

        raise ValueError(
            f"No ingestor available for file extension: {file_extension}"
        )


if __name__ == "__main__":
    file_path = "/Users/m1/Desktop/ house-pricing-mlops-pipeline/data/archive.zip" 
    file_extension = os.path.splitext(file_path)[1]
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = data_ingestor.ingest(file_path)

    print(df.head())
    print(f"\nShape: {df.shape}")