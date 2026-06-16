import pandas as pd
from zenml import step
from src.data_ingestion import DataIngestorFactory
import os


@step
def data_ingestion(file_path : str) -> pd.DataFrame:
    file_extension = os.path.splitext(file_path)[1]
    data_ingestor= DataIngestorFactory.get_data_ingestor(file_extension=file_extension)
    df = data_ingestor.ingest(file_path=file_path)
    return df