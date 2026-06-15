from zenml import step, pipeline
from steps.data_ingestion import data_ingesting
from steps.missing_values import missing_values
from steps.feature_engineering import feature_engineering
from steps.outliers import outliers
from steps.data_splitting import data_splitting
from steps.model_building import model_building
from steps.model_evaluating import model_evaluating

@pipeline()
def my_pipeline():
    pass