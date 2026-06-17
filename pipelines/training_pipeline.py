from steps.data_ingestion import data_ingestion_step
from steps.data_splitting import data_splitter_step
from steps.feature_engineering import feature_engineering_step
from steps.missing_values import handle_missing_values_step
from steps.model_building import model_building_step
from steps.model_evaluating import model_evaluator_step
from steps.outliers import outlier_detection_step
from zenml import Model, pipeline


@pipeline(model=Model(name="prices_predictor",),)
def ml_pipeline():
    raw_data = data_ingestion_step(file_path="/Users/m1/Desktop/prices-predictor-system/data/archive.zip")
    filled_data = handle_missing_values_step(raw_data)
    engineered_data = feature_engineering_step(filled_data,strategy="log",features=["Gr Liv Area", "SalePrice"],)
    clean_data = outlier_detection_step(engineered_data,column_name="SalePrice",)
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data,target_column="SalePrice",)
    model = model_building_step(X_train=X_train,y_train=y_train,)
    evaluation_metrics, mse = model_evaluator_step(trained_model=model,X_test=X_test,y_test=y_test,)
    return model


if __name__ == "__main__":
    run = ml_pipeline()