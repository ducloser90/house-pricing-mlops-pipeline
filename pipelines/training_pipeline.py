from steps.data_ingestion import data_ingestion_step
from steps.data_splitting import data_splitter_step
from steps.feature_engineering import feature_engineering_step
from steps.missing_values import handle_missing_values_step
from steps.model_building import model_building_step
from steps.model_evaluating import model_evaluator_step
from steps.outliers import outlier_detection_step
from zenml import Model, pipeline


@pipeline(model=Model(name="prices_predictor"))
def ml_pipeline(
    model_type: str = "linear_regression",
    tune_hyperparameters: bool = False,
    n_trials: int = 30,
    cv_folds: int = 5,
):
    raw_data = data_ingestion_step(
        file_path="/Users/m1/Desktop/prices-predictor-system/data/archive.zip"
    )

    filled_data = handle_missing_values_step(raw_data)

    # Drop row/parcel identifiers — they carry no predictive signal and
    # only add noise if left in as numerical features.
    id_dropped_data = feature_engineering_step(
        filled_data,
        strategy="column_dropping",
        features=["Order", "PID"],
    )

    # Ordinally encode quality/condition columns (Exter Qual, Kitchen Qual,
    # Bsmt Qual, etc.) so their natural ordering (Ex > Gd > TA > Fa > Po) is
    # preserved as a single numeric column instead of being flattened into
    # independent one-hot dummies.
    ordinal_encoded_data = feature_engineering_step(
        id_dropped_data,
        strategy="ordinal_encoding",
        # mapping=None -> uses OrdinalEncoding.DEFAULT_MAPPING
    )

    engineered_data = feature_engineering_step(
        ordinal_encoded_data,
        strategy="log",
        features=["Gr Liv Area", "SalePrice"],
    )

    clean_data = outlier_detection_step(
        engineered_data,
        column_name="SalePrice",
    )

    X_train, X_test, y_train, y_test = data_splitter_step(
        clean_data,
        target_column="SalePrice",
    )

    model = model_building_step(
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
        tune_hyperparameters=tune_hyperparameters,
        n_trials=n_trials,
        cv_folds=cv_folds,
    )

    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model,
        X_test=X_test,
        y_test=y_test,
    )

    return model


if __name__ == "__main__":
    run = ml_pipeline(model_type="xgboost", tune_hyperparameters=True, n_trials=50, cv_folds=5)
