import logging
from typing import Annotated
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from zenml import step, ArtifactConfig
from zenml.enums import ArtifactType
from zenml.client import Client
from zenml import Model

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)

experiment_tracker = Client().active_stack.experiment_tracker
step_args = {"enable_cache": False, "model": model}
if experiment_tracker:
    step_args["experiment_tracker"] = experiment_tracker.name


def _build_estimator(model_type: str):
    if model_type == "linear_regression":
        return LinearRegression()
    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


@step(**step_args)
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "linear_regression",
) -> Annotated[
    Pipeline, ArtifactConfig(name="sklearn_pipeline", artifact_type=ArtifactType.MODEL)
]:
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns
    numerical_cols = X_train.select_dtypes(
        exclude=["object", "category"]
    ).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")
    logging.info(f"Model type: {model_type}")

    transformers = [
        ("num", SimpleImputer(strategy="mean"), numerical_cols),
    ]
    if len(categorical_cols) > 0:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    estimator = _build_estimator(model_type)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    try:
        if experiment_tracker:
            mlflow.sklearn.autolog()

        logging.info(f"Building and training the {model_type} model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        expected_columns = numerical_cols.tolist()
        if len(categorical_cols) > 0:
            onehot_encoder = (
                pipeline.named_steps["preprocessor"]
                .transformers_[1][1]
                .named_steps["onehot"]
            )
            cat_features = list(
                onehot_encoder.get_feature_names_out(categorical_cols)
            )
            expected_columns += cat_features

        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    return pipeline


if __name__ == "__main__":
    pass