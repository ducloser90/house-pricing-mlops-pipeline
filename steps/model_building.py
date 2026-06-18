import logging
from typing import Annotated, Optional
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
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

# Models that Optuna will tune. Linear regression has no meaningful
# hyperparameters to search here, so it's trained directly with defaults.
TUNABLE_MODEL_TYPES = {"random_forest", "xgboost"}


def _suggest_params(trial: optuna.Trial, model_type: str) -> dict:
    """Defines the Optuna search space per model type."""
    if model_type == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    else:
        raise ValueError(f"No Optuna search space defined for model_type: {model_type}")


def _build_estimator(model_type: str, params: Optional[dict] = None):
    """
    Returns an unfitted estimator instance based on model_type, with the
    given hyperparameters applied (params=None -> library defaults).
    """
    params = params or {}

    if model_type == "linear_regression":
        return LinearRegression()
    elif model_type == "random_forest":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params,
        )
    elif model_type == "xgboost":
        return XGBRegressor(
            random_state=42,
            n_jobs=-1,
            **params,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def _tune_with_optuna(
    model_type: str,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    cv_folds: int,
    random_state: int,
) -> dict:
    """
    Runs an Optuna study to find the best hyperparameters for `model_type`,
    evaluated via K-fold cross-validation on X_train/y_train (never touches
    the held-out test set). Returns the best params dict found.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_type)
        estimator = _build_estimator(model_type, params)

        candidate_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        # Negative MSE because Optuna's default direction here is
        # "maximize" and sklearn's scorer convention returns negative MSE
        # for minimization-style metrics.
        scores = cross_val_score(
            candidate_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=1,  # avoid nested parallelism with n_jobs=-1 estimators
        )
        return float(np.mean(scores))

    logging.info(
        f"Starting Optuna study for {model_type}: "
        f"{n_trials} trials, {cv_folds}-fold CV on X_train."
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logging.info(
        f"Optuna study completed. Best CV neg-MSE: {study.best_value:.6f}. "
        f"Best params: {study.best_params}"
    )

    return study.best_params


@step(**step_args)
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "linear_regression",
    tune_hyperparameters: bool = False,
    n_trials: int = 30,
    cv_folds: int = 5,
    random_state: int = 42,
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
    logging.info(f"Tune hyperparameters: {tune_hyperparameters}")

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

    best_params = None
    if tune_hyperparameters and model_type in TUNABLE_MODEL_TYPES:
        # NOTE: cross_val_score below re-fits a fresh ColumnTransformer on
        # each fold internally (since it's inside the candidate_pipeline),
        # so there's no leakage between CV folds despite reusing the
        # `preprocessor` object's configuration here.
        best_params = _tune_with_optuna(
            model_type=model_type,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state,
        )
    elif tune_hyperparameters and model_type not in TUNABLE_MODEL_TYPES:
        logging.warning(
            f"tune_hyperparameters=True but no Optuna search space is "
            f"defined for model_type='{model_type}'. Skipping tuning."
        )

    estimator = _build_estimator(model_type, best_params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    try:
        # NOTE: Do NOT call mlflow.set_tracking_uri(...) or
        # mlflow.sklearn.autolog() with a custom URI here. ZenML's MLflow
        # experiment tracker (configured via the `experiment_tracker` step
        # arg above) already opens and manages an MLflow run scoped to
        # this step, using the tracking URI configured on the active
        # stack. Overriding it manually causes "Run not found" errors
        # when ZenML tries to terminate its own run afterward.
        if experiment_tracker:
            mlflow.sklearn.autolog()
            if best_params:
                mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})

        logging.info(f"Building and training the final {model_type} model.")
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