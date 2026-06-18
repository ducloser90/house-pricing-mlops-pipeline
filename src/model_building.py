import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# Idea:
# This code implements a flexible model training system using the Strategy design pattern.
# It allows different machine learning models to be built and trained through a unified interface.
# Implementations include Linear Regression with feature scaling, Random Forest, and XGBoost.
# Tree-based strategies (Random Forest, XGBoost) do not require feature scaling and tend to
# capture non-linear relationships and feature interactions better than a plain linear model.
# The design supports easy extension to other models without modifying the core training pipeline.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        pass


class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")

        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Linear Regression model with scaling.")

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

        logging.info("Training Linear Regression model.")

        pipeline.fit(X_train, y_train)

        logging.info("Model training completed.")

        return pipeline


class RandomForestStrategy(ModelBuildingStrategy):
    """
    Random Forest regressor. Unlike linear regression, it does not require
    feature scaling, can capture non-linear relationships and interaction
    effects (e.g. Overall Qual mattering more when Gr Liv Area is large),
    and is more robust to outliers and multicollinearity among predictors.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = None,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")

        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info(
            f"Initializing Random Forest model "
            f"(n_estimators={self.n_estimators}, max_depth={self.max_depth})."
        )

        pipeline = Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    ),
                ),
            ]
        )

        logging.info("Training Random Forest model.")

        pipeline.fit(X_train, y_train)

        logging.info("Model training completed.")

        return pipeline


class XGBoostStrategy(ModelBuildingStrategy):
    """
    XGBoost gradient-boosted regressor. Like Random Forest, it does not
    require feature scaling and handles non-linear relationships and
    feature interactions well. Unlike Random Forest, it builds trees
    sequentially, each correcting the residual errors of the previous
    ones, which often gives it an edge on tabular regression tasks like
    this one.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")

        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info(
            f"Initializing XGBoost model "
            f"(n_estimators={self.n_estimators}, max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate})."
        )

        pipeline = Pipeline(
            [
                (
                    "model",
                    XGBRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        subsample=self.subsample,
                        colsample_bytree=self.colsample_bytree,
                        reg_alpha=self.reg_alpha,
                        reg_lambda=self.reg_lambda,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    ),
                ),
            ]
        )

        logging.info("Training XGBoost model.")

        pipeline.fit(X_train, y_train)

        logging.info("Model training completed.")

        return pipeline


class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> RegressorMixin:

        logging.info("Building and training the model using the selected strategy.")

        return self._strategy.build_and_train_model(X_train, y_train)


if __name__ == "__main__":
    pass