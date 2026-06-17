import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Idea:
# This code implements a flexible model training system using the Strategy design pattern.
# It allows different machine learning models to be built and trained through a unified interface.
# Implementations include Linear Regression with feature scaling, and Random Forest, which
# does not require scaling and tends to capture non-linear relationships and feature
# interactions better than a plain linear model. The design supports easy extension to
# other models (e.g. gradient boosting) without modifying the core training pipeline.

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