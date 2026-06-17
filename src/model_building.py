import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Idea:
# This code implements a flexible model training system using the Strategy design pattern.
# It allows different machine learning models to be built and trained through a unified interface.
# The current implementation provides a Linear Regression strategy with feature scaling,
# but the design supports easy extension to other models such as tree-based or ensemble methods
# without modifying the core training pipeline.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self,X_train: pd.DataFrame,y_train: pd.Series,) -> RegressorMixin:
        pass


class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self,X_train: pd.DataFrame,y_train: pd.Series,)-> Pipeline:

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