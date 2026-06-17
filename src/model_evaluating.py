import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score


# Idea:
# This code implements a flexible model evaluation system using the Strategy design pattern.
# It supports interchangeable evaluation strategies for machine learning models,
# with the current implementation focusing on regression metrics such as MSE and R-squared.
# The structure separates evaluation logic from execution, enabling easy extension
# to other model types or custom metric definitions without modifying the core evaluator.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self,
        model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        pass


class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self,
        model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:

        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "Mean Squared Error": mse,
            "R-Squared": r2,
        }

        logging.info(f"Model Evaluation Metrics: {metrics}")

        return metrics


class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(
        self,
        model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:

        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    pass