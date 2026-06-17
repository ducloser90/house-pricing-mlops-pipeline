import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Idea:
# This code implements a flexible outlier detection and handling system using the
# Strategy design pattern. Different detection techniques, such as Z-score and IQR,
# are encapsulated as interchangeable strategies, while the OutlierDetector manages
# detection, visualization, and treatment of outliers. The design allows outlier
# handling methods to be easily switched, extended, and integrated into machine
# learning pipelines without modifying the core workflow.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the Z-score method.")
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info(
            f"Outliers detected with Z-score threshold: {self.threshold}."
        )
        return outliers


class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the IQR method.")
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        outliers = (
            (df < (q1 - 1.5 * iqr))
            | (df > (q3 + 1.5 * iqr))
        )
        logging.info("Outliers detected using the IQR method.")
        return outliers


class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing outlier detection strategy.")
        return self._strategy.detect_outliers(df)

    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: list = None,
        method="remove",
        **kwargs,
    ) -> pd.DataFrame:

        detection_df = df[columns] if columns is not None else df
        outliers = self.detect_outliers(detection_df)

        if method == "remove":
            logging.info("Removing outliers from the dataset.")
            row_mask = (~outliers).all(axis=1)
            df_cleaned = df[row_mask]
        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.copy()
            cols_to_cap = columns if columns is not None else df.columns
            lower = df[cols_to_cap].quantile(0.01)
            upper = df[cols_to_cap].quantile(0.99)
            df_cleaned[cols_to_cap] = df[cols_to_cap].clip(
                lower=lower, upper=upper, axis=1
            )
        else:
            logging.warning(
                f"Unknown method '{method}'. "
                "No outlier handling performed."
            )
            return df

        logging.info("Outlier handling completed.")
        return df_cleaned

    def visualize_outliers(
        self,
        df: pd.DataFrame,
        features: list,
    ):
        logging.info(
            f"Visualizing outliers for features: {features}"
        )
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")


if __name__ == "__main__":
    pass