import logging
from abc import ABC, abstractmethod
import pandas as pd

# Idea:
# This code implements a flexible missing value handling system using the Strategy design pattern.
# Different strategies can be used to process missing data, such as dropping rows/columns
# or filling missing values with statistical measures (mean, median, mode) or constants.
# The handling logic is decoupled from the execution logic, making it easy to switch
# strategies and extend the system with new missing value treatment methods in the future.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"Dropping missing values with axis={self.axis} and thresh={self.thresh}"
        )
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df_cleaned


class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()

        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(
                include="number"
            ).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )

        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(
                include="number"
            ).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )

        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(
                    df[column].mode().iloc[0],
                    inplace=True,
                )

        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)

        else:
            logging.warning(
                f"Unknown method '{self.method}'. No missing values handled."
            )

        logging.info("Missing values filled.")
        return df_cleaned


class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)


if __name__ == "__main__":
    df = pd.read_csv("/Users/m1/Desktop/ house-pricing-mlops-pipeline/extracted_data/AmesHousing.csv")
    missing_handler = MissingValueHandler(FillMissingValuesStrategy())
    print(df.isnull().sum())
    df_test = missing_handler.handle_missing_values(df)
    print(df_test.isnull().sum())