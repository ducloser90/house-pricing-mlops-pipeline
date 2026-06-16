from abc import ABC, abstractmethod
import pandas as pd

# Idea:
# This code defines a flexible data inspection system using the Strategy design pattern.
# It allows you to switch between different types of dataset analysis (such as data types
# inspection or summary statistics) without changing the core inspection logic.
# New inspection methods can be easily added by creating new strategies, making the system
# extensible and suitable for scalable data analysis workflows.


class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        pass


class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nDataTypesInspection: ")
        print(df.info())


class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["object"]))


class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        self._strategy.inspect(df)


if __name__ == "__main__":
    df = pd.read_csv("/Users/m1/Desktop/ house-pricing-mlops-pipeline/extracted_data/AmesHousing.csv")
    data_inspector = DataInspector(strategy=SummaryStatisticsInspectionStrategy())
    data_inspector.execute_inspection(df)