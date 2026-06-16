from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Idea:
# This code implements a flexible univariate analysis system using the Strategy design pattern.
# It separates analysis logic into interchangeable strategies for numerical and categorical features.
# This allows dynamic switching between visualization approaches (e.g., histogram/KDE for numerical data
# and count plots for categorical data) without modifying the core analyzer.
# The design makes the system easily extensible for adding new types of univariate analyses in the future.


class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        pass


class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        self._strategy.analyze(df, feature)


if __name__ == "__main__":
    df = pd.read_csv("/Users/m1/Desktop/ house-pricing-mlops-pipeline/extracted_data/AmesHousing.csv")
    uni_analysis = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    uni_analysis.execute_analysis(df, "House Style")