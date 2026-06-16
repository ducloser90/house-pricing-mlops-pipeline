from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Idea:
# This code implements a template-based system for analyzing missing values in a dataset.
# It uses the Template Method design pattern, where the overall analysis workflow is fixed
# (identify missing values → visualize missing values), but the specific implementation
# of each step can vary depending on the concrete class.
# This makes it easy to extend with different missing value analysis strategies
# (e.g., advanced reporting, statistical imputation diagnostics, or custom visualizations)
# without modifying the core workflow structure.


class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame):
        print("\nVisualizing Missing Values...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("/Users/m1/Desktop/ house-pricing-mlops-pipeline/extracted_data/AmesHousing.csv")
    simple_missing_analysis= SimpleMissingValuesAnalysis()
    simple_missing_analysis.analyze(df)